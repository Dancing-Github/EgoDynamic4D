import math
import sys
from typing import List, Optional, Tuple, Union
from time import time
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    CLIPVisionModel,
    LlamaConfig,
    LlamaModel,
    LlamaForCausalLM,
)

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower
from llava.model.multimodal_encoder.video_encoder import RGBDVideoTower

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from llava.constants import (
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    LOC_TOKEN_INDEX,
)

import pickle as pkl
import numpy as np
from pathlib import Path

from llava.model.language_model.show_scene import (
    align_voxel_grid_with_clip,
    get_transform_matrix,
    load_point_cloud,
    get_pixel_feature,
)


class LlavaConfig(LlamaConfig):
    model_type = "llava_monst3r"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)

class PoseCompressor(nn.Module):
    def __init__(self, hidden_dim=1024, out_dim=1024, num_tokens=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(7, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
        self.num_tokens = num_tokens
        # Define and initialize nn.Parameter
        self.query_embd = nn.Parameter(torch.empty(num_tokens, out_dim))
        nn.init.normal_(self.query_embd, mean=0.0, std=0.02)
        self.compress_attn = nn.MultiheadAttention(embed_dim=out_dim, num_heads=8, batch_first=True)

    def forward(self, poses):  # poses: (N, 7)
        e = self.mlp(poses).unsqueeze(0)  # (1, N, out_dim)
        q = self.query_embd.unsqueeze(0)       # (1, M, out_dim)
        compressed, _ = self.compress_attn(q, e, e)  # (1, M, out_dim)
        assert not torch.isnan(compressed).any(), "NaN detected in compressed camera token!"
        return compressed.squeeze(0)        # (M, out_dim)

    def post_init(self):
        print("Initializing PoseCompressor...")
        # Initialize query_embd
        nn.init.normal_(self.query_embd, mean=0.0, std=0.02)
        
        for module in self.modules():
            if isinstance(module, nn.MultiheadAttention):
                module._reset_parameters()
            elif isinstance(module, nn.Sequential):
                for submodule in module:
                    if isinstance(submodule, nn.Linear):
                        submodule.reset_parameters()


class VisionResampler(nn.Module):
    OBJECT_ID_EMBD_SIZE = 8
    ABLISION_STUDY = 'mlp_attention'
    DOWNSAMPLE_METHOD = 'random'  # 'octree' | 'voxel' | 'random'

    def __init__(self, identity_embedding_dim=OBJECT_ID_EMBD_SIZE, hidden_dim=1024, dropout=0.1):
        super().__init__()
        self.clip_dim = 1024
        
        if self.ABLISION_STUDY == 'mlp_no_time_obj_cam':
            self.camera_encoder = None
            self.object_resampler = None
            self.fusion_mlp = None
        elif self.ABLISION_STUDY == 'mlp_no_obj_cam':
            self.camera_encoder = None
            self.object_resampler = None
            self.fusion_mlp = nn.Sequential(
                nn.Linear(self.clip_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, self.clip_dim),
            )
        elif self.ABLISION_STUDY == 'mlp_no_cam':
            self.camera_encoder = None
            self.object_resampler = nn.Linear(identity_embedding_dim, self.clip_dim)
            self.fusion_mlp = nn.Sequential(
                nn.Linear(self.clip_dim * 3, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, self.clip_dim),
            )
        elif self.ABLISION_STUDY == 'mlp_attention_no_cam':
            self.attn_dim = 1024
            self.camera_encoder = None
            self.position_embd_proj = nn.Linear(self.clip_dim, self.attn_dim)
            self.object_resampler = nn.Linear(identity_embedding_dim, self.attn_dim)
            self.fusion_attention = nn.MultiheadAttention(embed_dim=self.attn_dim, num_heads=8, batch_first=True)
            self.fusion_mlp = nn.Sequential(
                nn.Linear(self.attn_dim * 3, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, self.clip_dim),
            )
        elif self.ABLISION_STUDY == 'mlp_attention':
            self.attn_dim = 1024
            self.camera_encoder = PoseCompressor(hidden_dim, self.clip_dim, num_tokens=8)
            self.position_embd_proj = nn.Linear(self.clip_dim, self.attn_dim)
            self.object_resampler = nn.Linear(identity_embedding_dim, self.attn_dim)
            self.fusion_attention = nn.MultiheadAttention(embed_dim=self.attn_dim, num_heads=8, batch_first=True)
            self.fusion_mlp = nn.Sequential(
                nn.Linear(self.attn_dim * 3, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, self.clip_dim),
            )
        else:
            raise ValueError(f"Invalid ABLISION_STUDY value: {self.ABLISION_STUDY}")

        self.fusion_alpha = 0.5
        # self.div_term = None

    def post_init(self):
        print("Initializing VisionResampler...")
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()
            elif isinstance(module, nn.MultiheadAttention):
                module._reset_parameters()
            elif isinstance(module, nn.Sequential):
                for submodule in module:
                    if isinstance(submodule, nn.Linear):
                        submodule.reset_parameters()
        if self.camera_encoder is not None:
            self.camera_encoder.post_init()

    def sinusoidal_time_embedding(self, t, dim):
        """
        计算 Sinusoidal 时间嵌入。
        
        Args:
            t: Tensor of shape (total_timestamps,) - 展平后的时间戳
            dim: int - 嵌入维度，默认为 1024
        Returns:
            Tensor of shape (total_timestamps, 1024) - Sinusoidal 编码
        """
        device = t.device
        dtype = t.dtype
        position = t.unsqueeze(-1)
        # if self.div_term is None:
        #     self.div_term = torch.exp(
        #         torch.arange(0, dim, 2, device=device, dtype=dtype) * -(math.log(10000.0) / dim)
        #     )
        div_term = torch.exp(
            torch.arange(0, dim, 2, device=device, dtype=dtype) * -(math.log(10000.0) / dim)
        )
        
        sinusoid = torch.zeros(t.size(0), dim, device=device, dtype=dtype)
        sinusoid[:, 0::2] = torch.sin(position * div_term)
        sinusoid[:, 1::2] = torch.cos(position * div_term)
        return sinusoid

    def encode_valid_timestamps(self, valid_timestamps, device, dtype, dim):
        """
        处理 NestedTensor 时间戳并生成聚合嵌入，输出按原始点顺序排列。优化版本：全局去重时间戳以减少计算量。
        
        Args:
            valid_timestamps: NestedTensor, 形状为 (num_points, num_occurs), num_occurs 可变
            device: torch.device，计算设备
            dtype: torch.dtype，数据类型
            dim: int - 嵌入维度，默认为 1024
        Returns:
            Tensor, shape (num_points, 1024)，每个点的聚合时间嵌入，按原始顺序
        """
        assert isinstance(valid_timestamps, torch.Tensor) and valid_timestamps.is_nested, \
            "Input must be a NestedTensor"
        num_points = valid_timestamps.size(0)
        flat_timestamps = valid_timestamps.values()
        flat_timestamps = flat_timestamps.to(device=device, dtype=dtype)
        if flat_timestamps.numel() == 0:
            zero_emb = torch.zeros(dim, device=device, dtype=dtype)
            return zero_emb.expand(num_points, -1)

        lengths = torch.tensor([ts.size(0) for ts in valid_timestamps], device=device)
        point_indices = torch.arange(num_points, device=device).repeat_interleave(lengths)

        max_emb = torch.full((num_points, dim), float('-inf'), device=device, dtype=dtype)
        sum_emb = torch.zeros(num_points, dim, device=device, dtype=dtype)
        total_lengths = torch.zeros(num_points, device=device, dtype=torch.long)

        total_timestamps_num = flat_timestamps.size(0)
        fake_one = torch.ones_like(point_indices, dtype=torch.long)

        time1 = time()
        unique_timestamps, inverse_indices = torch.unique(flat_timestamps, return_inverse=True)
        sinusoid_unique = self.sinusoidal_time_embedding(unique_timestamps, dim)
        assert inverse_indices.size(0) == point_indices.size(0), \
            f"indices size not match {inverse_indices.size(0)} != {point_indices.size(0)}"

        for i in range(unique_timestamps.size(0)):
            mask = (inverse_indices == i)
            curr_point_indices = point_indices[mask]
            if curr_point_indices.numel() > 0:
                curr_embedding = sinusoid_unique[i].unsqueeze(0)
                max_emb[curr_point_indices] = torch.maximum(max_emb[curr_point_indices], curr_embedding)
                sum_emb.index_add_(0, curr_point_indices, curr_embedding.expand(curr_point_indices.size(0), -1))
                total_lengths.index_add_(0, curr_point_indices, fake_one[:curr_point_indices.size(0)])

        time2 = time()

        max_emb = max_emb.masked_fill(total_lengths.unsqueeze(-1) == 0, 0.0)
        sum_emb = sum_emb.masked_fill(total_lengths.unsqueeze(-1) == 0, 0.0)
        avg_emb = sum_emb / total_lengths.unsqueeze(-1).clamp(min=1)
        
        time_emb = self.fusion_alpha * max_emb + (1 - self.fusion_alpha) * avg_emb
        assert not torch.isnan(time_emb).any(), "NaN detected!"
        return time_emb
    
    def encode_object_id(self, valid_obj_embedding):
        object_features = self.object_resampler(valid_obj_embedding)
        return object_features

    def forward(self, valid_obj_embedding, valid_timestamps, pos_feat):
        device = valid_obj_embedding.device
        dtype = valid_obj_embedding.dtype

        if self.ABLISION_STUDY == 'mlp_no_obj_cam':
            time_feat = self.encode_valid_timestamps(valid_timestamps, device, dtype, dim=self.clip_dim)
            fusion_input = torch.cat([time_feat, pos_feat], dim=-1)
            fused_features = self.fusion_mlp(fusion_input)
        elif self.ABLISION_STUDY == 'mlp_no_cam':
            obj_feat = self.encode_object_id(valid_obj_embedding)
            time_feat = self.encode_valid_timestamps(valid_timestamps, device, dtype, dim=self.clip_dim)
            fusion_input = torch.cat([obj_feat, time_feat, pos_feat], dim=-1)
            fused_features = self.fusion_mlp(fusion_input)
        elif self.ABLISION_STUDY == 'mlp_attention_no_cam' or self.ABLISION_STUDY == 'mlp_attention':
            obj_feat = self.encode_object_id(valid_obj_embedding)
            time_feat = self.encode_valid_timestamps(valid_timestamps, device, dtype, dim=self.attn_dim)
            fused_features = self.process_attention_and_proj(pos_feat, obj_feat, time_feat)
        elif self.ABLISION_STUDY == 'mlp_no_time_obj_cam':
            fused_features = pos_feat
        else:
            raise ValueError(f"Invalid ABLISION_STUDY value: {self.ABLISION_STUDY}")

        assert not torch.isnan(fused_features).any(), "NaN detected!"
        return fused_features
        
    def process_attention_and_proj(self, pos_feat, obj_feat, time_feat):
        """
        Returns:
            object_features: 形状 (N, 1024)
        """
        pos_feat = self.position_embd_proj(pos_feat)
        context_feat = torch.stack([pos_feat, obj_feat, time_feat], dim=1)
        attn_output, _ = self.fusion_attention(
            query=context_feat,
            key=context_feat,
            value=context_feat
        )
        assert not torch.isnan(attn_output).any(), "NaN detected in attn_output!"
        attn_output = attn_output.flatten(1)
        attn_output = self.fusion_mlp(attn_output)
        return attn_output



class Llava3dForMonst3r(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # vision_resampler encodes object_id and time(t) to clip feature size
        self.vision_resampler = VisionResampler()

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: List[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        videos_timestamps=None,
        videos_points_features=None,
        videos_points=None,
        videos_objects_embeddings=None,
        videos_points_timestamps=None,
        trajactory_prompts=None,
        voxel_indices=None,
        videos_extrinsic=None,
        clicks: List[List[float]] | None = None,
        return_dict: bool | None = None,
        **kwargs
    ):
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                videos_timestamps,
                videos_points_features,
                videos_points,
                videos_objects_embeddings,
                videos_points_timestamps,
                trajactory_prompts,
                voxel_indices,
                videos_extrinsic,
                clicks=clicks,
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: torch.Tensor,
        videos_timestamps,
        videos_points_features,
        videos_points,
        videos_objects_embeddings,
        videos_points_timestamps,
        trajactory_prompts,
        voxel_indices = None,
        videos_extrinsic=None,
        clicks: List[List[float]] | None = None,
        **kwargs,
    ):
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        assert getattr(self.config, "tokenizer_padding_side")== "left", \
            "Llava3dForMonst3r only supports left padding in generation."
        
        if videos_points is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _) = (
                self.prepare_inputs_labels_for_multimodal(
                    inputs,
                    position_ids,
                    attention_mask,
                    None,
                    None,
                    videos_timestamps,
                    videos_points_features,
                    videos_points,
                    videos_objects_embeddings,
                    videos_points_timestamps,
                    trajactory_prompts,
                    voxel_indices,
                    videos_extrinsic,
                    clicks=clicks,
                )
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    def handle_3d_input(self, voxel_grid: dict, frames_extrinsic, max_tokens=2560):
        pc_point, pc_feature = [], []
        for k, v in voxel_grid.items():
            if v["feature"].any():  # skip all zero feature
                # pc_point.append(torch.tensor(k))
                pc_point.append(v["avg_point"])
                pc_feature.append(v["feature"])

        pc_point = torch.stack(pc_point, dim=0)  # (N, 3)
        pc_feature = torch.stack(pc_feature, dim=0)  # (N, 1024)

        # cut off the feature N to max_tokens, following the same logic as in the video
        if pc_point.shape[0] > max_tokens:
            print(f"cut off the feature from {pc_point.shape[0]} to {max_tokens}",
                pc_point.shape, pc_feature.shape,)  # (2560, 3) (2560, 1024)
            indices = torch.randperm(pc_feature.size(0))[:max_tokens]
            indices, _ = torch.sort(indices)
            pc_point = pc_point[indices]
            pc_feature = pc_feature[indices]

        if self.vision_resampler.camera_encoder is not None:
            # concatenate extrinsic feature as new tokens
            frames_extrinsic = frames_extrinsic.to(self.device, dtype=self.dtype)  # (num frames, 3+4,)
            extrinsic_feature = self.vision_resampler.camera_encoder(frames_extrinsic)  # (num cam token, 1152)
            pc_feature = torch.cat([pc_feature, extrinsic_feature], dim=0)  # (N + num cam token, 1152)

        pc_feature = self.get_model().mm_projector(pc_feature)  # (N, 4096)
        return pc_feature

    def prepare_inputs_labels_for_multimodal(
        self,
        input_ids,
        position_ids,
        attention_mask,
        past_key_values,
        labels,
        videos_timestamps,
        videos_points_features,
        videos_points,
        videos_objects_embeddings,
        videos_points_timestamps,
        trajactory_prompts,
        voxel_indices,
        videos_extrinsic,
        clicks=None,
    ):

        # import pdb; pdb.set_trace()
        if self.get_vision_tower() is None or videos_timestamps is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels
        
        # import pdb; pdb.set_trace()
        # print("batch_size", input_ids.shape, len(videos_timestamps))
        ##### TODO: point cloud input ###################################################
        clip_vision_tower: CLIPVisionTower = self.get_vision_tower()  # ClipVisionTower
        if isinstance(clip_vision_tower.vision_tower, CLIPVisionModel):
            # delete clip_vision_tower to save memory
            old_model = clip_vision_tower.vision_tower
            # 安全删除模型参数数据
            for param in old_model.parameters():
                param.data = torch.empty(0, dtype=param.dtype, device=param.device)
            for buffer in old_model.buffers():
                buffer.data = torch.empty(0, dtype=param.dtype, device=param.device)
            # 替换为 Identity
            del old_model
            clip_vision_tower.vision_tower=nn.Identity().to(self.device)
            torch.cuda.empty_cache()
            print("CLIP vision model deleted and replaced.")
        # print("clip_vision_tower", clip_vision_tower)
        rgbd_video_tower: RGBDVideoTower = self.get_video_tower()  # RGBDVideoTower
        # print("rgbd_video_tower", rgbd_video_tower)
        pos_encoder = rgbd_video_tower.video_tower.positional_embedding.position_embedding_head
        # print("pos_encoder", pos_encoder) # in_features=3, out_features=1024
        prompt_encoder = rgbd_video_tower.prompt_encoder
        # print("prompt_encoder", prompt_encoder) # in_features=3, out_features=1024

        pc_features = []
        # self.vision_resampler.DOWNSAMPLE_METHOD # 可选 'octree' | 'voxel' | 'random'
        downsample_method = self.vision_resampler.DOWNSAMPLE_METHOD.lower()
        assert downsample_method in ['octree', 'voxel', 'random']
        print(f"[Info] Downsample method: {downsample_method}")

        for img_timestamps_ns, valid_features, valid_points, valid_obj_embedding, valid_timestamps, traj_prompt, octree_indice, frames_extrinsic, in zip(
            videos_timestamps, videos_points_features, videos_points, videos_objects_embeddings, videos_points_timestamps, trajactory_prompts, voxel_indices, videos_extrinsic, strict=True,
        ):
            if len(pc_features) > 0 and (valid_features == videos_points_features[0]).all():
                pc_features.append(pc_features[0])
                continue
                
            # time1 = time()
            voxel_grid = align_voxel_grid_with_clip(
                img_timestamps_ns,
                valid_features,
                valid_points,
                valid_obj_embedding,
                valid_timestamps,
                traj_prompt,
                pos_encoder,
                self.vision_resampler,
                device=self.device,
                dtype=self.dtype,
                octree_indice=octree_indice,
                # downsample_method="octree", # 可选 'octree' | 'voxel' | 'random'
                downsample_method=downsample_method,
                voxel_size=0.7 if downsample_method=="voxel" else 0.5, # 体素尺寸，单位米
            )
            # time2 = time()
            # print("align_voxel_grid_with_clip time:", time2 - time1)
            
            # {
            #   coordinate(tuple(int)):
            #     {"points_num": num_point(float),
            #      "feature": feature(torch.tensor),
            #      "avg_point":numpy.array},
            #   ...
            # }
            pc_feature = self.handle_3d_input(voxel_grid, frames_extrinsic, max_tokens=1600)  # 3D input batch_size*[num_points, 4096]
            print("pc_feature", pc_feature.shape)  # torch.Size([2817, 4096])
            pc_features.append(pc_feature)
        image_features = pc_features  # test 3D input
        # print("image_features.requires_grad", image_features[0].requires_grad)
        ##########################################################################

        if (
            clicks is None
        ):  # 1. no video data 2. the video data does not contain the click case in the batch
            pseudo_clicks = torch.zeros(
                (0, 3), dtype=self.dtype, device=self.device
            )
            prompt_features = self.encode_prompts(pseudo_clicks)  # (0, 3)
        else:  # 3. part of the video data contain click  4. all the video data contain click
            prompt_features = self.encode_prompts(clicks)  # (click_num, 3)
        # import pdb; pdb.set_trace()
        # print("prompt_features.shape", prompt_features.shape)  # torch.Size([0, 4096])

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(
            self.config, "mm_use_im_start_end", False
        ):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(
                0, input_ids.shape[1], dtype=torch.long, device=input_ids.device
            )
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [
            cur_input_ids[cur_attention_mask]
            for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask, strict=True)
        ]
        labels = [
            cur_labels[cur_attention_mask]
            for cur_labels, cur_attention_mask in zip(labels, attention_mask, strict=True)
        ]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        cur_prompt_idx = 0

        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            # ------------------------------------------------------
            num_prompts = (cur_input_ids == LOC_TOKEN_INDEX).sum()
            num_specials = num_images + num_prompts
            # ------------------------------------------------------

            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat(
                    [cur_input_embeds_1, cur_image_features[0:0]], dim=0
                )
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            # -----------------------------------------------------------------------------
            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[
                0
            ].tolist()
            prompt_token_indices = torch.where(cur_input_ids == LOC_TOKEN_INDEX)[
                0
            ].tolist()
            special_token_indices = sorted(image_token_indices + prompt_token_indices)
            special_tokens = [cur_input_ids[indice] for indice in special_token_indices]
            special_token_indices = (
                [-1] + special_token_indices + [cur_input_ids.shape[0]]
            )
            # -----------------------------------------------------------------------------

            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(special_token_indices) - 1):
                cur_input_ids_noim.append(
                    cur_input_ids[
                        special_token_indices[i] + 1 : special_token_indices[i + 1]
                    ]
                )
                cur_labels_noim.append(
                    cur_labels[
                        special_token_indices[i] + 1 : special_token_indices[i + 1]
                    ]
                )

            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(
                torch.cat(cur_input_ids_noim)
            )
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_specials + 1):  # num_images = 1? [0, 1]
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_specials:
                    # print(f"Batch Index: {batch_idx}\n, Current Image Index: {cur_image_idx}\n, Num Images: {num_images}")
                    special_token = special_tokens[i]
                    if special_token == IMAGE_TOKEN_INDEX:
                        cur_image_features = image_features[cur_image_idx]  # (N, C)
                        cur_image_idx += 1
                        cur_new_input_embeds.append(cur_image_features)
                        cur_new_labels.append(
                            torch.full(
                                (cur_image_features.shape[0],),
                                IGNORE_INDEX,
                                device=cur_labels.device,
                                dtype=cur_labels.dtype,
                            )
                        )
                    elif special_token == LOC_TOKEN_INDEX:
                        cur_prompt_features = prompt_features[cur_prompt_idx].unsqueeze(
                            0
                        )  # (1, C)
                        cur_prompt_idx += 1
                        cur_new_input_embeds.append(cur_prompt_features)
                        cur_new_labels.append(
                            torch.full(
                                (cur_prompt_features.shape[0],),
                                IGNORE_INDEX,
                                device=cur_labels.device,
                                dtype=cur_labels.dtype,
                            )
                        )
                    else:
                        raise NotImplementedError

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            if num_prompts == 0:
                cur_new_input_embeds = torch.cat(
                    [cur_new_input_embeds, prompt_features[0:0]], dim=0
                )

            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(
            self.config, "tokenizer_model_max_length", None
        )
        if tokenizer_model_max_length is not None:
            new_input_embeds = [
                x[:tokenizer_model_max_length] for x in new_input_embeds
            ]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full(
            (batch_size, max_len),
            IGNORE_INDEX,
            dtype=new_labels[0].dtype,
            device=new_labels[0].device,
        )
        attention_mask = torch.zeros(
            (batch_size, max_len),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        position_ids = torch.zeros(
            (batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device
        )

        for i, (cur_new_embed, cur_new_labels) in enumerate(
            zip(new_input_embeds, new_labels, strict=True)
        ):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                            cur_new_embed,
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(
                        0, cur_len, dtype=position_ids.dtype, device=position_ids.device
                    )
            else:
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            cur_new_embed,
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(
                        0, cur_len, dtype=position_ids.dtype, device=position_ids.device
                    )

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None
        # import pdb; pdb.set_trace()
        # if 1: 
        if new_input_embeds.shape[1] > 4096: 
            print(
                "inputs are very long",
                "image_features.shape:", [pc_feature.shape for pc_feature in image_features],
                "new_input_embeds.shape:", new_input_embeds.shape,
                "new_input_embeds requires_grad:", new_input_embeds.requires_grad,
                "vision_resampler mode:", self.vision_resampler.ABLISION_STUDY,
            )  # torch.Size([1, 989, 4096])
        return (
            None,
            position_ids,
            attention_mask,
            past_key_values,
            new_input_embeds,
            new_labels,
        )


AutoConfig.register("llava_monst3r", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, Llava3dForMonst3r)
