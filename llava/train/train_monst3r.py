'''
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
torchrun --nproc_per_node=6 \
    llava/train/train_monst3r.py  \
    --model_name_or_path=ChaimZhu/LLaVA-3D-7B  \
    --version=v1 \
    --mm_projector_type=mlp2x_gelu \
    --learning_rate 5e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --output_dir=./combined_mlp_no_time_obj_cam \
    --data_path=/data1/syh_data/Datasets/combined_dataset  \
    --vision_tower=clip  \
    --video_tower=clip  \
    --freeze_backbone  \
    --tune_mm_mlp_adapter  \
    --tune_video_tower  \
    --tune_vision_resampler  \
    --mm_vision_select_layer=-2 \
    --mm_use_im_start_end=False \
    --mm_use_im_patch_token=False \
    --num_train_epochs=2  \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps=100  \
    --logging_steps=10  \
    --eval_steps=100  \
    --save_total_limit 10 \
    --lr_scheduler_type "cosine" \
    --model_max_length 5000 \
    --bf16  \
    --dataloader_drop_last True \
    --gradient_checkpointing  \
    --gradient_accumulation_steps 1 \
    --lora_enable \
    --deepspeed scripts/zero2_offload.json \
    --seed 42 \

'''
# from datetime import timedelta
# import torch.distributed as dist
# dist.init_process_group(timeout=timedelta(hours=1))

from collections import OrderedDict
import random
from filelock import FileLock

import copy
from dataclasses import dataclass, field
import json
import logging
import pickle as pkl
from time import time
from typing import Dict, Optional, Sequence, List
from pathlib import Path
import torch
# Monkey-patch torch to replace dict with OrderedDict
original_module_init = torch.nn.Module.__init__
def patched_module_init(self, *args, **kwargs):
    original_module_init(self, *args, **kwargs)
    super(torch.nn.Module, self).__setattr__('_parameters', OrderedDict())
    # super(torch.nn.Module, self).__setattr__('_buffers', OrderedDict())
    # super(torch.nn.Module, self).__setattr__('_modules', OrderedDict())
torch.nn.Module.__init__ = patched_module_init

import numpy as np
from tqdm import tqdm
import transformers
import tokenizers

from llava.constants import (IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, 
                             DEFAULT_IM_END_TOKEN, DEFAULT_VIDEO_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN, 
                             DEFAULT_VID_START_TOKEN, DEFAULT_VIDEO_PATCH_TOKEN, DEFAULT_VID_END_TOKEN,
                             DEFAULT_LOC_START_TOKEN, DEFAULT_LOC_END_TOKEN, DEFAULT_BOX_TOKEN)
from torch.utils.data import Dataset
from llava.train.llava_trainer import (
    LLaVATrainer, find_all_linear_names, 
    get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3, 
    # maybe_zero_3, get_mm_adapter_state_maybe_zero_3, 
    safe_save_model_for_hf_trainer
)
from llava import conversation as conversation_lib
from llava.model import *
from llava.mm_utils import tokenizer_image_token, map_obj, PlainBoxFormatter, tokenizer_special_token
from PIL import Image

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


from packaging import version
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="ChaimZhu/LLaVA-3D-7B")
    version: Optional[str] = field(default="v1")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-2)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=False)
    mm_patch_merge_type: Optional[str] = field(default='flat')
    mm_vision_select_feature: Optional[str] = field(default="patch")

    # ===================================================================
    video_tower: Optional[str] = field(default=None)
    tune_video_tower: bool = field(default=False)
    num_frames: int = 16
    num_sample_tokens: int = 1024
    # ===================================================================


@dataclass
class DataArguments:
    lazy_preprocess: bool = True
    is_multimodal: bool = True
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    # ===================================================================
    data_path: Optional[List[str]] = field(default=None, metadata={"help": "Path to the training data."})
    video_folder: Optional[str] = field(default=None)
    # ===================================================================


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    tune_vision_resampler: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")

    dataloader_num_workers: int = 0
    dataloader_pin_memory: bool = True
    dataloader_drop_last: bool = True
    if dataloader_num_workers >= 1:
        # number of samples preloaded == dataloader_prefetch_factor * batch_size 
        dataloader_prefetch_factor : int = 2  
        dataloader_persistent_workers : bool = True

    # will be overwrited by input args
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 1  # global batch size = num_gpus * per_device_train_batch_size * gradient_accumulation_steps

    model_max_length: int = field(
        default=4096,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    gradient_checkpointing: bool = field(
        default=True,
        metadata={"help": "Whether to use gradient checkpointing."}
    )
    gradient_checkpointing_kwargs: Dict = field(
        default_factory=lambda: {"use_reentrant": False},  # should be true for Zero3
        metadata={"help": "Keyword arguments for gradient checkpointing. Should be True for Zero3 and False for Zero2."}
    )


    lora_enable: bool = field(default=False)
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = None
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value'] or DEFAULT_VIDEO_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_VIDEO_TOKEN, DEFAULT_IMAGE_TOKEN)
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            # Here we replace the <video> token with <image> token to reduce the coding 
            replace_token, video_replace_token = DEFAULT_IMAGE_TOKEN, DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end: # false
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)
    return sources


def preprocess_target_prompts(
    sources: Sequence[str],
    targets: Sequence,
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for idx, source in enumerate(sources):
        target = targets[idx]
        if target is not None and 'boxes' in target:
            boxes = target['boxes']
            clicks = []
            for box in boxes:
                if len(box) == 6:
                    delta_x = np.random.uniform(-box[3]/2, box[3]/2)
                    delta_y = np.random.uniform(-box[4]/2, box[4]/2)
                    delta_z = np.random.uniform(-box[5]/2, box[5]/2)
                    click = [box[0]+delta_x, box[1]+delta_y, box[2]+delta_z]
                    click = [round(coord, 3) for coord in click]
                elif len(box) == 9:
                    click = [round(coord, 3) for coord in box[:3]]
                clicks.append(click)
        else:
            clicks = []
        for sentence in source:
            words = sentence['value']
            boxes_seq = sentence.get('boxes_seq', None)
            if boxes_seq is not None:
                boxes = boxes_seq[0]
                objs_num = len(boxes)
                obj_placeholder =  DEFAULT_BOX_TOKEN + ', '
                objs_str = obj_placeholder * objs_num
                objs_str = objs_str.rstrip(', ')
                converted = words.replace(DEFAULT_BOX_TOKEN, objs_str)
                words = converted
            if boxes_seq is not None:
                sentence['value'] = words
    return sources, clicks


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        # print(f"source: {source}") 
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())   # list of combination conversation(including <image>/n)

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_special_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_special_token(rou, tokenizer))
                instruction_len = len(tokenizer_special_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )




def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    else:
        raise NotImplementedError(f"Unsupported version: {conversation_lib.default_conversation.version}")


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        data_path = data_path[0]
        self.tokenizer: transformers.PreTrainedTokenizer = tokenizer
        self.data_args = data_args
        self.scene_cache = OrderedDict()  # 非共享内存的版本
        self.current_rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        rank0_print("Formatting inputs...Skip in lazy mode")
        self.list_data_dict = self.load_json(data_path)
        assert len(self.list_data_dict) > 0, f"Empty dataset: {data_path}"
        rank0_print(f"Loaded {len(self.list_data_dict)} samples from {data_path}")
        self.local_offset = len(self.list_data_dict) // world_size
        print(f"local_offset: {self.local_offset}, current_rank: {self.current_rank}, world_size: {world_size}")
        self.load_all_scenes_for_rank()

    def load_json(self, data_folder, qa_per_item=500, max_conv_len=2500, add_reasoning=False):
        list_data_dict = []
        all_token_length = []

        if add_reasoning: # use new reasoning data
            train_folder = Path(data_folder, 'train_reasoning_new')
        else:
            train_folder = Path(data_folder, 'train_reasoning')
        rank0_print(f"Loading data from {train_folder}")
            
        for json_path in train_folder.glob('*.json'):
            with open(json_path, 'r') as f:
                qa_data = json.load(f)
            random.seed(42)
            random.shuffle(qa_data)
            if add_reasoning and ('capture' not in json_path.name.lower()): # sample 50% of reasoning data
                qa_data = qa_data[:len(qa_data)//2]
            npz_path = Path(data_folder, "clip_new") / json_path.name.replace('_qa_pairs.json', '_scene_data.npz')
            if not npz_path.exists():
                raise FileNotFoundError(f"Scene data not found: {npz_path}")

            conversations = []
            current_length = 0

            def flush():
                nonlocal current_length
                if conversations:
                    video_template = {
                        "id": -1,
                        "video": npz_path,
                        "conversations": conversations.copy(),
                    }
                    list_data_dict.append(video_template)
                    all_token_length.append(current_length)
                    conversations.clear()
                    current_length = 0

            for qa_pair in qa_data:
                q:str = qa_pair["question"]
                a:str = qa_pair["answer"]

                if add_reasoning:
                    r:str = qa_pair["reasoning"]
                    q += q + ' Please think step by step before answering.'
                    a = '<think>\n' + r + '\n</think>\n<answer>\n' + a + '\n</answer>'

                # 用tokenizer计算长度
                qa_len = len(self.tokenizer.encode(q)) + len(self.tokenizer.encode(a))

                # 如果单条 QA 本身就超限，先冲刷已有对话，再把它单独成一组
                if qa_len >= max_conv_len:
                    # 冲刷已有对话（如果有的话）
                    flush()
                    # 把超长 QA 作为一组
                    conversations.extend([
                        {"from": "human", "value": f"<video>\n{q}"},
                        {"from": "gpt",   "value": a}
                    ])
                    # print(f"QA pair too long {qa_len} tokens")
                    # 立即冲刷，保证它独立成组
                    flush()
                    continue

                # 普通 QA：如果加进去会超限，就先冲刷
                if current_length + qa_len >= max_conv_len:
                    flush()

                # 加入当前对话
                if not conversations:
                    conversations.extend([
                        {"from": "human", "value": f"<video>\n{q}"},
                        {"from": "gpt",   "value": a}
                    ])
                else:
                    conversations.extend([
                        {"from": "human", "value": q},
                        {"from": "gpt",   "value": a}
                    ])
                current_length += qa_len

                # 达到条数或长度上限，也冲刷
                if len(conversations) >= qa_per_item * 2 or current_length >= max_conv_len:
                    flush()

            # 循环结束后冲刷剩余（此时长度 < max_conv_len）
            flush()
            # break  # debug
        rank0_print(f"Loaded max_token_length {max(all_token_length)}")
        # resample 20% of the data for training
        # list_data_dict = random.sample(list_data_dict, int(len(list_data_dict) * 0.2))
        return list_data_dict
         
    def load_scene(self, scene_file, max_cache_size=60):
        if scene_file in self.scene_cache:
            scene = self.scene_cache[scene_file]
            self.scene_cache.move_to_end(scene_file)
            return scene
        
        ## 读取场景数据, 避免IO阻塞
        with FileLock(str(scene_file)+".lock"):
            print("loading scene data from", scene_file)
            valid_scene_data = np.load(scene_file, allow_pickle=True)
            img_timestamps_ns = valid_scene_data["img_timestamps_ns"]
            valid_features = valid_scene_data["valid_features"]
            valid_points = valid_scene_data["valid_points"]
            valid_obj_embedding = valid_scene_data["valid_obj_embedding"]
            valid_timestamps = valid_scene_data["valid_timestamps"]
            voxel_indices = valid_scene_data["voxel_indices"].item()

            pts_max = valid_scene_data["pts_max"] 
            pts_min = valid_scene_data["pts_min"]
            intrinsic = valid_scene_data["intrinsic"]
            # extrinsic = {
            #     "translation":frame['sensor']["translation"],
            #     "rotation":frame['sensor']["rotation"]
            # }
            extrinsic = valid_scene_data["extrinsic"]

        videos_timestamps = torch.tensor(img_timestamps_ns)
        videos_points_features = torch.from_numpy(valid_features)
        videos_points = torch.from_numpy(valid_points)
        videos_objects_embeddings = torch.from_numpy(valid_obj_embedding)
        videos_points_timestamps = [torch.from_numpy(timestamps) for timestamps in valid_timestamps]
        extrinsic = torch.tensor([ex['translation']+ex['rotation'] for ex in extrinsic])  # (3+4,)

        count_points = sum(len(v) for v in voxel_indices.values())
        assert count_points == len(videos_points), f"count_points: {count_points}, len(videos_points): {len(videos_points)}"

        trajactory_prompts = None

        # 缓存控制（FIFO 机制）
        if len(self.scene_cache) > max_cache_size:
            tmp = self.scene_cache.popitem(last=False)
            print("remove scene data from", tmp[0])
            del tmp

        scene = (videos_timestamps, videos_points_features, videos_points, videos_objects_embeddings, 
                videos_points_timestamps, trajactory_prompts, voxel_indices, extrinsic)
        self.scene_cache[scene_file] = scene
        return scene
    

    def load_all_scenes_for_rank(self):
        """
        依据 __getitem__ 里 i 映射规则，预加载该进程(rank)负责的所有视频场景文件，
        并缓存到 self.scene_cache 中。
        """
        if self.current_rank == -1:
            print("Warning: current_rank==-1, no distributed rank info")
            return

        print(f"Rank {self.current_rank} loading scenes with local_offset={self.local_offset}")

        # 先清空缓存
        self.scene_cache.clear()

        # 计算该rank负责的全局索引范围
        # i = i % local_offset + current_rank * local_offset
        # i 的范围在 [current_rank * local_offset, (current_rank+1)*local_offset)
        start_idx = self.current_rank * self.local_offset
        end_idx = min((self.current_rank + 1) * self.local_offset, len(self.list_data_dict))

        for idx in tqdm(range(start_idx, end_idx), desc=f"Rank {self.current_rank} loading scenes"):
            sample = self.list_data_dict[idx]
            scene_file = sample.get("video", None)
            if scene_file is None:
                continue
            # 避免重复加载
            if scene_file not in self.scene_cache:
                self.load_scene(scene_file)

        print(f"Rank {self.current_rank} loaded scenes from idx {start_idx} to {end_idx} into cache.")



    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        raise NotImplementedError
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if ('imgs' in sample or 'video' in sample) else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        time1 = time()
        # 重新映射所取的i，根据local_rank, 为每个进程减少cache的场景
        if self.current_rank != -1: # distributed training
            i = i % self.local_offset + self.current_rank * self.local_offset
            if i >= len(self.list_data_dict):
                raise IndexError(f"Index {i} out of range for rank {self.current_rank}")

        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "wrapped to a list"
        
        if 'video' in sources[0]:
            targets = [e.get("target", None) for e in sources]
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
            sources, clicks = preprocess_target_prompts(sources, targets, self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('images' in self.list_data_dict[i] or 'video' in self.list_data_dict[i]))
    
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # add point cloud data into data_dict
        if 'video' in self.list_data_dict[i]:
            data_dict['clicks'] = clicks  # list of list 
            (
            videos_timestamps,videos_points_features,videos_points,videos_objects_embeddings,
            videos_points_timestamps,trajactory_prompts,voxel_indices, extrinsic
            ) = self.load_scene(self.list_data_dict[i]["video"])
            
            data_dict['videos_timestamps'] = videos_timestamps
            data_dict['videos_points_features'] = videos_points_features
            data_dict['videos_points'] = videos_points
            data_dict['videos_objects_embeddings'] = videos_objects_embeddings
            data_dict['videos_points_timestamps'] = videos_points_timestamps
            data_dict['trajactory_prompts'] = trajactory_prompts
            data_dict['voxel_indices'] = voxel_indices
            data_dict['videos_extrinsic'] = extrinsic  # (3+4,)

        time2 = time()
        # print(f"get data item {i}, time:{time2-time1}, rank:{self.current_rank}")
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        time1 = time()
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        
        instances = [instance for instance in instances if 'videos_points' in instance]  # find the point cloud
        if len(instances) > 0:  # exist point cloud data in the batch data
            videos_timestamps = [t['videos_timestamps'] for t in instances]
            videos_points_features = [t['videos_points_features'] for t in instances]
            videos_points = [t['videos_points'] for t in instances]
            videos_objects_embeddings = [t['videos_objects_embeddings'] for t in instances]
            videos_points_timestamps = [t['videos_points_timestamps'] for t in instances]
            trajactory_prompts = [t['trajactory_prompts'] for t in instances]
            voxel_indices = [t['voxel_indices'] for t in instances]
            videos_extrinsic = [t['videos_extrinsic'] for t in instances]  # (batch_size, n frames, 3+4)

            batch['videos_timestamps'] = videos_timestamps  # (batch_size, vary num_frames)
            batch['videos_points_features'] = videos_points_features  # (batch_size, vary num_points, clip_dim)
            batch['videos_points'] = videos_points  # (batch_size, vary num_points, 3)
            batch['videos_objects_embeddings'] = videos_objects_embeddings  # (batch_size, vary num_points, embd_dim)
            batch['videos_points_timestamps'] = videos_points_timestamps  # (batch_size, vary num_points, [vary num of time])
            batch['trajactory_prompts'] = trajactory_prompts  # (batch_size, None)
            batch['voxel_indices'] = voxel_indices  # dict
            batch['videos_extrinsic'] = videos_extrinsic  # (batch_size, n frames, 3+4)

            clicks = []
            for instance in instances:
                clicks.extend(instance['clicks'])
            clicks = torch.tensor(clicks)  # (num_clicks, 3)
            if clicks.numel() != 0:  # valid tensor
                batch['clicks'] = clicks  # (num_clicks, 3)
        time2 = time()
        # print(f"collate time: {time2-time1}", "getting batch data", batch.keys())
        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def train(attn_implementation=None):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))


    # ==========================================================================
    if model_args.vision_tower is not None or model_args.video_tower is not None:
    # ==========================================================================
        if 'mpt' in model_args.model_name_or_path:
            raise NotImplementedError("MPT model is not supported for now, please use Llava3dForMonst3r instead.")
        else:
            model = Llava3dForMonst3r.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                **bnb_model_from_pretrained_args,
            )
            model.vision_resampler.post_init()
    else:
        raise NotImplementedError(f"Unsupported model: {model_args.model_name_or_path}")
    model.config.use_cache = False
    # =============================================================================
    model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_projector_lr = training_args.mm_projector_lr
    training_args.use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
    model.config.num_frames = model_args.num_frames
    model.config.num_sample_tokens = model_args.num_sample_tokens
    # =============================================================================


    if model_args.freeze_backbone:
        model.model.requires_grad_(False)
        model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model, PeftModel
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)

        # 加载已有权重
        if training_args.lora_weight_path is not None:
            rank0_print("Adding LoRA adapters...")
            model = PeftModel.from_pretrained(
                model,
                training_args.lora_weight_path,
                # torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                # device_map={"": training_args.device},
                is_trainable=True,
            )
        else:
            all_linear_names = find_all_linear_names(model)
            rank0_print("LoRA target modules: ", all_linear_names)
            lora_config = LoraConfig(
                r=training_args.lora_r,
                lora_alpha=training_args.lora_alpha,
                target_modules=all_linear_names,
                lora_dropout=training_args.lora_dropout,
                bias=training_args.lora_bias,
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_config)

        # set requires_grad to True for LoRA parameters
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.requires_grad_(True)

    if 'mpt' in model_args.model_name_or_path:
        raise NotImplementedError("MPT model is not supported for now, please use Llava3dForMonst3r instead.")
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            # use_fast=False,
            # padding_side="left",
            use_fast=True,
            trust_remote_code=True,
        )

    if model_args.version != "v1":
        raise NotImplementedError(f"Unsupported version: {model_args.version}")
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    # ======================================================================================
    if model_args.vision_tower is not None or model_args.video_tower is not None:
        # load vision tower model
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )
        
        if model_args.vision_tower is not None:
            vision_tower = model.get_vision_tower()
            # to gpu device
            vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

            data_args.image_processor = vision_tower.image_processor
            data_args.is_multimodal = True
    
        if model_args.video_tower is not None:
            video_tower = model.get_video_tower()  # class not str
            video_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)
            promp_encoder = model.get_prompt_encoder()
            promp_encoder.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

            data_args.video_processor = video_tower.video_processor
            data_args.is_multimodal = True
            data_args.box_processor = PlainBoxFormatter()
    # ======================================================================================

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter

        if model_args.tune_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad_(True)
        # =========================================================================
        if model_args.tune_video_tower:
            for p in model.get_model().video_tower.parameters():
                p.requires_grad_(True)

        # model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        model.config.freeze_mm_mlp_adapter = not model_args.tune_mm_mlp_adapter
        
        if training_args.tune_vision_resampler:
            model.vision_resampler.requires_grad_(True)
        else:
            model.vision_resampler.requires_grad_(False)
        # =========================================================================                              

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)
    
    total_params = sum(p.ds_numel if hasattr(p, "ds_numel") else p.numel() for p in model.parameters())
    trainable_params = sum(p.ds_numel if hasattr(p, "ds_numel") else p.numel() for p in model.parameters() if p.requires_grad)
    rank0_print(f"Total parameters: ~{total_params/1e6:.2f} MB)")
    rank0_print(f"Trainable parameters: ~{trainable_params/1e6:.2f} MB)")
    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    # this is a dict
    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)
    trainer = LLaVATrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    **data_module)

    # checkpoint_paths = list(Path(training_args.output_dir).glob("checkpoint-*"))
    if training_args.lora_weight_path is not None:
        # checkpoint_paths.sort(key=lambda x: int(x.name.split("-")[-1]))
        # model_path = checkpoint_paths[-1]
        print(f"Loading checkpoint from {training_args.lora_weight_path}")
        mm_projector_weights = torch.load(Path(training_args.lora_weight_path, 'non_lora_trainables.bin'), map_location='cpu')
        mm_projector_weights = {k: v.to(torch.bfloat16) for k, v in mm_projector_weights.items()}
        keys_not_loaded = model.load_state_dict(mm_projector_weights, strict=False)
        # print("missing keys:", keys_not_loaded.missing_keys)
        assert len(keys_not_loaded.unexpected_keys) == 0, f"Unexpected keys: {keys_not_loaded.unexpected_keys}"
            
        trainer.train()

        # trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    
    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, Path(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
