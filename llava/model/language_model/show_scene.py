from collections import defaultdict
import json
import random
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation
import pickle
import numpy as np
import torchvision
import trimesh
from PIL import Image
from trimesh.scene.scene import Scene
import torch

import cv2

import numpy as np
import torch

from PIL import Image
from sam2.utils.amg import rle_to_mask, mask_to_rle_pytorch

from tqdm import tqdm, trange
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

import pickle as pkl
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union

import open3d as o3d
from time import time
from torch_scatter import scatter_mean

DEPTH_SCALE = 1000.0  # Depth is saved in 16-bit PNG in millimeters, uint16
OPENGL = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
GRID_SIZE = 16  # Set the grid size, adjust as necessary
CKPT_PATH = Path("./concept-fusion/examples/checkpoints")
CAM_COLORS = [
    (255, 0, 0),
    (0, 0, 255),
    (0, 255, 0),
    (255, 0, 255),
    (255, 204, 0),
    (0, 204, 204),
    (128, 255, 255),
    (255, 128, 255),
    (255, 255, 128),
    (0, 0, 0),
    (128, 128, 128),
]


@dataclass
class ProgramArgs:
    # Torch device to run computation on (E.g., "cpu")
    device: str = "cuda"

    # SAM1 checkpoint and model params
    sam1_checkpoint_path: Union[str, Path] = CKPT_PATH / "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    # SAM2 checkpoint and model params
    sam2_checkpoint_path = "../monst3r/third_party/sam2/checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    # Ignore masks that have valid pixels less than this fraction (of the image area)
    bbox_area_thresh: float = 0.0005
    # Number of query points (grid size) to be sampled by SAM
    points_per_side: int = 32

    # CLIP model config
    open_clip_model = "ViT-H-14"
    open_clip_pretrained_dataset = "laion2b_s32b_b79k"
    cache_dir = CKPT_PATH

    # Directory to save extracted features
    save_dir: str = "saved-feat"


def get_sam2_masks(imagebasedir, basedir, level):
    pklbasedir = Path(basedir,level,'final-output')
    image_path_list = list(Path(imagebasedir).rglob('*.jpg'))
    image_path_list.sort()
    pkl_path_list = list(Path(pklbasedir).rglob('*.pkl'))
    pkl_path_list.sort()
    print([str(p.name) for p in image_path_list], [str(p.name) for p in pkl_path_list])
    assert len(pkl_path_list) == len(image_path_list)  # frame number should be the same
    
    def load_pkl(pkl_path):
        with open(pkl_path, 'rb') as f:
            uncompressed_rle = pkl.load(f)
        mask_list = [rle_to_mask(ur) for ur in uncompressed_rle]
        return np.array(mask_list, dtype=bool)
    npy_list = [load_pkl(p) for p in pkl_path_list]
    image_list = [Image.open(p).convert('RGB') for p in image_path_list]

    for objects_seg, pkl_path in zip(npy_list, pkl_path_list, strict=True):
        print(f"Processing {pkl_path}, shape: {objects_seg.shape}, dtype: {objects_seg.dtype}")



def get_sam1_masks(args, images):
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

    sam = sam_model_registry[args.model_type](checkpoint=Path(args.sam1_checkpoint_path))
    sam.to(device=args.device)
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=args.points_per_side,
        pred_iou_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
    )

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    print("Extracting SAM1 masks...")
    mask_files = []
    for idx in trange(len(images)):
        # img = cv2.imread(imgfile)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = images[idx]  # rgb image
        if img.max() <= 1:
            img = (img * 255).astype(np.uint8)
        masks = mask_generator.generate(img)
        cur_mask = masks[0]["segmentation"]
        _savefile = Path(args.save_dir) / Path(f"{idx:04d}.pkl")
        with open(_savefile, "wb") as f:
            pkl.dump(masks, f)
        mask_files.append(_savefile)
    return mask_files


def get_clip_feature(args, images, mask_files=None):
    import open_clip

    print(
        f"Initializing OpenCLIP model: {args.open_clip_model} pre-trained on {args.open_clip_pretrained_dataset}..."
    )
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.open_clip_model,
        args.open_clip_pretrained_dataset,
        cache_dir=args.cache_dir,
    )
    model.cuda()
    model.eval()

    print("Computing clip pixel-aligned features...")
    feature_files = []
    for idx in trange(len(images)):
        maskfile = mask_files[idx] if mask_files else Path(args.save_dir) / Path(f"{idx:04d}.pkl")
        with open(maskfile, "rb") as f:
            masks = pkl.load(f)
        
        # img = cv2.imread(imgfile)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = images[idx]  # rgb image
        img = (img * 255).astype(np.uint8)
        LOAD_IMG_HEIGHT, LOAD_IMG_WIDTH = img.shape[0], img.shape[1]

        with torch.amp.autocast("cuda"):
            # print("Extracting global CLIP features...")

            # _img = preprocess(Image.open(imgfile)).unsqueeze(0)
            _img = preprocess(Image.fromarray(img)).unsqueeze(0)
            global_feat = model.encode_image(_img.cuda())
            global_feat /= global_feat.norm(dim=-1, keepdim=True)
            # tqdm.write(f"Image feature dims: {global_feat.shape} \n")
        global_feat = global_feat.bfloat16().cuda()
        global_feat = F.normalize(global_feat, dim=-1)  # --> (1, 1024)
        feat_dim = global_feat.shape[-1]
        cosine_similarity = torch.nn.CosineSimilarity(dim=-1)

        feat_per_roi = []
        roi_nonzero_inds = []
        similarity_scores = []
        for maskidx in range(len(masks)):
            _x, _y, _w, _h = tuple(masks[maskidx]["bbox"])  # xywh bounding box
            seg = masks[maskidx]["segmentation"]
            nonzero_inds = torch.argwhere(torch.from_numpy(seg))
            # Note: Image is (H, W, 3). In SAM output, y coords are along height, x along width
            img_roi = img[_y : _y + _h, _x : _x + _w, :]
            img_roi = Image.fromarray(img_roi)
            img_roi = preprocess(img_roi).unsqueeze(0).cuda()
            roifeat = model.encode_image(img_roi)
            roifeat = F.normalize(roifeat, dim=-1)
            feat_per_roi.append(roifeat)
            roi_nonzero_inds.append(nonzero_inds)
            _sim = cosine_similarity(global_feat, roifeat)
            similarity_scores.append(_sim)

        similarity_scores = torch.cat(similarity_scores)
        softmax_scores = F.softmax(similarity_scores, dim=0)
        outfeat = torch.zeros(
            LOAD_IMG_HEIGHT, LOAD_IMG_WIDTH, feat_dim, dtype=torch.bfloat16
        )
        for maskidx in range(len(masks)):
            _weighted_feat = (
                softmax_scores[maskidx] * global_feat
                + (1 - softmax_scores[maskidx]) * feat_per_roi[maskidx]
            )
            _weighted_feat = F.normalize(_weighted_feat, dim=-1)
            outfeat[
                roi_nonzero_inds[maskidx][:, 0], roi_nonzero_inds[maskidx][:, 1]
            ] += (_weighted_feat[0].detach().cpu().bfloat16())
            outfeat[
                roi_nonzero_inds[maskidx][:, 0], roi_nonzero_inds[maskidx][:, 1]
            ] = F.normalize(
                outfeat[
                    roi_nonzero_inds[maskidx][:, 0], roi_nonzero_inds[maskidx][:, 1]
                ].float(),
                dim=-1,
            ).bfloat16()

        outfeat = outfeat.unsqueeze(0).float()
        outfeat = outfeat.permute(0, 3, 1, 2)  # 1, H, W, feat_dim -> 1, feat_dim, H, W
        # outfeat = F.interpolate(
        #     outfeat, [args.desired_height, args.desired_width], mode="nearest"
        # )
        outfeat = outfeat.permute(0, 2, 3, 1)  # 1, feat_dim, H, W --> 1, H, W, feat_dim
        outfeat = F.normalize(outfeat, dim=-1)
        outfeat = outfeat[0]  # --> H, W, feat_dim

        savefile = Path(args.save_dir) / Path(f"{idx:04d}.pt")
        print(f"Saving pixel-aligned feature of shape {outfeat.shape} to {savefile}...")
        torch.save(outfeat.detach().cpu(), savefile)
        feature_files.append(savefile)
    return feature_files

def align_voxel_grid(
    transform_matrix, points, masks, feature_files, device="cuda", pos_encoder=None
):
    # Reshape and calculate global bounding box using transformed points
    pts3d_flat = np.array(points).reshape(-1, 3)
    pts3d_flat = get_transformed_pts(transform_matrix, pts3d_flat)
    pts_min = torch.tensor(pts3d_flat.min(axis=0), device=device, dtype=torch.bfloat16)
    pts_max = torch.tensor(pts3d_flat.max(axis=0), device=device, dtype=torch.bfloat16)
    bbox_size = pts_max - pts_min
    voxel_size = bbox_size / GRID_SIZE

    # Voxel grid with sparse structure (kept on CPU)
    voxel_grid = defaultdict(
        lambda: {"points_num": 0, "feature": None, "sum_point": np.zeros(3)}
    )

    # Process each frame
    for p, m, feature_file in zip(points, masks, feature_files, strict=True):
        # Load feature file
        with open(feature_file, "rb") as f:
            feat = torch.load(f).to(device).bfloat16()
            FEAT_DIM = feat.shape[-1]

        # Transform points to the global coordinate system
        valid_points = get_transformed_pts(transform_matrix, p[m])
        valid_points = torch.tensor(valid_points, device=device, dtype=torch.bfloat16)
        # Filter out all-zero tensors and corresponding points
        non_zero_mask = feat[m].any(dim=-1)
        valid_features = feat[m][non_zero_mask]
        valid_points = valid_points[non_zero_mask]

        if pos_encoder is not None:
            pos_emb = pos_encoder(valid_points)
            valid_features += pos_emb

        # Compute voxel indices for all points in bulk
        voxel_indices = torch.floor((valid_points - pts_min) / voxel_size).to(
            dtype=int, device=device
        )
        # voxel_indices = torch.tensor(voxel_indices, device=device)

        # Find unique voxel indices and inverse mapping
        unique_indices, inverse_indices = torch.unique(
            voxel_indices, return_inverse=True, dim=0
        )

        # Initialize tensors for aggregation (on device)
        voxel_features = torch.zeros(
            (unique_indices.shape[0], FEAT_DIM), device=device, dtype=torch.bfloat16
        )
        voxel_counts = torch.zeros(
            unique_indices.shape[0], dtype=torch.float32, device=device
        )

        # Precompute the weights for incremental normalization
        weights = 1 / torch.bincount(
            inverse_indices, minlength=unique_indices.shape[0]
        ).float().to(device)

        # Aggregate features with normalization
        normalized_features = valid_features * weights[inverse_indices].unsqueeze(1)
        voxel_features.index_add_(0, inverse_indices, normalized_features.bfloat16())

        # Count remains the same
        voxel_counts.index_add_(
            0, inverse_indices, torch.ones_like(inverse_indices, dtype=torch.float32)
        )

        # Update global voxel grid on GPU
        for idx, voxel_idx in enumerate(unique_indices):
            voxel_idx_tuple = tuple(voxel_idx.tolist())
            sum_point = valid_points[inverse_indices == idx].sum(axis=0)

            if voxel_grid[voxel_idx_tuple]["feature"] is None:
                # Initialize voxel
                voxel_grid[voxel_idx_tuple]["feature"] = voxel_features[idx]
                voxel_grid[voxel_idx_tuple]["points_num"] = voxel_counts[idx].item()
                voxel_grid[voxel_idx_tuple]["sum_point"] = sum_point
            else:
                # Weighted feature aggregation and point sum update
                prev_count = voxel_grid[voxel_idx_tuple]["points_num"]
                total_count = prev_count + voxel_counts[idx].item()

                beta = voxel_counts[idx].item() / total_count
                existing_feature = voxel_grid[voxel_idx_tuple]["feature"].to(device)
                updated_feature = (1 - beta) * existing_feature + beta * voxel_features[
                    idx
                ]

                # Update global voxel grid
                voxel_grid[voxel_idx_tuple]["feature"] = updated_feature.cpu()
                voxel_grid[voxel_idx_tuple]["points_num"] = total_count
                voxel_grid[voxel_idx_tuple]["sum_point"] += sum_point

    # Convert to normal dict and compute averages
    voxel_grid = dict(voxel_grid)
    for key in voxel_grid.keys():
        voxel_grid[key]["feature"] = voxel_grid[key]["feature"].cpu()
        voxel_grid[key]["average_point"] = (
            voxel_grid[key]["sum_point"] / voxel_grid[key]["points_num"]
            if voxel_grid[key]["points_num"] > 0
            else None
        )

    count_points = sum([voxel_grid[key]["points_num"] for key in voxel_grid.keys()])
    print(
        f"Voxel grid size: {GRID_SIZE}x{GRID_SIZE}x{GRID_SIZE}, total voxels: {len(voxel_grid)}, total points: {count_points}"
    )
    return voxel_grid


@torch.no_grad()
def get_pixel_feature(
    transform_matrix, points, images, masks, model="clip", pos_encoder=None
):
    args = ProgramArgs

    # mask_files = get_sam1_masks(args, images)
    # # mask_files = None
    # match model:
    #     case "blip2":
    #         feature_files = get_blip_feature(args, images, mask_files)
    #     case "clip":
    #         feature_files = get_clip_feature(args, images, mask_files)
    #     case _:
    #         raise ValueError(f"Unknown model: {model}, choose from 'blip2' or 'clip'")

    feature_files = [
        Path(args.save_dir) / Path(f"{idx:04d}.pt") for idx in range(len(images))
    ]
    voxel_grid = align_voxel_grid(
        transform_matrix, points, masks, feature_files, pos_encoder=pos_encoder
    )
    with open(Path(args.save_dir) / "voxel_grid.pkl", "wb") as f:
        pkl.dump(voxel_grid, f)

    return voxel_grid


def get_global_feat(images, clip_vision_tower):
    # images: (N, H, W, C)
    # 在adt数据集中，图片被逆时针旋转90度，所以要先顺时针旋转90度摆正   
    images = torch.rot90(images, -1, (1, 2))
    # make sure images are in 0-255, set do_rescale = True
    images = clip_vision_tower.image_processor(images, return_tensors='pt', do_rescale=True)["pixel_values"]
    images = images.to(dtype=clip_vision_tower.dtype, device=clip_vision_tower.device)
    global_feat = clip_vision_tower.vision_tower(images).pooler_output 
    global_feat /= global_feat.norm(dim=-1, keepdim=True)
    global_feat = global_feat.bfloat16().cuda()
    global_feat = F.normalize(global_feat, dim=-1)
    return global_feat


def chunk_pixel_clip_feat(image: torch.Tensor, global_feat: torch.Tensor, object_seg:dict, clip_vision_tower,  obj_identity_dict, chunk_size:int, identity_embedding_dim:int):
    # 参考adt数据集的图片，图片被逆时针旋转90度，所以要先顺时针旋转90度摆正
    image = torch.rot90(image, -1, (0, 1))  # image in (H, W, C)
    LOAD_IMG_HEIGHT, LOAD_IMG_WIDTH = image.shape[0], image.shape[1]
    feat_dim = global_feat.shape[-1]

    # Collect ROIs and corresponding nonzero indices
    img_rois = []
    roi_nonzero_inds = []
    seg_obj_map = torch.zeros(LOAD_IMG_HEIGHT, LOAD_IMG_WIDTH, dtype=torch.int64, device=clip_vision_tower.device)
    print(f"Extracting pixel-aligned CLIP features for {len(object_seg)} sam mask")
    for obj_id, seg in object_seg.items():
        if isinstance(seg, dict):
            seg = rle_to_mask(seg)
        if isinstance(seg, np.ndarray):
            seg = torch.from_numpy(seg)
        # 像图片一样，seg也是逆时针旋转90度的，所以要先顺时针旋转90度摆正
        seg = torch.rot90(seg, -1, (0, 1))
        # Create identity embedding for the object across all frames
        if obj_id not in obj_identity_dict:
            obj_identity_dict[obj_id] = torch.randn(identity_embedding_dim, device=clip_vision_tower.device)
        seg_obj_map[seg] = obj_id
        # Get bounding box from mask
        # 计算bounding box
        nonzero_y, nonzero_x = torch.nonzero(seg, as_tuple=True)
        if len(nonzero_x) == 0 or len(nonzero_y) == 0:
            continue  # 空掩码，跳过
        _x, _y = nonzero_x.min().item(), nonzero_y.min().item()
        _w, _h = nonzero_x.max().item() - _x, nonzero_y.max().item() - _y
        if _w <= 10 or _h <= 10:
            continue  # 过小的掩码，跳过
        # Extract ROI
        nonzero_inds = torch.argwhere(seg)
        img_roi = image[_y : _y + _h, _x : _x + _w, :]
        # print(f"ROI shape: {img_roi.shape}")
        img_rois.append(img_roi)
        roi_nonzero_inds.append(nonzero_inds)
    img_rois = clip_vision_tower.image_processor(img_rois, return_tensors='pt')["pixel_values"]

    # Process ROIs in chunks
    outfeat = torch.zeros(
        LOAD_IMG_HEIGHT, LOAD_IMG_WIDTH, feat_dim, device=clip_vision_tower.device
    )
    for i in range(0, len(img_rois), chunk_size):
        chunk_img_rois = img_rois[i:i+chunk_size]
        chunk_img_rois = chunk_img_rois.to(dtype=clip_vision_tower.dtype, device=clip_vision_tower.device)
        roifeats_chunk = clip_vision_tower.vision_tower(chunk_img_rois).pooler_output
        roifeats_chunk = F.normalize(roifeats_chunk, dim=-1)
        similarity_scores = F.cosine_similarity(global_feat.expand(roifeats_chunk.size(0), -1), roifeats_chunk, dim=-1)
        # for j, maskidx in enumerate(range(i, min(i + chunk_size, len(sam_mask)))):
        #     _weighted_feat = (
        #         similarity_scores[j] * global_feat
        #         + (1 - similarity_scores[j]) * roifeats_chunk[j].unsqueeze(0)
        #     )

        #     _weighted_feat = F.normalize(_weighted_feat, dim=-1)
        #     index_pair = roi_nonzero_inds[maskidx][:, 0], roi_nonzero_inds[maskidx][:, 1]
        #     outfeat[index_pair] += _weighted_feat[0]
        
        # Compute weighted features for the entire batch, equivalent to the commented code above
        weighted_feats = similarity_scores.unsqueeze(-1) * global_feat + (1 - similarity_scores).unsqueeze(-1) * roifeats_chunk
        # Normalize the weighted features
        weighted_feats = F.normalize(weighted_feats, dim=-1)
        # Convert roi_nonzero_inds to a tensor of all indices in the chunk, record the corresponding feature index
        all_index_pairs, correpsonding_feat_index = [], []
        for j, maskidx in enumerate(range(i, min(i + chunk_size, len(roi_nonzero_inds)))):
            index_pairs = roi_nonzero_inds[maskidx]
            all_index_pairs.append(index_pairs)
            correpsonding_feat_index.extend([j] * len(index_pairs))
        all_index_pairs = torch.cat(all_index_pairs, dim=0)
        correpsonding_feat_index = torch.tensor(correpsonding_feat_index, device=clip_vision_tower.device)
        # Update outfeat by adding the corresponding weighted feature for each index pair
        # import pdb; pdb.set_trace()
        outfeat[all_index_pairs[:, 0], all_index_pairs[:, 1]] += weighted_feats[correpsonding_feat_index]

    outfeat = F.normalize(outfeat, dim=-1).to(clip_vision_tower.dtype)   # H, W, feat_dim

    # Rotate back to original orientation
    outfeat = torch.rot90(outfeat, 1, (0, 1))
    seg_obj_map = torch.rot90(seg_obj_map, 1, (0, 1))
    return outfeat, seg_obj_map


def get_transformed_pts_torch(transform_matrix, pts):
    # Convert points to homogeneous coordinates
    ones = torch.ones((pts.shape[0], 1), device=pts.device, dtype=pts.dtype)
    pts_homogeneous = torch.cat([pts, ones], dim=1)  # Add a column of ones
    pts_homogeneous = pts_homogeneous.to(dtype=transform_matrix.dtype,device=transform_matrix.device)
    # Transform the points
    # transformed_pts = (transform_matrix @ pts_homogeneous.T).T  # Apply transformation
    transformed_pts = pts_homogeneous @ transform_matrix.T  # Apply transformation
    transformed_pts = transformed_pts[:, :3]  # Convert back to 3D coordinates
    return transformed_pts

def get_point_cloud_size(pts3d):   
    # Reshape and calculate global bounding box using transformed points
    pts3d_flat = pts3d.reshape(-1, 3)
    pts3d_flat = pts3d_flat[~torch.isnan(pts3d_flat).any(dim=-1)]  # remove NaN points
    # Calculate min and max along the transformed points
    pts_min = pts3d_flat.min(dim=0).values  # (min_x, min_y, min_z)
    pts_max = pts3d_flat.max(dim=0).values  # (max_x, max_y, max_z)
    return pts_max, pts_min



def check_valid_inputs(valid_features, valid_points, valid_obj_embedding, valid_timestamps, octree_indice, max_num=200000, device="cuda", dtype=torch.bfloat16):
    # Convert to tensor if necessary
    if isinstance(valid_features, np.ndarray):
        valid_features = torch.from_numpy(valid_features)
    if isinstance(valid_points, np.ndarray):
        valid_points = torch.from_numpy(valid_points)
    if isinstance(valid_obj_embedding, np.ndarray):
        valid_obj_embedding = torch.from_numpy(valid_obj_embedding)
    if isinstance(valid_timestamps[0], np.ndarray):
        valid_timestamps = [torch.from_numpy(ts) for ts in valid_timestamps]

    # Move to device and dtype, except for valid_timestamps(handle later)
    valid_points = valid_points.to(dtype=dtype, device=device)
    valid_features = valid_features.to(dtype=dtype, device=device)
    valid_obj_embedding = valid_obj_embedding.to(dtype=dtype, device=device)
    N = valid_features.shape[0]
    if N <= max_num:
        return valid_features, valid_points, valid_obj_embedding, valid_timestamps, octree_indice
    
    time1 = time()
    print(f"Too many points ({valid_features.shape[0]}), sampling {max_num} points...")
    indices = torch.randperm(N, device=device, dtype=torch.long)[:max_num]
    indices, _ = torch.sort(indices)  # sort to keep the order of valid_points
    valid_features = valid_features[indices]
    valid_points = valid_points[indices]
    valid_obj_embedding = valid_obj_embedding[indices]
    valid_timestamps = [valid_timestamps[i] for i in indices.cpu().numpy()]  # faster
    time2 = time()
    # print(f"Time to subsample feature: {time2 - time1:.2f}s")

    # 如果 octree_indice 存在，则对其进行 torch 的 batch 操作
    if octree_indice is not None:
        time1 = time()
        # 将原始的字典转换成一个 tensor：每个位置存储该点所属的voxel_origin
        group_labels = torch.empty((N,3), dtype=dtype)  # in cpu, avoid frequent device transfer
        for group, inds in octree_indice.items():
            if not isinstance(group, torch.Tensor):
                group = torch.tensor(group, dtype=dtype)
            group_labels[inds] = group
        sampled_group_labels = group_labels.to(device=device)[indices]
        time2 = time()
        # print(f"Time to subsample octree_indice: {time2 - time1:.2f}s")
        unique_groups, inverse_indices = torch.unique(sampled_group_labels, return_inverse=True, dim=0)
        # 利用 batch 操作构建新的字典：键为组号，值为采样后对应的索引列表
        new_octree_indice = {}
        for i, group in enumerate(unique_groups):
            new_indices = (inverse_indices == i).nonzero(as_tuple=False).view(-1)
            # print(new_indices.shape, new_indices.device)
            new_octree_indice[tuple(group)] = new_indices#.tolist()
        octree_indice = new_octree_indice
        time3 = time()
        # print(f"Time to new octree_indice: {time3 - time3:.2f}s")
    return valid_features, valid_points, valid_obj_embedding, valid_timestamps, octree_indice
        

def align_voxel_grid_with_clip(
    img_timestamps_ns, valid_features, valid_points, valid_obj_embedding, valid_timestamps, traj_prompt,
    pos_encoder, vision_resampler, device, dtype, octree_indice = None,
    downsample_method: str = "octree",  # 新增: 可选 'octree' | 'voxel' | 'random'
    voxel_size: float = 0.5,           # voxel 下采样体素尺寸
    random_ratio: float = 0.01,           # 随机下采样比例 (0~1)
):
    if traj_prompt:
        raise NotImplementedError("Trajectory prompt is not implemented yet.")
    
    time1 = time()
    valid_features, valid_points, valid_obj_embedding, valid_timestamps, octree_indice = check_valid_inputs(
        valid_features, valid_points, valid_obj_embedding, valid_timestamps, octree_indice, 
        int(180000 * random_ratio) if downsample_method == "random" else 180000, device, dtype
    )
    time2 = time()
    # print(f"Time to check valid inputs: {time2 - time1:.2f}s")
    
    time1 = time()
    # Normalize timestamps
    timestamps_min_ns = img_timestamps_ns.min()
    # timestamps_max_ns = img_timestamps_ns.max()
    if isinstance(valid_timestamps, list):
        valid_timestamps = torch.nested.nested_tensor(valid_timestamps, device=device)  # don't change dtype
    valid_timestamps = (valid_timestamps - timestamps_min_ns) / 1e9 # Convert to seconds
    time2 = time()
    # print(f"Time to get nested_tensor: {time2 - time1:.2f}s，valid_timestamps {valid_timestamps.dtype}")
    
    # if octree_indice is None:
    #     # Convert points to numpy for Open3D
    #     time1 = time()
    #     if valid_features.dtype == torch.bfloat16:
    #         points_np = valid_points.cpu().detach().half().numpy()
    #     else:
    #         points_np = valid_points.cpu().detach().numpy()
    #     assert points_np.dtype == np.half, f"Invalid dtype: {points_np.dtype}"
    #     time2 = time()
    #     print(f"Time to convert points to numpy: {time2 - time1:.2f}s")
        
    #     # Build Octree
    #     time3 = time()
    #     temp_pcd = o3d.geometry.PointCloud()
    #     temp_pcd.points = o3d.utility.Vector3dVector(points_np)
    #     octree = o3d.geometry.Octree(max_depth=4)
    #     octree.convert_from_point_cloud(temp_pcd, size_expand=0.01)
    #     time4 = time()
    #     print(f"Time to build octree: {time4 - time3:.3f}s")

    #     octree_indice = {}
    #     def f_traverse(node, node_info):
    #         if isinstance(node, o3d.geometry.OctreeLeafNode):
    #             if isinstance(node, o3d.geometry.OctreePointColorLeafNode):
    #                 indices = node.indices  # Points in this voxel
    #                 if len(indices) > 0:
    #                     voxel_origin = tuple(node_info.origin.tolist())
    #                     octree_indice[voxel_origin] = indices
    #     octree.traverse(f_traverse)

    # 构建体素索引（GPU 实现）
    if downsample_method == "octree":
        if octree_indice is None:
            # Open3D 只能在 CPU 上执行（保留原逻辑）
            import open3d as o3d
            print(f"[Info] Using CPU Octree (Open3D)")
            points_np = valid_points.cpu().numpy()
            temp_pcd = o3d.geometry.PointCloud()
            temp_pcd.points = o3d.utility.Vector3dVector(points_np)
            octree = o3d.geometry.Octree(max_depth=4)
            octree.convert_from_point_cloud(temp_pcd, size_expand=0.01)
            octree_indice = {}

            def f_traverse(node, node_info):
                if isinstance(node, o3d.geometry.OctreeLeafNode):
                    if isinstance(node, o3d.geometry.OctreePointColorLeafNode):
                        indices = node.indices
                        if len(indices) > 0:
                            voxel_origin = tuple(node_info.origin.tolist())
                            octree_indice[voxel_origin] = indices
            octree.traverse(f_traverse)

    else:
        octree_indice = {} # reset octree_indice
        # -------------- GPU-based voxel 分组 --------------
        sampled_idx = torch.arange(valid_points.size(0), device=device)
        sampled_points = valid_points

        # 对点进行 voxel 量化 (GPU)
        coords = torch.floor(sampled_points / voxel_size).long()  # [K,3]
        # 获取唯一 voxel（GPU 实现的 unique_rows）
        unique_coords, inverse = torch.unique(coords, dim=0, return_inverse=True)

        # 构造 GPU 上的索引映射，不用 Python dict
        # 每个点对应的 voxel_id = inverse[i]
        voxel_ids = inverse
        num_voxels = unique_coords.size(0)

        # 计算每个 voxel 的均值（scatter_mean）
        voxel_features = scatter_mean(valid_features[sampled_idx], voxel_ids, dim=0, dim_size=num_voxels)
        voxel_points = scatter_mean(sampled_points, voxel_ids, dim=0, dim_size=num_voxels)
        points_num = torch.bincount(voxel_ids, minlength=num_voxels)

        # 把 voxel 原点（浮点坐标）恢复出来
        voxel_origins = unique_coords.float() * voxel_size

        # 转成与 octree 相同格式（如需兼容）
        octree_indice = {
            tuple(voxel_origins[i].tolist()): sampled_idx[voxel_ids == i].tolist()
            for i in range(num_voxels)
        }

    assert len(octree_indice) > 0, f"No voxel indices found! {octree_indice}"

    # Add positional and extra encoding
    # time1 = time()
    # valid_features += pos_encoder(valid_points)
    # valid_features += vision_resampler(valid_obj_embedding, valid_timestamps)
    valid_features += vision_resampler(valid_obj_embedding, valid_timestamps, pos_encoder(valid_points))
    assert not torch.isnan(valid_features).any(), "NaN detected!"
    # time2 = time()
    # print(f"Time to add encodings: {time2 - time1:.2f}s")

    voxel_grid = {}
    time5 = time()
    # Compute averages after traversal
    # for voxel_origin, indices in octree_indice.items():
    #     voxel_features = valid_features[indices].mean(dim=0)
    #     avg_point = valid_points[indices].mean(dim=0)
    #     voxel_grid[voxel_origin] = {
    #         "feature": voxel_features,
    #         "points_num": len(indices),
    #         "avg_point": avg_point
    #     }
 
    # using torch_scatter to speed up
    voxel_origins = list(octree_indice.keys())  # 所有体素原点的列表
    num_voxels = len(voxel_origins)  # 体素总数
    voxel_origin_to_idx = {origin: idx for idx, origin in enumerate(voxel_origins)}
    point_to_voxel_idx = torch.full((valid_features.size(0),), -1, dtype=torch.long, device=device)
    for voxel_origin, indices in octree_indice.items():
        voxel_idx = voxel_origin_to_idx[voxel_origin]
        point_to_voxel_idx[indices] = voxel_idx
    assert (point_to_voxel_idx >= 0).all(), "Some points are not assigned to any voxel!"
    voxel_features = scatter_mean(valid_features, point_to_voxel_idx, dim=0, dim_size=num_voxels)
    voxel_points = scatter_mean(valid_points, point_to_voxel_idx, dim=0, dim_size=num_voxels)
    # voxel_pos_embedding = scatter_mean(pos_embedding, point_to_voxel_idx, dim=0, dim_size=num_voxels)
    points_num = torch.bincount(point_to_voxel_idx, minlength=num_voxels)
    voxel_grid = {}
    for idx, voxel_origin in enumerate(voxel_origins):
        voxel_grid[voxel_origin] = {
            "feature": voxel_features[idx],
            "points_num": points_num[idx].item(),
            "avg_point": voxel_points[idx],
            # "avg_pos_embedding": voxel_pos_embedding[idx],
        }

    # 记录结束时间并打印
    time6 = time()
    # print(f"Time to mean feature with torch_scatter: {time6 - time5:.3f}s")

    # 记录结束时间并打印
    # time6 = time()
    # print(f"Time to mean feature: {time6 - time5:.3f}s")

    # Statistics
    # count_points = sum([voxel_grid[key]["points_num"] for key in voxel_grid.keys()])
    # print(f"Total voxels: {len(voxel_grid)}, total points: {count_points}")

    return voxel_grid


def to_device(batch, device, callback=None, non_blocking=False):
    """Transfer some variables to another device (i.e. GPU, CPU:torch, CPU:numpy).

    batch: list, tuple, dict of tensors or other things
    device: pytorch device or 'numpy'
    callback: function that would be called on every sub-elements.
    """
    if callback:
        batch = callback(batch)

    if isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}

    if isinstance(batch, (tuple, list)):
        return type(batch)(to_device(x, device) for x in batch)

    x = batch
    if device == "numpy":
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
    elif x is not None:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if torch.is_tensor(x):
            x = x.to(device, non_blocking=non_blocking)
    return x


def to_numpy(x):
    return to_device(x, "numpy")


def to_cpu(x):
    return to_device(x, "cpu")


def to_cuda(x):
    return to_device(x, "cuda")


def geotrf(Trf, pts, ncol=None, norm=False):
    """Apply a geometric transformation to a list of 3-D points.

    H: 3x3 or 4x4 projection matrix (typically a Homography)
    p: numpy/torch/tuple of coordinates. Shape must be (...,2) or (...,3)

    ncol: int. number of columns of the result (2 or 3)
    norm: float. if != 0, the resut is projected on the z=norm plane.

    Returns an array of projected 2d points.
    """
    assert Trf.ndim >= 2
    if isinstance(Trf, np.ndarray):
        pts = np.asarray(pts)
    elif isinstance(Trf, torch.Tensor):
        pts = torch.as_tensor(pts, dtype=Trf.dtype)

    # adapt shape if necessary
    output_reshape = pts.shape[:-1]
    ncol = ncol or pts.shape[-1]

    # optimized code
    if (
        isinstance(Trf, torch.Tensor)
        and isinstance(pts, torch.Tensor)
        and Trf.ndim == 3
        and pts.ndim == 4
    ):
        d = pts.shape[3]
        if Trf.shape[-1] == d:
            pts = torch.einsum("bij, bhwj -> bhwi", Trf, pts)
        elif Trf.shape[-1] == d + 1:
            pts = (
                torch.einsum("bij, bhwj -> bhwi", Trf[:, :d, :d], pts)
                + Trf[:, None, None, :d, d]
            )
        else:
            raise ValueError(f"bad shape, not ending with 3 or 4, for {pts.shape=}")
    else:
        if Trf.ndim >= 3:
            n = Trf.ndim - 2
            assert Trf.shape[:n] == pts.shape[:n], "batch size does not match"
            Trf = Trf.reshape(-1, Trf.shape[-2], Trf.shape[-1])

            if pts.ndim > Trf.ndim:
                # Trf == (B,d,d) & pts == (B,H,W,d) --> (B, H*W, d)
                pts = pts.reshape(Trf.shape[0], -1, pts.shape[-1])
            elif pts.ndim == 2:
                # Trf == (B,d,d) & pts == (B,d) --> (B, 1, d)
                pts = pts[:, None, :]

        if pts.shape[-1] + 1 == Trf.shape[-1]:
            Trf = Trf.swapaxes(-1, -2)  # transpose Trf
            pts = pts @ Trf[..., :-1, :] + Trf[..., -1:, :]
        elif pts.shape[-1] == Trf.shape[-1]:
            Trf = Trf.swapaxes(-1, -2)  # transpose Trf
            pts = pts @ Trf
        else:
            pts = Trf @ pts.T
            if pts.ndim >= 2:
                pts = pts.swapaxes(-1, -2)

    if norm:
        pts = pts / pts[..., -1:]  # DONT DO /= BECAUSE OF WEIRD PYTORCH BUG
        if norm != 1:
            pts *= norm

    res = pts[..., :ncol].reshape(*output_reshape, ncol)
    return res


@torch.no_grad()
def add_scene_cam(
    scene,
    pose_c2w,
    edge_color,
    image=None,
    focal=None,
    imsize=None,
    screen_width=0.03,
    marker=None,
):
    if image is not None:
        image = np.asarray(image)
        H, W, THREE = image.shape
        assert THREE == 3
        if image.dtype != np.uint8:
            image = np.uint8(255 * image)
    elif imsize is not None:
        W, H = imsize
    elif focal is not None:
        H = W = focal / 1.1
    else:
        H = W = 1

    if isinstance(focal, np.ndarray):
        focal = focal[0]
    if not focal:
        focal = min(H, W) * 1.1  # default value

    # create fake camera
    height = max(screen_width / 10, focal * screen_width / H)
    width = screen_width * 0.5**0.5
    rot45 = np.eye(4)
    rot45[:3, :3] = Rotation.from_euler("z", np.deg2rad(45)).as_matrix()
    rot45[2, 3] = -height  # set the tip of the cone = optical center
    aspect_ratio = np.eye(4)
    aspect_ratio[0, 0] = W / H
    transform = pose_c2w @ OPENGL @ aspect_ratio @ rot45
    cam = trimesh.creation.cone(width, height, sections=4)  # , transform=transform)

    # this is the image
    if image is not None:
        vertices = geotrf(transform, cam.vertices[[4, 5, 1, 3]])
        faces = np.array([[0, 1, 2], [0, 2, 3], [2, 1, 0], [3, 2, 0]])
        img = trimesh.Trimesh(vertices=vertices, faces=faces)
        uv_coords = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
        img.visual = trimesh.visual.TextureVisuals(
            uv_coords, image=Image.fromarray(image)
        )
        scene.add_geometry(img)

    # this is the camera mesh
    rot2 = np.eye(4)
    rot2[:3, :3] = Rotation.from_euler("z", np.deg2rad(2)).as_matrix()
    vertices = np.r_[cam.vertices, 0.95 * cam.vertices, geotrf(rot2, cam.vertices)]
    vertices = geotrf(transform, vertices)
    faces = []
    for face in cam.faces:
        if 0 in face:
            continue
        a, b, c = face
        a2, b2, c2 = face + len(cam.vertices)
        a3, b3, c3 = face + 2 * len(cam.vertices)

        # add 3 pseudo-edges
        faces.append((a, b, b2))
        faces.append((a, a2, c))
        faces.append((c2, b, c))

        faces.append((a, b, b3))
        faces.append((a, a3, c))
        faces.append((c3, b, c))

    # no culling
    faces += [(c, b, a) for a, b, c in faces]

    cam = trimesh.Trimesh(vertices=vertices, faces=faces)
    cam.visual.face_colors[:, :3] = edge_color
    scene.add_geometry(cam)

    if marker == "o":
        marker = trimesh.creation.icosphere(3, radius=screen_width / 4)
        marker.vertices += pose_c2w[:3, 3]
        marker.visual.face_colors[:, :3] = edge_color
        scene.add_geometry(marker)


def load_point_cloud(file_path):
    # Load the .pkl file
    with open(file_path, "rb") as fr:
        raw_scene = pickle.load(fr)
    print("Loaded a pkl file.")

    outdir = raw_scene["outdir"]
    imgs = raw_scene["imgs"]
    depths = raw_scene["depths"]
    pts3d = raw_scene["pts3d"]
    dynamic_masks = raw_scene["mask"]
    focals = raw_scene["focals"]
    cams2world = raw_scene["cams2world"]

    cam_size = raw_scene["cam_size"]
    show_cam = raw_scene["show_cam"]
    cam_color = raw_scene["cam_color"]
    as_pointcloud = raw_scene["as_pointcloud"]
    transparent_cams = raw_scene["transparent_cams"]
    silent = raw_scene["silent"]
    save_name = raw_scene["save_name"]

    pairs_movement = raw_scene["pairs_movement"]
    if pairs_movement is not None:
        pairs_movement = dict(sorted(pairs_movement.items()))
        print(len(pairs_movement), "Pair movement:", pairs_movement.keys())
    else:
        print("No pair movement.")

    assert len(pairs_movement) + 1 == len(pts3d) == len(dynamic_masks) <= len(imgs) <= len(cams2world) == len(focals)
    pts3d = to_numpy(pts3d)
    imgs = to_numpy(imgs)
    focals = to_numpy(focals)
    cams2world = to_numpy(cams2world)
    depths = to_numpy(depths)
    pairs_movement = to_numpy(pairs_movement)

    return (
        outdir,
        imgs,
        depths,
        pts3d,
        dynamic_masks,
        focals,
        cams2world,
        cam_size,
        show_cam,
        cam_color,
        as_pointcloud,
        transparent_cams,
        silent,
        save_name,
        pairs_movement,
    )


def get_transform_matrix(cams2world):
    rot = np.eye(4)
    rot[:3, :3] = Rotation.from_euler("y", np.deg2rad(180)).as_matrix()
    transform_matrix = np.linalg.inv(cams2world[0] @ OPENGL @ rot)
    return transform_matrix


def get_transformed_pts(transform_matrix, pts):
    # Convert points to homogeneous coordinates
    pts_homogeneous = np.hstack(
        [pts, np.ones((pts.shape[0], 1))]
    )  # Add a column of ones
    # Transform the points
    transformed_pts = (transform_matrix @ pts_homogeneous.T).T  # Apply transformation
    transformed_pts = transformed_pts[:, :3]  # Convert back to 3D coordinates
    return transformed_pts


def show_pkl(file_path):

    (
        _,
        imgs,
        _,
        pts3d,
        mask,
        focals,
        cams2world,
        cam_size,
        show_cam,
        cam_color,
        as_pointcloud,
        transparent_cams,
        _,
        _,
        _,
    ) = load_point_cloud(file_path)
    transform_matrix = get_transform_matrix(cams2world)
    # voxel_grid = get_pixel_feature(transform_matrix, pts3d, imgs, mask)

    # full pointcloud
    if not as_pointcloud:
        raise NotImplementedError("This function is not implemented yet.")

    scene = trimesh.Scene()

    # add each camera
    if show_cam:
        for i, pose_c2w in enumerate(cams2world):
            if isinstance(cam_color, list):
                camera_edge_color = cam_color[i]
            else:
                camera_edge_color = cam_color or CAM_COLORS[i % len(CAM_COLORS)]
            add_scene_cam(
                scene,
                pose_c2w,
                camera_edge_color,
                None if transparent_cams else imgs[i],
                focals[i],
                imsize=imgs[i].shape[1::-1],
                screen_width=cam_size,
            )

    # only align the cameras
    scene.apply_transform(transform_matrix)
    pts = np.concatenate([p[m] for p, m in zip(pts3d, mask)])
    col = np.concatenate([p[m] for p, m in zip(imgs, mask)])
    transformed_pts = get_transformed_pts(transform_matrix, pts)
    pct = trimesh.PointCloud(
        transformed_pts.reshape(-1, 3), colors=col.reshape(-1, 3)
    )
    scene.add_geometry(pct)        

    # Define the origin and the endpoints for the axes
    axis_length = 1.0  # Length of the axis lines
    origin = [0, 0, 0]  # World origin

    # X-axis: Red
    x_axis = np.array([origin, [axis_length, 0, 0]])
    x_color = [255, 0, 0, 255]  # Red

    # Y-axis: Green
    y_axis = np.array([origin, [0, axis_length, 0]])
    y_color = [0, 255, 0, 255]  # Green

    # Z-axis: Blue
    z_axis = np.array([origin, [0, 0, axis_length]])
    z_color = [0, 0, 255, 255]  # Blue

    # Create the axis lines as Line objects
    x_line = trimesh.load_path(x_axis)
    x_line.colors = [x_color] * len(x_line.entities)

    y_line = trimesh.load_path(y_axis)
    y_line.colors = [y_color] * len(y_line.entities)

    z_line = trimesh.load_path(z_axis)
    z_line.colors = [z_color] * len(z_line.entities)

    # Add the axes to the scene
    scene.add_geometry(x_line)
    scene.add_geometry(y_line)
    scene.add_geometry(z_line)

    scene.show()


def show_glb(file_path):
    # Load the .glb file
    scene: Scene = trimesh.load(file_path)

    # Check if the loaded file is a Scene
    if isinstance(scene, trimesh.Scene):
        print("Loaded a Scene object.")

        # Iterate over geometries in the scene
        for name, geom in scene.geometry.items():
            if isinstance(geom, trimesh.points.PointCloud):
                print(f"Found a PointCloud: {name}")
                # Check if colors exist
                if geom.colors is not None:
                    print("Colors are associated with this PointCloud.", geom.colors)
    else:
        print("The loaded file is not a Scene.")
    # Show the scene
    scene.show()


def show_depth(file_path):
    # load image
    img = Image.open(file_path)
    print(img.size)
    img.show()
    # to numpy
    img = np.array(img)
    print(img.shape, img[img != 0].shape)
    print(img[img != 0].max(), img[img != 0].min())


def show_movement(file_path):
    (
        _, imgs, _, pts3d, dynamic_mask, _, cams2world, _, _, _, 
        as_pointcloud, _, _, _, pairs_movement,
    ) = load_point_cloud(file_path)
    transform_matrix = get_transform_matrix(cams2world)
    # full pointcloud
    if not as_pointcloud:
        raise NotImplementedError("This function is not implemented yet.")
    SHORT, LONG = 0.01, 0.02

    # trace the movement of points
    print("moving points:", [int(dm.sum()) for dm in dynamic_mask])
    trajactory, distribution = {}, {}
    for pair, movement in tqdm(pairs_movement.items()):
        # print("pair:", pair, "movement:", movement.shape) # (H, W, y-x)
        valid_mask = (movement != -1).all(axis=-1)
        valid_mask &= dynamic_mask[pair[0]]  # only use the dynamic mask
        for y, x in zip(*np.where(valid_mask)):  # get coordinates
            key = (pair[0], (y, x))  # use the last movement as the key
            new_key = (pair[1], tuple(movement[y, x].astype(int)))
            if key not in trajactory:
                trajactory[new_key] = [key]
            else:  # change the key to trace latest movement
                trajactory[new_key] = trajactory.pop(key)
            trajactory[new_key].append(new_key)
    print("point trajactory:", len(trajactory))
    # count length distribution of trajactory
    for k, v in trajactory.items():
        distribution[len(v)] = distribution.get(len(v), 0) + 1
    print("distribution:", distribution)

    # show the trajactory in the scene
    pts, col, line = [], [], []
    for k, v in tqdm(trajactory.items()):
        if len(v) < 20:
            continue
        pts.extend([pts3d[i][y, x] for i, (y, x) in v])
        col.extend([imgs[i][y, x] for i, (y, x) in v])
        # draw line for one trace
        for i in range(1, len(v)):
            start_i, start_y, start_x = v[i-1][0], v[i-1][1][0], v[i-1][1][1]
            end_i, end_y, end_x = v[i][0], v[i][1][0], v[i][1][1]
            start_point = pts3d[start_i][start_y, start_x]
            end_point = pts3d[end_i][end_y, end_x]
            line.append([start_point, end_point])  # Add the line to the lines list
    pts, col, line = np.array(pts), np.array(col), np.array(line)
    pts = get_transformed_pts(transform_matrix, pts)
    line = get_transformed_pts(transform_matrix, line.reshape(-1, 3)).reshape(-1, 2, 3)
    delta = np.abs(line[:, 0] - line[:, 1]).sum(axis=-1)
    line = line[(delta > SHORT) & (delta < LONG)]  # skip too short or too long
    # random drop line for better visualization
    line = line[np.random.choice(len(line), min(20000, len(line)), replace=False)]
    print("pts:", pts.shape, "col:", col.shape, "line:", line.shape)
    scene = trimesh.Scene()
    scene.add_geometry(trimesh.PointCloud(pts, colors=col))
    line = trimesh.load_path(line)
    line.colors = np.array([[0, 255, 0, 255]] * len(line.entities)).astype(np.uint8)
    scene.add_geometry(line)
    scene.show()

    # show movement in the scene by pairs
    for idx, (pair, movement) in enumerate(pairs_movement.items()):
        scene = trimesh.Scene()
        pts = get_transformed_pts(transform_matrix, pts3d[pair[0]].reshape(-1, 3)).reshape(movement.shape[0], movement.shape[1], 3)
        corr_y, corr_x = movement.transpose(2, 0, 1)  # from (H, W, y-x) to (y-x, H, W)
        moved_pts = get_transformed_pts(transform_matrix, (pts3d[pair[1]][corr_y, corr_x]).reshape(-1, 3)).reshape(movement.shape[0], movement.shape[1], 3)
        movement_mask = (movement != -1).all(axis=-1)  # (H, W) -1 for no correspondence
        pts, moved_pts = pts[movement_mask], moved_pts[movement_mask]
        
        # Create line
        line = np.array([pts.reshape(-1,3), moved_pts.reshape(-1,3)]).transpose(1, 0, 2)
        delta = np.abs(line[:, 0] - line[:, 1]).sum(axis=-1)
        line = line[(delta > SHORT) & (delta < LONG)]  # skip too short or too long
        print("pair:", pair, "line", line.shape, "short line:", (delta < SHORT).sum(), "long line:", (delta > LONG).sum())
        line = trimesh.load_path(line)
        # set color by pair[0] (from blue to red for different pairs)
        line.colors = np.array([
            np.array([0, 100, 255, 255])*(pair[0]/len(pairs_movement)) + 
            np.array([255, 200, 0, 255])*(1-pair[0]/len(pairs_movement))
        ] * len(line.entities)).astype(np.uint8)
        scene.add_geometry(line)

        # Create point
        pct = trimesh.PointCloud(
            get_transformed_pts(transform_matrix, pts3d[pair[0]].reshape(-1, 3)),
            colors=imgs[pair[0]].reshape(-1, 3)
        )
        scene.add_geometry(pct)
        pct = trimesh.PointCloud(
            get_transformed_pts(transform_matrix, pts3d[pair[1]].reshape(-1, 3)),
            colors=imgs[pair[1]].reshape(-1, 3)
        )
        scene.add_geometry(pct)
        scene.show()



def make_llava3d_data(pkl_path, intrinsics_path):
    (
        _, imgs, depths, _, _, _, cams2world, _, _, _, _, _, _, _, _,
    ) = load_point_cloud(pkl_path)

    # Scale and convert depth images to 16-bit integers
    for idx, dep in enumerate(depths):
        # print("before scale:", dep.shape, dep.min(), dep.max())
        dep = (dep * DEPTH_SCALE).astype(np.uint16)
        # print("after scale:", dep.shape, dep.min(), dep.max())

        # Save each depth image as a 16-bit PNG
        name = f"./lady-running-llava3d/depth_{idx:05d}.png"
        cv2.imwrite(name, dep)
    
    # save image to jpg
    for idx, img in enumerate(imgs):
        name = f"./lady-running-llava3d/image_{idx:05d}.jpg"
        # from RGB to RGBA
        img = Image.fromarray((img*255).astype(np.uint8))
        img.save(name)

    # save pose to json
    poses = {}
    '''
    {
        "scannet/scene0191_00": {
            "scannet/posed_images/scene0191_00/00000.jpg": {"pose": [[0.246516, -0.470365, 0.847341, 3.043058], [-0.959136, 0.006886, 0.282862, 2.955299], [-0.138884, -0.882445, -0.449446, 1.551102], [0.0, 0.0, 0.0, 1.0]], "depth": "scannet/posed_images/scene0191_00/00000.png"}, 
            "scannet/posed_images/scene0191_00/00010.jpg": {"pose": [[0.234267, -0.467486, 0.852394, 3.041167], [-0.965482, -0.009191, 0.260306, 2.922879], [-0.113855, -0.883953, -0.453503, 1.551035], [0.0, 0.0, 0.0, 1.0]], "depth": "scannet/posed_images/scene0191_00/00010.png"},
            ...
            "scannet/posed_images/scene0191_00/01080.jpg": {"pose": [[-0.864832, 0.106215, -0.490697, 4.665287], [0.499826, 0.274274, -0.821552, 3.041187], [0.047324, -0.955768, -0.290291, 1.414955], [0.0, 0.0, 0.0, 1.0]], "depth": "scannet/posed_images/scene0191_00/01080.png"}, 
            "axis_align_matrix": [[-0.21644, 0.976296, 0.0, -1.01457], [-0.976296, -0.21644, 0.0, 3.91808], [0.0, 0.0, 1.0, -0.070665], [0.0, 0.0, 0.0, 1.0]], 
            "depth_intrinsic": [[577.87060546875, 0.0, 319.5, 0.0], [0.0, 577.87060546875, 239.5, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], 
            "intrinsic": [[1170.18798828125, 0.0, 647.75, 0.0], [0.0, 1170.18798828125, 483.75, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        },
        ... 
    }
    '''
    for idx, pose in enumerate(cams2world):
        pose = pose.tolist()
        poses[f"scannet/posed_images/lady-running-llava3d/image_{idx:05d}.jpg"] = {
            "pose": pose,
            "depth": f"scannet/posed_images/lady-running-llava3d/depth_{idx:05d}.png"
        }
    poses["axis_align_matrix"] = get_transform_matrix(cams2world).tolist()
    with open(intrinsics_path) as f:
        text = f.readlines()[0]
        # 440.493164 0.000000 256.000000 0.000000 440.493164 144.000000 0.000000 0.000000 1.000000
    intrinsic = [float(i) for i in text.split(" ")]
    intrinsic = np.array(intrinsic)[:8].reshape(2,4)
    intrinsic = np.concatenate([intrinsic, np.array([[0,0,1,0],[0,0,0,1]])], axis=0)
    poses["depth_intrinsic"] = intrinsic.tolist()
    poses["intrinsic"] = intrinsic.tolist()
    data = {"scannet/lady-running-llava3d":poses}

    name = f"./lady-running-llava3d/custom_annotation.json"
    with open(name, "w") as f:
        json.dump(data, f)


if __name__ == "__main__":
    # show_glb("./lady-running/scene.glb")
    # show_pkl("./lady-running/raw_scene.pkl")
    # make_llava3d_data("./lady-running/raw_scene.pkl", "./lady-running/pred_intrinsics.txt")
    # show_depth("./00000.png")
    # show_depth("./lady-running-llava3d/depth_00000.png")
    # load_point_cloud("./raw_scene.pkl")
    # show_movement("./raw_scene.pkl")
    get_sam2_masks("/data1/syh_data/Code/monst3r/demo_data/lady-running", 
                   "/data1/syh_data/Code/LLaVA-3D/demo/lady-running-seg",
                   "large")


