import json
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
import open3d as o3d

def unproject(depth, intrinsic, correct_scale=False):
    """
    Convert depth map to 3D points in camera coordinate system.

    Args:
        depth: np.ndarray of shape (H, W), dtype uint16, in millimeters
        intrinsic: np.ndarray [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        correct_scale: bool, whether to apply scale correction
    Returns:
        pts: np.ndarray of shape (H, W, 3) in meters, 3D points
        valid_mask: np.ndarray of shape (H, W), boolean mask for valid points
    """
    H, W = depth.shape
    if correct_scale:  # for THUD sequences
        f_scale = 0.7  # We scaled images from original data
        u, v = np.meshgrid(np.arange(W), np.arange(H - 1, -1, -1))
    else:
        f_scale = 1
        u, v = np.meshgrid(np.arange(W), np.arange(H))
    
    valid_mask = depth < np.max(depth) - 10  # avoid too large depth
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    x_r = (u - cx * f_scale) / (fx * f_scale)
    y_r = (v - cy * f_scale) / (fy * f_scale)
    rays = np.stack((x_r, y_r, np.ones_like(x_r)), axis=-1)    
    z_mm = depth.astype(np.float32)
    pts_mm = rays * z_mm[..., np.newaxis]
    if correct_scale:  # for THUD sequences
        # following original process
        pts_mm[..., 0] /= 2.5
        pts_mm[..., 1] /= 2
        pts_mm[..., 2] /= 6.5
    
    pts = pts_mm / 1000.0  # Convert to meters
    return pts, valid_mask

def transform_to_scene(vertices_cam, T_scene_cam, correct_shift=False):
    """
    Transform 3D points from camera to scene coordinate system.

    Args:
        vertices_cam: np.ndarray of shape (N, 3), points in camera coordinates
        T_scene_cam: np.ndarray of shape (4, 4), transformation matrix
    Returns:
        points_world: np.ndarray of shape (N, 3), points in scene coordinates
    """
    p_in_cam_homo = np.hstack([vertices_cam, np.ones((vertices_cam.shape[0], 1))])
    p_in_scene_homo = (T_scene_cam @ p_in_cam_homo.T).T  # [N, 4]
    points_world = p_in_scene_homo[:, :3] / p_in_scene_homo[:, 3:4]
    if correct_shift:  # for THUD++
        points_world[:, 0] -= 0.3  #  align to gt bounding box 3d
    return points_world

def rle_to_mask(rle: dict) -> np.ndarray:
    """Convert RLE segmentation to binary mask."""
    h, w = rle["size"]
    mask = np.empty(h * w, dtype=bool)
    idx = 0
    parity = False
    for count in rle["counts"]:
        mask[idx: idx + count] = parity
        idx += count
        parity ^= True
    mask = mask.reshape(w, h)
    return mask.transpose()

def get_seg_obj_map(image_shape, object_seg):
    """Generate instance segmentation map from segmentation data."""
    H, W = image_shape[:2]
    seg_obj_map = np.zeros((H, W), dtype=np.int64)
    for obj_id, seg in object_seg.items():
        if isinstance(seg, dict):
            seg = rle_to_mask(seg)
        seg_obj_map[seg] = int(obj_id)
    return seg_obj_map

def export_one_scan(sequence_name, interleave = 10):
    """Export one sequence to ScanNet-like format with 3D point clouds and bounding boxes."""
    sequence_path = Path(DATASET_PATH, sequence_name)
    depth_paths = sorted(sequence_path.glob('depth*.png'), key=lambda x: x.name)
    image_paths = sorted(sequence_path.glob('image*.jpg'), key=lambda x: x.name)
    with open(sequence_path / 'processed_poses.json', 'r') as f:
        poses_dict = json.load(f)
    with open(sequence_path / 'segs.json', 'r') as f:
        segs_list = json.load(f)

    all_poses = dict(sorted(poses_dict[sequence_name].items()))
    all_poses = {k: v for k, v in all_poses.items() if '.jpg' in k}
    intrinsic = np.array(poses_dict['intrinsic'])
    rotate = '_Capture_' not in sequence_name  # only rotate for ADT sequences

    # Collect all 3D points, colors, and labels
    all_points = []
    all_colors = []
    all_instance_ids = []

    # Sort all_poses keys by file name
    sorted_pose_keys = sorted(all_poses.keys(), key=lambda x: Path(x).name)
    # Now, iterate through the sorted lists and keys
    assert len(depth_paths)==len(image_paths)==len(segs_list)==len(sorted_pose_keys)

    for idx, (depth_path, image_path, segs, pose_key) in tqdm(enumerate(zip(depth_paths, image_paths, segs_list, sorted_pose_keys)), total=len(sorted_pose_keys)):
        # Verify that the image file name matches the pose key
        assert Path(image_path).name == Path(pose_key).name, f"Mismatch between image {Path(image_path).name} and pose key {Path(pose_key).name}"

        if idx % interleave != 0: continue

        depth = np.array(Image.open(depth_path))
        image = np.array(Image.open(image_path))
        pose = np.array(all_poses[pose_key]["pose"])

        # Generate 3D point cloud
        p_in_cam, valid_mask = unproject(depth, intrinsic, not rotate)
        points_world = transform_to_scene(p_in_cam.reshape(-1, 3), pose, not rotate)
        frame_points = points_world.reshape(depth.shape[0], depth.shape[1], 3)
        frame_points[~valid_mask] = np.nan

        # Generate instance labels
        seg_obj_map = get_seg_obj_map(image.shape, segs)
        if rotate:
            image = np.rot90(image, -1, axes=(0, 1))
            frame_points = np.rot90(frame_points, -1, axes=(0, 1))
            seg_obj_map = np.rot90(seg_obj_map, -1, axes=(0, 1))

        # Flatten and filter valid 3D points
        points = frame_points.reshape(-1, 3)
        colors = image.reshape(-1, 3)
        instance_ids = seg_obj_map.reshape(-1)
        valid = ~np.isnan(points).any(axis=1)
        points = points[valid]
        colors = colors[valid]
        instance_ids = instance_ids[valid]

        all_points.append(points)
        all_colors.append(colors)
        all_instance_ids.append(instance_ids)

    print([(p.shape,c.shape,l.shape) for p,c,l in zip(all_points, all_colors, all_instance_ids, strict=True)])
    return all_points, all_colors, all_instance_ids


def get_object_ids(sequence_name):
    """
    Extract unique object IDs from segs.pkl for a given sequence.

    Args:
        sequence_name: str, name of the sequence
    Returns:
        list, unique object IDs
    """
    sequence_path = Path(DATASET_PATH, sequence_name)
    segs_file = sequence_path / 'segs.json'
    if not segs_file.exists():
        print(f"Warning: {segs_file} not found, returning empty object_ids")
        return []

    try:
        with open(segs_file, 'r') as f:
            segs_list = json.load(f)
        object_ids = set()
        for segs in segs_list:
            if isinstance(segs, dict):
                object_ids.update(str(k) for k in segs.keys())
        return sorted(list(object_ids))
    except Exception as e:
        print(f"Error reading {segs_file}: {e}")
        return []
    
def visualize_point_cloud(points, colors):
    """
    Visualize 3D point cloud using Open3D.

    Args:
        points: np.ndarray of shape (N, 3), 3D points
        colors: np.ndarray of shape (N, 3), RGB colors
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
    o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    DATASET_PATH = 'LLaVA_format'  # Set your dataset path here, like 'EgoDynamic4D/converted_data'
    all_points, all_colors, all_instance_ids = export_one_scan("Apartment_release_golden_skeleton_seq100_10s_sample_M1292_rect")  # Example usage
    visualize_point_cloud(np.vstack(all_points), np.vstack(all_colors))
    