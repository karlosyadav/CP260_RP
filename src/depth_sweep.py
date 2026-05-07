"""
Plane-sweep multi-view stereo depth estimation.

Given a reference frame and a set of neighbour frames, we sweep a fronto-
parallel plane through a discrete range of depths in front of the
reference camera and measure photometric consistency with each warped
neighbour. The depth chosen for a pixel is the one with the lowest
aggregated patch error.

This module is intentionally written without using OpenCV's stereo or
triangulation primitives so that it stays distinct from a feature-based
approach.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
import numpy as np

from . import settings as S
from . import asset_io


# -------------------------------------------------------------------------
# Helpers for warping
# -------------------------------------------------------------------------
def _build_pixel_grid(h: int, w: int) -> np.ndarray:
    """Return a (3, H*W) grid of homogeneous pixel coordinates."""
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    grid = np.stack([xs.ravel(), ys.ravel(), np.ones(h * w)], axis=0)
    return grid.astype(np.float32)


def _homography_for_depth(K_ref: np.ndarray,
                          K_src: np.ndarray,
                          pose_ref: np.ndarray,
                          pose_src: np.ndarray,
                          depth: float) -> np.ndarray:
    """
    Plane-induced homography that maps pixels in the reference camera to
    pixels in the source camera, assuming all 3D points lie on a fronto-
    parallel plane at distance `depth` in front of the reference camera.

    For a 3-D point P in the reference camera frame that lies on the plane
    n^T P = depth (with n = (0, 0, 1)^T), the source camera coordinates are
    P_src = R_rel P + t_rel = (R_rel + t_rel n^T / depth) P.
    Hence H = K_src (R_rel + t_rel n^T / depth) K_ref^-1.
    """
    w2c_ref = asset_io.world_to_cam(pose_ref)
    w2c_src = asset_io.world_to_cam(pose_src)
    rel = w2c_src @ np.linalg.inv(w2c_ref)
    R_rel = rel[:3, :3]
    t_rel = rel[:3, 3:4]
    n = np.array([[0.0], [0.0], [1.0]])
    H = K_src @ (R_rel + (t_rel @ n.T) / depth) @ np.linalg.inv(K_ref)
    return H.astype(np.float32)


def _warp(src_img: np.ndarray, H: np.ndarray, out_size: Tuple[int, int]) -> np.ndarray:
    """Warp source image into the reference camera frame via homography."""
    h, w = out_size
    return cv2.warpPerspective(src_img, H, (w, h),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=0)


# -------------------------------------------------------------------------
# Photometric cost — patch-NCC-style, but vectorised with box filters
# -------------------------------------------------------------------------
def _patch_cost(ref_gray: np.ndarray, warped_gray: np.ndarray,
                radius: int) -> np.ndarray:
    """
    Compute negative normalised cross correlation per pixel using
    summed-area filtering. Returns cost in [0, 2] where 0 means perfect
    correlation.
    """
    win = 2 * radius + 1
    box = (win, win)

    mu_r = cv2.boxFilter(ref_gray, ddepth=-1, ksize=box)
    mu_w = cv2.boxFilter(warped_gray, ddepth=-1, ksize=box)

    var_r = cv2.boxFilter(ref_gray ** 2, -1, box) - mu_r ** 2
    var_w = cv2.boxFilter(warped_gray ** 2, -1, box) - mu_w ** 2
    cov = cv2.boxFilter(ref_gray * warped_gray, -1, box) - mu_r * mu_w

    eps = 1e-6
    ncc = cov / np.sqrt(np.maximum(var_r * var_w, eps))
    cost = 1.0 - np.clip(ncc, -1.0, 1.0)
    return cost.astype(np.float32)


# -------------------------------------------------------------------------
# Main entry
# -------------------------------------------------------------------------
def neighbour_pose_distance(pose_a: np.ndarray, pose_b: np.ndarray) -> float:
    """L2 distance between camera centres in world coordinates."""
    return float(np.linalg.norm(pose_a[:3, 3] - pose_b[:3, 3]))


def select_neighbour_frames(ref_id: int,
                            poses: Dict[int, np.ndarray],
                            k: int = S.PHOTO_REF_FRAMES) -> List[int]:
    """Pick the k frames whose camera centres are closest to the reference."""
    pose_ref = poses[ref_id]
    ranked = sorted(
        [fid for fid in poses if fid != ref_id],
        key=lambda fid: neighbour_pose_distance(pose_ref, poses[fid])
    )
    return ranked[:k]


def estimate_depth_map(ref_id: int,
                       images: Dict[int, np.ndarray],
                       poses: Dict[int, np.ndarray],
                       K: np.ndarray,
                       *,
                       n_levels: int = S.DEPTH_LEVELS,
                       d_near: float = S.DEPTH_NEAR,
                       d_far: float = S.DEPTH_FAR,
                       neighbours: List[int] | None = None,
                       progress: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate a per-pixel depth map and confidence map for the reference frame.

    Returns
    -------
    depth_map : (H, W) float32, in metres
    conf_map  : (H, W) float32, in [0, 1] — higher means more reliable
    """
    if neighbours is None:
        neighbours = select_neighbour_frames(ref_id, poses)

    ref = images[ref_id]
    h, w = ref.shape[:2]
    ref_gray = cv2.cvtColor((ref * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    ref_gray = ref_gray.astype(np.float32) / 255.0

    inv_d = np.linspace(1.0 / d_far, 1.0 / d_near, n_levels)
    depths = 1.0 / inv_d

    cost_volume = np.zeros((n_levels, h, w), dtype=np.float32)
    valid_count = np.zeros((n_levels, h, w), dtype=np.float32)

    iterator = enumerate(depths)
    if progress:
        from tqdm.auto import tqdm
        iterator = tqdm(list(iterator), desc=f"  Sweeping depths (frame {ref_id})")

    for level, d in iterator:
        for nid in neighbours:
            H_mat = _homography_for_depth(K, K, poses[ref_id], poses[nid], d)
            warped = _warp(images[nid], H_mat, (h, w))
            warped_gray = cv2.cvtColor((warped * 255).astype(np.uint8),
                                       cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            mask = (warped.sum(axis=-1) > 0).astype(np.float32)
            cost = _patch_cost(ref_gray, warped_gray, radius=S.PATCH_RADIUS)
            cost_volume[level] += cost * mask
            valid_count[level] += mask

    cost_volume /= np.maximum(valid_count, 1.0)
    cost_volume[valid_count < 1] = 1.0  # large penalty if no neighbour saw it

    best_level = np.argmin(cost_volume, axis=0)
    depth_map = depths[best_level].astype(np.float32)

    # Confidence: ratio between best and second-best cost (peak ratio)
    sorted_costs = np.sort(cost_volume, axis=0)
    best = sorted_costs[0]
    second = sorted_costs[min(1, n_levels - 1)]
    conf_map = np.clip(1.0 - best / np.maximum(second, 1e-3), 0.0, 1.0)

    # A simple median filter cleans up speckle without blurring depth edges
    depth_map = cv2.medianBlur(depth_map, 5)

    return depth_map, conf_map


# -------------------------------------------------------------------------
# Lift a depth map to a world-frame point cloud
# -------------------------------------------------------------------------
def depth_to_world_cloud(depth_map: np.ndarray,
                         conf_map: np.ndarray,
                         K: np.ndarray,
                         pose_c2w: np.ndarray,
                         conf_thresh: float = 0.05,
                         stride: int = 4) -> np.ndarray:
    """Convert (H,W) depth into an (N,3) world-frame point cloud."""
    h, w = depth_map.shape
    ys = np.arange(0, h, stride)
    xs = np.arange(0, w, stride)
    xx, yy = np.meshgrid(xs, ys)
    z = depth_map[yy, xx]
    c = conf_map[yy, xx]

    keep = (c > conf_thresh) & (z > 0)
    xx, yy, z = xx[keep], yy[keep], z[keep]
    if xx.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    X = (xx - cx) * z / fx
    Y = (yy - cy) * z / fy
    pts_cam = np.stack([X, Y, z], axis=-1)            # (N, 3)
    pts_h = np.concatenate([pts_cam, np.ones((len(pts_cam), 1))], axis=-1)
    pts_world = (pose_c2w @ pts_h.T).T[:, :3]
    return pts_world.astype(np.float32)
