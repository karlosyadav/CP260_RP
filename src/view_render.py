"""
Image-based view synthesis using estimated depth maps.

Instead of training a neural radiance field, this module performs
**depth-aware forward warping**: every source frame contributes a warped
RGB image to the novel viewpoint by re-projecting its pixels through its
own depth map. The contributions are blended with weights that depend on
viewing-angle similarity and per-pixel depth confidence.

This is fast, deterministic, and entirely classical — and it gives a
sensible novel view from a small number of input images.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
import numpy as np

from . import settings as S
from . import asset_io


# -------------------------------------------------------------------------
# Camera-centre based source ranking
# -------------------------------------------------------------------------
def _rank_source_views(target_pose: np.ndarray,
                       source_poses: Dict[int, np.ndarray],
                       k: int = S.NUM_SOURCE_VIEWS) -> List[int]:
    target_c = target_pose[:3, 3]
    target_z = target_pose[:3, 2]  # forward axis

    def _score(fid: int) -> float:
        p = source_poses[fid]
        # Distance + angular alignment
        dist = np.linalg.norm(p[:3, 3] - target_c)
        ang = 1.0 - float(np.clip(np.dot(p[:3, 2], target_z), -1, 1))
        return dist + 0.5 * ang

    return sorted(source_poses.keys(), key=_score)[:k]


# -------------------------------------------------------------------------
# Forward warp a single source frame
# -------------------------------------------------------------------------
def _forward_warp(rgb_src: np.ndarray,
                  depth_src: np.ndarray,
                  conf_src: np.ndarray,
                  K_src: np.ndarray,
                  pose_src: np.ndarray,
                  K_tgt: np.ndarray,
                  pose_tgt: np.ndarray,
                  out_h: int,
                  out_w: int,
                  conf_thresh: float = 0.05
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Warp an RGB+depth source view into the target camera frame.

    Returns
    -------
    rgb_tgt   : (H, W, 3) blended RGB (zeros where empty)
    depth_tgt : (H, W) depth in target frame (inf where empty)
    weight    : (H, W) per-pixel weight contributed by the source
    """
    h_src, w_src = depth_src.shape
    ys, xs = np.where((conf_src > conf_thresh) & (depth_src > 0))
    if xs.size == 0:
        return (np.zeros((out_h, out_w, 3), dtype=np.float32),
                np.full((out_h, out_w), np.inf, dtype=np.float32),
                np.zeros((out_h, out_w), dtype=np.float32))

    z = depth_src[ys, xs]
    fx_s, fy_s = K_src[0, 0], K_src[1, 1]
    cx_s, cy_s = K_src[0, 2], K_src[1, 2]
    Xc = (xs - cx_s) * z / fx_s
    Yc = (ys - cy_s) * z / fy_s
    Zc = z
    pts_cam = np.stack([Xc, Yc, Zc, np.ones_like(Xc)], axis=-1)
    pts_world = (pose_src @ pts_cam.T).T

    w2c_t = asset_io.world_to_cam(pose_tgt)
    pts_tgt = (w2c_t @ pts_world.T).T[:, :3]
    valid = pts_tgt[:, 2] > 1e-3
    pts_tgt = pts_tgt[valid]
    rgb_vals = rgb_src[ys[valid], xs[valid]]

    fx_t, fy_t = K_tgt[0, 0], K_tgt[1, 1]
    cx_t, cy_t = K_tgt[0, 2], K_tgt[1, 2]
    u = (pts_tgt[:, 0] * fx_t / pts_tgt[:, 2]) + cx_t
    v = (pts_tgt[:, 1] * fy_t / pts_tgt[:, 2]) + cy_t
    u_i = np.round(u).astype(np.int32)
    v_i = np.round(v).astype(np.int32)
    in_bounds = (u_i >= 0) & (u_i < out_w) & (v_i >= 0) & (v_i < out_h)
    u_i, v_i = u_i[in_bounds], v_i[in_bounds]
    rgb_vals = rgb_vals[in_bounds]
    z_tgt = pts_tgt[in_bounds, 2]

    rgb_acc = np.zeros((out_h, out_w, 3), dtype=np.float32)
    depth_acc = np.full((out_h, out_w), np.inf, dtype=np.float32)
    weight = np.zeros((out_h, out_w), dtype=np.float32)

    # Z-buffer style update — keep nearest depth per pixel
    flat_idx = v_i * out_w + u_i
    order = np.argsort(z_tgt)[::-1]   # paint far first, then near overwrites
    for k in order:
        f = flat_idx[k]
        rr, cc = v_i[k], u_i[k]
        if z_tgt[k] < depth_acc[rr, cc]:
            depth_acc[rr, cc] = z_tgt[k]
            rgb_acc[rr, cc] = rgb_vals[k]
            weight[rr, cc] = 1.0

    # Light hole-fill via dilation of the colour image (radius 1)
    weight_dil = cv2.dilate(weight, np.ones((3, 3), np.float32))
    rgb_dil = cv2.dilate(rgb_acc, np.ones((3, 3), np.float32))
    fill = (weight == 0) & (weight_dil > 0)
    rgb_acc[fill] = rgb_dil[fill]
    weight[fill] = 0.4
    return rgb_acc, depth_acc, weight


# -------------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------------
def synthesize_view(target_pose: np.ndarray,
                    K: np.ndarray,
                    images: Dict[int, np.ndarray],
                    poses: Dict[int, np.ndarray],
                    depth_maps: Dict[int, np.ndarray],
                    conf_maps: Dict[int, np.ndarray],
                    out_size: Tuple[int, int] | None = None,
                    n_sources: int = S.NUM_SOURCE_VIEWS,
                    soft_tau: float = S.SOFT_BLEND_TAU) -> np.ndarray:
    """
    Render a novel RGB image at `target_pose`.

    Parameters
    ----------
    out_size : (H, W) — defaults to the original capture resolution.
    """
    if out_size is None:
        any_img = next(iter(images.values()))
        h, w = any_img.shape[:2]
    else:
        h, w = out_size

    src_ids = _rank_source_views(target_pose, poses, k=n_sources)
    target_c = target_pose[:3, 3]

    rgb_sum = np.zeros((h, w, 3), dtype=np.float32)
    weight_sum = np.zeros((h, w), dtype=np.float32)

    for sid in src_ids:
        if sid not in depth_maps:
            continue
        cam_dist = np.linalg.norm(poses[sid][:3, 3] - target_c)
        view_w = float(np.exp(-cam_dist / soft_tau))

        rgb_w, _, w_pix = _forward_warp(
            images[sid], depth_maps[sid], conf_maps[sid], K, poses[sid],
            K, target_pose, h, w
        )
        rgb_sum += rgb_w * (w_pix * view_w)[..., None]
        weight_sum += w_pix * view_w

    out = np.zeros_like(rgb_sum)
    valid = weight_sum > 1e-6
    out[valid] = rgb_sum[valid] / weight_sum[valid, None]
    return np.clip(out, 0.0, 1.0)
