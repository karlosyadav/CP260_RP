"""
Asset reader: loads images, intrinsics, and pose dictionary from disk.

Pose convention: each entry in `poses.json` is a 4x4 camera-to-world
matrix in OpenCV / robotics convention (X right, Y down, Z forward).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import cv2
import numpy as np

from . import settings as S


# -------------------------------------------------------------------------
# Pose helpers
# -------------------------------------------------------------------------
def read_pose_dict(path: Path | None = None) -> Dict[int, np.ndarray]:
    """Return {frame_id: 4x4 c2w matrix} for every pose in the JSON."""
    path = path or S.POSE_FILE
    with open(path, "r") as fh:
        raw = json.load(fh)
    return {int(k): np.asarray(v, dtype=np.float64) for k, v in raw.items()}


def world_to_cam(pose_c2w: np.ndarray) -> np.ndarray:
    """Invert a c2w transform to obtain w2c (assuming SE(3))."""
    R = pose_c2w[:3, :3]
    t = pose_c2w[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def projection_matrix(K: np.ndarray, pose_c2w: np.ndarray) -> np.ndarray:
    """Build P = K [R|t] (world → image)."""
    w2c = world_to_cam(pose_c2w)
    return K @ w2c[:3, :4]


# -------------------------------------------------------------------------
# Intrinsics
# -------------------------------------------------------------------------
def read_intrinsics(path: Path | None = None) -> np.ndarray:
    """Load 3x3 K matrix; fall back to nominal values if file missing."""
    path = path or S.INTRINSIC_FILE
    if path.exists():
        with open(path, "r") as fh:
            payload = json.load(fh)
        # Accept either a `camera_matrix` key or a flat list of 9 floats
        if "camera_matrix" in payload:
            return np.asarray(payload["camera_matrix"], dtype=np.float64)
        if isinstance(payload, list):
            return np.asarray(payload, dtype=np.float64).reshape(3, 3)
    return S.NOMINAL_K.copy()


def scale_intrinsics(K: np.ndarray, sx: float, sy: float | None = None) -> np.ndarray:
    """Adjust focal lengths and principal point for a resized image."""
    sy = sy if sy is not None else sx
    K_new = K.copy()
    K_new[0, 0] *= sx
    K_new[1, 1] *= sy
    K_new[0, 2] *= sx
    K_new[1, 2] *= sy
    return K_new


# -------------------------------------------------------------------------
# Image reading
# -------------------------------------------------------------------------
def _frame_filename(idx: int) -> str:
    return f"frame_{idx:06d}.png"


def read_one_image(idx: int, asset_dir: Path | None = None,
                   resize: float = 1.0) -> np.ndarray | None:
    """Read a single frame as RGB float32 in [0, 1] (or None if missing)."""
    asset_dir = asset_dir or S.ASSET_DIR
    fpath = asset_dir / _frame_filename(idx)
    if not fpath.exists():
        return None
    bgr = cv2.imread(str(fpath))
    if bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if resize != 1.0:
        rgb = cv2.resize(rgb, None, fx=resize, fy=resize,
                         interpolation=cv2.INTER_AREA)
    return (rgb.astype(np.float32) / 255.0)


def read_image_stack(frames: Iterable[int] | None = None,
                     resize: float = 1.0) -> Dict[int, np.ndarray]:
    """Read every available frame into a dictionary of float32 RGB arrays."""
    frames = frames if frames is not None else S.AVAILABLE_FRAMES
    out: Dict[int, np.ndarray] = {}
    for i in frames:
        img = read_one_image(i, resize=resize)
        if img is not None:
            out[i] = img
    return out


# -------------------------------------------------------------------------
# Convenience bundle
# -------------------------------------------------------------------------
def load_full_capture(resize: float = 1.0
                      ) -> Tuple[Dict[int, np.ndarray],
                                 Dict[int, np.ndarray],
                                 np.ndarray]:
    """One-call helper that returns (images, poses, K)."""
    K = read_intrinsics()
    if resize != 1.0:
        K = scale_intrinsics(K, resize)
    poses_all = read_pose_dict()
    images = read_image_stack(resize=resize)
    poses = {i: poses_all[i] for i in images.keys() if i in poses_all}
    return images, poses, K
