"""
Helpers for plotting boxes on images, drawing 3D scenes, and saving JSON
in the expected submission format.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

from . import asset_io


# -------------------------------------------------------------------------
# 3D corner enumeration
# -------------------------------------------------------------------------
def obb_corners(center: np.ndarray, extent: np.ndarray,
                rotation: np.ndarray) -> np.ndarray:
    """
    Return the 8 corners of an OBB in world coordinates.

    Convention used by the course evaluator (matching the OLD course
    submission format):
        corner_i = center + R @ (sign_i * extent)
    where ``extent`` are half-edge lengths and ``sign_i`` runs through all
    eight combinations of ±1.
    """
    signs = np.array([
        [-1, -1, -1], [-1, -1, +1], [-1, +1, -1], [-1, +1, +1],
        [+1, -1, -1], [+1, -1, +1], [+1, +1, -1], [+1, +1, +1],
    ], dtype=np.float64)
    local = signs * extent
    world = (rotation @ local.T).T + center
    return world


def project_corners(corners: np.ndarray, K: np.ndarray,
                    pose_c2w: np.ndarray) -> np.ndarray | None:
    w2c = asset_io.world_to_cam(pose_c2w)
    pts_cam = (w2c[:3, :3] @ corners.T + w2c[:3, 3:4]).T
    if (pts_cam[:, 2] <= 0).any():
        return None
    proj = (K @ pts_cam.T).T
    return proj[:, :2] / proj[:, 2:3]


# -------------------------------------------------------------------------
# 2D drawing
# -------------------------------------------------------------------------
def draw_box_overlay(img_uint8: np.ndarray, corners_2d: np.ndarray,
                     colour: Tuple[int, int, int] = (255, 50, 50),
                     label: str = "") -> np.ndarray:
    out = img_uint8.copy()
    edges = [(0, 1), (0, 2), (0, 4),
             (1, 3), (1, 5),
             (2, 3), (2, 6),
             (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
    for i, j in edges:
        p1 = tuple(np.round(corners_2d[i]).astype(int))
        p2 = tuple(np.round(corners_2d[j]).astype(int))
        cv2.line(out, p1, p2, colour, 2, cv2.LINE_AA)
    if label:
        anchor = tuple(np.round(corners_2d.min(axis=0)).astype(int))
        cv2.putText(out, label, (anchor[0], max(20, anchor[1] - 6)),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, colour, 2, cv2.LINE_AA)
    return out


def draw_mask_overlay(img_uint8: np.ndarray, mask: np.ndarray,
                      colour: Tuple[int, int, int] = (60, 200, 60),
                      alpha: float = 0.4) -> np.ndarray:
    out = img_uint8.copy().astype(np.float32)
    layer = np.zeros_like(out)
    layer[mask] = colour
    out = out * (1 - alpha * mask[..., None]) + layer * alpha
    return out.clip(0, 255).astype(np.uint8)


# -------------------------------------------------------------------------
# 3D scene plot (matplotlib)
# -------------------------------------------------------------------------
def plot_scene_3d(poses: Dict[int, np.ndarray],
                  records: List[Dict],
                  save_path: Path | str | None = None,
                  point_cloud: np.ndarray | None = None) -> None:
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    if point_cloud is not None and len(point_cloud) > 0:
        sample = point_cloud
        if len(sample) > 5000:
            idx = np.random.choice(len(sample), 5000, replace=False)
            sample = sample[idx]
        ax.scatter(sample[:, 0], sample[:, 1], sample[:, 2],
                   s=0.5, c=sample[:, 2], cmap="viridis", alpha=0.4)

    for fid, p in poses.items():
        c = p[:3, 3]
        forward = p[:3, 2]
        ax.scatter(*c, c="tab:blue", s=22)
        ax.quiver(*c, *forward, length=0.05, color="tab:blue", alpha=0.6)

    palette = ["tab:red", "tab:green", "tab:orange", "tab:purple", "tab:cyan"]
    for i, rec in enumerate(records):
        obb = rec["obb"]
        corners = obb_corners(np.asarray(obb["center"]),
                              np.asarray(obb["extent"]),
                              np.asarray(obb["rotation"]))
        col = palette[i % len(palette)]
        edges = [(0, 1), (0, 2), (0, 4),
                 (1, 3), (1, 5),
                 (2, 3), (2, 6),
                 (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
        for a, b in edges:
            ax.plot(*zip(corners[a], corners[b]), color=col, lw=1.5)
        ax.text(*np.asarray(obb["center"]), "  " + rec["entity"], color=col,
                fontsize=8)

    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.set_title("Scene reconstruction — cameras and oriented bounding boxes")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


# -------------------------------------------------------------------------
# Submission JSON
# -------------------------------------------------------------------------
def write_answers_json(records: List[Dict], path: Path | str) -> None:
    with open(path, "w") as fh:
        json.dump(records, fh, indent=2)
    print(f"  • saved {len(records)} entities → {path}")
