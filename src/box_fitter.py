"""
Build oriented bounding boxes from segmentation masks + depth.

This module does **not** triangulate features.  Instead, for each entity
it gathers the 3D points produced by the depth-sweep that fall inside the
projection of the connector mask, fits a RANSAC plane to those points
(this is the back-panel surface), and constructs an OBB whose first two
axes lie in the panel and the third is the panel normal.

The third extent is supplied as a physical prior (the connector body
protrudes by a few millimetres), which matches how connectors actually
behave in the real world.

Output convention is identical to the one used by the course evaluation:

    {
        "entity":  str,
        "obb": {
            "center":   [x, y, z],          # metres, world frame
            "extent":   [ex, ey, ez],       # half-edge lengths, metres
            "rotation": [[r00, r01, r02],
                         [r10, r11, r12],
                         [r20, r21, r22]]   # rows are the box axes
        }
    }
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from . import settings as S
from . import asset_io
from .vlm_segmenter import MaskHit


# -------------------------------------------------------------------------
# Per-entity 3D point gathering
# -------------------------------------------------------------------------
def _back_project_mask_pixels(mask: np.ndarray,
                              depth_map: np.ndarray,
                              conf_map: np.ndarray,
                              K: np.ndarray,
                              pose_c2w: np.ndarray,
                              conf_thresh: float = 0.05) -> np.ndarray:
    """
    Lift the masked pixels of a single frame to world-space coordinates,
    using the depth map estimated for that same frame.
    """
    ys, xs = np.where(mask & (conf_map > conf_thresh) & (depth_map > 0))
    if xs.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    z = depth_map[ys, xs]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    X = (xs - cx) * z / fx
    Y = (ys - cy) * z / fy
    pts_cam = np.stack([X, Y, z], axis=-1)
    pts_h = np.concatenate([pts_cam, np.ones((len(pts_cam), 1))], axis=-1)
    pts_world = (pose_c2w @ pts_h.T).T[:, :3]
    return pts_world.astype(np.float32)


def gather_entity_points(entity_hits: List[MaskHit],
                         depth_maps: Dict[int, np.ndarray],
                         conf_maps: Dict[int, np.ndarray],
                         K: np.ndarray,
                         poses: Dict[int, np.ndarray]) -> np.ndarray:
    """
    Combine 3D points produced by every hit of a given entity (across
    multiple frames) into a single point cloud in the world frame.
    """
    chunks = []
    for hit in entity_hits:
        if hit.frame_id not in depth_maps:
            continue
        chunks.append(_back_project_mask_pixels(
            hit.mask,
            depth_maps[hit.frame_id],
            conf_maps[hit.frame_id],
            K, poses[hit.frame_id]
        ))
    if not chunks:
        return np.zeros((0, 3), dtype=np.float32)
    return np.concatenate(chunks, axis=0)


# -------------------------------------------------------------------------
# RANSAC plane fitting
# -------------------------------------------------------------------------
def _ransac_plane(points: np.ndarray,
                  thresh: float = S.RANSAC_PLANE_THRESH,
                  iters: int = S.RANSAC_ITERS,
                  rng: np.random.Generator | None = None
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Robustly fit a plane to a 3D point cloud.

    Returns
    -------
    normal       : (3,) unit normal vector
    centroid     : (3,) centroid of the inliers
    inlier_mask  : (N,) bool, points within `thresh` of the plane
    """
    if rng is None:
        rng = np.random.default_rng(0)
    n = len(points)
    if n < 3:
        return (np.array([0.0, 0.0, 1.0]),
                points.mean(axis=0) if n else np.zeros(3),
                np.zeros(n, dtype=bool))

    best_inliers = None
    best_count = -1
    for _ in range(iters):
        idx = rng.choice(n, size=3, replace=False)
        p0, p1, p2 = points[idx]
        v1, v2 = p1 - p0, p2 - p0
        nrm = np.cross(v1, v2)
        norm_len = np.linalg.norm(nrm)
        if norm_len < 1e-9:
            continue
        nrm /= norm_len
        d = -nrm @ p0
        dists = np.abs(points @ nrm + d)
        inliers = dists < thresh
        if inliers.sum() > best_count:
            best_count = int(inliers.sum())
            best_inliers = inliers

    if best_inliers is None or best_count < 3:
        # Degenerate fall-back: use full PCA
        centroid = points.mean(axis=0)
        _, _, vt = np.linalg.svd(points - centroid, full_matrices=False)
        return vt[-1], centroid, np.ones(n, dtype=bool)

    inlier_pts = points[best_inliers]
    centroid = inlier_pts.mean(axis=0)
    _, _, vt = np.linalg.svd(inlier_pts - centroid, full_matrices=False)
    normal = vt[-1] / np.linalg.norm(vt[-1])
    return normal, centroid, best_inliers


# -------------------------------------------------------------------------
# Build the OBB from in-plane spread
# -------------------------------------------------------------------------
@dataclass
class OBB:
    center: np.ndarray
    extent: np.ndarray
    rotation: np.ndarray


def _build_obb_on_plane(points: np.ndarray,
                        normal: np.ndarray,
                        centroid: np.ndarray,
                        protrusion: float = S.PROTRUSION_PRIOR) -> OBB:
    """
    Project the inlier points into the plane defined by `normal`
    (anchored at `centroid`), find the dominant in-plane direction with
    PCA, then size the box to the extreme projections plus a thickness
    given by the protrusion prior.
    """
    n = normal / np.linalg.norm(normal)

    # Pick any vector not parallel to n
    helper = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = helper - n * (helper @ n)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    v /= np.linalg.norm(v)

    rel = points - centroid
    coords_uv = np.stack([rel @ u, rel @ v], axis=-1)

    # Refine in-plane orientation with PCA over 2D coords_uv
    if len(coords_uv) >= 2:
        c = coords_uv.mean(axis=0)
        cov = np.cov((coords_uv - c).T)
        evals, evecs = np.linalg.eigh(cov)
        order = np.argsort(evals)[::-1]
        evecs = evecs[:, order]
        # major axis in 3D
        major_2d = evecs[:, 0]
        u_new = (u * major_2d[0] + v * major_2d[1])
        u_new /= np.linalg.norm(u_new)
        v_new = np.cross(n, u_new)
        v_new /= np.linalg.norm(v_new)
        u, v = u_new, v_new
        coords_uv = np.stack([rel @ u, rel @ v], axis=-1)

    if len(coords_uv) == 0:
        half_u = half_v = 0.005
        center_uv = np.zeros(2)
    else:
        u_min, u_max = np.percentile(coords_uv[:, 0], [2, 98])
        v_min, v_max = np.percentile(coords_uv[:, 1], [2, 98])
        half_u = max((u_max - u_min) / 2.0, 0.003)
        half_v = max((v_max - v_min) / 2.0, 0.003)
        center_uv = np.array([(u_min + u_max) / 2.0, (v_min + v_max) / 2.0])

    centre_world = centroid + center_uv[0] * u + center_uv[1] * v

    # The OBB axes are u, v, n.  The course evaluator computes corners as
    #     corner = center + R @ (sign * extent)
    # which means ``R`` must have the box axes as its COLUMNS so that the
    # k-th column of R is the world-frame direction of the k-th local axis.
    R = np.stack([u, v, n], axis=1)  # columns are axes

    # Force a consistent normal direction (column 2) so that the box
    # convention remains stable across frames.
    if R[0, 2] < 0:
        R[:, 2] = -R[:, 2]
        # Re-orthogonalise column 1 to keep the matrix right-handed
        R[:, 1] = np.cross(R[:, 2], R[:, 0])
        R[:, 1] /= np.linalg.norm(R[:, 1])

    extent = np.array([half_u, half_v, protrusion], dtype=np.float64)
    return OBB(center=centre_world.astype(np.float64),
               extent=extent,
               rotation=R.astype(np.float64))


# -------------------------------------------------------------------------
# Top-level
# -------------------------------------------------------------------------
def fit_box_for_entity(entity: str,
                       hits: List[MaskHit],
                       depth_maps: Dict[int, np.ndarray],
                       conf_maps: Dict[int, np.ndarray],
                       K: np.ndarray,
                       poses: Dict[int, np.ndarray]) -> OBB | None:
    pts = gather_entity_points(hits, depth_maps, conf_maps, K, poses)
    if len(pts) < 30:
        print(f"  [skip] `{entity}` produced only {len(pts)} 3D points.")
        return None
    normal, centroid, inliers = _ransac_plane(pts)
    inlier_pts = pts[inliers] if inliers.any() else pts
    return _build_obb_on_plane(inlier_pts, normal, centroid)


def fit_all_boxes(hits_by_entity: Dict[str, List[MaskHit]],
                  depth_maps: Dict[int, np.ndarray],
                  conf_maps: Dict[int, np.ndarray],
                  K: np.ndarray,
                  poses: Dict[int, np.ndarray]
                  ) -> List[Dict]:
    """Return a list of submission-format records for every entity."""
    records = []
    for entity, hits in hits_by_entity.items():
        if not hits:
            continue
        box = fit_box_for_entity(entity, hits, depth_maps, conf_maps, K, poses)
        if box is None:
            continue
        records.append({
            "entity": entity,
            "obb": {
                "center":   [float(v) for v in box.center],
                "extent":   [float(v) for v in box.extent],
                "rotation": [[float(v) for v in row] for row in box.rotation],
            },
        })
    return records
