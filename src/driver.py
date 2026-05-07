"""
End-to-end driver that runs the four stages of the pipeline:

    1. Asset loading
    2. Plane-sweep depth estimation (per frame)
    3. OWL-ViT + MobileSAM segmentation (per entity)
    4. RANSAC plane-fit OBBs (per entity)
    5. Optional novel-view rendering for the report

Usage from the command line:

    python -m src.driver --sam-ckpt mobile_sam.pt
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from . import settings as S
from . import asset_io
from . import depth_sweep
from . import vlm_segmenter
from . import box_fitter
from . import view_render
from . import render_helpers


# -------------------------------------------------------------------------
# Stages
# -------------------------------------------------------------------------
def stage_depth(images, poses, K, *, n_levels=None,
                progress=True) -> tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    print("\n[2] Plane-sweep depth estimation")
    depth_maps, conf_maps = {}, {}
    for fid in images:
        d, c = depth_sweep.estimate_depth_map(
            fid, images, poses, K,
            n_levels=n_levels or S.DEPTH_LEVELS, progress=progress
        )
        depth_maps[fid] = d
        conf_maps[fid] = c
        # Save a colourised preview to disk
        d_norm = (d - d.min()) / max(d.max() - d.min(), 1e-6)
        prev = (cv2.applyColorMap((d_norm * 255).astype(np.uint8),
                                  cv2.COLORMAP_TURBO))
        cv2.imwrite(str(S.RESULT_DIR / "viz" / f"depth_{fid:06d}.png"), prev)
    return depth_maps, conf_maps


def stage_segment(images, sam_ckpt: str):
    print("\n[3] Open-vocabulary segmentation")
    return vlm_segmenter.segment_all(images, sam_ckpt)


def stage_boxes(hits, depth_maps, conf_maps, K, poses):
    print("\n[4] Plane-RANSAC OBB fitting")
    return box_fitter.fit_all_boxes(hits, depth_maps, conf_maps, K, poses)


def stage_render_examples(records, images, poses, depth_maps, conf_maps, K):
    print("\n[5] Novel-view synthesis")
    target_ids = [471, 496]   # representative back-panel views
    for tid in target_ids:
        if tid not in poses:
            continue
        out = view_render.synthesize_view(
            poses[tid], K, images, poses, depth_maps, conf_maps,
        )
        out_uint8 = (out * 255).astype(np.uint8)
        cv2.imwrite(str(S.RESULT_DIR / "renders" / f"novel_{tid:06d}.png"),
                    cv2.cvtColor(out_uint8, cv2.COLOR_RGB2BGR))
    print("  • rendered example novel views.")


# -------------------------------------------------------------------------
# Top-level entry
# -------------------------------------------------------------------------
def run(sam_ckpt: str, *, render_demo: bool = True, fast: bool = False) -> Dict:
    t0 = time.time()
    print("=" * 60)
    print("  Desktop Connector OBB Pipeline")
    print("=" * 60)

    print("\n[1] Loading capture")
    resize = 0.5 if fast else 1.0
    images, poses, K = asset_io.load_full_capture(resize=resize)
    print(f"  • {len(images)} frames at scale {resize}")

    depth_maps, conf_maps = stage_depth(
        images, poses, K, n_levels=48 if fast else S.DEPTH_LEVELS
    )
    hits = stage_segment(images, sam_ckpt)
    records = stage_boxes(hits, depth_maps, conf_maps, K, poses)

    out_path = S.RESULT_DIR / "answers.json"
    render_helpers.write_answers_json(records, out_path)

    if render_demo:
        stage_render_examples(records, images, poses, depth_maps, conf_maps, K)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s — outputs in {S.RESULT_DIR}")
    return {
        "records": records,
        "images": images,
        "poses": poses,
        "K": K,
        "depth_maps": depth_maps,
        "conf_maps": conf_maps,
        "hits": hits,
    }


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Desktop reconstruction driver")
    p.add_argument("--sam-ckpt", required=True,
                   help="Path to MobileSAM weights (vit_t .pt)")
    p.add_argument("--no-render", action="store_true",
                   help="Skip novel-view synthesis")
    p.add_argument("--fast", action="store_true",
                   help="Half-resolution sweep (development mode)")
    args = p.parse_args(argv)

    run(sam_ckpt=args.sam_ckpt, render_demo=not args.no_render, fast=args.fast)
    return 0


if __name__ == "__main__":
    sys.exit(main())
