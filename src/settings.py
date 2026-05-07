"""
Pipeline settings for the desktop scene reconstruction project.

This module centralises configurable parameters used across the pipeline.
Values are deliberately kept in one place so that experimentation does not
require editing multiple files.
"""
from pathlib import Path
import numpy as np


# -------------------------------------------------------------------------
# Filesystem layout
# -------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
ASSET_DIR = ROOT / "Data"
RESULT_DIR = ROOT / "outputs"
INTRINSIC_FILE = ROOT / "intrinsic.json"
POSE_FILE = ASSET_DIR / "poses.json"

RESULT_DIR.mkdir(exist_ok=True, parents=True)
(RESULT_DIR / "renders").mkdir(exist_ok=True)
(RESULT_DIR / "masks").mkdir(exist_ok=True)
(RESULT_DIR / "viz").mkdir(exist_ok=True)


# -------------------------------------------------------------------------
# Capture geometry (provided with the dataset)
# -------------------------------------------------------------------------
SENSOR_W = 2560
SENSOR_H = 1440

# Pinhole intrinsics — read from intrinsic.json at load time, but the
# nominal values below are kept here for reference / fallback.
NOMINAL_K = np.array([
    [1477.00974684544, 0.0, 1298.2501500778505],
    [0.0,              1480.4424455584467, 686.8201623541711],
    [0.0,              0.0,                1.0],
], dtype=np.float64)


# Frames available in the data folder
AVAILABLE_FRAMES = (319, 333, 353, 359, 365, 371, 390, 400,
                    426, 449, 461, 468, 471, 496, 515, 531)


# -------------------------------------------------------------------------
# Plane-sweep depth estimation parameters
# -------------------------------------------------------------------------
# We sweep through a discrete set of fronto-parallel depth planes and pick
# the one with the lowest patch-photometric error per pixel.
DEPTH_NEAR = 0.05      # metres — closest plane considered
DEPTH_FAR = 1.20       # metres — farthest plane considered
DEPTH_LEVELS = 96      # number of inverse-depth slices in the sweep
PATCH_RADIUS = 3       # NCC patch is (2*r+1) x (2*r+1)
PHOTO_REF_FRAMES = 4   # number of neighbour frames used as references


# -------------------------------------------------------------------------
# Semantic segmentation parameters (zero-shot, prompt-based)
# -------------------------------------------------------------------------
# We rely on text prompts that describe each connector visually, then take
# the highest-confidence mask returned by the open-vocabulary detector.
PROMPTS = {
    "power_socket":     "an IEC power inlet on a computer back panel",
    "ethernet_socket":  "an RJ45 ethernet port on a computer back panel",
    "vga_socket":       "a blue VGA D-sub connector on a computer back panel",
    "hdmi_socket_left": "the leftmost HDMI port on a computer back panel",
    "usb_socket_top_right": "the top-right USB-A port on a computer back panel",
}

DETECTOR_THRESHOLD = 0.18
SEG_MIN_AREA = 100  # ignore tiny mask fragments


# -------------------------------------------------------------------------
# Pose-fitting parameters
# -------------------------------------------------------------------------
# Each connector is essentially planar — width and height come from the
# segmentation mask back-projected onto the back panel; the third extent
# is the physical protrusion of the connector itself, which we treat as a
# constant prior.
PROTRUSION_PRIOR = 0.006   # metres — half-thickness of the connector body
RANSAC_PLANE_THRESH = 0.004   # metres — distance threshold for plane RANSAC
RANSAC_ITERS = 1500


# -------------------------------------------------------------------------
# Novel-view rendering parameters
# -------------------------------------------------------------------------
RENDER_WIDTH = 1280
RENDER_HEIGHT = 720
SOFT_BLEND_TAU = 0.04        # softness for view-blending weights (metres)
NUM_SOURCE_VIEWS = 5         # closest source frames used for re-projection
