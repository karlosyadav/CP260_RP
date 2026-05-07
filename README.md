# Scene Reconstruction with Plane-Sweep Depth and Open-Vocabulary Segmentation

End-to-end pipeline that takes 16 posed RGB photographs of a desktop computer and produces:

1. **Depth maps** for every input view (estimated by a multi-view plane-sweep stereo).
2. **Pixel-precise masks** for named connectors on the back panel (OWL-ViT detections refined by MobileSAM).
3. **Oriented 3D bounding boxes** for those connectors (RANSAC plane fit + in-plane PCA + protrusion prior).
4. **Novel-view RGB images** at user-specified poses (depth-aware forward warping + soft view-blending).

This is the deliverable for the CP260 “Robot Perception” final project.

---

## Why this design?

The dataset is **sparse** — only sixteen RGB images covering a back-panel of a workstation. That ruled out training-heavy methods like NeRF or Gaussian Splatting from scratch (they overfit badly with so few views in the few hours available on a free Colab GPU).

Instead the pipeline is built around three simple ideas:

| Question | Answer |
|---|---|
| How do we get geometry? | A *plane-sweep* MVS — for every reference frame we sweep a fronto-parallel depth plane and pick the depth whose patch-NCC against the closest neighbours is best. |
| How do we know which pixels belong to which connector? | An *open-vocabulary* detector (OWL-ViT) fed with text prompts like *“an RJ45 ethernet port on a computer back panel”*, refined to a tight mask using MobileSAM. |
| How do we go from pixels + depth to an OBB? | Back-project the masked pixels of every detected frame, fit a RANSAC plane (the panel surface), align two box axes inside the plane via PCA, and use a 6 mm protrusion prior for the third axis. |

No part of the pipeline needs hand-labelled bounding boxes; adding a new entity at evaluation time is a one-line change to a prompt dictionary.

---

## Repository layout

```
.
├── src/
│   ├── __init__.py
│   ├── settings.py          # all parameters in one place
│   ├── asset_io.py          # image / pose / intrinsic loaders
│   ├── depth_sweep.py       # plane-sweep MVS
│   ├── vlm_segmenter.py     # OWL-ViT + MobileSAM segmentation
│   ├── box_fitter.py        # RANSAC plane → OBB
│   ├── view_render.py       # novel-view synthesis
│   ├── render_helpers.py    # plotting + JSON I/O
│   └── driver.py            # CLI entry point that runs everything
├── notebook/
│   └── pipeline.ipynb       # Colab walk-through (mirrors driver.py)
├── outputs/
│   ├── answers.json         # final OBBs in the submission format
│   ├── viz/                 # depth previews, 3-D scene plot
│   └── renders/             # synthesised novel views
├── docs/
│   └── report.pdf           # full project report
└── requirements.txt
```

---

## Running

### Colab (recommended, ~25 min on a free T4)

1. Open `notebook/pipeline.ipynb`.
2. Run the cells top-to-bottom. The notebook installs dependencies, downloads MobileSAM weights, asks you to upload `Data.zip`, and produces `outputs/answers.json`.

### Local

```bash
pip install -r requirements.txt
# Place the dataset at ./Data/ and download MobileSAM weights:
wget https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt
python -m src.driver --sam-ckpt mobile_sam.pt
```

Add `--fast` to halve the resolution of the depth sweep while debugging.

---

## Output format

`outputs/answers.json` is a list, one entry per detected entity:

```json
[
  {
    "entity": "power_socket",
    "obb": {
      "center":   [x, y, z],
      "extent":   [ex, ey, ez],
      "rotation": [[r00, r01, r02],
                   [r10, r11, r12],
                   [r20, r21, r22]]
    }
  },
  ...
]
```

This matches the convention the course evaluator expects (`corner_i = center + R^T @ (sign_i * extent)`).

---

## Adding new entities at evaluation time

Open `src/settings.py` and append a row to the `PROMPTS` dictionary:

```python
PROMPTS["audio_jack"] = "a green or pink 3.5 mm audio jack on a computer back panel"
```

Re-run the driver — the new entity is picked up automatically.

---

## Dependencies

The headline packages are listed in `requirements.txt`. The pipeline targets:

* PyTorch 2.x with CUDA 11/12
* `transformers >= 4.41`
* `mobile-sam` (installed from the upstream GitHub repo)
* OpenCV, NumPy, Matplotlib, tqdm

A T4 (16 GB) is enough for the full-resolution run.

---

## Limitations and possible follow-ups

* The protrusion of every connector is a constant 6 mm prior. A photometric refinement step that searches along the panel normal for an actual change in colour gradient would remove this assumption.
* The plane-sweep is fronto-parallel — slanted-plane sweeps would help on the side of the case.
* Forward warping leaves small holes near depth discontinuities. A learned in-painting network would clean those up.
* For higher accuracy on novel-view PSNR, one could initialise a 3D Gaussian Splat from the dense depth point cloud and do 2 000 iterations of refinement. This is left as a stretch goal.

---

## Acknowledgements

* OWL-ViT (Minderer et al., 2022) — open-vocabulary detector used out of the box.
* MobileSAM (Zhang et al., 2023) — small, fast Segment-Anything variant.
* Course staff for the dataset and the OBB convention.
