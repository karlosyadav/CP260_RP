"""
Open-vocabulary connector segmentation.

Strategy: for each text prompt we use OWL-ViT (a CLIP-style open-vocabulary
detector) to obtain candidate boxes per frame, then refine each box into a
pixel-precise mask using MobileSAM. The result is a dictionary that maps
{entity_name: {frame_id: BinaryMask}}.

We deliberately use OWL-ViT + MobileSAM rather than Grounding-DINO + SAM-H
because the smaller variants are easier to run on a free Colab GPU and
because they constitute a different architectural family.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from . import settings as S


# -------------------------------------------------------------------------
# Lightweight result container — kept module-level so it can be imported
# without pulling in heavy dependencies.
# -------------------------------------------------------------------------
@dataclass
class MaskHit:
    box_xyxy: Tuple[int, int, int, int]
    score: float
    mask: np.ndarray          # (H, W) bool
    frame_id: int


# -------------------------------------------------------------------------
# Lazy-loaded model singletons
# -------------------------------------------------------------------------
_owlvit_state = {"model": None, "processor": None}
_sam_state = {"predictor": None}


def _device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_owlvit():
    if _owlvit_state["model"] is None:
        import torch
        from transformers import Owlv2Processor, Owlv2ForObjectDetection

        name = "google/owlv2-base-patch16-ensemble"

        proc = Owlv2Processor.from_pretrained(name)

        mdl = (
            Owlv2ForObjectDetection
            .from_pretrained(name)
            .to(_device())
            .eval()
        )

        _owlvit_state["processor"] = proc
        _owlvit_state["model"] = mdl

    return _owlvit_state["processor"], _owlvit_state["model"]


def _load_mobile_sam(checkpoint_path: str):
    if _sam_state["predictor"] is None:
        # MobileSAM uses the same SamPredictor wrapper but a smaller backbone
        from mobile_sam import sam_model_registry, SamPredictor

        sam = sam_model_registry["vit_t"](checkpoint=checkpoint_path)

        sam.to(_device()).eval()

        _sam_state["predictor"] = SamPredictor(sam)

    return _sam_state["predictor"]


# -------------------------------------------------------------------------
# Detection
# -------------------------------------------------------------------------
def _detect_in_frame(
    rgb_uint8: np.ndarray,
    prompt: str,
    thresh: float
) -> List[Tuple[Tuple[int, int, int, int], float]]:

    import torch
    from PIL import Image

    proc, mdl = _load_owlvit()

    pil = Image.fromarray(rgb_uint8)

    with torch.no_grad():

        # -------------------------------------------------
        # Clean prompt for OWLv2 stability
        # -------------------------------------------------
        prompt = str(prompt).strip()

        # Skip empty prompts
        if not prompt:
            return []

        # Prevent tokenizer / embedding mismatch issues
        prompt = prompt[:64]

        inputs = proc(
            images=pil,
            text=[prompt],
            return_tensors="pt"
        ).to(_device())

        outputs = mdl(**inputs)

        target_size = torch.tensor([pil.size[::-1]]).to(_device())

        results = proc.post_process_object_detection(
            outputs=outputs,
            target_sizes=target_size,
            threshold=thresh,
        )[0]

    boxes = results["boxes"].cpu().numpy()
    scores = results["scores"].cpu().numpy()

    out = []

    for box, sc in zip(boxes, scores):
        x1, y1, x2, y2 = [int(round(v)) for v in box]

        out.append(((x1, y1, x2, y2), float(sc)))

    return out


# -------------------------------------------------------------------------
# Segmentation
# -------------------------------------------------------------------------
def _segment_with_box(
    rgb_uint8: np.ndarray,
    box: Tuple[int, int, int, int],
    sam_ckpt: str
) -> np.ndarray:

    pred = _load_mobile_sam(sam_ckpt)

    pred.set_image(rgb_uint8)

    masks, scores, _ = pred.predict(
        box=np.array(box, dtype=np.float32),
        multimask_output=False,
    )

    return masks[0].astype(bool)


# -------------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------------
def segment_entity(
    entity: str,
    prompt: str,
    images: Dict[int, np.ndarray],
    sam_ckpt: str,
    thresh: float = S.DETECTOR_THRESHOLD,
    keep_top_k: int = 1
) -> List[MaskHit]:
    """
    Run OWL-ViT + MobileSAM on every frame for a given entity prompt.

    Returns a list of MaskHits sorted by detector score (descending).
    """

    hits: List[MaskHit] = []

    for fid, img in images.items():

        rgb_uint8 = (img * 255).astype(np.uint8)

        detections = _detect_in_frame(
            rgb_uint8,
            prompt,
            thresh
        )

        if not detections:
            continue

        # Use highest-confidence detection
        detections.sort(key=lambda x: x[1], reverse=True)

        for box, score in detections[:keep_top_k]:

            try:
                mask = _segment_with_box(
                    rgb_uint8,
                    box,
                    sam_ckpt
                )

            except Exception as exc:                       # pragma: no cover
                print(f"  [warn] SAM failed on frame {fid}: {exc}")
                continue

            if mask.sum() < S.SEG_MIN_AREA:
                continue

            hits.append(
                MaskHit(
                    box_xyxy=box,
                    score=score,
                    mask=mask,
                    frame_id=fid,
                )
            )

    hits.sort(key=lambda h: h.score, reverse=True)

    return hits


def segment_all(
    images: Dict[int, np.ndarray],
    sam_ckpt: str,
    prompts: Dict[str, str] | None = None
) -> Dict[str, List[MaskHit]]:
    """
    Run segmentation for every entity in the prompt dictionary.
    """

    prompts = prompts or S.PROMPTS

    out: Dict[str, List[MaskHit]] = {}

    for entity, prompt in prompts.items():

        print(f"  • detecting `{entity}` …")

        try:
            out[entity] = segment_entity(
                entity,
                prompt,
                images,
                sam_ckpt
            )

            print(
                f"     → {len(out[entity])} frames with detection"
            )

        except Exception as exc:
            print(f"     [warn] detection failed for {entity}: {exc}")
            out[entity] = []

    return out
