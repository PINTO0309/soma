"""Public integration API: use SOMA from external programs in a few lines.

Two integration levels:

Level B — tracker only, bring your own detector (numpy-only)::

    from soma import Detection, SomaTracker, tracker_config

    trk = SomaTracker(tracker_config("somar-pv", fps=25), record_rows=False)
    for frame_dets in stream:                       # your detector + ReID
        tracks = trk.update([Detection(box=(x1, y1, x2, y2), score=s,
                                       embedding=e) for ... in frame_dets])
        for t in tracks:
            ...  # t.tid, t.box, t.score, t.ghost

Level A — full pipeline, frames in / tracks out (requires the ``perception``
extra: opencv + onnxruntime)::

    from soma import SomaVideoTracker

    vt = SomaVideoTracker(preset="somar-pv",
                          detector="models/yolov9_e_wholebody28_refine_Nx3HxW.onnx",
                          reid="models/personvit_vits16_ain_unified_aug_n.onnx",
                          fps=30)
    for frame_bgr in video:
        tracks = vt.update(frame_bgr)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .presets import VARIANTS, tracker_config
from .tokens import N_SLOTS, AnatomicalToken
from .tracker import SomaTracker, TrackResult


@dataclass
class Detection:
    """A plain detection for the bring-your-own-detector path.

    ``box`` is (x1, y1, x2, y2) in frame coordinates. ``embedding`` (optional)
    should be an L2-normalized appearance vector; it is renormalized
    defensively. Without wholebody structure the part-OKS / head-orientation
    channels stay silent and association runs on IoU, embedding, attributes
    and the memory/revival stack.
    """

    box: tuple[float, float, float, float]
    score: float
    embedding: "np.ndarray | None" = None
    gender: int = -1        # 0 male, 1 female, -1 unknown
    generation: int = -1    # 0 adult, 1 child, -1 unknown


def token_from_detection(det: Detection) -> AnatomicalToken:
    """Build a parts-less anatomical token from a plain detection."""
    box = np.asarray(det.box, dtype=np.float32)
    emb = None
    if det.embedding is not None:
        emb = np.asarray(det.embedding, dtype=np.float32)
        n = float(np.linalg.norm(emb))
        emb = emb / n if n > 1e-6 else None
    return AnatomicalToken(
        anchor="body",
        body_box=box,
        body_score=float(det.score),
        box_proxy=box.copy(),
        presence=np.zeros(N_SLOTS, dtype=bool),
        points=np.full((N_SLOTS, 2), np.nan, dtype=np.float32),
        generation=int(det.generation),
        gender=int(det.gender),
        embedding=emb,
        head_box=None,
        head_dir=None,
        head_dir_conf=0.0,
    )


class SomaVideoTracker:
    """Frames in, tracks out — the full SOMA / SOMA-R pipeline as one object.

    Wraps the perception layer (wholebody detector + optional ReID embedder)
    and the tracker with a shipped preset. Requires the ``perception`` extra
    (opencv-python + onnxruntime); the import cost is paid at construction,
    not at ``import soma``.
    """

    def __init__(
        self,
        preset: str = "somar-pv",
        detector: str = "models/yolov9_e_wholebody28_refine_Nx3HxW.onnx",
        reid: "str | None" = None,
        reid_whiten: float = 0.0,
        fps: int = 30,
        backend: str = "tensorrt",
        reid_backend: "str | None" = None,
        mode: str = "stretch",
        input_size: tuple[int, int] = (640, 640),
        body_score_floor: float = 0.10,
        **config_overrides: Any,
    ) -> None:
        from .perception import Perception  # perception extra

        self.perception = Perception(
            model_path=detector,
            mode=mode,
            body_score_floor=body_score_floor,
            input_size=input_size,
            execution_provider=backend,
            reid_path=reid,
            reid_backend=reid_backend or backend,
            reid_whiten=reid_whiten,
        )
        self.tracker = SomaTracker(
            tracker_config(preset, fps=fps, **config_overrides),
            record_rows=False,
        )

    @classmethod
    def from_variant(cls, variant: str = "pv", fps: int = 30,
                     detector: str = "models/yolov9_e_wholebody28_refine_Nx3HxW.onnx",
                     backend: str = "tensorrt", **kwargs: Any) -> "SomaVideoTracker":
        """Construct from a shipped variant (det / pv / os) — the same
        model + preset + whitening wiring as ``soma-eval video``."""
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant: {variant!r}; expected one of: "
                             f"{', '.join(sorted(VARIANTS))}")
        var = VARIANTS[variant]
        preset = {"pv": "somar-pv", "os": "somar-os", "det": "soma"}[variant]
        return cls(preset=preset, detector=detector, reid=var["reid"],
                   reid_whiten=var["whiten"], fps=fps, backend=backend, **kwargs)

    def update(self, frame_bgr: np.ndarray) -> list[TrackResult]:
        """Track one BGR frame; returns this frame's emissions."""
        result = self.perception(frame_bgr)
        self.last_perception = result
        return self.tracker.update(result.tokens)

    def head_boxes(self, min_score: float = 0.20) -> np.ndarray:
        """Head boxes of the LAST processed frame (privacy masking helper)."""
        det = self.last_perception.detections
        mask = (det.labels == 7) & (det.scores >= min_score)
        return det.boxes[mask]

    def reset(self) -> None:
        self.tracker.reset()
