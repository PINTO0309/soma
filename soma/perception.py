"""L0 perception pipeline: frame -> anatomical tokens (single 640 inference)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import constants as C
from .assembly import AssemblyResult, assemble
from .detector import Detections, Detector
from .tokens import AnatomicalToken, build_token

DEFAULT_MODEL = "models/yolov9_e_wholebody28_refine_Nx3HxW.onnx"

# Classes the active tracking stack actually consumes: body anchors,
# adult/child/male/female (the attr_penalty disagreement channel reads
# generation/gender — dropping them costs ~2 IDF1, measured), head, the 8
# orientation attributes, and the keypoint vocabulary (incl. L/R children).
# face(16-20) is referenced by no code path, and hand(32-34)/foot(45-47)
# are objects, not token slots — those rows feed nothing (exp024).
# Auto-enabled for the mask-less YOLO kinds only; DETR paths keep full rows.
LEAN_CLASS_IDS = np.array(sorted({*range(0, 16),
                                  *range(21, 32), *range(35, 45)}), dtype=np.int32)


@dataclass
class FrameResult:
    detections: Detections
    assembly: AssemblyResult
    tokens: list[AnatomicalToken]


class Perception:
    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        mode: str = "stretch",
        score_threshold: float = C.DEFAULT_SCORE_THRESHOLD,
        attr_score_threshold: float = C.DEFAULT_ATTR_SCORE_THRESHOLD,
        body_score_floor: float | None = None,
        input_size: tuple[int, int] | None = None,
        execution_provider: str | None = None,
        tensorrt_precision: str | None = None,
        reid_path: str | None = None,
        reid_backend: str = "tensorrt",
        reid_norm: str = "half",
        reid_whiten: float = 0.0,
        reid_batch: int = 1,
    ):
        # causal embedding whitening — running EMA mean/var of frame
        # embeddings; e' = normalize((e - mu) / sd). Needed for embedders
        # whose raw cosine cone is compressed (OSNet family); PersonViT v3
        # normalizes internally and runs RAW (whiten 0).
        self.reid_whiten = reid_whiten
        self._wh_mu = None
        self._wh_var = None
        # SOMA-R: external ReID embedder; None keeps the ReID-less stack.
        self.reid = None
        if reid_path:
            from .reid import ReIDEmbedder
            self.reid = ReIDEmbedder(reid_path, execution_provider=reid_backend,
                                     norm=reid_norm, batch_max=reid_batch)
        kw = {}
        if execution_provider is not None:
            kw["execution_provider"] = execution_provider
        if tensorrt_precision is not None:
            kw["tensorrt_precision"] = tensorrt_precision
        self.detector = Detector(model_path, mode=mode, input_size=input_size, **kw)
        self.score_threshold = score_threshold
        self.attr_score_threshold = attr_score_threshold
        self.body_score_floor = body_score_floor

    def __call__(self, img_bgr: np.ndarray) -> FrameResult:
        det = self.detector(img_bgr)
        keep = np.isin(det.labels, LEAN_CLASS_IDS)
        if not keep.all():
            det = Detections(
                labels=det.labels[keep], boxes=det.boxes[keep],
                scores=det.scores[keep], mask_probs=None,
                scale=det.scale, offset=det.offset)
        asm = assemble(det, score_threshold=self.score_threshold,
                       attr_score_threshold=self.attr_score_threshold,
                       body_score_floor=self.body_score_floor)
        tokens = []
        for person in asm.persons:
            tok = build_token(person, det)
            if tok is not None:
                tokens.append(tok)
        if self.reid is not None and tokens:
            with_box = [(t, t.body_box) for t in tokens if t.body_box is not None]
            if with_box:
                boxes = np.stack([b for _, b in with_box])
                embs = self.reid(img_bgr, boxes)
                if self.reid_whiten > 0:
                    ok = (embs * embs).sum(1) > 0.5
                    if ok.any():
                        fm = embs[ok].mean(0)
                        fv = embs[ok].var(0)
                        a = self.reid_whiten
                        self._wh_mu = (fm if self._wh_mu is None
                                       else a * self._wh_mu + (1 - a) * fm)
                        self._wh_var = (fv if self._wh_var is None
                                        else a * self._wh_var + (1 - a) * fv)
                    if self._wh_mu is not None:
                        sd = np.sqrt(np.maximum(self._wh_var, 1e-8))
                        w = (embs - self._wh_mu) / sd
                        n = np.linalg.norm(w, axis=1, keepdims=True)
                        embs = np.where((n > 1e-6) & (((embs * embs).sum(
                            1, keepdims=True)) > 0.5), w / np.maximum(n, 1e-6),
                            0.0)
                for (t, _), e in zip(with_box, embs):
                    if float(e @ e) > 0.5:       # skip degenerate crops
                        t.embedding = e
        return FrameResult(detections=det, assembly=asm, tokens=tokens)
