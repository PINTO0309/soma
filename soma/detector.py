"""Single-shot whole-frame YOLOv9-E Wholebody28-Refine ONNX inference.

Design directive: NO tiling. One 640x640 inference per frame; small/distant
people that fall below the detector's resolving power are accepted losses.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import onnxruntime as ort

# cv2 / onnxruntime are imported lazily inside the functions that need them:
# `import soma` (and the numpy-only tracking core) must work without the
# `perception` extra installed.

INPUT_SIZE = 640

# Inference backends (initialization pattern follows PINTO0309/LINEAE
# demo_lineae.py). Default is TensorRT; the engine cache is written next to
# the .onnx file. NOTE: the first run per model+shape builds the engine
# (minutes for the large detectors) — subsequent runs load the cache.
EXECUTION_PROVIDERS = ("cpu", "cuda", "tensorrt")
DEFAULT_EXECUTION_PROVIDER = "tensorrt"
TENSORRT_PRECISIONS = ("fp16", "bf16", "fp32")
DEFAULT_TENSORRT_PRECISION = "fp16"


def build_providers(execution_provider: str, model_path: str,
                    tensorrt_precision: str = DEFAULT_TENSORRT_PRECISION,
                    ) -> tuple["ort.SessionOptions", list[Any]]:
    """SessionOptions + provider list for cpu / cuda / tensorrt."""
    import onnxruntime as ort

    if execution_provider not in EXECUTION_PROVIDERS:
        raise ValueError(f"unsupported execution provider: {execution_provider}; "
                         f"expected one of: {', '.join(EXECUTION_PROVIDERS)}")
    if tensorrt_precision not in TENSORRT_PRECISIONS:
        raise ValueError(f"unsupported TensorRT precision: {tensorrt_precision}; "
                         f"expected one of: {', '.join(TENSORRT_PRECISIONS)}")

    available = ort.get_available_providers()
    required = {"cpu": "CPUExecutionProvider",
                "cuda": "CUDAExecutionProvider",
                "tensorrt": "TensorrtExecutionProvider"}[execution_provider]
    if required not in available:
        raise RuntimeError(f"{required} is unavailable; available providers: {available}")

    options = ort.SessionOptions()
    if execution_provider == "cpu":
        return options, ["CPUExecutionProvider"]

    # HEURISTIC algo search: the default EXHAUSTIVE re-benchmarks every conv
    # for every NEW input shape — with dynamic batch (ReID crops vary per
    # frame) that turned into a 30-60x slowdown (exp027).
    cuda = ("CUDAExecutionProvider", {"use_tf32": "0",
                                      "cudnn_conv_algo_search": "HEURISTIC"})
    if execution_provider == "cuda":
        # 参照実装は CPU fallback を禁止するが、動的形状グラフ(Nx3xHxW)は
        # shape 系ノードが CPU 常駐のため cuda 単独では初期化不能 — cuda の
        # ときのみ fallback を許容する(tensorrt は CUDA が受け皿なので厳格)。
        return options, [cuda, "CPUExecutionProvider"]

    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")

    if "CUDAExecutionProvider" not in available:
        raise RuntimeError("TensorRT execution requires CUDAExecutionProvider fallback")
    tensorrt_options = {
        "trt_engine_cache_enable": True,
        # デフォルト: onnx ファイルと同じ階層にエンジンキャッシュを出力
        "trt_engine_cache_path": os.path.dirname(os.path.abspath(model_path)) or ".",
        "trt_op_types_to_exclude": "NonMaxSuppression,NonZero,RoiAlign",
    }
    if tensorrt_precision == "fp16":
        tensorrt_options["trt_fp16_enable"] = True
    elif tensorrt_precision == "bf16":
        tensorrt_options["trt_bf16_enable"] = True
    return options, [("TensorrtExecutionProvider", tensorrt_options), cuda]
MASK_GRID = 80  # 640 / stride 8
_MASK_STRIDE = INPUT_SIZE // MASK_GRID

# Wholebody28(-Refine) class id -> Wholebody49 class id. 0..20 identical
# (body..ear incl. the 8 head-orientation classes); the remaining 7 parts map
# onto the same WB49 parents the WB34 table uses for those names.
WB28_TO_WB49 = np.arange(49, dtype=np.int32)
for _a, _b in ((21, 22), (22, 26), (23, 32), (24, 33), (25, 34), (26, 39), (27, 45)):
    WB28_TO_WB49[_a] = _b


@dataclass
class Detections:
    """All 1240 output rows of one frame, boxes mapped back to frame coords.

    Rows are top-k (query, class) pairs — the same query can appear in
    several rows (e.g. body + its attribute classes).
    """

    labels: np.ndarray          # (N,) int32
    boxes: np.ndarray           # (N, 4) float32 x1,y1,x2,y2 in frame coords
    scores: np.ndarray          # (N,) float32
    # (N, 80, 80) float32 SIGMOID PROBABILITIES (verified: range [0,1], sparse,
    # exact 0 outside the instance), meaningful for body rows only.
    # None for mask-less detector variants (Wholebody34).
    mask_probs: "np.ndarray | None"
    # frame -> 640-input mapping: p640 = p_frame * scale + offset
    scale: tuple[float, float]
    offset: tuple[float, float]
    # (N, 256) detector-native appearance embeddings (gathered mask_embed),
    # present when the "*_embeds.onnx" model is used; else None.
    embeddings: "np.ndarray | None" = None
    # (256, 80, 80) stride-8 appearance map (mask-pooling source); optional.
    mask_features: "np.ndarray | None" = None

    def centers(self) -> np.ndarray:
        return np.stack(
            [(self.boxes[:, 0] + self.boxes[:, 2]) * 0.5,
             (self.boxes[:, 1] + self.boxes[:, 3]) * 0.5], axis=1)

    def to_mask_grid(self, points_frame: np.ndarray) -> np.ndarray:
        """Map (M,2) frame-coordinate points into 80x80 mask-grid coordinates."""
        sx, sy = self.scale
        ox, oy = self.offset
        gx = (points_frame[:, 0] * sx + ox) / _MASK_STRIDE
        gy = (points_frame[:, 1] * sy + oy) / _MASK_STRIDE
        return np.stack([gx, gy], axis=1)

    def mask_prob_at(self, row: int, points_frame: np.ndarray) -> np.ndarray:
        """Bilinear-sampled mask probability of `row` at (M,2) frame points."""
        assert self.mask_probs is not None    # only called on mask-capable models
        grid = self.to_mask_grid(points_frame)
        return _bilinear(self.mask_probs[row], grid)

    def mask_probs_matrix(self, rows: np.ndarray, points_frame: np.ndarray) -> np.ndarray:
        """(len(rows), M) mask probabilities at shared frame points."""
        assert self.mask_probs is not None    # only called on mask-capable models
        grid = self.to_mask_grid(points_frame)
        return np.stack([_bilinear(self.mask_probs[r], grid) for r in rows]) \
            if len(rows) else np.zeros((0, len(points_frame)), np.float32)


def _bilinear(img: np.ndarray, pts: np.ndarray) -> np.ndarray:
    h, w = img.shape
    x = np.clip(pts[:, 0] - 0.5, 0.0, w - 1.001)
    y = np.clip(pts[:, 1] - 0.5, 0.0, h - 1.001)
    x0 = x.astype(np.int64)
    y0 = y.astype(np.int64)
    fx = (x - x0).astype(np.float32)
    fy = (y - y0).astype(np.float32)
    return (img[y0, x0] * (1 - fx) * (1 - fy)
            + img[y0, x0 + 1] * fx * (1 - fy)
            + img[y0 + 1, x0] * (1 - fx) * fy
            + img[y0 + 1, x0 + 1] * fx * fy)


class Detector:
    """wb28 (YOLOv9-E Wholebody28-Refine, dynamic-res raw head) detector."""

    def __init__(self, model_path: str, mode: str = "stretch",
                 input_size: tuple[int, int] | None = None,
                 execution_provider: str = DEFAULT_EXECUTION_PROVIDER,
                 tensorrt_precision: str = DEFAULT_TENSORRT_PRECISION):
        import onnxruntime as ort

        if mode not in ("stretch", "letterbox"):
            raise ValueError(f"unknown preprocess mode: {mode}")
        self.mode = mode
        ort.set_default_logger_severity(3)
        sess_options, provs = build_providers(execution_provider, model_path,
                                              tensorrt_precision)
        sess_options.log_severity_level = 3
        self.session = ort.InferenceSession(model_path, sess_options=sess_options,
                                            providers=provs)
        self.input_name = self.session.get_inputs()[0].name
        self.in_w, self.in_h = input_size if input_size is not None else (INPUT_SIZE,
                                                                          INPUT_SIZE)

    def __call__(self, img_bgr: np.ndarray) -> Detections:
        import cv2

        h, w = img_bgr.shape[:2]
        in_w, in_h = self.in_w, self.in_h
        if self.mode == "stretch":
            sx, sy = in_w / w, in_h / h
            ox = oy = 0.0
            canvas = cv2.resize(img_bgr, (in_w, in_h))
        else:
            s = min(in_w / w, in_h / h)
            sx = sy = s
            nw, nh = round(w * s), round(h * s)
            ox, oy = (in_w - nw) / 2.0, (in_h - nh) / 2.0
            canvas = np.full((in_h, in_w, 3), 114, np.uint8)
            canvas[int(oy):int(oy) + nh, int(ox):int(ox) + nw] = cv2.resize(img_bgr, (nw, nh))
        blob = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        out = self.session.run(["output0"],
                               {self.input_name: np.ascontiguousarray(blob)})[0][0]
        box, cls = out[:4], out[4:]                     # (4,A), (C,A)
        ci, ai = np.where(cls >= 0.05)                  # (class, anchor) pairs
        sc = cls[ci, ai]
        cx, cy, w2, h2 = box[0, ai], box[1, ai], box[2, ai] / 2, box[3, ai] / 2
        boxes = np.stack([cx - w2, cy - h2, cx + w2, cy + h2], axis=1)
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - ox) / sx
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - oy) / sy
        nc = cls.shape[0] - 1
        return Detections(
            labels=WB28_TO_WB49[np.clip(ci.astype(np.int32), 0, nc)],
            boxes=boxes.astype(np.float32),
            scores=sc.astype(np.float32),
            mask_probs=None,
            embeddings=None,
            mask_features=None,
            scale=(sx, sy),
            offset=(ox, oy),
        )
