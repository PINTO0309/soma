"""SOMA-R external ReID embedder: person crops -> L2-normalized embeddings.

Validated config (ptrack exp054/056): TensorRT fp16 with batch_max=1 — every
chunk is padded to one static [1,3,H,W] engine shape (parity vs CUDA >=.998,
~2.8x faster than batched CUDA). Preprocessing: RGB, (x/255 - 0.5) / 0.5
("half" norm) for the PersonViT / OSNet-AIN aug models."""
from __future__ import annotations

import numpy as np

from .detector import DEFAULT_TENSORRT_PRECISION, build_providers

# cv2 / onnxruntime are imported lazily (see soma/detector.py).

_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)
# exp050: per-model normalization for the legacy crop path
_NORMS = {"imagenet": (_MEAN, _STD),
          "half": (np.array([0.5, 0.5, 0.5], np.float32),
                   np.array([0.5, 0.5, 0.5], np.float32))}
# ImageNet mean as BGR uint8: fills masked-out background with the exact
# color that normalizes to ~0 in every channel (exp034 bodymask variant)


class ReIDEmbedder:
    def __init__(self, model_path: str, execution_provider: str = "cuda",
                 tensorrt_precision: str = DEFAULT_TENSORRT_PRECISION,
                 batch_max: int = 1, norm: str = "half"):
        import onnxruntime as ort

        self.norm_mean, self.norm_std = _NORMS[norm]
        # NOTE: default is cuda, not tensorrt — the pre-reset project hit a
        # TRT EP bug on a dynamic-batch ViT embedder (NaN/wrong rows). Enable
        # tensorrt only after a row-wise parity check against cuda.
        ort.set_default_logger_severity(3)
        sess_options, provs = build_providers(execution_provider, model_path,
                                              tensorrt_precision)
        sess_options.log_severity_level = 3
        self.session = ort.InferenceSession(model_path, sess_options=sess_options,
                                            providers=provs)
        i = self.session.get_inputs()[0]
        self.input_name = i.name
        self.in_h, self.in_w = int(i.shape[2]), int(i.shape[3])
        self.batch_max = batch_max
        # exp039: output dim from the graph (CDNet 768, OSNet 512, ...)
        od = self.session.get_outputs()[0].shape[-1]
        self.out_dim = int(od) if isinstance(od, int) else 768

    def __call__(self, img_bgr: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """boxes: (N,4) x1y1x2y2 frame coords -> (N,D) L2-normalized fp32.
        Boxes are clipped to the frame; degenerate crops get zero vectors."""
        import cv2

        H, W = img_bgr.shape[:2]
        n = len(boxes)
        out = np.zeros((n, self.out_dim), dtype=np.float32)
        crops, idx = [], []
        for k, b in enumerate(np.asarray(boxes, dtype=np.float64)):
            x1, y1 = max(int(b[0]), 0), max(int(b[1]), 0)
            x2, y2 = min(int(b[2]), W), min(int(b[3]), H)
            if x2 - x1 < 4 or y2 - y1 < 8:
                continue
            crops.append(img_bgr[y1:y2, x1:x2])
            idx.append(k)
        # resize + /255 + mean subtraction + BGR->RGB + NCHW packing in one C
        # call (blobFromImages supports only a scalar scale, so the
        # per-channel STD division is a single broadcast pass afterwards)
        std_nchw = self.norm_std.reshape(1, 3, 1, 1)
        for s in range(0, len(crops), self.batch_max):
            chunk = crops[s:s + self.batch_max]
            x = cv2.dnn.blobFromImages(chunk, scalefactor=1.0 / 255.0,
                                       size=(self.in_w, self.in_h),
                                       mean=(self.norm_mean * 255.0).tolist(),
                                       swapRB=True)
            x /= std_nchw
            # pad every chunk to batch_max: ONE batch shape for the whole run
            n_real = len(x)
            if n_real < self.batch_max:
                x = np.concatenate([x, np.zeros((self.batch_max - n_real, *x.shape[1:]),
                                                dtype=np.float32)])
            e = self.session.run(None, {self.input_name: x})[0][:n_real]
            e = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9)
            if not np.isfinite(e).all():                     # non-finite guard
                e = np.where(np.isfinite(e), e, 0.0)
            for j, k in enumerate(idx[s:s + self.batch_max]):
                out[k] = e[j]
        return out
