"""Constant-velocity Kalman filter over (cx, cy, w, h) + part-point predictor."""
from __future__ import annotations

import numpy as np


class BoxKalman:
    """8-state CV filter. State: [cx, cy, w, h, vcx, vcy, vw, vh]."""

    def __init__(self, box_xyxy: np.ndarray, q_pos: float = 0.5, r_scale: float = 1.0):
        cx, cy, w, h = _to_cwh(box_xyxy)
        self.x = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=np.float64)
        self.P = np.diag([10, 10, 10, 10, 1e4, 1e4, 1e4, 1e4]).astype(np.float64)
        self.F = np.eye(8)
        self.F[:4, 4:] = np.eye(4)
        self.H = np.zeros((4, 8))
        self.H[:4, :4] = np.eye(4)
        self.q_pos = q_pos
        self.r_scale = r_scale

    def predict(self) -> np.ndarray:
        # SORT-style guard: never extrapolate size through zero
        if self.x[2] + self.x[6] <= 1.0:
            self.x[6] = 0.0
        if self.x[3] + self.x[7] <= 1.0:
            self.x[7] = 0.0
        self.x = self.F @ self.x
        s = max(self.x[3], 1.0)
        q = np.diag([self.q_pos * s / 20, self.q_pos * s / 20, s / 100, s / 100,
                     self.q_pos * s / 40, self.q_pos * s / 40, s / 200, s / 200]) ** 2
        self.P = self.F @ self.P @ self.F.T + q
        return self.box()

    def update(self, box_xyxy: np.ndarray, r_inflate: float = 1.0) -> None:
        z = np.array(_to_cwh(box_xyxy), dtype=np.float64)
        s = max(z[3], 1.0)
        r = (np.diag([s / 20, s / 20, s / 10, s / 10]) * self.r_scale * r_inflate) ** 2
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + r
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P
        self.x[2] = max(self.x[2], 1.0)
        self.x[3] = max(self.x[3], 1.0)

    def shift(self, dx: float, dy: float) -> None:
        """Position-only correction (e.g. from part-based translation)."""
        self.x[0] += dx
        self.x[1] += dy

    def box(self) -> np.ndarray:
        cx, cy, w, h = self.x[:4]
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float32)

    def center(self) -> np.ndarray:
        return self.x[:2].astype(np.float32)


class PartPoints:
    """Per-slot EMA constant-velocity predictor for token part points."""

    def __init__(self, n_slots: int, alpha_v: float = 0.5):
        self.pos = np.full((n_slots, 2), np.nan, dtype=np.float64)
        self.vel = np.zeros((n_slots, 2), dtype=np.float64)
        self.seen = np.zeros(n_slots, dtype=bool)
        self.alpha_v = alpha_v

    def predict(self) -> np.ndarray:
        out = self.pos + self.vel
        return out.astype(np.float32)

    def update(self, points_abs: np.ndarray, presence: np.ndarray) -> None:
        for si in np.where(presence)[0]:
            p = points_abs[si]
            if self.seen[si]:
                v = p - self.pos[si]
                self.vel[si] = self.alpha_v * v + (1 - self.alpha_v) * self.vel[si]
            self.pos[si] = p
            self.seen[si] = True

    def shift(self, dx: float, dy: float) -> None:
        self.pos[self.seen] += (dx, dy)


def batched_predict(kfs: list["BoxKalman"]) -> list[np.ndarray]:
    """exp063: one vectorized dispatch for the per-frame predict loop.

    With max_age 6s the tracker predicts ~150+ filters per frame; per-call
    numpy dispatch dominated the KF cost (exp063 profile: 2.9s/13s dense).
    All filters share the default q_pos/r_scale (the tracker never overrides
    them), asserted below. Falls back to the scalar path when they differ.
    Verified bit-identical to [kf.predict() for kf in kfs] on all 33
    CrowdTrack sequences (same kernels, slice-wise matmul).
    """
    if len(kfs) < 8:                    # dispatch overhead not worth it
        return [kf.predict() for kf in kfs]
    q_pos = kfs[0].q_pos
    if any(kf.q_pos != q_pos for kf in kfs):
        return [kf.predict() for kf in kfs]
    F = kfs[0].F
    X = np.stack([kf.x for kf in kfs])                  # (T,8)
    P = np.stack([kf.P for kf in kfs])                  # (T,8,8)
    m = X[:, 2] + X[:, 6] <= 1.0
    X[m, 6] = 0.0
    m = X[:, 3] + X[:, 7] <= 1.0
    X[m, 7] = 0.0
    X = X @ F.T
    s = np.maximum(X[:, 3], 1.0)
    t = q_pos * s
    qd = (np.stack([t / 20, t / 20, s / 100, s / 100,
                    t / 40, t / 40, s / 200, s / 200], axis=1)) ** 2
    P = F @ P @ F.T
    P[:, np.arange(8), np.arange(8)] += qd
    out = []
    for i, kf in enumerate(kfs):
        kf.x = X[i].copy()
        kf.P = P[i].copy()
        out.append(kf.box())
    return out


def _to_cwh(box: np.ndarray) -> tuple[float, float, float, float]:
    return (float(box[0] + box[2]) / 2, float(box[1] + box[3]) / 2,
            float(box[2] - box[0]), float(box[3] - box[1]))
