"""Dense Hungarian assignment (numpy only) for the tracker's small problems.

The tracking core keeps a numpy-only dependency policy; when scipy happens
to be importable (dev/eval environments) its C implementation is used as a
drop-in fast path — same optimum, ~100x faster on dense MOT20 problems
(profiled 9.9s/800 frames in pure numpy, exp030).
"""
from __future__ import annotations

import numpy as np

try:  # optional fast path
    from scipy.optimize import linear_sum_assignment as _scipy_lsa
except Exception:                                     # pragma: no cover
    _scipy_lsa = None

_BIG = 1e12


def linear_assignment(cost: np.ndarray, gate: float = np.inf) -> list[tuple[int, int]]:
    """Minimize total cost; entries >= gate (or non-finite) are forbidden.

    Returns matched (row, col) pairs; forbidden pairs are never returned.
    """
    if cost.size == 0:
        return []
    n_r, n_c = cost.shape
    c = cost.astype(np.float64, copy=True)
    c[~np.isfinite(c) | (c >= gate)] = _BIG
    if _scipy_lsa is not None:
        ri, ci = _scipy_lsa(c)
        return [(int(i), int(j)) for i, j in zip(ri, ci) if c[i, j] < _BIG / 2]
    n = max(n_r, n_c)
    a = np.full((n, n), _BIG, dtype=np.float64)
    a[:n_r, :n_c] = c
    row_of_col = _hungarian_square(a)
    out = []
    for j in range(n_c):
        i = row_of_col[j]
        if i < n_r and c[i, j] < _BIG / 2:
            out.append((i, j))
    return out


def _hungarian_square(a: np.ndarray) -> np.ndarray:
    """Kuhn-Munkres with potentials (e-maxx formulation), vectorized inner loop.

    Returns row_of_col: col j -> assigned row.
    """
    n = a.shape[0]
    u = np.zeros(n + 1)
    v = np.zeros(n + 1)
    p = np.zeros(n + 1, dtype=np.int64)          # p[j]: row (1-based) assigned to col j
    way = np.zeros(n + 1, dtype=np.int64)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n + 1, np.inf)
        used = np.zeros(n + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            free = ~used
            free[0] = False
            idx = np.where(free)[0]
            cur = a[i0 - 1, idx - 1] - u[i0] - v[idx]
            upd = cur < minv[idx]
            minv[idx[upd]] = cur[upd]
            way[idx[upd]] = j0
            k = int(np.argmin(minv[idx]))
            delta = minv[idx[k]]
            j1 = int(idx[k])
            u[p[used]] += delta
            v[used] -= delta
            minv[~used] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = int(way[j0])
            p[j0] = p[j1]
            j0 = j1
    row_of_col = np.empty(n, dtype=np.int64)
    for j in range(1, n + 1):
        row_of_col[j - 1] = p[j] - 1
    return row_of_col
