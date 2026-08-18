// Dense Hungarian assignment — port of soma/matching.py (numpy-only path).
// Minimizes total cost; entries >= gate (or non-finite) are forbidden.

const BIG = 1e12;

export function linearAssignment(
  cost: number[][],
  nRows: number,
  nCols: number,
  gate = Number.POSITIVE_INFINITY,
): Array<[number, number]> {
  if (nRows === 0 || nCols === 0) {
    return [];
  }
  const n = Math.max(nRows, nCols);
  const a: number[][] = Array.from({ length: n }, (_v, i) =>
    Array.from({ length: n }, (_w, j) => {
      if (i >= nRows || j >= nCols) {
        return BIG;
      }
      const c = cost[i][j];
      return Number.isFinite(c) && c < gate ? c : BIG;
    }),
  );
  const rowOfCol = hungarianSquare(a, n);
  const out: Array<[number, number]> = [];
  for (let j = 0; j < nCols; j += 1) {
    const i = rowOfCol[j];
    if (i < nRows && a[i][j] < BIG / 2) {
      out.push([i, j]);
    }
  }
  return out;
}

// Kuhn-Munkres with potentials (e-maxx formulation).
// Returns rowOfCol: col j -> assigned row.
function hungarianSquare(a: number[][], n: number): Int32Array {
  const u = new Float64Array(n + 1);
  const v = new Float64Array(n + 1);
  const p = new Int32Array(n + 1); // p[j]: row (1-based) assigned to col j
  const way = new Int32Array(n + 1);
  for (let i = 1; i <= n; i += 1) {
    p[0] = i;
    let j0 = 0;
    const minv = new Float64Array(n + 1).fill(Number.POSITIVE_INFINITY);
    const used = new Uint8Array(n + 1);
    for (;;) {
      used[j0] = 1;
      const i0 = p[j0];
      let delta = Number.POSITIVE_INFINITY;
      let j1 = -1;
      for (let j = 1; j <= n; j += 1) {
        if (used[j]) {
          continue;
        }
        const cur = a[i0 - 1][j - 1] - u[i0] - v[j];
        if (cur < minv[j]) {
          minv[j] = cur;
          way[j] = j0;
        }
        if (minv[j] < delta) {
          delta = minv[j];
          j1 = j;
        }
      }
      for (let j = 0; j <= n; j += 1) {
        if (used[j]) {
          u[p[j]] += delta;
          v[j] -= delta;
        } else {
          minv[j] -= delta;
        }
      }
      j0 = j1;
      if (p[j0] === 0) {
        break;
      }
    }
    while (j0 !== 0) {
      const j1 = way[j0];
      p[j0] = p[j1];
      j0 = j1;
    }
  }
  const rowOfCol = new Int32Array(n);
  for (let j = 1; j <= n; j += 1) {
    rowOfCol[j - 1] = p[j] - 1;
  }
  return rowOfCol;
}
