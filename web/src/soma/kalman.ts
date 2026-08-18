// Constant-velocity Kalman filter over (cx, cy, w, h) + part-point predictor.
// Port of soma/kalman.py. State: [cx, cy, w, h, vcx, vcy, vw, vh].

import type { Box } from './types';

function toCwh(box: Box): [number, number, number, number] {
  return [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2, box[2] - box[0], box[3] - box[1]];
}

// 4x4 inverse via Gauss-Jordan with partial pivoting (S is well-conditioned:
// covariance + measurement noise).
function inv4(m: Float64Array): Float64Array {
  const a = new Float64Array(4 * 8);
  for (let i = 0; i < 4; i += 1) {
    for (let j = 0; j < 4; j += 1) {
      a[i * 8 + j] = m[i * 4 + j];
    }
    a[i * 8 + 4 + i] = 1;
  }
  for (let col = 0; col < 4; col += 1) {
    let piv = col;
    for (let r = col + 1; r < 4; r += 1) {
      if (Math.abs(a[r * 8 + col]) > Math.abs(a[piv * 8 + col])) {
        piv = r;
      }
    }
    if (piv !== col) {
      for (let j = 0; j < 8; j += 1) {
        const t = a[col * 8 + j];
        a[col * 8 + j] = a[piv * 8 + j];
        a[piv * 8 + j] = t;
      }
    }
    const d = a[col * 8 + col] || 1e-12;
    for (let j = 0; j < 8; j += 1) {
      a[col * 8 + j] /= d;
    }
    for (let r = 0; r < 4; r += 1) {
      if (r === col) {
        continue;
      }
      const f = a[r * 8 + col];
      if (f === 0) {
        continue;
      }
      for (let j = 0; j < 8; j += 1) {
        a[r * 8 + j] -= f * a[col * 8 + j];
      }
    }
  }
  const out = new Float64Array(16);
  for (let i = 0; i < 4; i += 1) {
    for (let j = 0; j < 4; j += 1) {
      out[i * 4 + j] = a[i * 8 + 4 + j];
    }
  }
  return out;
}

export class BoxKalman {
  x: Float64Array; // (8,)
  P: Float64Array; // (8,8) row-major
  qPos: number;
  rScale: number;

  constructor(box: Box, qPos = 0.5, rScale = 1.0) {
    const [cx, cy, w, h] = toCwh(box);
    this.x = Float64Array.from([cx, cy, w, h, 0, 0, 0, 0]);
    this.P = new Float64Array(64);
    const diag = [10, 10, 10, 10, 1e4, 1e4, 1e4, 1e4];
    for (let i = 0; i < 8; i += 1) {
      this.P[i * 9] = diag[i];
    }
    this.qPos = qPos;
    this.rScale = rScale;
  }

  predict(): Box {
    const x = this.x;
    // SORT-style guard: never extrapolate size through zero
    if (x[2] + x[6] <= 1.0) {
      x[6] = 0.0;
    }
    if (x[3] + x[7] <= 1.0) {
      x[7] = 0.0;
    }
    for (let i = 0; i < 4; i += 1) {
      x[i] += x[i + 4];
    }
    const s = Math.max(x[3], 1.0);
    const qp = this.qPos;
    const qd = [
      (qp * s) / 20, (qp * s) / 20, s / 100, s / 100,
      (qp * s) / 40, (qp * s) / 40, s / 200, s / 200,
    ].map((v) => v * v);
    // P = F P F^T + Q with F = [[I, I], [0, I]] (4x4 blocks):
    // FP[i][j] = P[i][j] + (i<4 ? P[i+4][j] : 0); then add column blocks.
    const P = this.P;
    const FP = new Float64Array(64);
    for (let i = 0; i < 8; i += 1) {
      for (let j = 0; j < 8; j += 1) {
        FP[i * 8 + j] = P[i * 8 + j] + (i < 4 ? P[(i + 4) * 8 + j] : 0);
      }
    }
    for (let i = 0; i < 8; i += 1) {
      for (let j = 0; j < 8; j += 1) {
        P[i * 8 + j] = FP[i * 8 + j] + (j < 4 ? FP[i * 8 + j + 4] : 0);
      }
    }
    for (let i = 0; i < 8; i += 1) {
      P[i * 9] += qd[i];
    }
    return this.box();
  }

  update(box: Box, rInflate = 1.0): void {
    const z = toCwh(box);
    const s = Math.max(z[3], 1.0);
    const rf = this.rScale * rInflate;
    const rd = [s / 20, s / 20, s / 10, s / 10].map((v) => (v * rf) ** 2);
    const x = this.x;
    const P = this.P;
    const y = [z[0] - x[0], z[1] - x[1], z[2] - x[2], z[3] - x[3]];
    // S = P[0:4,0:4] + R
    const S = new Float64Array(16);
    for (let i = 0; i < 4; i += 1) {
      for (let j = 0; j < 4; j += 1) {
        S[i * 4 + j] = P[i * 8 + j];
      }
      S[i * 4 + i] += rd[i];
    }
    const Si = inv4(S);
    // K = P[:,0:4] @ Si  (8x4)
    const K = new Float64Array(32);
    for (let i = 0; i < 8; i += 1) {
      for (let j = 0; j < 4; j += 1) {
        let acc = 0;
        for (let k = 0; k < 4; k += 1) {
          acc += P[i * 8 + k] * Si[k * 4 + j];
        }
        K[i * 4 + j] = acc;
      }
    }
    for (let i = 0; i < 8; i += 1) {
      let acc = 0;
      for (let k = 0; k < 4; k += 1) {
        acc += K[i * 4 + k] * y[k];
      }
      x[i] += acc;
    }
    // P = (I - K H) P  ->  P[i][j] -= sum_k K[i][k] * P[k][j], k in 0..3
    const Pold = P.slice();
    for (let i = 0; i < 8; i += 1) {
      for (let j = 0; j < 8; j += 1) {
        let acc = 0;
        for (let k = 0; k < 4; k += 1) {
          acc += K[i * 4 + k] * Pold[k * 8 + j];
        }
        P[i * 8 + j] = Pold[i * 8 + j] - acc;
      }
    }
    x[2] = Math.max(x[2], 1.0);
    x[3] = Math.max(x[3], 1.0);
  }

  // Position-only correction (e.g. from part-based translation).
  shift(dx: number, dy: number): void {
    this.x[0] += dx;
    this.x[1] += dy;
  }

  box(): Box {
    const [cx, cy, w, h] = this.x;
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2];
  }

  center(): [number, number] {
    return [this.x[0], this.x[1]];
  }
}

// Per-slot EMA constant-velocity predictor for token part points.
export class PartPoints {
  pos: Float64Array;  // (nSlots, 2), NaN when unseen
  vel: Float64Array;  // (nSlots, 2)
  seen: boolean[];
  alphaV: number;

  constructor(nSlots: number, alphaV = 0.5) {
    this.pos = new Float64Array(nSlots * 2).fill(Number.NaN);
    this.vel = new Float64Array(nSlots * 2);
    this.seen = new Array<boolean>(nSlots).fill(false);
    this.alphaV = alphaV;
  }

  predict(): Float64Array {
    const out = new Float64Array(this.pos.length);
    for (let i = 0; i < this.pos.length; i += 1) {
      out[i] = this.pos[i] + this.vel[i];
    }
    return out;
  }

  update(pointsAbs: Float64Array, presence: boolean[]): void {
    for (let si = 0; si < presence.length; si += 1) {
      if (!presence[si]) {
        continue;
      }
      const px = pointsAbs[si * 2];
      const py = pointsAbs[si * 2 + 1];
      if (this.seen[si]) {
        const vx = px - this.pos[si * 2];
        const vy = py - this.pos[si * 2 + 1];
        this.vel[si * 2] = this.alphaV * vx + (1 - this.alphaV) * this.vel[si * 2];
        this.vel[si * 2 + 1] = this.alphaV * vy + (1 - this.alphaV) * this.vel[si * 2 + 1];
      }
      this.pos[si * 2] = px;
      this.pos[si * 2 + 1] = py;
      this.seen[si] = true;
    }
  }

  shift(dx: number, dy: number): void {
    for (let si = 0; si < this.seen.length; si += 1) {
      if (this.seen[si]) {
        this.pos[si * 2] += dx;
        this.pos[si * 2 + 1] += dy;
      }
    }
  }

  scaleVel(f: number): void {
    for (let i = 0; i < this.vel.length; i += 1) {
      this.vel[i] *= f;
    }
  }
}
