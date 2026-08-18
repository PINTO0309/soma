// SOMA-R external ReID embedder: person crops -> L2-normalized embeddings.
// Port of soma/reid.py (batch 1, "half" normalization (x/255 - 0.5) / 0.5)
// plus the causal embedding whitening from soma/perception.py.

import { create2dContext, type Any2DContext } from '../runtime/canvas';
import type { EngineModel } from '../runtime/engine';
import type { Box } from './types';

export interface ReidFrameSource {
  // Draws the sub-rectangle (sx, sy, sw, sh) of the current frame into ctx
  // scaled to (dw, dh).
  drawCrop: (
    ctx: Any2DContext,
    sx: number,
    sy: number,
    sw: number,
    sh: number,
    dw: number,
    dh: number,
  ) => void;
  width: number;
  height: number;
}

export class WebReidEmbedder {
  private engine: EngineModel;
  private ctx: Any2DContext;
  private input: Float32Array;
  private outDim: number;

  constructor(engine: EngineModel) {
    this.engine = engine;
    this.ctx = create2dContext(engine.inWidth, engine.inHeight);
    this.input = new Float32Array(engine.inWidth * engine.inHeight * 3);
    // static output dim when the runtime exposes it; else established from
    // the first inference result
    const outShape = engine.outputDims?.[0];
    const last = outShape && outShape.length > 0 ? outShape[outShape.length - 1] : 0;
    this.outDim = last > 1 ? last : 0;
  }

  // boxes: frame coords -> (N, outDim); degenerate crops get zero vectors.
  async embed(src: ReidFrameSource, boxes: Box[]): Promise<Float32Array[]> {
    const { inWidth: w, inHeight: h, layout } = this.engine;
    const results: Array<Float32Array | null> = boxes.map(() => null);
    for (let k = 0; k < boxes.length; k += 1) {
      const b = boxes[k];
      const x1 = Math.max(Math.floor(b[0]), 0);
      const y1 = Math.max(Math.floor(b[1]), 0);
      const x2 = Math.min(Math.floor(b[2]), src.width);
      const y2 = Math.min(Math.floor(b[3]), src.height);
      if (x2 - x1 < 4 || y2 - y1 < 8) {
        continue;
      }
      src.drawCrop(this.ctx, x1, y1, x2 - x1, y2 - y1, w, h);
      const rgba = this.ctx.getImageData(0, 0, w, h).data;
      const n = w * h;
      const inp = this.input;
      // RGB, (x/255 - 0.5) / 0.5
      if (layout === 'nchw') {
        for (let i = 0; i < n; i += 1) {
          inp[i] = rgba[i * 4] / 127.5 - 1.0;
          inp[n + i] = rgba[i * 4 + 1] / 127.5 - 1.0;
          inp[2 * n + i] = rgba[i * 4 + 2] / 127.5 - 1.0;
        }
      } else {
        for (let i = 0; i < n; i += 1) {
          inp[i * 3] = rgba[i * 4] / 127.5 - 1.0;
          inp[i * 3 + 1] = rgba[i * 4 + 1] / 127.5 - 1.0;
          inp[i * 3 + 2] = rgba[i * 4 + 2] / 127.5 - 1.0;
        }
      }
      const outputs = await this.engine.run(this.input);
      let e = outputs[0].data;
      if (this.outDim > 0 && e.length !== this.outDim) {
        // Some exports emit [1, D]; some emit extra outputs — take the first
        // output whose length matches the declared dim.
        const match = outputs.find((o) => o.data.length === this.outDim);
        if (match) {
          e = match.data;
        }
      }
      if (this.outDim === 0) {
        this.outDim = e.length;
      }
      let norm = 0;
      for (let i = 0; i < e.length; i += 1) {
        norm += e[i] * e[i];
      }
      norm = Math.sqrt(norm) + 1e-9;
      const dst = new Float32Array(this.outDim);
      const dim = Math.min(e.length, this.outDim);
      for (let i = 0; i < dim; i += 1) {
        const v = e[i] / norm;
        // non-finite guard, matching soma/reid.py
        dst[i] = Number.isFinite(v) ? v : 0;
      }
      results[k] = dst;
    }
    const dim = Math.max(this.outDim, 1);
    return results.map((r) => r ?? new Float32Array(dim));
  }
}

// Causal embedding whitening — running EMA mean/var of frame embeddings;
// e' = normalize((e - mu) / sd). Needed for embedders whose raw cosine cone
// is compressed (OSNet family); PersonViT v3 runs RAW (whiten 0).
//
// Few-person guard (deliberate web-only deviation from soma/perception.py):
// the statistics are computed ACROSS the people in a frame, so sparse scenes
// degenerate them — with 1 person the frame variance is identically zero
// (EMA decays to the floor) and the mean absorbs that person's identity,
// collapsing same-person whitened cosine to ~0 and mis-firing the calibrated
// somar-os gates. Stats therefore update only on frames with at least
// MIN_STATS_ROWS valid embeddings; sparse frames whiten with the FROZEN
// stats (measured: same-person cosine stays at the dense-regime ~0.42 for
// K=1-2 instead of collapsing to 0-0.2). Until a dense-enough frame has been
// seen, embeddings are withheld (zeroed) — raw OSNet cosines (~0.94 between
// DIFFERENT people) must never reach the whitened-space thresholds.
export const WHITEN_MIN_STATS_ROWS = 4;

export class CausalWhitener {
  private alpha: number;
  private minStatsRows: number;
  private mu: Float64Array | null = null;
  private varr: Float64Array | null = null;

  constructor(alpha: number, minStatsRows = WHITEN_MIN_STATS_ROWS) {
    this.alpha = alpha;
    this.minStatsRows = minStatsRows;
  }

  get enabled(): boolean {
    return this.alpha > 0;
  }

  apply(embs: Float32Array[]): Float32Array[] {
    if (this.alpha <= 0 || embs.length === 0) {
      return embs;
    }
    const dim = embs[0].length;
    const ok = embs.map((e) => {
      let s = 0;
      for (let i = 0; i < dim; i += 1) {
        s += e[i] * e[i];
      }
      return s > 0.5;
    });
    let cnt = 0;
    for (let k = 0; k < embs.length; k += 1) {
      if (ok[k]) {
        cnt += 1;
      }
    }
    if (cnt >= this.minStatsRows) {
      const fm = new Float64Array(dim);
      const fv = new Float64Array(dim);
      for (let k = 0; k < embs.length; k += 1) {
        if (!ok[k]) {
          continue;
        }
        for (let i = 0; i < dim; i += 1) {
          fm[i] += embs[k][i] / cnt;
        }
      }
      for (let k = 0; k < embs.length; k += 1) {
        if (!ok[k]) {
          continue;
        }
        for (let i = 0; i < dim; i += 1) {
          const d = embs[k][i] - fm[i];
          fv[i] += (d * d) / cnt;
        }
      }
      const a = this.alpha;
      if (this.mu === null || this.varr === null) {
        this.mu = fm;
        this.varr = fv;
      } else {
        for (let i = 0; i < dim; i += 1) {
          this.mu[i] = a * this.mu[i] + (1 - a) * fm[i];
          this.varr[i] = a * this.varr[i] + (1 - a) * fv[i];
        }
      }
    }
    if (this.mu === null || this.varr === null) {
      // whitening requested but no dense-enough frame seen yet: withhold
      // appearance evidence entirely (the tracker falls back to geometry)
      return embs.map(() => new Float32Array(dim));
    }
    const mu = this.mu;
    const varr = this.varr;
    return embs.map((e, k) => {
      if (!ok[k]) {
        return new Float32Array(dim);
      }
      const w = new Float64Array(dim);
      let n = 0;
      for (let i = 0; i < dim; i += 1) {
        const sd = Math.sqrt(Math.max(varr[i], 1e-8));
        w[i] = (e[i] - mu[i]) / sd;
        n += w[i] * w[i];
      }
      n = Math.sqrt(n);
      const outE = new Float32Array(dim);
      if (n > 1e-6) {
        for (let i = 0; i < dim; i += 1) {
          outE[i] = w[i] / Math.max(n, 1e-6);
        }
      }
      return outE;
    });
  }
}
