// SOMA-R external ReID embedder: person crops -> L2-normalized embeddings.
// Port of soma/reid.py (batch 1, "half" normalization (x/255 - 0.5) / 0.5)
// plus the causal embedding whitening from soma/perception.py.

import type { LoadedModel } from '../runtime/litert';
import { runModel } from '../runtime/litert';
import type { Box } from './types';

export interface ReidFrameSource {
  // Draws the sub-rectangle (sx, sy, sw, sh) of the current frame into ctx
  // scaled to (dw, dh).
  drawCrop: (
    ctx: CanvasRenderingContext2D,
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
  private loaded: LoadedModel;
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private input: Float32Array;
  readonly outDim: number;

  constructor(loaded: LoadedModel) {
    this.loaded = loaded;
    this.canvas = document.createElement('canvas');
    this.canvas.width = loaded.inWidth;
    this.canvas.height = loaded.inHeight;
    const ctx = this.canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) {
      throw new Error('2D canvas context unavailable');
    }
    this.ctx = ctx;
    this.input = new Float32Array(loaded.inWidth * loaded.inHeight * 3);
    const outShape = loaded.model.getOutputDetails()[0].shape;
    const last = outShape[outShape.length - 1];
    this.outDim = last > 0 ? last : 768;
  }

  // boxes: frame coords -> (N, outDim); degenerate crops get zero vectors.
  async embed(src: ReidFrameSource, boxes: Box[]): Promise<Float32Array[]> {
    const { inWidth: w, inHeight: h, layout } = this.loaded;
    const out: Float32Array[] = boxes.map(() => new Float32Array(this.outDim));
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
      const results = await runModel(this.loaded, this.input);
      let e = results[0];
      if (e.length !== this.outDim) {
        // Some exports emit [1, D]; some emit extra outputs — take the first
        // output whose length matches the declared dim.
        const match = results.find((r) => r.length === this.outDim);
        if (match) {
          e = match;
        }
      }
      let norm = 0;
      for (let i = 0; i < e.length; i += 1) {
        norm += e[i] * e[i];
      }
      norm = Math.sqrt(norm) + 1e-9;
      const dst = out[k];
      const dim = Math.min(e.length, this.outDim);
      let finite = true;
      for (let i = 0; i < dim; i += 1) {
        const v = e[i] / norm;
        dst[i] = Number.isFinite(v) ? v : 0;
        if (!Number.isFinite(v)) {
          finite = false;
        }
      }
      if (!finite) {
        // non-finite guard, matching soma/reid.py
        for (let i = 0; i < dim; i += 1) {
          if (!Number.isFinite(dst[i])) {
            dst[i] = 0;
          }
        }
      }
    }
    return out;
  }
}

// Causal embedding whitening — running EMA mean/var of frame embeddings;
// e' = normalize((e - mu) / sd). Needed for embedders whose raw cosine cone
// is compressed (OSNet family); PersonViT v3 runs RAW (whiten 0).
export class CausalWhitener {
  private alpha: number;
  private mu: Float64Array | null = null;
  private varr: Float64Array | null = null;

  constructor(alpha: number) {
    this.alpha = alpha;
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
    if (ok.some(Boolean)) {
      const fm = new Float64Array(dim);
      const fv = new Float64Array(dim);
      let cnt = 0;
      for (let k = 0; k < embs.length; k += 1) {
        if (ok[k]) {
          cnt += 1;
        }
      }
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
      return embs;
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
