// Whole-frame wholebody detector on the shared inference engine (LiteRT or
// onnxruntime-web). Port of soma/detector.py (stretch preprocessing, raw
// YOLO head postprocess, wb28 -> wb49 id mapping) extended to also accept
// post-processed [N, 7] detector exports (batchno_classid_score_x1y1x2y2 —
// the PINTO_model_zoo "post" variants).

import * as C from './constants';
import { create2dContext, type Any2DContext } from '../runtime/canvas';
import type { EngineModel } from '../runtime/engine';
import type { Detections } from './types';

const RAW_SCORE_THRESHOLD = 0.05;

export interface DetectorFrameSource {
  // Draws the current frame into ctx at (0,0) with the given size.
  draw: (ctx: Any2DContext, w: number, h: number) => void;
  width: number;
  height: number;
}

export class WebDetector {
  private engine: EngineModel;
  private ctx: Any2DContext;
  private input: Float32Array;
  private classCount: number | null = null;

  constructor(engine: EngineModel) {
    this.engine = engine;
    this.ctx = create2dContext(engine.inWidth, engine.inHeight);
    this.input = new Float32Array(engine.inWidth * engine.inHeight * 3);
  }

  get inputSize(): [number, number] {
    return [this.engine.inWidth, this.engine.inHeight];
  }

  // stretch preprocessing: RGB, x/255, NCHW or NHWC to match the model.
  private preprocess(src: DetectorFrameSource): void {
    const { inWidth: w, inHeight: h, layout } = this.engine;
    src.draw(this.ctx, w, h);
    const rgba = this.ctx.getImageData(0, 0, w, h).data;
    const n = w * h;
    const out = this.input;
    if (layout === 'nchw') {
      for (let i = 0; i < n; i += 1) {
        out[i] = rgba[i * 4] / 255;
        out[n + i] = rgba[i * 4 + 1] / 255;
        out[2 * n + i] = rgba[i * 4 + 2] / 255;
      }
    } else {
      for (let i = 0; i < n; i += 1) {
        out[i * 3] = rgba[i * 4] / 255;
        out[i * 3 + 1] = rgba[i * 4 + 1] / 255;
        out[i * 3 + 2] = rgba[i * 4 + 2] / 255;
      }
    }
  }

  private mapLabel(cid: number, numClasses: number): number {
    const clamped = Math.min(Math.max(cid, 0), 48);
    if (numClasses === 28) {
      return C.WB28_TO_WB49[clamped];
    }
    if (numClasses === 25) {
      return C.WB25_TO_WB49[clamped];
    }
    return clamped;
  }

  async detect(src: DetectorFrameSource): Promise<Detections> {
    const { inWidth, inHeight } = this.engine;
    const sx = inWidth / src.width;
    const sy = inHeight / src.height;
    this.preprocess(src);
    const outputs = await this.engine.run(this.input);

    const labels: number[] = [];
    const boxes: number[] = [];
    const scores: number[] = [];

    // Pick the primary output: prefer the largest tensor of rank >= 2.
    let out = outputs[0].data;
    let dims = outputs[0].dims;
    for (const o of outputs) {
      if (o.dims.length >= 2 && o.data.length >= out.length) {
        out = o.data;
        dims = o.dims;
      }
    }
    while (dims.length > 2 && dims[0] === 1) {
      dims = dims.slice(1);
    }

    if (dims.length === 2 && dims[1] === 7) {
      // post-processed rows: [batchno, classid, score, x1, y1, x2, y2] in
      // detector-input pixel coordinates.
      const nRows = dims[0];
      for (let r = 0; r < nRows; r += 1) {
        const o = r * 7;
        const score = out[o + 2];
        if (score < RAW_SCORE_THRESHOLD) {
          continue;
        }
        const cid = Math.round(out[o + 1]);
        labels.push(this.mapLabel(cid, 49));
        scores.push(score);
        boxes.push(out[o + 3] / sx, out[o + 4] / sy, out[o + 5] / sx, out[o + 6] / sy);
      }
    } else if (dims.length === 2) {
      // raw YOLO head: (4+C, A) or (A, 4+C); the channel dim is the small one.
      let ch = dims[0];
      let anchors = dims[1];
      let chFirst = true;
      if (dims[0] > dims[1]) {
        ch = dims[1];
        anchors = dims[0];
        chFirst = false;
      }
      const numClasses = ch - 4;
      if (numClasses < 1) {
        throw new Error(`Cannot interpret detector output shape [${dims.join(', ')}]`);
      }
      this.classCount = numClasses;
      const at = (c: number, a: number): number => (chFirst ? out[c * anchors + a] : out[a * ch + c]);
      for (let a = 0; a < anchors; a += 1) {
        for (let c = 0; c < numClasses; c += 1) {
          const score = at(4 + c, a);
          if (score < RAW_SCORE_THRESHOLD) {
            continue;
          }
          const cx = at(0, a);
          const cy = at(1, a);
          const w2 = at(2, a) / 2;
          const h2 = at(3, a) / 2;
          labels.push(this.mapLabel(c, numClasses));
          scores.push(score);
          boxes.push((cx - w2) / sx, (cy - h2) / sy, (cx + w2) / sx, (cy + h2) / sy);
        }
      }
    } else {
      throw new Error(`Cannot interpret detector output shape [${dims.join(', ')}]`);
    }

    // LEAN filter (soma/perception.py): keep only the classes the tracking
    // stack consumes.
    const keptLabels: number[] = [];
    const keptScores: number[] = [];
    const keptBoxes: number[] = [];
    for (let i = 0; i < labels.length; i += 1) {
      if (C.LEAN_CLASS_IDS.has(labels[i])) {
        keptLabels.push(labels[i]);
        keptScores.push(scores[i]);
        keptBoxes.push(boxes[i * 4], boxes[i * 4 + 1], boxes[i * 4 + 2], boxes[i * 4 + 3]);
      }
    }

    return {
      labels: Int32Array.from(keptLabels),
      boxes: Float32Array.from(keptBoxes),
      scores: Float32Array.from(keptScores),
      scale: [sx, sy],
      offset: [0, 0],
    };
  }

  get numClasses(): number | null {
    return this.classCount;
  }
}
