// L0 perception + tracking pipeline: frame -> tokens -> tracker step.
// Port of soma/perception.py Perception + the cli.py video driver.

import { assemble } from './assembly';
import { WebDetector, type DetectorFrameSource } from './detector';
import { BODY_SCORE_FLOOR } from './presets';
import { CausalWhitener, WebReidEmbedder, type ReidFrameSource } from './reid';
import { buildToken } from './tokens';
import { SomaTracker, type TrackerConfig } from './tracker';
import * as C from './constants';
import type { AnatomicalToken, Box, TrackRow } from './types';
import { detBox } from './types';

export interface FrameStats {
  detectMs: number;
  assembleMs: number;
  reidMs: number;
  trackMs: number;
  nDetections: number;
  nTokens: number;
  nTracks: number;
}

export interface FrameOutput {
  rows: TrackRow[];
  headBoxes: Box[];
  tokens: AnatomicalToken[];
  stats: FrameStats;
}

export type FrameSource = DetectorFrameSource & ReidFrameSource;

export class SomaPipeline {
  private detector: WebDetector;
  private reid: WebReidEmbedder | null;
  private whitener: CausalWhitener;
  tracker: SomaTracker;
  private frameId = 0;

  constructor(
    detector: WebDetector,
    reid: WebReidEmbedder | null,
    whiten: number,
    trackerCfg: Partial<TrackerConfig>,
  ) {
    this.detector = detector;
    this.reid = reid;
    this.whitener = new CausalWhitener(whiten);
    this.tracker = new SomaTracker(trackerCfg);
  }

  get frame(): number {
    return this.frameId;
  }

  async process(src: FrameSource): Promise<FrameOutput> {
    this.frameId += 1;
    const t0 = performance.now();
    const det = await this.detector.detect(src);
    const t1 = performance.now();

    const asm = assemble(det, { bodyScoreFloor: BODY_SCORE_FLOOR });
    const tokens: AnatomicalToken[] = [];
    for (const person of asm.persons) {
      const tok = buildToken(person);
      if (tok !== null) {
        tokens.push(tok);
      }
    }
    // Head boxes for the privacy mosaic (labels == HEAD, score >= 0.20 —
    // cli.py cmd_video privacy-lean band).
    const headBoxes: Box[] = [];
    for (let r = 0; r < det.labels.length; r += 1) {
      if (det.labels[r] === C.HEAD && det.scores[r] >= 0.2) {
        headBoxes.push(detBox(det, r));
      }
    }
    const t2 = performance.now();

    if (this.reid !== null && tokens.length > 0) {
      const withBox = tokens.filter((t) => t.bodyBox !== null);
      if (withBox.length > 0) {
        let embs = await this.reid.embed(src, withBox.map((t) => t.bodyBox as Box));
        embs = this.whitener.apply(embs);
        withBox.forEach((t, i) => {
          const e = embs[i];
          let s = 0;
          for (let k = 0; k < e.length; k += 1) {
            s += e[k] * e[k];
          }
          if (s > 0.5) { // skip degenerate crops
            t.embedding = e;
          }
        });
      }
    }
    const t3 = performance.now();

    const rows = this.tracker.step(this.frameId, tokens);
    const t4 = performance.now();

    return {
      rows,
      headBoxes,
      tokens,
      stats: {
        detectMs: t1 - t0,
        assembleMs: t2 - t1,
        reidMs: t3 - t2,
        trackMs: t4 - t3,
        nDetections: det.labels.length,
        nTokens: tokens.length,
        nTracks: this.tracker.tracks.length,
      },
    };
  }
}

export function makeTrackerConfig(
  preset: Partial<TrackerConfig>,
  fps: number,
): Partial<TrackerConfig> {
  const cfg: Partial<TrackerConfig> = { ...preset, fps: Math.max(1, Math.round(fps)) };
  const maxAgeSec = cfg.maxAgeSec ?? 0;
  cfg.maxAge = maxAgeSec > 0
    ? Math.round(maxAgeSec * (cfg.fps as number))
    : 2 * (cfg.fps as number);
  return cfg;
}
