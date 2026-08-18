// SOMA: training-free multi-channel tracker over anatomical tokens.
// Faithful port of soma/tracker.py. One deliberate difference for the live
// web app: step() RETURNS the rows emitted for the current frame instead of
// accumulating the full MOT row history in memory.

import { BoxKalman, PartPoints } from './kalman';
import { linearAssignment } from './matching';
import { N_SLOTS, applyAmodal } from './tokens';
import type { AnatomicalToken, Box, TrackRow } from './types';

export interface TrackerConfig {
  detThresh: number;
  initThresh: number;
  minHits: number;
  maxAge: number;
  maxAgeSec: number;
  simGate: number;
  partGate: number;
  fps: number;
  wIou: number;
  wOks: number;
  oksKappa: number;
  wEmb: number;
  embLo: number;
  embHi: number;
  embVeto: number;
  embAlpha: number;
  wDir: number;
  dirMinConf: number;
  dirAlpha: number;
  dirVeto: number;
  dirMaxStepDeg: number;
  attrPenalty: number;
  centerGateMult: number;
  gateGrow: number;
  amodalAlpha: number;
  amodalGamma: number;
  dloBeta: number;
  tokenFloor: number;
  sizePriorThr: number;
  sizePriorQuality: number;
  sizePriorScale: number;
  lostVelDecay: number;
  lostEmbGateSec: number;
  lostEmbGateCos: number;
  reviveCos: number;
  reviveMinGap: number;
  reviveDistMult: number;
  memTtlSec: number;
  memDistMult: number;
  memGrow: number;
  memSizeTol: number;
  memMargin: number;
  memCos: number;
  embUpdateCrowdMax: number;
  embCenterLambda: number;
  emitKf: boolean;
  ghostEmitMaxS: number;
  ghostScoreMult: number;
  ghostCrowdMax: number;
}

export function defaultConfig(): TrackerConfig {
  return {
    detThresh: 0.45,
    initThresh: 0.55,
    minHits: 3,
    maxAge: 60,
    maxAgeSec: 0,
    simGate: 0.3,
    partGate: 0.45,
    fps: 30,
    wIou: 1.0,
    wOks: 0.8,
    oksKappa: 0.15,
    wEmb: 1.0,
    embLo: 0.6,
    embHi: 0.98,
    embVeto: 0.5,
    embAlpha: 0.9,
    wDir: 0.0,
    dirMinConf: 0.6,
    dirAlpha: 0.8,
    dirVeto: 0.0,
    dirMaxStepDeg: 30.0,
    attrPenalty: 0.1,
    centerGateMult: 2.0,
    gateGrow: 0.0,
    amodalAlpha: 0.0,
    amodalGamma: 0.9,
    dloBeta: 0.0,
    tokenFloor: 0.0,
    sizePriorThr: 0.0,
    sizePriorQuality: 0.12,
    sizePriorScale: 0.5,
    lostVelDecay: 1.0,
    lostEmbGateSec: 0.0,
    lostEmbGateCos: 0.55,
    reviveCos: 0.0,
    reviveMinGap: 15,
    reviveDistMult: 3.0,
    memTtlSec: 0.0,
    memDistMult: 1.6,
    memGrow: 0.5,
    memSizeTol: 0.35,
    memMargin: 0.08,
    memCos: 0.0,
    embUpdateCrowdMax: 1.0,
    embCenterLambda: 0.0,
    emitKf: false,
    ghostEmitMaxS: 0.0,
    ghostScoreMult: 0.5,
    ghostCrowdMax: 1.0,
  };
}

interface MemoryEntry {
  tid: number;
  lastFrame: number;
  center: [number, number];
  h: number;
  vel: [number, number];
  occTid: number;
  occOff: [number, number] | null;
  bank: Float32Array[];
  gender: number;
  generation: number;
  genderVotes: [number, number];
  genVotes: [number, number];
  headDir: [number, number] | null;
  headDirConf: number;
}

function center(box: Box): [number, number] {
  return [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2];
}

function absPoints(tok: AnatomicalToken): Float64Array {
  const bp = tok.boxProxy;
  const w = Math.max(bp[2] - bp[0], 1.0);
  const h = Math.max(bp[3] - bp[1], 1.0);
  const out = new Float64Array(N_SLOTS * 2);
  for (let si = 0; si < N_SLOTS; si += 1) {
    out[si * 2] = bp[0] + tok.points[si * 2] * w;
    out[si * 2 + 1] = bp[1] + tok.points[si * 2 + 1] * h;
  }
  return out;
}

function iou(a: Box, b: Box): number {
  const x1 = Math.max(a[0], b[0]);
  const y1 = Math.max(a[1], b[1]);
  const x2 = Math.min(a[2], b[2]);
  const y2 = Math.min(a[3], b[3]);
  if (x2 <= x1 || y2 <= y1) {
    return 0;
  }
  const inter = (x2 - x1) * (y2 - y1);
  const aa = (a[2] - a[0]) * (a[3] - a[1]);
  const bb = (b[2] - b[0]) * (b[3] - b[1]);
  return inter / (aa + bb - inter + 1e-9);
}

function dot(a: Float32Array, b: Float32Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i += 1) {
    s += a[i] * b[i];
  }
  return s;
}

function l2normalize(e: Float32Array): Float32Array {
  let n = 0;
  for (let i = 0; i < e.length; i += 1) {
    n += e[i] * e[i];
  }
  n = Math.sqrt(n) + 1e-9;
  const out = new Float32Array(e.length);
  for (let i = 0; i < e.length; i += 1) {
    out[i] = e[i] / n;
  }
  return out;
}

export class Track {
  tid: number;
  kf: BoxKalman;
  parts: PartPoints;
  hits = 1;
  age = 1;
  timeSinceUpdate = 0;
  lastFrame = 0;
  lastBox: Box | null = null;
  genderVotes: [number, number] = [0, 0];
  genVotes: [number, number] = [0, 0];
  anchor = 'body';
  score = 0;
  embedding: Float32Array | null = null;
  embPregap: Float32Array | null = null;
  embBest: Float32Array | null = null;
  bestScore = 0;
  headDir: [number, number] | null = null;
  headDirConf = 0;
  lastCrowding = 0;

  constructor(tid: number, kf: BoxKalman, parts: PartPoints) {
    this.tid = tid;
    this.kf = kf;
    this.parts = parts;
  }

  embCos(e: Float32Array | null): number | null {
    if (e === null || this.embedding === null) {
      return null;
    }
    return dot(this.embedding, e);
  }

  gender(): number {
    const [a, b] = this.genderVotes;
    return a + b >= 3 ? (a >= b ? 0 : 1) : -1;
  }

  generation(): number {
    const [a, b] = this.genVotes;
    return a + b >= 3 ? (a >= b ? 0 : 1) : -1;
  }

  absorb(tok: AnatomicalToken, cfg: TrackerConfig, full = true): void {
    const pts = absPoints(tok);
    if (full) {
      this.kf.update(tok.bodyBox as Box);
      this.lastBox = [...(tok.bodyBox as Box)] as Box;
      this.score = tok.bodyScore;
      this.anchor = tok.anchor;
    } else {
      // stage-2 partial continuation: translation-only observation from the
      // common part slots ("a head alone can carry the ID")
      const pred = this.parts.predict();
      const dx: number[] = [];
      const dy: number[] = [];
      for (let si = 0; si < N_SLOTS; si += 1) {
        if (tok.presence[si] && this.parts.seen[si]) {
          dx.push(pts[si * 2] - pred[si * 2]);
          dy.push(pts[si * 2 + 1] - pred[si * 2 + 1]);
        }
      }
      if (dx.length > 0) {
        const median = (arr: number[]): number => {
          const s = [...arr].sort((x, y) => x - y);
          const mid = Math.floor(s.length / 2);
          return s.length % 2 === 1 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
        };
        const mdx = median(dx);
        const mdy = median(dy);
        if (Number.isFinite(mdx) && Number.isFinite(mdy)) {
          this.kf.shift(mdx, mdy);
        }
      }
    }
    const crowding = tok.crowding;
    if (full) {
      this.lastCrowding = crowding;
    }
    if (tok.embedding !== null && crowding <= cfg.embUpdateCrowdMax) {
      if (this.embedding === null) {
        this.embedding = tok.embedding.slice();
      } else {
        const e = new Float32Array(this.embedding.length);
        for (let i = 0; i < e.length; i += 1) {
          e[i] = cfg.embAlpha * this.embedding[i] + (1 - cfg.embAlpha) * tok.embedding[i];
        }
        this.embedding = l2normalize(e);
      }
      if (tok.bodyScore >= this.bestScore) {
        this.bestScore = tok.bodyScore;
        this.embBest = tok.embedding.slice();
      }
    }
    if (tok.headDir !== null && tok.headDirConf >= cfg.dirMinConf) {
      if (this.headDir === null) {
        this.headDir = [...tok.headDir];
      } else {
        const vx = cfg.dirAlpha * this.headDir[0] + (1 - cfg.dirAlpha) * tok.headDir[0];
        const vy = cfg.dirAlpha * this.headDir[1] + (1 - cfg.dirAlpha) * tok.headDir[1];
        const n = Math.hypot(vx, vy);
        if (n > 1e-6) {
          this.headDir = [vx / n, vy / n];
        }
      }
      this.headDirConf = tok.headDirConf;
    }
    this.parts.update(pts, tok.presence);
    if (tok.gender >= 0) {
      this.genderVotes[tok.gender as 0 | 1] += 1;
    }
    if (tok.generation >= 0) {
      this.genVotes[tok.generation as 0 | 1] += 1;
    }
    this.hits += 1;
    this.timeSinceUpdate = 0;
  }
}

export class SomaTracker {
  cfg: TrackerConfig;
  tracks: Track[] = [];
  nextId = 1;
  private memory: MemoryEntry[] = [];
  private geomSamples: Array<[number, number]> = [];
  private geomFit: [number, number] | null = null;
  private geomNextFit = 100;
  private rngState = 0x9e3779b9;

  constructor(cfg?: Partial<TrackerConfig>) {
    this.cfg = { ...defaultConfig(), ...cfg };
  }

  // deterministic LCG for the size-prior sampling (numpy rng replacement)
  private rand(): number {
    this.rngState = (1664525 * this.rngState + 1013904223) >>> 0;
    return this.rngState / 0x100000000;
  }

  // ---- online scene-geometry size prior ----------------------------------
  private geomUpdate(tokens: AnatomicalToken[], matchedTokens: Set<number>): void {
    for (const di of matchedTokens) {
      const tok = tokens[di];
      if (tok.bodyBox !== null && tok.bodyScore >= 0.5) {
        this.geomSamples.push([tok.bodyBox[3], tok.bodyBox[3] - tok.bodyBox[1]]);
      }
    }
    if (this.geomSamples.length > 4000) {
      this.geomSamples = this.geomSamples.slice(-2000);
    }
    if (this.geomSamples.length >= this.geomNextFit) {
      this.geomNextFit = this.geomSamples.length + 300;
      const n = this.geomSamples.length;
      const slopes: number[] = [];
      for (let k = 0; k < 2000; k += 1) {
        const i = Math.floor(this.rand() * n);
        const j = Math.floor(this.rand() * n);
        const dy = this.geomSamples[i][0] - this.geomSamples[j][0];
        if (Math.abs(dy) > 5) {
          slopes.push((this.geomSamples[i][1] - this.geomSamples[j][1]) / dy);
        }
      }
      if (slopes.length < 50) {
        return;
      }
      const median = (arr: number[]): number => {
        const s = [...arr].sort((x, y) => x - y);
        const mid = Math.floor(s.length / 2);
        return s.length % 2 === 1 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
      };
      const a = median(slopes);
      const b = median(this.geomSamples.map(([y, h]) => h - a * y));
      const resid = this.geomSamples.map(([y, h]) => {
        const pred = Math.max(a * y + b, 1.0);
        return Math.abs(h - pred) / pred;
      });
      const quality = median(resid);
      this.geomFit = quality <= this.cfg.sizePriorQuality ? [a, b] : null;
    }
  }

  private sizeResidual(box: Box): number {
    if (this.geomFit === null) {
      return 0;
    }
    const [a, b] = this.geomFit;
    const pred = Math.max(a * box[3] + b, 1.0);
    return Math.abs(box[3] - box[1] - pred) / pred;
  }

  private oks(tr: Track, tok: AnatomicalToken, predH: number): number | null {
    const pred = tr.parts.predict();
    const obs = absPoints(tok);
    const s2 = (this.cfg.oksKappa * Math.max(predH, 8.0)) ** 2;
    let sum = 0;
    let cnt = 0;
    for (let si = 0; si < N_SLOTS; si += 1) {
      if (tok.presence[si] && tr.parts.seen[si]) {
        const dx = pred[si * 2] - obs[si * 2];
        const dy = pred[si * 2 + 1] - obs[si * 2 + 1];
        sum += Math.exp(-(dx * dx + dy * dy) / (2 * s2));
        cnt += 1;
      }
    }
    return cnt > 0 ? sum / cnt : null;
  }

  // ---- stage-1 similarity ------------------------------------------------
  private similarityMatrix(
    tokens: AnatomicalToken[],
    fullIdx: number[],
    predBoxes: Box[],
  ): number[][] {
    const cfg = this.cfg;
    const T = this.tracks.length;
    const D = fullIdx.length;
    const toks = fullIdx.map((di) => tokens[di]);
    const predPts = this.tracks.map((tr) => tr.parts.predict());
    const obsPts = toks.map((t) => absPoints(t));
    const s: number[][] = Array.from({ length: T }, () => new Array<number>(D).fill(0));

    for (let ti = 0; ti < T; ti += 1) {
      const tr = this.tracks[ti];
      const pb = predBoxes[ti];
      const pcx = (pb[0] + pb[2]) * 0.5;
      const pcy = (pb[1] + pb[3]) * 0.5;
      const ph = pb[3] - pb[1];
      const diag = Math.hypot(pb[2] - pb[0], ph);
      const gap = Math.max(tr.timeSinceUpdate - 1, 0);
      const gapS = gap / Math.max(cfg.fps, 1);
      const gateLim = cfg.centerGateMult * (1.0 + cfg.gateGrow * gapS) * diag;
      const pa = Math.max(pb[2] - pb[0], 0) * Math.max(ph, 0);
      const s2 = (cfg.oksKappa * Math.max(ph, 8.0)) ** 2;
      const trGender = tr.gender();
      const trGen = tr.generation();
      const wcTrack = (tr.anchor === 'wheelchair' || tr.anchor === 'crutches') && tr.hits > 5;
      const longLost = cfg.lostEmbGateSec > 0 && gapS >= cfg.lostEmbGateSec;

      for (let dj = 0; dj < D; dj += 1) {
        const tok = toks[dj];
        const tb = tok.bodyBox !== null ? tok.bodyBox : tok.boxProxy;
        const tcx = (tok.boxProxy[0] + tok.boxProxy[2]) * 0.5;
        const tcy = (tok.boxProxy[1] + tok.boxProxy[3]) * 0.5;
        const inGate = Math.hypot(tcx - pcx, tcy - pcy) <= gateLim;

        // IoU (pred box x token body/proxy box)
        const x1 = Math.max(pb[0], tb[0]);
        const y1 = Math.max(pb[1], tb[1]);
        const x2 = Math.min(pb[2], tb[2]);
        const y2 = Math.min(pb[3], tb[3]);
        const inter = Math.max(x2 - x1, 0) * Math.max(y2 - y1, 0);
        const ta = Math.max(tb[2] - tb[0], 0) * Math.max(tb[3] - tb[1], 0);
        const iouV = inter / Math.max(pa + ta - inter, 1e-9);

        // part OKS. NOTE: even at wOks 0 the OKS feeds the availability floor.
        let oksSum = 0;
        let oksCnt = 0;
        for (let si = 0; si < N_SLOTS; si += 1) {
          if (tok.presence[si] && tr.parts.seen[si]) {
            const dx = predPts[ti][si * 2] - obsPts[dj][si * 2];
            const dy = predPts[ti][si * 2 + 1] - obsPts[dj][si * 2 + 1];
            oksSum += Math.exp(-(dx * dx + dy * dy) / (2 * s2));
            oksCnt += 1;
          }
        }
        const hasOks = oksCnt > 0;
        const oksV = hasOks ? oksSum / oksCnt : 0;

        // ReID embedding cosine (EMA)
        const cos = tr.embCos(tok.embedding);
        const cosValid = cos !== null;

        // head-direction channel
        let dirValid = false;
        let dcos = 0;
        if ((cfg.wDir > 0 || cfg.dirVeto > 0)
          && tr.headDir !== null && tr.headDirConf >= cfg.dirMinConf
          && tok.headDir !== null && tok.headDirConf >= cfg.dirMinConf) {
          dirValid = true;
          dcos = tr.headDir[0] * tok.headDir[0] + tr.headDir[1] * tok.headDir[1];
        }

        let num = cfg.wIou * iouV;
        let den = cfg.wIou;
        if (hasOks) {
          num += cfg.wOks * oksV;
          den += cfg.wOks;
        }
        if (cosValid && cfg.wEmb > 0) {
          const emb = Math.min(Math.max((cos - cfg.embLo) / (cfg.embHi - cfg.embLo), 0), 1);
          num += cfg.wEmb * emb;
          den += cfg.wEmb;
        }
        if (cfg.wDir > 0 && dirValid) {
          num += cfg.wDir * 0.5 * (1.0 + dcos);
          den += cfg.wDir;
        }
        let sim = num / den;
        if (cosValid && cos < cfg.embVeto) {
          sim -= 0.15;
        }
        if (cfg.dirVeto > 0 && dirValid) {
          const tsu = Math.max(tr.timeSinceUpdate, 1);
          const cap = Math.min(180.0, cfg.dirMaxStepDeg * tsu + 45.0);
          if (dcos < Math.cos((cap * Math.PI) / 180.0)) {
            sim -= cfg.dirVeto;
          }
        }

        // availability floor: no geometric or part evidence -> no stage-1 match
        const floor = iouV < 0.05 && (!hasOks || oksV < 0.35);

        // attribute penalties (skipped for floored cells)
        if (!floor) {
          if (trGender >= 0 && tok.gender >= 0 && trGender !== tok.gender) {
            sim -= cfg.attrPenalty;
          }
          if (trGen >= 0 && tok.generation >= 0 && trGen !== tok.generation) {
            sim -= cfg.attrPenalty * 0.5;
          }
          if (wcTrack && tok.anchor === 'body') {
            sim -= cfg.attrPenalty;
          }
        }

        // "no look, no re-latch": long-lost rows require embedding confirmation
        if (longLost && (!cosValid || (cos as number) < cfg.lostEmbGateCos)) {
          sim = 0;
        }

        sim = Math.max(sim, 0);
        if (!inGate || floor) {
          sim = 0;
        }
        s[ti][dj] = sim;
      }
    }
    return s;
  }

  // ---- main step ----------------------------------------------------------
  step(frameId: number, tokensIn: AnatomicalToken[]): TrackRow[] {
    const cfg = this.cfg;
    const emitted: TrackRow[] = [];
    const emit = (tr: Track, box: Box, ghost = false): void => {
      if (!ghost && tr.hits < cfg.minHits) {
        return;
      }
      emitted.push({
        frame: frameId,
        tid: tr.tid,
        x: box[0],
        y: box[1],
        w: box[2] - box[0],
        h: box[3] - box[1],
        score: ghost ? tr.score * cfg.ghostScoreMult : tr.score,
        ghost,
      });
    };

    let tokens = tokensIn;
    if (cfg.tokenFloor > 0) {
      tokens = tokens.filter((t) => t.anchor === 'orphan' || t.bodyScore >= cfg.tokenFloor);
    }
    const predBoxes = this.tracks.map((tr) => tr.kf.predict());
    for (const tr of this.tracks) {
      tr.age += 1;
      tr.timeSinceUpdate += 1;
    }

    if (cfg.amodalAlpha > 0) {
      for (const t of tokens) {
        applyAmodal(t, cfg.amodalAlpha, cfg.amodalGamma);
      }
    }

    if (cfg.embCenterLambda > 0) {
      const embs = tokens.filter((t) => t.embedding !== null).map((t) => t.embedding as Float32Array);
      if (embs.length >= 3) {
        const dim = embs[0].length;
        const mean = new Float32Array(dim);
        for (const e of embs) {
          for (let i = 0; i < dim; i += 1) {
            mean[i] += e[i] / embs.length;
          }
        }
        for (const t of tokens) {
          if (t.embedding !== null) {
            const e = new Float32Array(dim);
            for (let i = 0; i < dim; i += 1) {
              e[i] = t.embedding[i] - cfg.embCenterLambda * mean[i];
            }
            t.embedding = l2normalize(e);
          }
        }
      }
    }

    if (cfg.embUpdateCrowdMax < 1.0) {
      const bodies = tokens.filter((t) => t.anchor !== 'orphan');
      for (let i = 0; i < bodies.length; i += 1) {
        const bi = bodies[i].bodyBox !== null ? (bodies[i].bodyBox as Box) : bodies[i].boxProxy;
        let crowd = 0;
        for (let j = 0; j < bodies.length; j += 1) {
          if (i === j) {
            continue;
          }
          const bj = bodies[j].bodyBox !== null ? (bodies[j].bodyBox as Box) : bodies[j].boxProxy;
          crowd = Math.max(crowd, iou(bi, bj));
        }
        bodies[i].crowding = crowd;
      }
    }

    // DLO-lite: a low-score body detection overlapping a predicted track is
    // probably a real (occluded) person — lift it into the stage-1 pool.
    const poolScores = tokens.map((t) => t.bodyScore);
    if (cfg.dloBeta > 0 && predBoxes.length > 0 && tokens.length > 0) {
      for (let i = 0; i < tokens.length; i += 1) {
        const t = tokens[i];
        if (t.anchor === 'orphan' || poolScores[i] >= cfg.detThresh) {
          continue;
        }
        const cb = t.bodyBox !== null ? t.bodyBox : t.boxProxy;
        let best = 0;
        for (const pb of predBoxes) {
          best = Math.max(best, iou(pb, cb));
        }
        poolScores[i] = Math.max(poolScores[i], cfg.dloBeta * best);
      }
    }

    const sizeBad = new Array<boolean>(tokens.length).fill(false);
    if (cfg.sizePriorThr > 0 && this.geomFit !== null) {
      for (let i = 0; i < tokens.length; i += 1) {
        const t = tokens[i];
        if (t.anchor === 'orphan') {
          continue;
        }
        const box = t.bodyBox !== null ? t.bodyBox : t.boxProxy;
        if (this.sizeResidual(box) > cfg.sizePriorThr) {
          sizeBad[i] = true;
          poolScores[i] *= cfg.sizePriorScale;
        }
      }
    }

    const fullIdx: number[] = [];
    for (let i = 0; i < tokens.length; i += 1) {
      if (tokens[i].anchor !== 'orphan' && poolScores[i] >= cfg.detThresh) {
        fullIdx.push(i);
      }
    }
    const fullSet = new Set(fullIdx);
    const restIdx: number[] = [];
    for (let i = 0; i < tokens.length; i += 1) {
      if (!fullSet.has(i)) {
        restIdx.push(i);
      }
    }

    // stage 1
    const nT = this.tracks.length;
    const matchedTracks = new Set<number>();
    const matchedTokens = new Set<number>();
    if (nT > 0 && fullIdx.length > 0) {
      const sim = this.similarityMatrix(tokens, fullIdx, predBoxes);
      const cost = sim.map((row) => row.map((v) => 1.0 - v));
      for (const [ti, dj] of linearAssignment(cost, nT, fullIdx.length, 1.0 - cfg.simGate)) {
        const tr = this.tracks[ti];
        const di = fullIdx[dj];
        tr.absorb(tokens[di], cfg);
        matchedTracks.add(ti);
        matchedTokens.add(di);
        emit(tr, cfg.emitKf ? tr.kf.box() : (tokens[di].bodyBox as Box));
      }
    }

    // revival: long-lost tracks x unmatched strong tokens, embedding-only
    if (cfg.reviveCos > 0) {
      const revTracks: number[] = [];
      for (let ti = 0; ti < nT; ti += 1) {
        const tr = this.tracks[ti];
        if (!matchedTracks.has(ti) && tr.timeSinceUpdate - 1 >= cfg.reviveMinGap
          && tr.embedding !== null) {
          revTracks.push(ti);
        }
      }
      const revTokens = fullIdx.filter(
        (di) => !matchedTokens.has(di) && tokens[di].embedding !== null
          && tokens[di].bodyScore >= cfg.initThresh,
      );
      if (revTracks.length > 0 && revTokens.length > 0) {
        const sim: number[][] = revTracks.map((ti) => {
          const tr = this.tracks[ti];
          const pb = predBoxes[ti];
          const diag = Math.hypot(pb[2] - pb[0], pb[3] - pb[1]);
          const pc = center(pb);
          return revTokens.map((di) => {
            const tok = tokens[di];
            const tc = center(tok.boxProxy);
            if (Math.hypot(tc[0] - pc[0], tc[1] - pc[1]) > cfg.reviveDistMult * diag) {
              return 0;
            }
            const c = tr.embCos(tok.embedding);
            return c !== null && c >= cfg.reviveCos ? c : 0;
          });
        });
        const cost = sim.map((row) => row.map((v) => 1.0 - v));
        for (const [a, b] of linearAssignment(cost, revTracks.length, revTokens.length,
          1.0 - cfg.reviveCos)) {
          const ti = revTracks[a];
          const di = revTokens[b];
          const tr = this.tracks[ti];
          tr.kf = new BoxKalman(tokens[di].bodyBox as Box); // motion state is stale
          tr.absorb(tokens[di], cfg);
          matchedTracks.add(ti);
          matchedTokens.add(di);
          emit(tr, cfg.emitKf ? tr.kf.box() : (tokens[di].bodyBox as Box));
        }
      }
    }

    // stage 2: part-only continuation (orphan part groups + low-score tokens)
    const remTracks: number[] = [];
    if (cfg.partGate < 1.0) {
      for (let ti = 0; ti < nT; ti += 1) {
        if (!matchedTracks.has(ti)) {
          remTracks.push(ti);
        }
      }
    }
    const remTokens = restIdx.filter((di) => !matchedTokens.has(di));
    if (remTracks.length > 0 && remTokens.length > 0) {
      const sim: number[][] = remTracks.map((ti) => {
        const tr = this.tracks[ti];
        const pb = predBoxes[ti];
        const diag = Math.hypot(pb[2] - pb[0], pb[3] - pb[1]);
        const pc = center(pb);
        return remTokens.map((di) => {
          const tc = center(tokens[di].boxProxy);
          if (Math.hypot(tc[0] - pc[0], tc[1] - pc[1]) > cfg.centerGateMult * diag) {
            return 0;
          }
          return this.oks(tr, tokens[di], pb[3] - pb[1]) ?? 0;
        });
      });
      const cost = sim.map((row) => row.map((v) => 1.0 - v));
      for (const [a, b] of linearAssignment(cost, remTracks.length, remTokens.length,
        1.0 - cfg.partGate)) {
        const ti = remTracks[a];
        const di = remTokens[b];
        const tr = this.tracks[ti];
        tr.absorb(tokens[di], cfg, false);
        matchedTracks.add(ti);
        matchedTokens.add(di);
        emit(tr, tr.kf.box()); // amodal: KF box
      }
    }

    // post-death memory: bind dying-out identities to would-be births.
    const reborn = new Map<number, MemoryEntry>();
    if (cfg.memTtlSec > 0) {
      const ttl = cfg.memTtlSec * Math.max(cfg.fps, 1);
      this.memory = this.memory.filter((e) => frameId - e.lastFrame <= ttl).slice(-300);
      const birthDis = fullIdx.filter(
        (di) => !matchedTokens.has(di) && !sizeBad[di]
          && tokens[di].bodyScore >= cfg.initThresh,
      );
      if (birthDis.length > 0 && this.memory.length > 0) {
        const M = this.memory.length;
        const D = birthDis.length;
        const score: number[][] = Array.from({ length: M }, () => new Array<number>(D).fill(0));
        const liveOf = new Map<number, Track>(this.tracks.map((t) => [t.tid, t]));
        const birthGeo = birthDis.map((di) => {
          const tb0 = tokens[di].bodyBox !== null
            ? (tokens[di].bodyBox as Box) : tokens[di].boxProxy;
          return { box: tb0, center: center(tb0) };
        });
        this.memory.forEach((e, mi) => {
          const gapF = frameId - e.lastFrame;
          const lim = cfg.memDistMult * e.h * (1.0 + (cfg.memGrow * gapF) / Math.max(cfg.fps, 1));
          const occ = e.occTid >= 0 ? liveOf.get(e.occTid) : undefined;
          let oc: [number, number] | null = null;
          if (occ !== undefined && e.occOff !== null) {
            const ob = occ.lastBox !== null ? occ.lastBox : occ.kf.box();
            const obc = center(ob);
            oc = [obc[0] + e.occOff[0], obc[1] + e.occOff[1]]; // riding the occluder
          }
          birthDis.forEach((di, dj) => {
            const tok = tokens[di];
            const { box: tb, center: tc } = birthGeo[dj];
            let d = Math.hypot(tc[0] - e.center[0], tc[1] - e.center[1]);
            const pcx = e.center[0] + e.vel[0] * gapF; // linear continuation
            const pcy = e.center[1] + e.vel[1] * gapF;
            d = Math.min(d, Math.hypot(tc[0] - pcx, tc[1] - pcy));
            if (oc !== null) {
              d = Math.min(d, Math.hypot(tc[0] - oc[0], tc[1] - oc[1]));
            }
            if (d > lim) {
              return;
            }
            const h = Math.max(tb[3] - tb[1], 1e-3);
            const dlh = Math.abs(Math.log(h / e.h));
            if (dlh > Math.log1p(cfg.memSizeTol)) {
              return;
            }
            if (tok.gender >= 0 && e.gender >= 0 && tok.gender !== e.gender) {
              return;
            }
            if (tok.generation >= 0 && e.generation >= 0 && tok.generation !== e.generation) {
              return;
            }
            const posS = 1.0 - d / lim;
            const sizeS = 1.0 - dlh / Math.log1p(cfg.memSizeTol);
            if (cfg.memCos > 0) {
              if (tok.embedding === null || e.bank.length === 0) {
                return; // SOMA-R: no look, no bind
              }
              let cos = Number.NEGATIVE_INFINITY;
              for (const be of e.bank) {
                cos = Math.max(cos, dot(be, tok.embedding));
              }
              if (cos < cfg.memCos) {
                return;
              }
              score[mi][dj] = 0.45 * posS + 0.2 * sizeS + 0.35 * cos;
            } else {
              score[mi][dj] = 0.6 * posS + 0.4 * sizeS;
            }
          });
        });
        const secondMax = (arr: number[]): number => {
          let m1 = Number.NEGATIVE_INFINITY;
          let m2 = Number.NEGATIVE_INFINITY;
          for (const v of arr) {
            if (v > m1) {
              m2 = m1;
              m1 = v;
            } else if (v > m2) {
              m2 = v;
            }
          }
          return m2;
        };
        const usedMi = new Set<number>();
        const cost = score.map((row) => row.map((v) => 1.0 - v));
        for (const [mi, dj] of linearAssignment(cost, M, D, 1.0 - 1e-9)) {
          if (score[mi][dj] <= 0) {
            continue;
          }
          const runR = D > 1 ? secondMax(score[mi]) : 0;
          const runC = M > 1 ? secondMax(score.map((row) => row[dj])) : 0;
          if (score[mi][dj] - Math.max(runR, runC) < cfg.memMargin) {
            continue;
          }
          reborn.set(birthDis[dj], this.memory[mi]);
          usedMi.add(mi);
        }
        if (usedMi.size > 0) {
          this.memory = this.memory.filter((_e, k) => !usedMi.has(k));
        }
      }
    }

    // births
    for (const di of fullIdx) {
      if (matchedTokens.has(di)) {
        continue;
      }
      const tok = tokens[di];
      if (tok.bodyScore < cfg.initThresh) {
        continue;
      }
      if (sizeBad[di]) {
        continue; // scale-implausible: no birth
      }
      const e = reborn.get(di);
      const tr = new Track(
        e !== undefined ? e.tid : this.nextId,
        new BoxKalman(tok.bodyBox as Box),
        new PartPoints(N_SLOTS),
      );
      tr.absorb(tok, cfg);
      tr.hits = 1;
      if (e !== undefined) { // inherit identity evidence
        tr.genderVotes = [...e.genderVotes];
        tr.genVotes = [...e.genVotes];
        if (e.headDir !== null) {
          tr.headDir = [...e.headDir];
          tr.headDirConf = e.headDirConf;
        }
      } else {
        this.nextId += 1;
      }
      this.tracks.push(tr);
      if (cfg.minHits <= 1) {
        emit(tr, tok.bodyBox as Box);
      }
    }

    // lost-track bookkeeping
    for (let ti = 0; ti < nT; ti += 1) {
      const tr = this.tracks[ti];
      if (matchedTracks.has(ti)) {
        continue;
      }
      if (tr.timeSinceUpdate === 1 && tr.embedding !== null) {
        tr.embPregap = tr.embedding.slice(); // last clean look before the gap
      }
      if (cfg.lostVelDecay < 1.0) {
        tr.kf.x[4] *= cfg.lostVelDecay;
        tr.kf.x[5] *= cfg.lostVelDecay;
        tr.parts.scaleVel(cfg.lostVelDecay);
      }
    }

    if (cfg.sizePriorThr > 0) {
      this.geomUpdate(tokens, matchedTokens);
    }

    for (const ti of matchedTracks) {
      this.tracks[ti].lastFrame = frameId;
    }

    // deaths (finished confirmed tracklets feed the identity memory)
    const alive: Track[] = [];
    for (const tr of this.tracks) {
      if (tr.timeSinceUpdate <= cfg.maxAge) {
        alive.push(tr);
      } else if (cfg.memTtlSec > 0 && tr.hits >= cfg.minHits && tr.lastBox !== null) {
        this.memory.push(this.memEntry(tr));
      }
    }
    this.tracks = alive;

    // ghost output: keep emitting the KF prediction through SHORT losses
    if (cfg.ghostEmitMaxS > 0) {
      const lim = Math.max(Math.round(cfg.ghostEmitMaxS * Math.max(cfg.fps, 1)), 1);
      for (const tr of this.tracks) {
        if (tr.hits >= cfg.minHits && tr.timeSinceUpdate >= 1 && tr.timeSinceUpdate <= lim
          && tr.lastCrowding <= cfg.ghostCrowdMax) {
          emit(tr, tr.kf.box(), true);
        }
      }
    }

    return emitted;
  }

  // Identity memory captured at death — anchored on the LAST OBSERVED box.
  // Also records the OCCLUDER (max-overlap live track at death).
  private memEntry(tr: Track): MemoryEntry {
    const b = tr.lastBox as Box;
    const c = center(b);
    let occTid = -1;
    let occOff: [number, number] | null = null;
    let best = 0.15;
    for (const o of this.tracks) {
      if (o.tid === tr.tid || o.timeSinceUpdate > 1) {
        continue;
      }
      const ob = o.lastBox !== null ? o.lastBox : o.kf.box();
      const v = iou(b, ob);
      if (v > best) {
        best = v;
        occTid = o.tid;
        const obc = center(ob);
        occOff = [c[0] - obc[0], c[1] - obc[1]];
      }
    }
    const bank: Float32Array[] = [];
    for (const e of [tr.embedding, tr.embPregap, tr.embBest]) {
      if (e !== null) {
        bank.push(e);
      }
    }
    return {
      tid: tr.tid,
      lastFrame: tr.lastFrame,
      center: c,
      h: Math.max(b[3] - b[1], 8.0),
      vel: [tr.kf.x[4], tr.kf.x[5]],
      occTid,
      occOff,
      bank,
      gender: tr.gender(),
      generation: tr.generation(),
      genderVotes: [...tr.genderVotes],
      genVotes: [...tr.genVotes],
      headDir: tr.headDir === null ? null : [...tr.headDir],
      headDirConf: tr.headDirConf,
    };
  }
}
