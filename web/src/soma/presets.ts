// Tracker presets — port of soma/cli.py VARIANTS + PRESETS.
// SOMA: geometry/parts/orientation channels only (no ReID).
// SOMA-R: appearance-dominant stage-1 + memory/revival/ghost stack; the
// embedding thresholds are quantile-calibrated per ReID model.

import type { TrackerConfig } from './tracker';

export type VariantId = 'det' | 'pv' | 'os';

const SOMAR_BASE: Partial<TrackerConfig> = {
  detThresh: 0.35,
  initThresh: 0.55,
  simGate: 0.3,
  wEmb: 4.0,
  wOks: 0.0,
  amodalAlpha: 4.0,
  dloBeta: 0.65,
  tokenFloor: 0.12,
  wDir: 0.0,
  dirVeto: 0.0,
  sizePriorThr: 0.5,
  embCenterLambda: 0.25,
  memTtlSec: 12.0,
  memMargin: 0.1,
  lostVelDecay: 0.85,
  gateGrow: 0.6,
  maxAgeSec: 6.0,
  lostEmbGateSec: 1.2,
  emitKf: true,
  partGate: 1.01,
  embUpdateCrowdMax: 0.6,
  ghostEmitMaxS: 0.1,
  ghostCrowdMax: 0.3,
};

export const PRESETS: Record<string, Partial<TrackerConfig>> = {
  soma: {
    detThresh: 0.45,
    initThresh: 0.55,
    simGate: 0.2,
    amodalAlpha: 4.0,
    dloBeta: 0.65,
    tokenFloor: 0.25,
    wDir: 0.5,
    dirVeto: 0.15,
    sizePriorThr: 0.5,
  },
  'somar-pv': {
    ...SOMAR_BASE,
    embLo: 0.398,
    embHi: 0.945,
    embVeto: 0.479,
    lostEmbGateCos: 0.398,
    memCos: 0.642,
    reviveCos: 0.56,
  },
  'somar-os': {
    ...SOMAR_BASE,
    embLo: 0.25,
    embHi: 0.899,
    embVeto: 0.314,
    lostEmbGateCos: 0.25,
    memCos: 0.475,
    reviveCos: 0.384,
  },
};

export interface VariantInfo {
  id: VariantId;
  label: string;
  preset: keyof typeof PRESETS;
  usesReid: boolean;
  // causal whitening EMA keep-rate (soma/cli.py VARIANTS whiten)
  whiten: number;
}

export const VARIANTS: VariantInfo[] = [
  { id: 'det', label: 'SOMA (no ReID)', preset: 'soma', usesReid: false, whiten: 0.0 },
  { id: 'pv', label: 'SOMA-R / PersonViT (raw)', preset: 'somar-pv', usesReid: true, whiten: 0.0 },
  { id: 'os', label: 'SOMA-R / OSNet-AIN (whitened)', preset: 'somar-os', usesReid: true, whiten: 0.98 },
];

// perception defaults used by the python cache/video paths
export const BODY_SCORE_FLOOR = 0.1;
