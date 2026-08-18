// Dedicated inference worker (default execution mode, following the
// PINTO0309/screen-eye-tracking `--web-inference-worker dedicated` design):
// the whole perception + tracking pipeline lives here, so model inference
// and tensor conversion never block the UI thread. The main thread sends
// one RGBA frame per message and receives tracker rows + head boxes back.

import { setAssetBaseUrl, type Accelerator, type EngineModel, type RuntimeId } from '../runtime/engine';
// static imports: iife worker bundles forbid code-splitting; importing both
// engines is harmless — only the selected one is ever initialized
import { loadLitertModel } from '../runtime/litert';
import { loadOrtModel } from '../runtime/ort';
import { SomaPipeline, makeTrackerConfig, type FrameOutput, type FrameSource } from '../soma/pipeline';
import { WebDetector } from '../soma/detector';
import { WebReidEmbedder } from '../soma/reid';
import type { TrackerConfig } from '../soma/tracker';

export interface WorkerInitMessage {
  type: 'init';
  runtime: RuntimeId;
  accelerator: Accelerator;
  numThreads: number;
  assetBaseUrl: string;
  detectorUrl: string;
  reidUrl: string | null;
  whiten: number;
  preset: Partial<TrackerConfig>;
}

export interface WorkerFrameMessage {
  type: 'frame';
  rgba: ArrayBuffer;
  width: number;
  height: number;
}

export type MainToWorkerMessage = WorkerInitMessage | WorkerFrameMessage | { type: 'stop' };

export type WorkerToMainMessage =
  | { type: 'ready'; accelerator: Accelerator; note: string | null }
  | { type: 'initError'; message: string }
  | { type: 'result'; output: FrameOutput }
  | { type: 'frameError'; message: string };

let pipeline: SomaPipeline | null = null;
let engines: EngineModel[] = [];
let frameCtx: OffscreenCanvasRenderingContext2D | null = null;
let emaFps = 0;
let lastT = 0;

const post = (message: WorkerToMainMessage): void => {
  self.postMessage(message);
};

async function fetchBytes(url: string): Promise<Uint8Array> {
  const response = await fetch(url, { cache: 'no-store' });
  if (!response.ok) {
    throw new Error(`Failed to fetch model: ${response.status} ${response.statusText}`);
  }
  return new Uint8Array(await response.arrayBuffer());
}

async function initialize(msg: WorkerInitMessage): Promise<void> {
  setAssetBaseUrl(msg.assetBaseUrl);
  const loadEngine = msg.runtime === 'ort' ? loadOrtModel : loadLitertModel;

  let accel: Accelerator = msg.accelerator;
  let note: string | null = null;

  const loadWithFallback = async (bytes: Uint8Array): Promise<EngineModel> => {
    try {
      return await loadEngine(bytes, accel, msg.numThreads);
    } catch (error) {
      if (accel === 'webgpu') {
        note = `WebGPU init failed — fell back to WASM: ${
          error instanceof Error ? error.message : String(error)
        }`;
        accel = 'wasm';
        return loadEngine(bytes, accel, msg.numThreads);
      }
      throw error;
    }
  };

  const detEngine = await loadWithFallback(await fetchBytes(msg.detectorUrl));
  engines.push(detEngine);
  let reidEngine: EngineModel | null = null;
  if (msg.reidUrl !== null) {
    reidEngine = await loadWithFallback(await fetchBytes(msg.reidUrl));
    engines.push(reidEngine);
  }

  const detector = new WebDetector(detEngine);
  const reid = reidEngine !== null ? new WebReidEmbedder(reidEngine) : null;
  pipeline = new SomaPipeline(detector, reid, msg.whiten, makeTrackerConfig(msg.preset, 30));
  emaFps = 0;
  lastT = 0;
  post({ type: 'ready', accelerator: accel, note });
}

async function processFrame(msg: WorkerFrameMessage): Promise<void> {
  if (pipeline === null) {
    post({ type: 'frameError', message: 'worker not initialized' });
    return;
  }
  if (
    frameCtx === null ||
    frameCtx.canvas.width !== msg.width ||
    frameCtx.canvas.height !== msg.height
  ) {
    const canvas = new OffscreenCanvas(msg.width, msg.height);
    frameCtx = canvas.getContext('2d', { willReadFrequently: false });
    if (frameCtx === null) {
      post({ type: 'frameError', message: '2D canvas context unavailable in worker' });
      return;
    }
  }
  frameCtx.putImageData(new ImageData(new Uint8ClampedArray(msg.rgba), msg.width, msg.height), 0, 0);
  const frameCanvas = frameCtx.canvas;
  const source: FrameSource = {
    width: msg.width,
    height: msg.height,
    draw: (ctx, w, h) => ctx.drawImage(frameCanvas, 0, 0, w, h),
    drawCrop: (ctx, sx, sy, sw, sh, dw, dh) => ctx.drawImage(frameCanvas, sx, sy, sw, sh, 0, 0, dw, dh),
  };

  const output = await pipeline.process(source);

  // adapt the tracker's frame-rate-derived horizons (max_age etc. count in
  // PROCESSED frames) to the achieved inference rate
  const now = performance.now();
  if (lastT > 0) {
    const fps = 1000 / Math.max(now - lastT, 1);
    emaFps = emaFps === 0 ? fps : 0.9 * emaFps + 0.1 * fps;
    const cfg = pipeline.tracker.cfg;
    cfg.fps = Math.max(1, Math.round(emaFps));
    cfg.maxAge = cfg.maxAgeSec > 0 ? Math.round(cfg.maxAgeSec * cfg.fps) : 2 * cfg.fps;
  }
  lastT = now;

  post({ type: 'result', output });
}

// Surface the FIRST underlying failure: ort caches its wasm-init rejection
// and later rethrows a generic "previous call to 'initWasm()' failed", so
// capture every console.error and unhandled rejection seen during init.
const initDiagnostics: string[] = [];
for (const level of ['error', 'warn'] as const) {
  const original = console[level].bind(console);
  console[level] = (...args: unknown[]) => {
    initDiagnostics.push(
      args
        .map((a) => (a instanceof Error ? `${a.message}\n${a.stack ?? ''}` : String(a)))
        .join(' ')
        .slice(0, 500),
    );
    original(...args);
  };
}
self.addEventListener('unhandledrejection', (event) => {
  const reason = (event as PromiseRejectionEvent).reason;
  initDiagnostics.push(
    `unhandledrejection: ${reason instanceof Error ? `${reason.message}\n${reason.stack ?? ''}` : String(reason)}`.slice(0, 500),
  );
});

self.onmessage = (event: MessageEvent<MainToWorkerMessage>) => {
  const msg = event.data;
  if (msg.type === 'init') {
    initialize(msg).catch((error) => {
      const detail = initDiagnostics.length > 0 ? ` | diagnostics: ${initDiagnostics.slice(0, 4).join(' || ')}` : '';
      post({
        type: 'initError',
        message: `${error instanceof Error ? `${error.message}\n${error.stack ?? ''}` : String(error)}${detail}`,
      });
    });
  } else if (msg.type === 'frame') {
    processFrame(msg).catch((error) => {
      post({
        type: 'frameError',
        message: error instanceof Error ? error.message : String(error),
      });
    });
  } else if (msg.type === 'stop') {
    for (const engine of engines) {
      try {
        engine.dispose();
      } catch {
        // already disposed
      }
    }
    engines = [];
    pipeline = null;
  }
};
