// Inference-engine abstraction: the SOMA pipeline runs on either LiteRT.js
// (.tflite) or onnxruntime-web (.onnx), selected with the --runtime option,
// on the main thread or (default) inside a dedicated inference worker.

export type Accelerator = 'webgpu' | 'wasm';
export type RuntimeId = 'litert' | 'ort';
export type WorkerMode = 'dedicated' | 'main';

export interface EngineOutput {
  data: Float32Array;
  dims: number[];
}

export interface EngineModel {
  runtime: RuntimeId;
  accelerator: Accelerator;
  layout: 'nchw' | 'nhwc';
  inHeight: number;
  inWidth: number;
  // static output dims when the runtime exposes them (dims <= 0 normalized
  // to 1); null when unknown before the first run
  outputDims: number[][] | null;
  run(input: Float32Array): Promise<EngineOutput[]>;
  // N-batch execution for engines whose models carry a dynamic batch dim
  // (onnxruntime-web with the _aug_n exports); absent on LiteRT, whose
  // runtime pins the compiled input shape to batch 1.
  runBatched?(input: Float32Array, batch: number): Promise<EngineOutput[]>;
  dispose(): void;
}

// Absolute base URL that ./models/ and ./wasm/ resolve against. Workers have
// no document and their own script URL lives under dist/assets/, so the main
// thread passes document.baseURI in the worker init message.
let assetBaseUrl: string =
  typeof document !== 'undefined' ? document.baseURI : self.location.href;

export function setAssetBaseUrl(url: string): void {
  assetBaseUrl = url;
}

export function assetUrl(relative: string): string {
  return new URL(relative, assetBaseUrl).href;
}

export function activeRuntime(): RuntimeId {
  return new URLSearchParams(window.location.search).get('runtime') === 'ort' ? 'ort' : 'litert';
}

export function activeWorkerMode(): WorkerMode {
  return new URLSearchParams(window.location.search).get('worker') === 'main' ? 'main' : 'dedicated';
}

export function modelExtension(runtime: RuntimeId): string {
  return runtime === 'ort' ? '.onnx' : '.tflite';
}
