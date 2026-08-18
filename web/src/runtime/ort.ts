// onnxruntime-web engine: .onnx models on the wasm or webgpu execution
// provider. Counterpart of the LiteRT loader in litert.ts, exposed through
// the shared EngineModel interface.

// the /webgpu bundle registers both the webgpu (JSEP) and wasm backends
import * as ortNs from 'onnxruntime-web/webgpu';
import { assetUrl, type Accelerator, type EngineModel, type EngineOutput } from './engine';

const ort = ortNs;

let envConfigured = false;

function ensureOrtEnv(): void {
  if (envConfigured) {
    return;
  }
  // absolute URL: ort resolves relative wasmPaths against the BUNDLE's own
  // import.meta.url, not the document/worker location
  ort.env.wasm.wasmPaths = assetUrl('wasm/ort/');
  // Single-threaded wasm, matching the validated configuration of
  // PINTO0309/screen-eye-tracking: ort's thread pool spawns Workers from the
  // wasm .mjs URL, which fails from file:// pages and nested workers.
  ort.env.wasm.numThreads = 1;
  ort.env.logLevel = 'error';
  envConfigured = true;
}

interface ValueMetadataLike {
  name?: string;
  isTensor?: boolean;
  type?: unknown;
  shape?: ReadonlyArray<number | string>;
}

function resolveDims(shape: ReadonlyArray<number | string> | undefined): number[] {
  if (!shape || shape.length !== 4) {
    throw new Error(
      `Expected a 4-D image input, got shape [${(shape ?? []).join(', ')}] — ` +
      'the model must expose static-enough input metadata.',
    );
  }
  // symbolic / dynamic dims: batch -> 1, spatial -> 640 (the python
  // package's detector default; ReID exports carry static H/W anyway)
  return shape.map((d, i) => {
    if (typeof d === 'number' && d > 0) {
      return d;
    }
    return i === 0 ? 1 : 640;
  });
}

export async function loadOrtModel(
  bytes: Uint8Array,
  accelerator: Accelerator,
  _numThreads: number,
): Promise<EngineModel> {
  if (accelerator === 'webgpu' && !('gpu' in navigator)) {
    throw new Error('WebGPU is not available in this runtime.');
  }
  ensureOrtEnv();

  const session = await ort.InferenceSession.create(bytes, {
    executionProviders: accelerator === 'webgpu' ? ['webgpu'] : ['wasm'],
    graphOptimizationLevel: 'all',
  });

  const inputName = session.inputNames[0];
  const inputMeta = (session as unknown as { inputMetadata?: ReadonlyArray<ValueMetadataLike> })
    .inputMetadata?.[0];
  const dims = resolveDims(inputMeta?.shape);
  if (inputMeta?.type !== undefined && inputMeta.type !== 'float32') {
    session.release();
    throw new Error(
      `Model input dtype is ${String(inputMeta.type)} — the web runtime assumes float32 models.`,
    );
  }
  const layout: 'nchw' | 'nhwc' = dims[1] === 3 ? 'nchw' : 'nhwc';
  if (layout === 'nhwc' && dims[3] !== 3) {
    session.release();
    throw new Error(`Expected a 3-channel image input, got shape [${dims.join(', ')}]`);
  }
  const inHeight = layout === 'nchw' ? dims[2] : dims[1];
  const inWidth = layout === 'nchw' ? dims[3] : dims[2];

  const outputMeta = (session as unknown as { outputMetadata?: ReadonlyArray<ValueMetadataLike> })
    .outputMetadata;
  const outputDims = outputMeta
    ? outputMeta.map((m) =>
        (m.shape ?? []).map((d) => (typeof d === 'number' && d > 0 ? d : 1)),
      )
    : null;

  // dynamic batch dim: the export declares a symbolic/negative first dim
  const batchDynamic = (() => {
    const declared = inputMeta?.shape?.[0];
    return !(typeof declared === 'number' && declared > 0);
  })();

  const runWithDims = async (input: Float32Array, runDims: number[]): Promise<EngineOutput[]> => {
    const tensor = new ort.Tensor('float32', input, runDims);
    const results = await session.run({ [inputName]: tensor });
    return session.outputNames.map((name) => {
      const out = results[name];
      const data = out.data;
      return {
        data:
          data instanceof Float32Array
            ? data.slice()
            : Float32Array.from(data as unknown as ArrayLike<number>),
        dims: Array.from(out.dims, (d) => (d > 0 ? Number(d) : 1)),
      };
    });
  };

  return {
    runtime: 'ort',
    accelerator,
    layout,
    inHeight,
    inWidth,
    outputDims,
    run: (input: Float32Array) => runWithDims(input, dims),
    // Verified (RTX 3070, WebGPU EP): batch-4 matches four batch-1 runs at
    // cosine 1.0 and is ~2.9x faster — only exposed for dynamic-batch models.
    runBatched: batchDynamic
      ? (input: Float32Array, batch: number) => runWithDims(input, [batch, ...dims.slice(1)])
      : undefined,
    dispose(): void {
      void session.release();
    },
  };
}
