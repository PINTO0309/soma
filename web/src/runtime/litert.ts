// LiteRT.js runtime wrapper: loading, compilation, inference and the
// wasm/webgpu exception handling ported from PINTO0309/litertjs-test
// (src/bench/litertBenchmark.ts).

import {
  Tensor,
  getDefaultEnvironment,
  isWebGPUSupported,
  loadAndCompile,
  loadLiteRt,
  type Accelerator,
  type CompiledModel,
  type Tensor as LiteRtTensor,
  type TypedArray as LiteRtTypedArray,
} from '@litertjs/core';

const LITERT_WASM_PATH = './wasm/litert/';
const WEBGPU_INIT_WARN_PREFIX = 'Failed to create default WebGPU device:';
const STATIC_TENSOR_DELEGATE_WARNING_FRAGMENT =
  'Attempting to use a delegate that only supports static-sized tensors';
const LITERT_COMPILE_ERROR_MARKER = 'litert_compiled_model_next.cc:41';
const IGNORED_LITERT_LOG_FRAGMENTS = [
  STATIC_TENSOR_DELEGATE_WARNING_FRAGMENT,
  'litert_web.cc:110',
  LITERT_COMPILE_ERROR_MARKER,
];

const BUILTIN_OPERATOR_CODE_NAMES: Record<number, string> = {
  48: 'TOPK_V2',
  107: 'GATHER_ND',
};
const KNOWN_UNSUPPORTED_BUILTIN_OPERATOR_CODES = new Set<number>([48, 107]);
const TREAT_ANY_CUSTOM_OPERATOR_AS_UNSUPPORTED = true;

export type ParsedOperator =
  | { kind: 'builtin'; builtinCode: number; count: number }
  | { kind: 'custom'; customCode: string; count: number };

type OperatorAnalysisResult =
  | { ok: true; operators: ParsedOperator[] }
  | { ok: false; reason: string };

let liteRtLoadPromise: Promise<void> | null = null;
let liteRtLoadedWithThreads: boolean | null = null;

export async function ensureLiteRtLoaded(threads: boolean): Promise<void> {
  if (liteRtLoadPromise && liteRtLoadedWithThreads !== null && liteRtLoadedWithThreads !== threads) {
    // The runtime can only be initialized once per page; keep the first mode.
    threads = liteRtLoadedWithThreads;
  }
  if (!liteRtLoadPromise) {
    liteRtLoadedWithThreads = threads;
    liteRtLoadPromise = (async () => {
      const originalWarn = console.warn;
      console.warn = (...args: unknown[]) => {
        const [first] = args;
        if (typeof first === 'string' && first.startsWith(WEBGPU_INIT_WARN_PREFIX)) {
          return;
        }
        originalWarn(...args);
      };

      try {
        await loadLiteRt(LITERT_WASM_PATH, { threads });
      } finally {
        console.warn = originalWarn;
      }
    })().catch((error) => {
      liteRtLoadPromise = null;
      liteRtLoadedWithThreads = null;
      throw error;
    });
  }

  await liteRtLoadPromise;
}

function shouldIgnoreLiteRtLogMessage(firstArg: unknown): boolean {
  if (typeof firstArg !== 'string') {
    return false;
  }
  return IGNORED_LITERT_LOG_FRAGMENTS.some((fragment) => firstArg.includes(fragment));
}

export async function withFilteredLiteRtLogs<T>(work: () => Promise<T>): Promise<T> {
  const originalWarn = console.warn;
  const originalError = console.error;

  console.warn = (...args: unknown[]) => {
    if (shouldIgnoreLiteRtLogMessage(args[0])) {
      return;
    }
    originalWarn(...args);
  };

  console.error = (...args: unknown[]) => {
    if (shouldIgnoreLiteRtLogMessage(args[0])) {
      return;
    }
    originalError(...args);
  };

  try {
    return await work();
  } finally {
    console.warn = originalWarn;
    console.error = originalError;
  }
}

export function isStaticTensorDelegateCompileError(message: string): boolean {
  return (
    message.includes(STATIC_TENSOR_DELEGATE_WARNING_FRAGMENT) ||
    message.includes(LITERT_COMPILE_ERROR_MARKER)
  );
}

export function getBuiltinOperatorName(code: number): string {
  return BUILTIN_OPERATOR_CODE_NAMES[code] ?? `BUILTIN_${code}`;
}

export function formatParsedOperator(op: ParsedOperator): string {
  if (op.kind === 'custom') {
    return `CUSTOM:${op.customCode}`;
  }
  return `${getBuiltinOperatorName(op.builtinCode)}(${op.builtinCode})`;
}

export function detectKnownUnsupportedOperators(operators: ParsedOperator[]): ParsedOperator[] {
  return operators.filter((op) => {
    if (op.kind === 'custom') {
      return TREAT_ANY_CUSTOM_OPERATOR_AS_UNSUPPORTED;
    }
    return KNOWN_UNSUPPORTED_BUILTIN_OPERATOR_CODES.has(op.builtinCode);
  });
}

// Minimal FlatBuffer walk over the TFLite schema to enumerate operator codes
// BEFORE compilation — lets us fail with a readable message instead of a
// wasm-side crash for models with unsupported ops (e.g. DETR postprocessing).
export function analyzeTfliteOperators(modelBytes: Uint8Array): OperatorAnalysisResult {
  const decoder = new TextDecoder('utf-8');
  const view = new DataView(modelBytes.buffer, modelBytes.byteOffset, modelBytes.byteLength);

  const checkBounds = (offset: number, size: number): void => {
    if (offset < 0 || size < 0 || offset + size > view.byteLength) {
      throw new Error(`FlatBuffer out of bounds at offset=${offset}, size=${size}`);
    }
  };

  const readInt8 = (offset: number): number => {
    checkBounds(offset, 1);
    return view.getInt8(offset);
  };
  const readUint16 = (offset: number): number => {
    checkBounds(offset, 2);
    return view.getUint16(offset, true);
  };
  const readInt32 = (offset: number): number => {
    checkBounds(offset, 4);
    return view.getInt32(offset, true);
  };
  const readUint32 = (offset: number): number => {
    checkBounds(offset, 4);
    return view.getUint32(offset, true);
  };

  const getTableFieldOffset = (tableOffset: number, fieldIndex: number): number => {
    const vtableOffset = tableOffset - readInt32(tableOffset);
    checkBounds(vtableOffset, 4);
    const vtableSize = readUint16(vtableOffset);
    const entryOffset = vtableOffset + 4 + fieldIndex * 2;
    if (entryOffset + 2 > vtableOffset + vtableSize) {
      return 0;
    }
    return readUint16(entryOffset);
  };

  const getTableFieldPos = (tableOffset: number, fieldIndex: number): number => {
    const fieldOffset = getTableFieldOffset(tableOffset, fieldIndex);
    return fieldOffset === 0 ? 0 : tableOffset + fieldOffset;
  };

  const getIndirectOffset = (offsetPos: number): number => offsetPos + readUint32(offsetPos);

  const readString = (offsetPos: number): string => {
    const stringTable = getIndirectOffset(offsetPos);
    const length = readUint32(stringTable);
    const dataStart = stringTable + 4;
    checkBounds(dataStart, length);
    return decoder.decode(modelBytes.subarray(dataStart, dataStart + length));
  };

  const getVectorInfo = (offsetPos: number): { dataStart: number; length: number } | null => {
    if (offsetPos === 0) {
      return null;
    }
    const vectorTable = offsetPos + readUint32(offsetPos);
    const length = readUint32(vectorTable);
    return { dataStart: vectorTable + 4, length };
  };

  try {
    const modelTable = readUint32(0);
    const opCodesVec = getVectorInfo(getTableFieldPos(modelTable, 1));
    const subgraphsVec = getVectorInfo(getTableFieldPos(modelTable, 2));
    if (!opCodesVec || !subgraphsVec) {
      throw new Error('Missing operator_codes or subgraphs');
    }

    const opCodeRecords: { customCode: string | null; builtinCode: number }[] = [];
    for (let i = 0; i < opCodesVec.length; i += 1) {
      const opCodeTable = getIndirectOffset(opCodesVec.dataStart + i * 4);
      const deprecatedBuiltinPos = getTableFieldPos(opCodeTable, 0);
      const customCodePos = getTableFieldPos(opCodeTable, 1);
      const builtinCodePos = getTableFieldPos(opCodeTable, 3);

      const deprecatedBuiltinCode = deprecatedBuiltinPos === 0 ? 0 : readInt8(deprecatedBuiltinPos);
      const builtinCode = builtinCodePos === 0 ? deprecatedBuiltinCode : readInt32(builtinCodePos);
      const customCode = customCodePos === 0 ? null : readString(customCodePos);
      opCodeRecords.push({ builtinCode, customCode });
    }

    const usedOperators = new Map<string, ParsedOperator>();
    for (let subgraphIndex = 0; subgraphIndex < subgraphsVec.length; subgraphIndex += 1) {
      const subgraphTable = getIndirectOffset(subgraphsVec.dataStart + subgraphIndex * 4);
      const operatorsVec = getVectorInfo(getTableFieldPos(subgraphTable, 3));
      if (!operatorsVec) {
        continue;
      }

      for (let operatorIndex = 0; operatorIndex < operatorsVec.length; operatorIndex += 1) {
        const operatorTable = getIndirectOffset(operatorsVec.dataStart + operatorIndex * 4);
        const opcodeFieldPos = getTableFieldPos(operatorTable, 0);
        if (opcodeFieldPos === 0) {
          continue;
        }

        const opcodeIndex = readUint32(opcodeFieldPos);
        const opCode = opCodeRecords[opcodeIndex];
        if (!opCode) {
          continue;
        }

        if (opCode.customCode && opCode.customCode.length > 0) {
          const key = `custom:${opCode.customCode}`;
          const current = usedOperators.get(key);
          if (current && current.kind === 'custom') {
            current.count += 1;
          } else {
            usedOperators.set(key, { kind: 'custom', customCode: opCode.customCode, count: 1 });
          }
        } else {
          const key = `builtin:${opCode.builtinCode}`;
          const current = usedOperators.get(key);
          if (current && current.kind === 'builtin') {
            current.count += 1;
          } else {
            usedOperators.set(key, { kind: 'builtin', builtinCode: opCode.builtinCode, count: 1 });
          }
        }
      }
    }

    return { ok: true, operators: Array.from(usedOperators.values()) };
  } catch (error) {
    return { ok: false, reason: error instanceof Error ? error.message : String(error) };
  }
}

// Reads the input tensor shapes (shape + shape_signature) of subgraph 0
// straight from the FlatBuffer. Dynamic-shape exports (e.g. Nx3HxW) wedge the
// wasm compile on the renderer main thread, so they must be rejected BEFORE
// loadAndCompile with a readable message.
export function analyzeTfliteInputShapes(modelBytes: Uint8Array): number[][] | null {
  const view = new DataView(modelBytes.buffer, modelBytes.byteOffset, modelBytes.byteLength);
  const readInt32 = (o: number): number => view.getInt32(o, true);
  const readUint16 = (o: number): number => view.getUint16(o, true);
  const readUint32 = (o: number): number => view.getUint32(o, true);
  const fieldPos = (table: number, index: number): number => {
    const vtable = table - readInt32(table);
    const vsize = readUint16(vtable);
    const entry = vtable + 4 + index * 2;
    if (entry + 2 > vtable + vsize) {
      return 0;
    }
    const off = readUint16(entry);
    return off === 0 ? 0 : table + off;
  };
  const indirect = (pos: number): number => pos + readUint32(pos);
  const vector = (pos: number): { start: number; length: number } | null => {
    if (pos === 0) {
      return null;
    }
    const v = pos + readUint32(pos);
    return { start: v + 4, length: readUint32(v) };
  };
  try {
    const model = readUint32(0);
    const subgraphs = vector(fieldPos(model, 2));
    if (!subgraphs || subgraphs.length === 0) {
      return null;
    }
    const sg = indirect(subgraphs.start);
    const tensors = vector(fieldPos(sg, 0));
    const inputs = vector(fieldPos(sg, 1));
    if (!tensors || !inputs) {
      return null;
    }
    const shapes: number[][] = [];
    for (let i = 0; i < inputs.length; i += 1) {
      const tensorIdx = readInt32(inputs.start + i * 4);
      if (tensorIdx < 0 || tensorIdx >= tensors.length) {
        continue;
      }
      const tensor = indirect(tensors.start + tensorIdx * 4);
      const shapeVec = vector(fieldPos(tensor, 0));
      const sigVec = vector(fieldPos(tensor, 7)); // shape_signature: -1 = dynamic
      const dims: number[] = [];
      const n = sigVec?.length ?? shapeVec?.length ?? 0;
      for (let d = 0; d < n; d += 1) {
        const sig = sigVec ? readInt32(sigVec.start + d * 4) : Number.NaN;
        const val = shapeVec && d < shapeVec.length ? readInt32(shapeVec.start + d * 4) : sig;
        dims.push(sigVec && sig === -1 ? -1 : val);
      }
      shapes.push(dims);
    }
    return shapes;
  } catch {
    return null;
  }
}

export interface LoadedModel {
  model: CompiledModel;
  accelerator: Accelerator;
  inputShape: number[]; // normalized (dims <= 0 -> 1)
  layout: 'nchw' | 'nhwc';
  inHeight: number;
  inWidth: number;
  dispose: () => void;
}

export async function fetchModelBytes(url: string): Promise<Uint8Array> {
  const response = await fetch(url, { cache: 'no-store' });
  if (!response.ok) {
    throw new Error(`Failed to fetch model: ${response.status} ${response.statusText}`);
  }
  const buffer = await response.arrayBuffer();
  return new Uint8Array(buffer);
}

export async function loadModel(
  bytes: Uint8Array,
  accelerator: Accelerator,
  numThreads: number,
): Promise<LoadedModel> {
  // Pre-compilation input-shape screening: dynamic-shape exports (Nx3HxW)
  // wedge the wasm compile on the renderer main thread, so reject them with
  // a readable message instead.
  const inputShapes = analyzeTfliteInputShapes(bytes);
  if (inputShapes !== null) {
    for (const dims of inputShapes) {
      // A dynamic batch dim compiles fine (we run batch 1); dynamic spatial
      // dims (H/W) are what wedge the compile.
      if (dims.slice(1).some((d) => d <= 0)) {
        throw new Error(
          `Model has a dynamic input shape [${dims.join(', ')}] — LiteRT.js needs static spatial ` +
          'dims (e.g. 1x3x640x640). Pick the fixed-resolution .tflite variant.',
        );
      }
    }
  }

  // Pre-compilation operator screening (readable failures for models whose
  // ops LiteRT.js does not support in this build).
  const analysis = analyzeTfliteOperators(bytes);
  if (analysis.ok) {
    const unsupported = detectKnownUnsupportedOperators(analysis.operators);
    if (unsupported.length > 0) {
      throw new Error(
        `Model contains operators not supported by LiteRT.js in this build: ${unsupported
          .map((op) => formatParsedOperator(op))
          .join(', ')}`,
      );
    }
  }

  if (accelerator === 'webgpu' && !isWebGPUSupported()) {
    throw new Error('WebGPU is not available in this runtime.');
  }

  await ensureLiteRtLoaded(numThreads > 0);

  if (accelerator === 'webgpu' && !getDefaultEnvironment().webGpuDevice) {
    throw new Error('WebGPU adapter/device is unavailable in this runtime.');
  }

  try {
    const model = await withFilteredLiteRtLogs(async () =>
      loadAndCompile(bytes, {
        accelerator,
        cpuOptions: numThreads > 0 ? { numThreads } : undefined,
      }),
    );
    const detail = model.getInputDetails()[0];
    // Premise of the web runtime: fixed-resolution float32 models only.
    if (detail.dtype !== 'float32') {
      model.delete();
      throw new Error(
        `Model input dtype is ${detail.dtype} — the web runtime assumes float32 models. ` +
        'Pick the *_float32.tflite export.',
      );
    }
    const shape = Array.from(detail.shape).map((d) => (d > 0 ? d : 1));
    if (shape.length !== 4) {
      model.delete();
      throw new Error(`Expected a 4-D image input, got shape [${shape.join(', ')}]`);
    }
    const layout: 'nchw' | 'nhwc' = shape[1] === 3 ? 'nchw' : 'nhwc';
    const inHeight = layout === 'nchw' ? shape[2] : shape[1];
    const inWidth = layout === 'nchw' ? shape[3] : shape[2];
    if (layout === 'nhwc' && shape[3] !== 3) {
      model.delete();
      throw new Error(`Expected a 3-channel image input, got shape [${shape.join(', ')}]`);
    }
    return {
      model,
      accelerator,
      inputShape: shape,
      layout,
      inHeight,
      inWidth,
      dispose: () => model.delete(),
    };
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    if (isStaticTensorDelegateCompileError(message)) {
      throw new Error(
        'LiteRT delegate requires static-sized tensors for this graph, but this model has dynamic-sized tensors.',
      );
    }
    throw error instanceof Error ? error : new Error(message);
  }
}

export async function runModel(
  loaded: LoadedModel,
  input: Float32Array,
): Promise<Float32Array[]> {
  let tensor: LiteRtTensor = new Tensor(input as LiteRtTypedArray, loaded.inputShape);
  const toDispose: LiteRtTensor[] = [];
  try {
    if (loaded.accelerator === 'webgpu') {
      tensor = await tensor.moveTo('webgpu');
    }
    toDispose.push(tensor);
    const outputs = (await loaded.model.run([tensor])) as LiteRtTensor[];
    const results: Float32Array[] = [];
    for (const out of outputs) {
      toDispose.push(out);
      const data = await out.data();
      results.push(
        data instanceof Float32Array ? data.slice() : Float32Array.from(data as ArrayLike<number>),
      );
    }
    return results;
  } finally {
    for (const t of toDispose) {
      try {
        t.delete();
      } catch {
        // already moved/deleted
      }
    }
  }
}

export function outputShapes(loaded: LoadedModel): number[][] {
  return loaded.model.getOutputDetails().map((d) => Array.from(d.shape).map((v) => (v > 0 ? v : 1)));
}

export { isWebGPUSupported };
export type { Accelerator, CompiledModel };
