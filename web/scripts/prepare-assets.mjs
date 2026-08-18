// Copies the inference runtimes' wasm assets into public/ and builds the
// model manifest. Models are looked up in web/models/ AND ../models/ (the
// python package's model directory) — drop .tflite (LiteRT runtime) or
// .onnx (onnxruntime-web runtime, `--runtime=ort`) files into either and
// they appear in the UI.
import {
  cpSync,
  copyFileSync,
  existsSync,
  mkdirSync,
  readdirSync,
  unlinkSync,
  writeFileSync,
} from 'node:fs';
import path from 'node:path';

const root = process.cwd();
const MODEL_SOURCE_DIRS = [path.join(root, 'models'), path.join(root, '..', 'models')];

// The LiteRT runtime assumes FIXED-RESOLUTION float32 models: float16 /
// quantized exports are excluded from the catalog up front. The "_aug_n_"
// fine-tune variants are out of scope for the LiteRT catalog as well.
const EXCLUDED_TFLITE_PATTERN = /float16|int8|integer_quant|dynamic_range|_aug_n_|rank5-original|\(copy\)/i;
// onnxruntime-web handles dynamic shapes and the python package's canonical
// .onnx exports — exclude only backups/copies.
const EXCLUDED_ONNX_PATTERN = /rank5-original|\(copy\)/i;

function ensureDir(dirPath) {
  mkdirSync(dirPath, { recursive: true });
}

function listModelFiles(dir) {
  if (!existsSync(dir)) {
    return [];
  }
  return readdirSync(dir, { withFileTypes: true })
    .filter((entry) => entry.isFile())
    .filter((entry) => {
      if (entry.name.endsWith('.tflite')) {
        return !EXCLUDED_TFLITE_PATTERN.test(entry.name);
      }
      if (entry.name.endsWith('.onnx')) {
        return !EXCLUDED_ONNX_PATTERN.test(entry.name);
      }
      return false;
    })
    .map((entry) => path.join(dir, entry.name));
}

function copyModels() {
  const modelOutDir = path.join(root, 'public', 'models');
  ensureDir(modelOutDir);

  for (const entry of readdirSync(modelOutDir, { withFileTypes: true })) {
    if (
      entry.isFile() &&
      (entry.name.endsWith('.tflite') || entry.name.endsWith('.onnx') || entry.name === 'manifest.json')
    ) {
      unlinkSync(path.join(modelOutDir, entry.name));
    }
  }

  const seen = new Set();
  const fileNames = [];
  for (const dir of MODEL_SOURCE_DIRS) {
    for (const src of listModelFiles(dir)) {
      const name = path.basename(src);
      if (seen.has(name)) {
        continue;
      }
      seen.add(name);
      copyFileSync(src, path.join(modelOutDir, name));
      fileNames.push(name);
    }
  }
  fileNames.sort();

  writeFileSync(path.join(modelOutDir, 'manifest.json'), `${JSON.stringify(fileNames, null, 2)}\n`);
  console.log(`[prepare-assets] copied ${fileNames.length} model(s)`);
  if (fileNames.length === 0) {
    console.warn(
      '[prepare-assets] no models found — put detector/ReID .tflite or .onnx files into web/models/ or models/',
    );
  }
}

function copyWasm() {
  ensureDir(path.join(root, 'public', 'wasm'));

  const litertWasmSrc = path.join(root, 'node_modules', '@litertjs', 'core', 'wasm');
  const litertWasmDst = path.join(root, 'public', 'wasm', 'litert');
  if (existsSync(litertWasmSrc)) {
    cpSync(litertWasmSrc, litertWasmDst, { recursive: true, force: true });
  } else {
    console.warn(`[prepare-assets] LiteRT wasm directory missing: ${litertWasmSrc}`);
  }

  // onnxruntime-web runtime assets. The webgpu EP of ort 1.27 loads the
  // ASYNCIFY wasm variant at runtime — omitting it fails session creation
  // with a permanently cached "previous call to 'initWasm()' failed".
  const ortDistSrc = path.join(root, 'node_modules', 'onnxruntime-web', 'dist');
  const ortWasmDst = path.join(root, 'public', 'wasm', 'ort');
  const ortFiles = [
    'ort-wasm-simd-threaded.wasm',
    'ort-wasm-simd-threaded.mjs',
    'ort-wasm-simd-threaded.jsep.wasm',
    'ort-wasm-simd-threaded.jsep.mjs',
    'ort-wasm-simd-threaded.asyncify.wasm',
    'ort-wasm-simd-threaded.asyncify.mjs',
    'ort-wasm-simd-threaded.jspi.wasm',
    'ort-wasm-simd-threaded.jspi.mjs',
  ];
  if (existsSync(ortDistSrc)) {
    ensureDir(ortWasmDst);
    for (const file of ortFiles) {
      const src = path.join(ortDistSrc, file);
      if (existsSync(src)) {
        copyFileSync(src, path.join(ortWasmDst, file));
      } else {
        console.warn(`[prepare-assets] onnxruntime-web asset missing: ${src}`);
      }
    }
  } else {
    console.warn(`[prepare-assets] onnxruntime-web dist directory missing: ${ortDistSrc}`);
  }
}

copyModels();
copyWasm();

console.log('[prepare-assets] done');
