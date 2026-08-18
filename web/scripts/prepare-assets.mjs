// Copies the LiteRT wasm runtime into public/ and builds the model manifest.
// Models are looked up in web/models/ AND ../models/ (the python package's
// model directory) — drop .tflite files into either and they appear in the UI.
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

function ensureDir(dirPath) {
  mkdirSync(dirPath, { recursive: true });
}

// The LiteRT web runtime assumes FIXED-RESOLUTION float32 models: float16 /
// quantized exports are excluded from the catalog up front. The "_aug_n_"
// fine-tune variants are out of scope for the web runtime as well.
const EXCLUDED_NAME_PATTERN = /float16|int8|integer_quant|dynamic_range|_aug_n_|rank5-original|\(copy\)/i;

function listTfliteFiles(dir) {
  if (!existsSync(dir)) {
    return [];
  }
  return readdirSync(dir, { withFileTypes: true })
    .filter((entry) => entry.isFile() && entry.name.endsWith('.tflite'))
    .filter((entry) => !EXCLUDED_NAME_PATTERN.test(entry.name))
    .map((entry) => path.join(dir, entry.name));
}

function copyModels() {
  const modelOutDir = path.join(root, 'public', 'models');
  ensureDir(modelOutDir);

  for (const entry of readdirSync(modelOutDir, { withFileTypes: true })) {
    if (entry.isFile() && (entry.name.endsWith('.tflite') || entry.name === 'manifest.json')) {
      unlinkSync(path.join(modelOutDir, entry.name));
    }
  }

  const seen = new Set();
  const fileNames = [];
  for (const dir of MODEL_SOURCE_DIRS) {
    for (const src of listTfliteFiles(dir)) {
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
      '[prepare-assets] no .tflite models found — put detector/ReID .tflite files into web/models/ or models/',
    );
  }
}

function copyWasm() {
  const litertWasmSrc = path.join(root, 'node_modules', '@litertjs', 'core', 'wasm');
  const litertWasmDst = path.join(root, 'public', 'wasm', 'litert');

  ensureDir(path.join(root, 'public', 'wasm'));

  if (existsSync(litertWasmSrc)) {
    cpSync(litertWasmSrc, litertWasmDst, { recursive: true, force: true });
  } else {
    console.warn(`[prepare-assets] LiteRT wasm directory missing: ${litertWasmSrc}`);
  }
}

copyModels();
copyWasm();

console.log('[prepare-assets] done');
