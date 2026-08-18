import { app, BrowserWindow, session } from 'electron';
import path from 'node:path';

// dev = vite dev server (npm run dev passes --dev); otherwise load dist/.
const isDev = !app.isPackaged && process.argv.includes('--dev');

// Inference runtime selection: --runtime=litert (default, .tflite) or
// --runtime=ort (onnxruntime-web, .onnx); forwarded to the renderer as a
// query parameter.
const runtimeArg = process.argv.find((a) => a.startsWith('--runtime='));
const inferenceRuntime = runtimeArg?.split('=')[1] === 'ort' ? 'ort' : 'litert';

// Inference execution mode: dedicated worker by default (same design as
// screen-eye-tracking's --web-inference-worker dedicated);
// --web-inference-worker=main runs the engines on the UI thread instead.
const workerArg = process.argv.find((a) => a.startsWith('--web-inference-worker'));
const workerArgValue = workerArg?.includes('=')
  ? workerArg.split('=')[1]
  : process.argv[process.argv.indexOf(workerArg ?? '') + 1];
const inferenceWorkerMode = workerArgValue === 'main' ? 'main' : 'dedicated';

// GPU設定の最適化 (litertjs-test/electron/main.ts と同一構成)。
// これらのコマンドラインスイッチが無いと LiteRT.js の WebGPU アクセラレータ
// から GPU が正しく認識されない。
app.commandLine.appendSwitch('ignore-gpu-blocklist');
// enable-gpu-rasterization は force-cpu-rasterization と競合するため削除
app.commandLine.appendSwitch('enable-zero-copy');
app.commandLine.appendSwitch('disable-gpu-sandbox');
app.commandLine.appendSwitch('enable-unsafe-webgpu');
app.commandLine.appendSwitch('enable-webgpu-developer-features');
// SharedArrayBuffer: file:// pages are never cross-origin isolated (the
// COOP/COEP header interception below only covers http(s)), so enable SAB
// via the feature flag — onnxruntime-web's threaded wasm needs it.
const enabledGpuFeatures = ['WebGPU', 'WebGPUService', 'SharedArrayBuffer'];
if (process.platform !== 'win32' && process.platform !== 'darwin') {
  enabledGpuFeatures.unshift('Vulkan');
}
app.commandLine.appendSwitch('enable-features', enabledGpuFeatures.join(','));
app.commandLine.appendSwitch('use-webgpu-adapter', 'default');
app.commandLine.appendSwitch(
  'disable-features',
  'UseSkiaRenderer,UseChromeOSDirectVideoDecoder',
);

// wasm threads (SharedArrayBuffer) に必要な cross-origin isolation ヘッダ
function setIsolationHeaders(): void {
  session.defaultSession.webRequest.onHeadersReceived((details, callback) => {
    const responseHeaders = {
      ...details.responseHeaders,
      'Cross-Origin-Embedder-Policy': ['require-corp'],
      'Cross-Origin-Opener-Policy': ['same-origin'],
    };
    callback({ responseHeaders });
  });
}

function createWindow(): void {
  const win = new BrowserWindow({
    width: 1500,
    height: 980,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      nodeIntegration: false,
      contextIsolation: true,
      sandbox: false,
    },
  });

  // カメラ許可 (Electron はダイアログを出さないため明示的に許可する)
  session.defaultSession.setPermissionRequestHandler((_wc, permission, callback) => {
    callback(permission === 'media' || permission === 'mediaKeySystem');
  });

  if (isDev) {
    win.loadURL(`http://localhost:5273/?runtime=${inferenceRuntime}&worker=${inferenceWorkerMode}`);
    win.webContents.openDevTools({ mode: 'detach' });
    return;
  }

  win.loadFile(path.join(__dirname, '../dist/index.html'), {
    query: { runtime: inferenceRuntime, worker: inferenceWorkerMode },
  });
}

app.whenReady().then(() => {
  setIsolationHeaders();
  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});
