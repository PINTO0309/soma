import { useCallback, useEffect, useRef, useState } from 'react';
import { listCameras, openCamera, stopStream, type CameraDevice } from './runtime/camera';
import { fetchModelBytes, isWebGPUSupported, loadModel, type Accelerator, type LoadedModel } from './runtime/litert';
import { drawTracks, frostHeads } from './soma/draw';
import { SomaPipeline, makeTrackerConfig, type FrameSource } from './soma/pipeline';
import { PRESETS, VARIANTS, type VariantId } from './soma/presets';
import { WebDetector } from './soma/detector';
import { WebReidEmbedder } from './soma/reid';

interface ModelEntry {
  id: string;
  label: string;
  url: string;
}

type SourceKind = 'camera' | 'video';

interface RunStats {
  fps: number;
  detectMs: number;
  assembleMs: number;
  reidMs: number;
  trackMs: number;
  nTokens: number;
  nTracks: number;
  frame: number;
}

// Name-based classification of catalog entries.
function isReidName(label: string): boolean {
  return /personvit|osnet|reid/i.test(label);
}

function isDetectorEntry(m: ModelEntry): boolean {
  // dynamic-shape exports (Nx3HxW) are rejected at load time anyway — keep
  // them out of the list entirely
  if (/nx3hxw/i.test(m.label)) {
    return false;
  }
  return !isReidName(m.label);
}

// Model-size order for the detection list: n -> t -> s -> e, others last.
const DETECTOR_SIZE_ORDER: Record<string, number> = { n: 0, t: 1, s: 2, e: 3 };

function detectorRank(label: string): number {
  const m = label.match(/_([ntse])_wholebody/i);
  return m ? DETECTOR_SIZE_ORDER[m[1].toLowerCase()] : 9;
}

function defaultDetectorId(list: ModelEntry[]): string {
  const s = list.find((m) => /_s_wholebody/i.test(m.label));
  return s ? s.id : (list[0]?.id ?? '');
}

// ReID list: never detector models; under the pv/os variants only the
// matching embedder family is listed.
function isReidEntry(m: ModelEntry, variant: VariantId): boolean {
  if (!isReidName(m.label)) {
    return false;
  }
  const pv = /personvit/i.test(m.label);
  const os = /osnet/i.test(m.label);
  if (variant === 'pv' && os) {
    return false;
  }
  if (variant === 'os' && pv) {
    return false;
  }
  return true;
}

async function resolveBytes(entry: ModelEntry): Promise<Uint8Array> {
  return fetchModelBytes(entry.url);
}

export default function App() {
  const [models, setModels] = useState<ModelEntry[]>([]);
  const [catalogError, setCatalogError] = useState<string | null>(null);
  const [detectorId, setDetectorId] = useState<string>('');
  const [reidId, setReidId] = useState<string>('');
  const [variant, setVariant] = useState<VariantId>('pv');
  const [backend, setBackend] = useState<Accelerator>('webgpu');
  const [numThreads, setNumThreads] = useState<number>(0);
  const [sourceKind, setSourceKind] = useState<SourceKind>('camera');
  const [cameras, setCameras] = useState<CameraDevice[]>([]);
  const [cameraId, setCameraId] = useState<string>('');
  const [videoFileUrl, setVideoFileUrl] = useState<string | null>(null);
  const [videoFileName, setVideoFileName] = useState<string>('');
  const [privacy, setPrivacy] = useState<boolean>(true);
  const [running, setRunning] = useState<boolean>(false);
  const [status, setStatus] = useState<string>('Idle');
  const [stats, setStats] = useState<RunStats | null>(null);
  const [webgpuOk] = useState<boolean>(() => isWebGPUSupported());

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const runningRef = useRef<boolean>(false);
  const streamRef = useRef<MediaStream | null>(null);
  const loadedModelsRef = useRef<LoadedModel[]>([]);

  // ---- model catalog ------------------------------------------------------
  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const response = await fetch('./models/manifest.json', { cache: 'no-store' });
        if (!response.ok) {
          throw new Error(`manifest: ${response.status} ${response.statusText}`);
        }
        const names = (await response.json()) as unknown;
        if (!alive) {
          return;
        }
        if (!Array.isArray(names)) {
          throw new Error('Invalid model manifest format.');
        }
        const entries: ModelEntry[] = names
          .filter((v): v is string => typeof v === 'string' && v.endsWith('.tflite'))
          .map((name) => ({ id: name, label: name, url: `./models/${name}` }));
        setModels(entries);
        setCatalogError(null);
      } catch (error) {
        if (alive) {
          setCatalogError(error instanceof Error ? error.message : String(error));
        }
      }
    })();
    return () => {
      alive = false;
    };
  }, []);

  useEffect(() => {
    let alive = true;
    (async () => {
      const cams = await listCameras().catch(() => []);
      if (alive) {
        setCameras(cams);
        if (cams.length > 0) {
          setCameraId((prev) => prev || cams[0].deviceId);
        }
      }
    })();
    return () => {
      alive = false;
    };
  }, []);

  // Role/variant-filtered model lists for the two dropdowns; the detection
  // list is ordered n -> t -> s -> e.
  const detectorModels = models
    .filter(isDetectorEntry)
    .sort((a, b) => detectorRank(a.label) - detectorRank(b.label) || a.label.localeCompare(b.label));
  const reidModels = models.filter((m) => isReidEntry(m, variant));

  // Keep selections inside the filtered lists (default detector: the S
  // model; auto-repick when the variant switch or catalog load invalidates
  // the selection).
  useEffect(() => {
    if (!detectorModels.some((m) => m.id === detectorId)) {
      setDetectorId(defaultDetectorId(detectorModels));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [models, detectorId]);
  useEffect(() => {
    if (!reidModels.some((m) => m.id === reidId)) {
      setReidId(reidModels[0]?.id ?? '');
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [models, variant, reidId]);

  // ---- run loop -----------------------------------------------------------
  const stop = useCallback(() => {
    runningRef.current = false;
    setRunning(false);
    stopStream(streamRef.current);
    streamRef.current = null;
    for (const m of loadedModelsRef.current) {
      try {
        m.dispose();
      } catch {
        // already disposed
      }
    }
    loadedModelsRef.current = [];
    const video = videoRef.current;
    if (video) {
      video.pause();
      video.srcObject = null;
    }
  }, []);

  useEffect(() => stop, [stop]);

  const start = useCallback(async () => {
    if (runningRef.current) {
      return;
    }
    const variantInfo = VARIANTS.find((v) => v.id === variant);
    if (!variantInfo) {
      return;
    }
    const detEntry = models.find((m) => m.id === detectorId);
    if (!detEntry) {
      setStatus('Select a detection model (.tflite).');
      return;
    }
    const reidEntry = variantInfo.usesReid ? models.find((m) => m.id === reidId) : null;
    if (variantInfo.usesReid && !reidEntry) {
      setStatus('SOMA-R requires a ReID model (.tflite).');
      return;
    }

    setRunning(true);
    runningRef.current = true;
    setStats(null);

    let accel: Accelerator = backend;
    try {
      // ---- load models (with wasm fallback when webgpu init fails) --------
      setStatus(`Loading detection model (${accel}) ...`);
      const detBytes = await resolveBytes(detEntry);
      let detLoaded: LoadedModel;
      try {
        detLoaded = await loadModel(detBytes, accel, numThreads);
      } catch (error) {
        if (accel === 'webgpu') {
          const msg = error instanceof Error ? error.message : String(error);
          setStatus(`WebGPU init failed — falling back to WASM: ${msg}`);
          accel = 'wasm';
          detLoaded = await loadModel(detBytes, accel, numThreads);
        } else {
          throw error;
        }
      }
      loadedModelsRef.current.push(detLoaded);

      let reidLoaded: LoadedModel | null = null;
      if (variantInfo.usesReid && reidEntry) {
        setStatus(`Loading ReID model (${accel}) ...`);
        const reidBytes = await resolveBytes(reidEntry);
        try {
          reidLoaded = await loadModel(reidBytes, accel, numThreads);
        } catch (error) {
          if (accel === 'webgpu') {
            const msg = error instanceof Error ? error.message : String(error);
            setStatus(`Retrying the ReID model on WASM: ${msg}`);
            reidLoaded = await loadModel(reidBytes, 'wasm', numThreads);
          } else {
            throw error;
          }
        }
        loadedModelsRef.current.push(reidLoaded);
      }

      // ---- open the source ------------------------------------------------
      const video = videoRef.current;
      if (!video) {
        throw new Error('video element unavailable');
      }
      if (sourceKind === 'camera') {
        setStatus('Starting camera ...');
        const stream = await openCamera(cameraId || null);
        streamRef.current = stream;
        video.srcObject = stream;
        video.loop = false;
      } else {
        if (!videoFileUrl) {
          throw new Error('Select a video file.');
        }
        video.srcObject = null;
        video.src = videoFileUrl;
        video.loop = true;
      }
      video.muted = true;
      await video.play();
      const frameW = video.videoWidth;
      const frameH = video.videoHeight;
      if (!frameW || !frameH) {
        throw new Error('Could not determine the source resolution.');
      }

      const canvas = canvasRef.current;
      if (!canvas) {
        throw new Error('canvas element unavailable');
      }
      canvas.width = frameW;
      canvas.height = frameH;
      const ctx = canvas.getContext('2d');
      if (!ctx) {
        throw new Error('2D canvas context unavailable');
      }

      // ---- pipeline -------------------------------------------------------
      const detector = new WebDetector(detLoaded);
      const reid = reidLoaded ? new WebReidEmbedder(reidLoaded) : null;
      const preset = PRESETS[variantInfo.preset];
      const pipeline = new SomaPipeline(
        detector,
        reid,
        variantInfo.whiten,
        makeTrackerConfig(preset, 30),
      );

      const source: FrameSource = {
        width: frameW,
        height: frameH,
        draw: (c, w, h) => c.drawImage(video, 0, 0, w, h),
        drawCrop: (c, sx, sy, sw, sh, dw, dh) => c.drawImage(video, sx, sy, sw, sh, 0, 0, dw, dh),
      };

      setStatus(`Running — ${variantInfo.label} / ${accel}${accel === 'wasm' && numThreads > 0 ? ` (${numThreads} threads)` : ''}`);

      let emaFps = 0;
      let lastT = performance.now();
      const loop = async (): Promise<void> => {
        while (runningRef.current) {
          if (video.paused || video.ended) {
            await new Promise((r) => setTimeout(r, 50));
            continue;
          }
          const out = await pipeline.process(source);
          const now = performance.now();
          const dt = now - lastT;
          lastT = now;
          const fps = 1000 / Math.max(dt, 1);
          emaFps = emaFps === 0 ? fps : 0.9 * emaFps + 0.1 * fps;
          // adapt the tracker's frame-rate-derived horizons to the achieved
          // processing rate (max_age etc. are counted in processed frames)
          const cfg = pipeline.tracker.cfg;
          cfg.fps = Math.max(1, Math.round(emaFps));
          cfg.maxAge = cfg.maxAgeSec > 0
            ? Math.round(cfg.maxAgeSec * cfg.fps)
            : 2 * cfg.fps;

          // ---- render -------------------------------------------------
          ctx.drawImage(video, 0, 0, frameW, frameH);
          if (privacyRef.current) {
            frostHeads(ctx, video, out.headBoxes, frameW, frameH);
          }
          drawTracks(ctx, out.rows);

          setStats({
            fps: emaFps,
            detectMs: out.stats.detectMs,
            assembleMs: out.stats.assembleMs,
            reidMs: out.stats.reidMs,
            trackMs: out.stats.trackMs,
            nTokens: out.stats.nTokens,
            nTracks: out.stats.nTracks,
            frame: pipeline.frame,
          });
          // yield to the event loop so React/UI stays responsive
          await new Promise((r) => setTimeout(r, 0));
        }
      };
      void loop().catch((error) => {
        setStatus(`Runtime error: ${error instanceof Error ? error.message : String(error)}`);
        stop();
      });
    } catch (error) {
      setStatus(`Error: ${error instanceof Error ? error.message : String(error)}`);
      stop();
    }
  }, [backend, cameraId, detectorId, models, numThreads, reidId, sourceKind, stop, variant, videoFileUrl]);

  const privacyRef = useRef(privacy);
  useEffect(() => {
    privacyRef.current = privacy;
  }, [privacy]);

  const variantInfo = VARIANTS.find((v) => v.id === variant);

  return (
    <main className="layout">
      <section className="card controls-card">
        <h1>SOMA Web</h1>
        <p className="subtle">
          Electron + Vite + TypeScript + React + LiteRT ({webgpuOk ? 'WebGPU/WASM' : 'WASM only'}) —
          runs the SOMA / SOMA-R tracker in real time on a webcam or a video file.
        </p>

        <div className="control-grid">
          <label>
            Variant (preset)
            <select value={variant} onChange={(e) => setVariant(e.target.value as VariantId)} disabled={running}>
              {VARIANTS.map((v) => (
                <option key={v.id} value={v.id}>{v.label}</option>
              ))}
            </select>
          </label>

          <label>
            Detection model
            <select value={detectorId} onChange={(e) => setDetectorId(e.target.value)} disabled={running}>
              <option value="">-- select --</option>
              {detectorModels.map((m) => (
                <option key={m.id} value={m.id}>{m.label}</option>
              ))}
            </select>
          </label>

          <label className={variantInfo?.usesReid ? '' : 'dimmed'}>
            ReID model
            <select value={reidId} onChange={(e) => setReidId(e.target.value)} disabled={running || !variantInfo?.usesReid}>
              <option value="">-- select --</option>
              {reidModels.map((m) => (
                <option key={m.id} value={m.id}>{m.label}</option>
              ))}
            </select>
          </label>

          <label>
            Backend
            <select value={backend} onChange={(e) => setBackend(e.target.value as Accelerator)} disabled={running}>
              <option value="webgpu" disabled={!webgpuOk}>LiteRT WebGPU{webgpuOk ? '' : ' (unavailable)'}</option>
              <option value="wasm">LiteRT WASM</option>
            </select>
          </label>

          <label>
            WASM threads (0: default)
            <input
              type="number"
              min={0}
              value={numThreads}
              onChange={(e) => setNumThreads(Math.max(0, Math.floor(Number(e.target.value) || 0)))}
              disabled={running}
            />
          </label>

          <label>
            Source
            <select value={sourceKind} onChange={(e) => setSourceKind(e.target.value as SourceKind)} disabled={running}>
              <option value="camera">Webcam</option>
              <option value="video">Video file</option>
            </select>
          </label>

          {sourceKind === 'camera' ? (
            <label>
              Camera
              <select value={cameraId} onChange={(e) => setCameraId(e.target.value)} disabled={running}>
                {cameras.length === 0 && <option value="">(no camera found)</option>}
                {cameras.map((c) => (
                  <option key={c.deviceId} value={c.deviceId}>{c.label}</option>
                ))}
              </select>
            </label>
          ) : (
            <label>
              Video file
              <input
                type="file"
                accept="video/*"
                disabled={running}
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) {
                    if (videoFileUrl) {
                      URL.revokeObjectURL(videoFileUrl);
                    }
                    setVideoFileUrl(URL.createObjectURL(f));
                    setVideoFileName(f.name);
                  }
                }}
              />
              {videoFileName && <span className="subtle">{videoFileName}</span>}
            </label>
          )}

          <label className="check-row">
            <input type="checkbox" checked={privacy} onChange={(e) => setPrivacy(e.target.checked)} />
            <span>Head mosaic (privacy)</span>
          </label>
        </div>

        <div className="buttons">
          {!running ? (
            <button type="button" onClick={() => void start()}>Start</button>
          ) : (
            <button type="button" className="stop" onClick={stop}>Stop</button>
          )}
        </div>

        <p className="status">Status: {status}</p>
        {catalogError && <p className="subtle">Model catalog: {catalogError}</p>}
        {models.length === 0 && !catalogError && (
          <p className="subtle">
            No .tflite models found. Put them into web/models/ or models/ and run
            `pnpm run prepare:assets`.
          </p>
        )}

        {stats && (
          <div className="stats">
            <div><span>fps</span><b>{stats.fps.toFixed(1)}</b></div>
            <div><span>detect</span><b>{stats.detectMs.toFixed(1)} ms</b></div>
            <div><span>assemble</span><b>{stats.assembleMs.toFixed(1)} ms</b></div>
            <div><span>reid</span><b>{stats.reidMs.toFixed(1)} ms</b></div>
            <div><span>track</span><b>{stats.trackMs.toFixed(2)} ms</b></div>
            <div><span>tokens</span><b>{stats.nTokens}</b></div>
            <div><span>tracks</span><b>{stats.nTracks}</b></div>
            <div><span>frame</span><b>{stats.frame}</b></div>
          </div>
        )}
      </section>

      <section className="card view-card">
        <canvas ref={canvasRef} className="view" />
        <video ref={videoRef} className="hidden-video" playsInline />
      </section>
    </main>
  );
}
