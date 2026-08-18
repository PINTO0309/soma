// Main-thread client of the dedicated inference worker: captures RGBA
// frames, transfers them to the worker and awaits tracker results.

// inline (blob URL) worker: from file:// pages the CSP 'self' source cannot
// match worker scripts (opaque origin), while blob: workers stay allowed
import InferenceWorker from '../workers/inference.worker.ts?worker&inline';
import type { Accelerator, RuntimeId } from './engine';
import type { FrameOutput } from '../soma/pipeline';
import type { TrackerConfig } from '../soma/tracker';
import type { MainToWorkerMessage, WorkerToMainMessage } from '../workers/inference.worker';

export interface WorkerInitOptions {
  runtime: RuntimeId;
  accelerator: Accelerator;
  numThreads: number;
  detectorUrl: string;
  reidUrl: string | null;
  whiten: number;
  preset: Partial<TrackerConfig>;
}

export class WorkerPipeline {
  private worker: Worker;
  private waiter: ((msg: WorkerToMainMessage) => void) | null = null;
  private terminated = false;

  constructor() {
    this.worker = new InferenceWorker();
    this.worker.onmessage = (event: MessageEvent<WorkerToMainMessage>) => {
      const resolve = this.waiter;
      this.waiter = null;
      resolve?.(event.data);
    };
    this.worker.onerror = (event: ErrorEvent) => {
      const resolve = this.waiter;
      this.waiter = null;
      resolve?.({ type: 'frameError', message: `inference worker error: ${event.message}` });
    };
  }

  private post(message: MainToWorkerMessage, transfer?: Transferable[]): Promise<WorkerToMainMessage> {
    return new Promise((resolve) => {
      this.waiter = resolve;
      this.worker.postMessage(message, transfer ?? []);
    });
  }

  async init(opts: WorkerInitOptions): Promise<{ accelerator: Accelerator; note: string | null }> {
    const reply = await this.post({
      type: 'init',
      runtime: opts.runtime,
      accelerator: opts.accelerator,
      numThreads: opts.numThreads,
      assetBaseUrl: document.baseURI,
      detectorUrl: new URL(opts.detectorUrl, document.baseURI).href,
      reidUrl: opts.reidUrl === null ? null : new URL(opts.reidUrl, document.baseURI).href,
      whiten: opts.whiten,
      preset: opts.preset,
    });
    if (reply.type === 'ready') {
      return { accelerator: reply.accelerator, note: reply.note };
    }
    throw new Error(reply.type === 'initError' ? reply.message : `unexpected worker reply: ${reply.type}`);
  }

  async process(frame: ImageData): Promise<FrameOutput> {
    const reply = await this.post(
      { type: 'frame', rgba: frame.data.buffer as ArrayBuffer, width: frame.width, height: frame.height },
      [frame.data.buffer as ArrayBuffer],
    );
    if (reply.type === 'result') {
      return reply.output;
    }
    throw new Error(reply.type === 'frameError' ? reply.message : `unexpected worker reply: ${reply.type}`);
  }

  dispose(): void {
    if (!this.terminated) {
      this.worker.postMessage({ type: 'stop' } satisfies MainToWorkerMessage);
      this.worker.terminate();
      this.terminated = true;
    }
  }
}
