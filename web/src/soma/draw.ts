// Canvas overlay rendering: id-colored boxes + labels and the frosted-glass
// head mosaic (port of soma/cli.py _id_color / _frost).

import type { Box, TrackRow } from './types';

// Golden-ratio hue walk, HSV(h, 200/255, 1.0) — matches cli.py _id_color.
export function idColor(tid: number): string {
  const h = (tid * 0.618033988749895) % 1.0;
  const s = 200 / 255;
  const v = 1.0;
  const i = Math.floor(h * 6);
  const f = h * 6 - i;
  const p = v * (1 - s);
  const q = v * (1 - f * s);
  const t = v * (1 - (1 - f) * s);
  let r = 0;
  let g = 0;
  let b = 0;
  switch (i % 6) {
    case 0: r = v; g = t; b = p; break;
    case 1: r = q; g = v; b = p; break;
    case 2: r = p; g = v; b = t; break;
    case 3: r = p; g = q; b = v; break;
    case 4: r = t; g = p; b = v; break;
    default: r = v; g = p; b = q; break;
  }
  const to255 = (x: number): number => Math.round(x * 255);
  return `rgb(${to255(r)}, ${to255(g)}, ${to255(b)})`;
}

export function drawTracks(ctx: CanvasRenderingContext2D, rows: TrackRow[]): void {
  ctx.save();
  ctx.font = '13px monospace';
  ctx.textBaseline = 'top';
  for (const r of rows) {
    const col = idColor(r.tid);
    ctx.strokeStyle = col;
    ctx.lineWidth = 2;
    if (r.ghost) {
      ctx.setLineDash([6, 4]);
    } else {
      ctx.setLineDash([]);
    }
    ctx.strokeRect(r.x, r.y, r.w, r.h);
    if (!r.ghost) {
      const label = String(r.tid);
      const tw = ctx.measureText(label).width;
      ctx.fillStyle = col;
      ctx.fillRect(r.x, Math.max(r.y - 17, 0), tw + 8, 17);
      ctx.fillStyle = '#000';
      ctx.fillText(label, r.x + 4, Math.max(r.y - 15, 2));
    }
  }
  ctx.restore();
}

// Privacy: pixelated ellipse over each (padded) head box. Canvas variant of
// cli.py _frost — draw the region tiny, scale it back up unsmoothed, clip to
// an ellipse. The mosaic grid is a FIXED cells x cells (default 9x9)
// regardless of head size, so heads stay unidentifiable at any resolution.
export function frostHeads(
  ctx: CanvasRenderingContext2D,
  source: CanvasImageSource,
  boxes: Box[],
  frameW: number,
  frameH: number,
  pad = 0.1,
  cells = 9,
): void {
  for (const b of boxes) {
    const w = b[2] - b[0];
    const h = b[3] - b[1];
    const x1 = Math.max(0, Math.floor(b[0] - pad * w));
    const y1 = Math.max(0, Math.floor(b[1] - pad * h));
    const x2 = Math.min(frameW, Math.ceil(b[2] + pad * w));
    const y2 = Math.min(frameH, Math.ceil(b[3] + pad * h));
    const rw = x2 - x1;
    const rh = y2 - y1;
    if (rw < 2 || rh < 2) {
      continue;
    }
    const smallW = Math.max(1, Math.min(cells, rw));
    const smallH = Math.max(1, Math.min(cells, rh));
    const tmp = frostHeads.scratch ?? (frostHeads.scratch = document.createElement('canvas'));
    if (tmp.width < smallW || tmp.height < smallH) {
      tmp.width = Math.max(tmp.width, smallW);
      tmp.height = Math.max(tmp.height, smallH);
    }
    const tctx = tmp.getContext('2d');
    if (!tctx) {
      continue;
    }
    tctx.imageSmoothingEnabled = true;
    tctx.drawImage(source, x1, y1, rw, rh, 0, 0, smallW, smallH);

    ctx.save();
    ctx.beginPath();
    ctx.ellipse(x1 + rw / 2, y1 + rh / 2, rw / 2, rh / 2, 0, 0, Math.PI * 2);
    ctx.clip();
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(tmp, 0, 0, smallW, smallH, x1, y1, rw, rh);
    ctx.restore();
  }
}
// eslint-disable-next-line @typescript-eslint/no-namespace
export namespace frostHeads {
  export let scratch: HTMLCanvasElement | undefined;
}
