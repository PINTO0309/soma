"""soma-eval: cache tokens, replay/evaluate, render the benchmark table,
and run live tracking on arbitrary videos.

  soma-eval cache data/CrowdTrack/train --variant pv \\
      --out data/cache/pv_crowdtrack_train
  soma-eval bench                     # recompute SOMA rows -> results/eval_table.json
  soma-eval table                     # render the standing comparison table
  soma-eval video --video path.mp4    # live tracking (SOMA-R PersonViT) -> mp4
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time

import numpy as np

from .mot import find_sequences, frame_path, load_gt
from .perception import Perception
from .presets import PRESETS, VARIANTS
from .tokens import AnatomicalToken
from .tracker import SomaTracker, TrackerConfig

W28 = "models/yolov9_e_wholebody28_refine_Nx3HxW.onnx"
TABLE_JSON = "results/eval_table.json"
CROWDTRACK = "data/CrowdTrack/train"

_TOKEN_FIELDS = ("anchor", "body_box", "body_score", "box_proxy", "presence",
                 "points", "generation", "gender", "embedding", "head_box",
                 "head_dir", "head_dir_conf")


def _tok_to_dict(t: AnatomicalToken) -> dict:
    d = {f: getattr(t, f) for f in _TOKEN_FIELDS}
    if d["embedding"] is not None:
        d["embedding"] = d["embedding"].astype(np.float16)   # cache size
    return d


def _tok_from_dict(d: dict) -> AnatomicalToken:
    emb = d.get("embedding")
    return AnatomicalToken(
        anchor=str(d["anchor"]), body_box=d["body_box"],
        body_score=float(d["body_score"]),
        box_proxy=d["box_proxy"], presence=d["presence"], points=d["points"],
        generation=int(d["generation"]), gender=int(d["gender"]),
        embedding=None if emb is None else emb.astype(np.float32),
        head_box=d.get("head_box"),
        head_dir=d.get("head_dir"),
        head_dir_conf=float(d.get("head_dir_conf", 0.0)),
    )


def _run_sequence(seq, cache_dir: str, cfg_kw: dict) -> tuple[list[tuple], dict]:
    data = np.load(os.path.join(cache_dir, f"{seq.name}.npz"), allow_pickle=True)
    frames = data["frames"]
    sx, sy = data["scale"]
    fps = int(data["fps"])
    cfg = TrackerConfig(scale=(float(sx), float(sy)), **cfg_kw)
    cfg.fps = fps
    if cfg.max_age_sec > 0:
        cfg.max_age = int(round(cfg.max_age_sec * fps))
    else:
        cfg.max_age = 2 * fps
    tracker = SomaTracker(cfg)
    t0 = time.perf_counter()
    for fi in range(1, len(frames) + 1):
        tracker.step(fi, [_tok_from_dict(d) for d in frames[fi - 1]])
    dt = (time.perf_counter() - t0) / len(frames) * 1000
    return tracker.results(), {"track_ms": dt}


# ---- subcommands -----------------------------------------------------------

def cmd_cache(args) -> None:
    import cv2
    var = VARIANTS[args.variant]
    per = Perception(model_path=args.model, mode="stretch",
                     body_score_floor=0.10,
                     input_size=(640, 640),
                     execution_provider=args.backend,
                     reid_path=var["reid"], reid_backend=args.reid_backend,
                     reid_whiten=var["whiten"])
    os.makedirs(args.out, exist_ok=True)
    for seq in find_sequences(args.split):
        dst = os.path.join(args.out, f"{seq.name}.npz")
        if os.path.exists(dst) and not args.force:
            print(f"skip {seq.name} (exists)")
            continue
        frames = []
        t0 = time.perf_counter()
        for fid in range(1, seq.length + 1):
            fr = per(cv2.imread(frame_path(seq, fid)))
            frames.append([_tok_to_dict(t) for t in fr.tokens])
        sx, sy = fr.detections.scale
        np.savez_compressed(dst, frames=np.array(frames, dtype=object),
                            scale=np.array([sx, sy]), fps=seq.fps)
        print(f"cached {seq.name}: {seq.length} frames "
              f"({(time.perf_counter() - t0) / seq.length * 1000:.0f} ms/frame)",
              flush=True)


def cmd_run(args) -> None:
    os.makedirs(args.out, exist_ok=True)
    cfg = dict(PRESETS[args.preset])
    for seq in find_sequences(args.split):
        rows, info = _run_sequence(seq, args.cache, dict(cfg))
        with open(os.path.join(args.out, f"{seq.name}.txt"), "w") as f:
            for r in rows:
                f.write(f"{r[0]},{r[1]},{r[2]:.2f},{r[3]:.2f},{r[4]:.2f},"
                        f"{r[5]:.2f},{r[6]:.2f},-1,-1,-1\n")
        print(f"{seq.name}: {len(rows)} rows ({info['track_ms']:.1f} ms/f)")


# (row key, cache dir, preset) per SOMA-computed cell; section membership and
# the static external baselines live in results/eval_table.json.
BENCH_ROWS = {
    "soma": ("data/cache/l028b_crowdtrack_train", "soma"),
    "somar-pv": ("data/cache/l028pv3rb_crowdtrack_train", "somar-pv"),
    "somar-os": ("data/cache/l028oaug4Wrb_crowdtrack_train", "somar-os"),
}


def _eval_cell(cache: str, preset: str) -> dict:
    from .metrics import (RECOVERY_BUCKETS, combine, evaluate_sequence,
                          hota_accumulate, hota_final, hota_merge,
                          recovery_evaluate, recovery_merge)
    mets, haccs, recs = [], [], []
    for seq in find_sequences(CROWDTRACK):
        rows, _ = _run_sequence(seq, cache, dict(PRESETS[preset]))
        gt_frames, n_gt = load_gt(seq.path)
        m, mh = evaluate_sequence(gt_frames, n_gt, rows, return_matches=True)
        mets.append(m)
        haccs.append(hota_accumulate(gt_frames, rows))
        recs.append(recovery_evaluate(seq.path, seq.fps, mh))
    tot = combine(mets)
    h = hota_final(hota_merge(haccs))
    rec = recovery_merge(recs)
    return {"HOTA": round(h["HOTA"], 4), "DetA": round(h["DetA"], 4),
            "AssA": round(h["AssA"], 4), "MOTA": round(tot["MOTA"], 4),
            "IDF1": round(tot["IDF1"], 4), "IDSW": int(tot["IDSW"]),
            "FP": int(tot["FP"]), "FN": int(tot["FN"]),
            "recovery": {n: [int(v) for v in rec[n]]
                         for n, _, _ in RECOVERY_BUCKETS}}


def cmd_bench(args) -> None:
    with open(TABLE_JSON) as f:
        table = json.load(f)
    for row in args.rows.split(","):
        cache, preset = BENCH_ROWS[row]
        if table.get("cells", {}).get(row) is not None and not args.refresh:
            print(f"cached {row}")
            continue
        print(f"computing {row} ...", flush=True)
        cell = _eval_cell(cache, preset)
        table.setdefault("cells", {})[row] = cell
        with open(TABLE_JSON, "w") as f:
            json.dump(table, f, indent=1, sort_keys=True)
        print(f"  -> {cell['HOTA']:.1%} HOTA, {cell['IDF1']:.1%} IDF1")
    print("wrote", TABLE_JSON)


def cmd_table(args) -> None:
    with open(TABLE_JSON) as f:
        t = json.load(f)
    cells = t["cells"]
    gt_total = t["gt_total"]
    print(f"protocol: {t['protocol']}\n")
    rec_bins = ("~1s", "~3s", "~5s")
    for title, rows in t["sections"]:
        print(f"### {title}\n")
        print("| tracker | HOTA | DetA | AssA | MOTA | IDF1 | IDSW | sw/TP | "
              + " | ".join(f"same-id {b}" for b in rec_bins) + " |")
        print("|---|" + "---:|" * (7 + len(rec_bins)))
        for label, key in rows:
            c = cells.get(key)
            if c is None:
                print(f"| {label} | " + " | ".join(["—"] * (7 + len(rec_bins))) + " |")
                continue
            tp = gt_total - c["FN"]
            vals = [f"{c['HOTA']:.1%}", f"{c['DetA']:.1%}", f"{c['AssA']:.1%}",
                    f"{c['MOTA']:.1%}", f"{c['IDF1']:.1%}", f"{c['IDSW']:,}",
                    f"{c['IDSW'] / max(tp, 1):.2%}"]
            for b in rec_bins:
                n, same = c["recovery"][b][0], c["recovery"][b][1]
                vals.append(f"{same / n:.0%}" if n else "—")
            print(f"| {label} | " + " | ".join(vals) + " |")
        print()


# ---- live video (operation check) ------------------------------------------

def _frost(img, boxes, pad=0.1, cells=9) -> None:
    """Privacy: frosted-glass ellipse over each (padded) head box. The mosaic
    grid is a FIXED cells x cells regardless of head size, so heads stay
    unidentifiable at any resolution."""
    import cv2
    H, W = img.shape[:2]
    for b in boxes:
        w, h = b[2] - b[0], b[3] - b[1]
        x1, y1 = max(0, int(b[0] - pad * w)), max(0, int(b[1] - pad * h))
        x2, y2 = min(W, int(b[2] + pad * w)), min(H, int(b[3] + pad * h))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        roi = img[y1:y2, x1:x2]
        rw, rh = x2 - x1, y2 - y1
        small = cv2.resize(roi, (min(cells, rw), min(cells, rh)),
                           interpolation=cv2.INTER_AREA)
        mosaic = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)
        frosted = cv2.GaussianBlur(mosaic, (0, 0), max(1.5, min(rw, rh) / 12.0))
        mask = np.zeros((rh, rw), np.float32)
        cv2.ellipse(mask, (rw // 2, rh // 2), (rw // 2, rh // 2), 0, 0, 360, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), max(1.0, min(rw, rh) / 16.0))
        m3 = mask[..., None]
        img[y1:y2, x1:x2] = (frosted * m3 + roi * (1.0 - m3)).astype(np.uint8)


def _id_color(tid: int) -> tuple:
    import cv2
    h = (tid * 0.6180339887) % 1.0
    c = cv2.cvtColor(np.array([[[int(h * 180), 200, 255]]], np.uint8),
                     cv2.COLOR_HSV2BGR)[0, 0]
    return int(c[0]), int(c[1]), int(c[2])


def cmd_video(args) -> None:
    import cv2
    var = VARIANTS[args.variant]
    label = {"pv": "SOMA-R / PersonViT ViT-S/16 aug v3 (raw)",
             "os": "SOMA-R / OSNet-AIN aug v3 (whitened)",
             "det": "SOMA (no ReID)"}[args.variant]
    preset = {"pv": "somar-pv", "os": "somar-os", "det": "soma"}[args.variant]
    stem = os.path.splitext(os.path.basename(args.video))[0]
    out_mp4 = args.out or f"results/videos/{stem}_{args.variant}.mp4"
    os.makedirs(os.path.dirname(out_mp4), exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    per = Perception(model_path=args.model, mode="stretch",
                     body_score_floor=0.10, input_size=(640, 640),
                     execution_provider=args.backend,
                     reid_path=var["reid"], reid_backend=args.reid_backend,
                     reid_whiten=var["whiten"])
    tracker = None
    heads: list[np.ndarray] = []
    t0 = time.perf_counter()
    fi = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        fi += 1
        fr = per(img)
        if tracker is None:
            sx, sy = fr.detections.scale
            cfg = TrackerConfig(scale=(float(sx), float(sy)),
                                **PRESETS[preset])
            cfg.fps = int(round(fps))
            cfg.max_age = int(round(cfg.max_age_sec * cfg.fps)) \
                if cfg.max_age_sec > 0 else 2 * cfg.fps
            tracker = SomaTracker(cfg)
        d = fr.detections
        m = (d.labels == 7) & (d.scores >= 0.20)     # Head class, privacy-lean
        heads.append(d.boxes[m].astype(np.float32))
        tracker.step(fi, [_tok_from_dict(_tok_to_dict(t)) for t in fr.tokens])
    cap.release()
    n_frames = fi
    print(f"tracked {n_frames} frames "
          f"({(time.perf_counter() - t0) / max(n_frames, 1) * 1000:.0f} ms/f)")

    rows = tracker.results() if tracker else []
    by_frame: dict[int, list] = {}
    for r in rows:
        by_frame.setdefault(int(r[0]), []).append(r)

    cap = cv2.VideoCapture(args.video)
    ok, img0 = cap.read()
    H, W = img0.shape[:2]
    tmp = out_mp4 + ".raw.mp4"
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for fi in range(1, n_frames + 1):
        ok, img = cap.read()
        if not ok:
            break
        _frost(img, heads[fi - 1])
        for r in by_frame.get(fi, []):
            tid = int(r[1])
            x, y, w, h = (int(round(v)) for v in r[2:6])
            col = _id_color(tid)
            cv2.rectangle(img, (x, y), (x + w, y + h), col, 2)
            txt = str(tid)
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            ty = max(y, th + 4)
            cv2.rectangle(img, (x, ty - th - 4), (x + tw + 6, ty), col, -1)
            cv2.putText(img, txt, (x + 3, ty - 3), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.rectangle(img, (0, 0), (W, 26), (0, 0, 0), -1)
        cv2.putText(img, f"{label}   {stem}  frame {fi}/{n_frames}",
                    (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
        vw.write(img)
    cap.release()
    vw.release()
    try:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp,
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                        "-pix_fmt", "yuv420p", out_mp4], check=True)
        os.remove(tmp)
    except (FileNotFoundError, subprocess.CalledProcessError):
        os.replace(tmp, out_mp4)
    print("wrote", out_mp4)


def main() -> None:
    ap = argparse.ArgumentParser(prog="soma-eval")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("cache", help="run perception over a split and cache tokens")
    c.add_argument("split")
    c.add_argument("--out", required=True)
    c.add_argument("--variant", default="pv", choices=tuple(VARIANTS))
    c.add_argument("--model", default=W28)
    c.add_argument("--backend", default="tensorrt", choices=("cpu", "cuda", "tensorrt"))
    c.add_argument("--reid-backend", default="tensorrt",
                   choices=("cpu", "cuda", "tensorrt"))
    c.add_argument("--force", action="store_true")
    c.set_defaults(fn=cmd_cache)

    r = sub.add_parser("run", help="replay a cache and write MOT rows")
    r.add_argument("split")
    r.add_argument("--cache", required=True)
    r.add_argument("--preset", default="somar-pv", choices=tuple(PRESETS))
    r.add_argument("--out", required=True)
    r.set_defaults(fn=cmd_run)

    b = sub.add_parser("bench", help="recompute SOMA cells of the standing table")
    b.add_argument("--rows", default="soma,somar-pv,somar-os")
    b.add_argument("--refresh", action="store_true")
    b.set_defaults(fn=cmd_bench)

    tb = sub.add_parser("table", help="render the standing comparison table")
    tb.set_defaults(fn=cmd_table)

    v = sub.add_parser("video", help="live tracking on a video file -> mp4")
    v.add_argument("--video", required=True)
    v.add_argument("--variant", default="pv", choices=tuple(VARIANTS))
    v.add_argument("--model", default=W28)
    v.add_argument("--backend", default="tensorrt", choices=("cpu", "cuda", "tensorrt"))
    v.add_argument("--reid-backend", default="tensorrt",
                   choices=("cpu", "cuda", "tensorrt"))
    v.add_argument("--out", default=None)
    v.set_defaults(fn=cmd_video)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
