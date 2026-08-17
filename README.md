# SOMA

**S**tructured **O**utput **M**atching &amp; **A**ssociation: online multi-person
tracking built on the structured output of a multi-task wholebody detector,
one whole-frame pass per frame (plus per-crop ReID embeddings in SOMA-R) —
no tiling, no offline post-processing.

A person is not "one box": it is an **anatomical token** — body box, body
parts (head/shoulders/elbows/hands/knees/feet), head-orientation ring and
person attributes — assembled from one
[YOLOv9-E Wholebody28-Refine](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/468_YOLOv9-Wholebody28-Refine)
forward pass. Association fuses cheap identity channels (box IoU, part OKS,
orientation continuity, attribute vetoes) with anatomical amodal box
synthesis and an online scene-geometry size prior. **SOMA-R** adds an
external ReID embedding as the dominant stage-1 channel plus a post-death
identity memory, an embedding-only revival stage, appearance-locked track
extension and short KF ghost coasting — the long-gap same-id recovery stack.

Every decision is strictly causal (online). This repo is the cleaned
deployment core of the [ptrack](https://github.com/PINTO0309/ptrack)
research tracker: every mechanism present here is exercised by the shipped
presets, and the replay outputs are **bit-identical** to the research repo's
benchmark cells (verified over all 33 CrowdTrack sequences x 3 presets).

## Benchmark

CrowdTrack-MOT-720p train (33 static-CCTV sequences, 25 fps, 720p) — the
long-gap recovery battleground: its ~5s occlusion-episode pool is ~19x MOT17's.
Rendered by `soma-eval table` (protocol details inside
`results/eval_table.json`):

### 640x640 stretch — detector: wb28 — ReID: PersonViT ViT-S/16 aug v3 (raw)

| tracker | HOTA | DetA | AssA | MOTA | IDF1 | IDSW | sw/TP | same-id ~1s | ~3s | ~5s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SOMA | 29.2% | 28.5% | 30.2% | 31.1% | 31.2% | 7,964 | 3.32% | 8% | 0% | 0% |
| **SOMA-R** | **37.4%** | 30.6% | **46.1%** | **33.7%** | **45.2%** | 4,376 | 1.65% | **34%** | **30%** | **44%** |
| ByteTrack | 26.4% | 27.7% | 25.4% | 31.2% | 28.6% | 7,456 | 3.08% | 6% | 0% | 0% |
| BoostTrack++ | 28.9% | 26.1% | 32.2% | 28.9% | 30.4% | 4,500 | 2.08% | 5% | 0% | 0% |
| BoostTrack++-R | 30.7% | 26.3% | 36.0% | 29.3% | 33.2% | 3,426 | 1.58% | 7% | 0% | 0% |

### 640x640 stretch — detector: wb28 — ReID: OSNet-AIN aug v3 (whitened)

| tracker | HOTA | DetA | AssA | MOTA | IDF1 | IDSW | sw/TP | same-id ~1s | ~3s | ~5s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SOMA | 29.2% | 28.5% | 30.2% | 31.1% | 31.2% | 7,964 | 3.32% | 8% | 0% | 0% |
| **SOMA-R** | **36.7%** | 30.6% | **44.4%** | **33.5%** | **44.0%** | 4,938 | 1.87% | **32%** | **26%** | **32%** |
| ByteTrack | 26.4% | 27.7% | 25.4% | 31.2% | 28.6% | 7,456 | 3.08% | 6% | 0% | 0% |
| BoostTrack++ | 28.9% | 26.1% | 32.2% | 28.9% | 30.4% | 4,500 | 2.08% | 5% | 0% | 0% |
| BoostTrack++-R | 30.7% | 26.2% | 35.8% | 29.1% | 32.8% | 3,638 | 1.68% | 6% | 0% | 0% |

All rows share the same low-floor (0.10) wb28 detections; within a section
every ReID row is fed the SAME features (fairness pairing). External
baselines run official code (computed in the research repo via its shims).
`same-id ~Ns` = fraction of ~N-second occlusion episodes re-attached under
the SAME identity — the primary KPI.

## Setup

```bash
uv sync                                  # pinned deps (numpy/opencv/ort-gpu/scipy)
# models/ : copy the three ONNX files (gitignored) —
#   yolov9_e_wholebody28_refine_Nx3HxW.onnx   (detector, dynamic res)
#   personvit_vits16_ain_unified_aug_n.onnx   (SOMA-R ReID, 384-d, raw)
#   osnet_ain_x1_0_p_unified_aug_n.onnx       (SOMA-R ReID, 512-d, whitened)
# data/CrowdTrack-MOT-720p : the benchmark split (MOT layout)
```

ReID preprocessing is RGB `(x/255 - 0.5) / 0.5`; both embedders run
TensorRT fp16 with `batch_max=1` (one static engine shape — validated
config). The detector runs TensorRT fp16 (first run per shape builds the
engine; use `--backend cuda` to skip TensorRT entirely).

## Reproduce

```bash
# token caches (one per variant; ~15-20 min each on an RTX 3070)
soma-eval cache data/CrowdTrack-MOT-720p/train --variant det --out data/cache/l028b_crowdtrack_train
soma-eval cache data/CrowdTrack-MOT-720p/train --variant pv  --out data/cache/l028pv3rb_crowdtrack_train
soma-eval cache data/CrowdTrack-MOT-720p/train --variant os  --out data/cache/l028oaug3Wrb_crowdtrack_train

soma-eval bench --refresh                # recompute the SOMA rows
soma-eval table                          # render the table above

# operation check: live tracking on any video (head mosaic included)
soma-eval video --video path/to/video.mp4 --variant pv
```

`soma-eval run` writes per-sequence MOT rows for external evaluation.

## Package layout

```
soma/
  detector.py     wb28 ONNX inference (stretch/letterbox, TRT/CUDA/CPU)
  assembly.py     detections -> per-person part groups (bone joining)
  tokens.py       part groups -> anatomical tokens (+ amodal synthesis)
  reid.py         SOMA-R crop embedder (TRT fp16, batch_max=1)
  perception.py   frame -> tokens (+ causal whitening for OSNet)
  tracker.py      SOMA tracker (stage-1 fusion, memory, revival, ghost)
  kalman.py       box KF (batched predict) + part-point predictor
  matching.py     Hungarian (scipy fast path, numpy fallback)
  metrics.py      CLEAR/ID + HOTA + gap-recovery evaluator
  mot.py          MOT-format dataset access
  cli.py          soma-eval: cache / run / bench / table / video
```

License: MIT
