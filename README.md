# SOMA

**S**tructured **O**utput **M**atching &amp; **A**ssociation: online multi-person tracking built on the structured output of a multi-task wholebody detector, one whole-frame pass per frame (plus per-crop ReID embeddings in SOMA-R) — no tiling, no offline post-processing.

![GitHub](https://img.shields.io/github/license/PINTO0309/soma?color=2BAF2B) [![DOI](https://zenodo.org/badge/1337142718.svg)](https://doi.org/10.5281/zenodo.21986816) [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/PINTO0309/soma)

A person is not "one box": it is an **anatomical token** — body box, body parts (head/shoulders/elbows/hands/knees/feet), head-orientation ring and person attributes — assembled from one [YOLOv9-E Wholebody28-Refine](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/468_YOLOv9-Wholebody28-Refine) forward pass. Association fuses cheap identity channels (box IoU, part OKS, orientation continuity, attribute vetoes) with anatomical amodal box synthesis and an online scene-geometry size prior. **SOMA-R** adds an external ReID embedding as the dominant stage-1 channel plus a post-death identity memory, an embedding-only revival stage, appearance-locked track extension and short KF ghost coasting — the long-gap same-id recovery stack.

The detector slot is deliberately interchangeable, not wedded to that one checkpoint: the research lineage walked DEIMv2-Wholebody49 → Wholebody34 → wb28-Refine, and any wholebody detector that emits the same anatomical output family can take its place — [DEIM-Wholebody28](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/465_DEIM-Wholebody28), [YOLO-Wholebody34](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/471_YOLO-Wholebody34), [DEIMv2-Wholebody34](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/472_DEIMv2-Wholebody34), [DEIMv2-Wholebody40](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/485_DEIMv2-Wholebody40) and [DEIMv2-Wholebody49](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/488_DEIMv2-Wholebody49) are all drop-in candidates.

SOMA's center of gravity is **recovery from long occlusion loss**: when a person disappears behind an occluder for seconds and re-emerges, they should come back under the *same id*. Frame-to-frame matching is a largely solved problem that every modern tracker handles well; surviving a multi-second gap is not — and it is exactly what the SOMA-R stack above (identity memory, embedding-only revival, appearance-locked extension, ghost coasting) is built for, and what the primary KPI below (`same-id ~1s/~3s/~5s`) scores directly.

https://github.com/user-attachments/assets/9235a9b7-6e09-4860-a393-fd5c8af6d953

https://github.com/user-attachments/assets/0eae42b9-69fe-4382-bfcd-e7edd5707dbc

## Why SOMA?

The obvious alternative is the classic recipe — ByteTrack-style association on a YOLOX-X detector, plus BoostTrack++ — which looks overwhelmingly strong on the MOT17/MOT20 leaderboards. The research repo behind SOMA measured what those numbers are actually made of, ingredient by ingredient, and found that much of the margin is benchmark artifact, not deployment quality:

- **Train leakage** — the published `bytetrack_x_mot17` detector is trained on the *full* MOT17 train set, so any MOT17-train-derived evaluation partly re-detects its own training data (MOTA ~88-90).
- **Detector-benchmark co-adaptation** — that ByteTrack YOLOX-X is trained on a dataset recipe assembled *for this benchmark family*, and it regresses MOT's **amodal full-body boxes**: a person standing behind an occluder still gets a box stretched over their invisible legs. That convention serves exactly one consumer — MOT-style association — and is the wrong primitive for every other use of a person detector: crop such a box and a ReID embedder, pose estimator or privacy mask mostly sees the occluder; gate a region-entry counter with it and people walk through walls. A generic visible-extent detector is systematically IoU-penalized by MOT GT for refusing to hallucinate — a penalty that says nothing about real-world quality. SOMA keeps the detector generic and reusable beyond tracking (visible-extent boxes + parts) and synthesizes the amodal box *inside the tracker* instead.

  <img width="500" alt="amodal_bbox" src="https://github.com/user-attachments/assets/bef1de7e-3c12-43be-9531-7bf04c538431" />

- **The literature is detector-bound** — for well over five years the tracking-by-detection line of papers has presupposed the output of a specialized detection model (benchmark-adapted, amodal, high-resolution), so a "better tracker" result holds only inside that detector's output distribution. The association step moves single HOTA points while the detector choice moves tens — the research repo measured +14 HOTA / +19 MOTA on MOT20 from swapping the detector alone under the *same* SOMA-R tracker. To first order, tracker rankings are detector rankings.
- **The assumed ReID is weak** — when those papers do attach an appearance model, it is an off-the-shelf ReID checkpoint of very modest accuracy: the untuned official multi-source OSNet-AIN quoted in Setup below scores mAP ~0.46-0.74 on the official ReID splits, and that class of embedder is what the literature's lukewarm "ReID adds little" verdicts are built on. That verdict describes the embedder, not appearance itself: swap in a properly fine-tuned embedder (mAP 0.87-0.99 under the same protocol) and the embedding carries SOMA-R's entire long-gap recovery story — the difference between 0% and 44% in the ~5s bin.
- **Tuning and post-processing** — official thresholds were tuned on the very sequences being scored (official repos even carry per-sequence overrides), and published numbers include offline interpolation — unusable in a live system.
- **The compute budget is off-screen** — most published configurations assume high-resolution inputs (well above 720p; ByteTrack's official test size is 1440x800), where detector inference — not the tracker's few milliseconds of association — is nearly the entire end-to-end cost. A ranking bought at that input size says nothing about what holds up on a real deployment budget.
- **The metric barely sees the hardest failure** — losing a person for *seconds* and re-acquiring them with the same id is the failure that hurts in production, yet MOT17 val contains exactly **7** ground-truth occlusion episodes in the 4-6 s range.

SOMA is the counter-design, built for the question those leaderboards don't answer — *what tracks well behind a generic detector in a live system?*

1. **Structure over scale.** A multi-task wholebody detector that was never trained on MOT data already emits rich per-person structure in one whole-frame pass; SOMA turns it into an anatomical token and fuses many weak, cheap identity channels instead of relying on one strong, expensive one.
2. **Online only, end to end.** No offline interpolation ever, no per-sequence tuning, TensorRT throughout — every number in the table below is producible by the live pipeline.
3. **Low resolution by design.** SOMA assumes VGA-class inputs and below — the shipped presets run the detector at 640x640 stretch even on 720p footage — so the dominant cost term stays small enough for real-time edge deployment, and the detection-recall hit is taken openly instead of being hidden behind a high-resolution detector pass.
4. **Score what hurts in production.** The primary KPI is long-gap same-id recovery (`same-id ~1s/~3s/~5s`) plus the coverage-fair switch rate `sw/TP` — hence the [CrowdTrack](https://github.com/loseevaya/CrowdTrack) benchmark, whose ~5s occlusion-episode pool is ~19x MOT17's.

> [!IMPORTANT]
> **However, since object detection models, ReID models, and tracking algorithms each have distinct strengths and weaknesses depending on the situation, we make no exaggerated claims regarding the accuracy of the SOMA tracker. The sole focus of the discussion is whether or not practical use cases exist.**

## Benchmark

[CrowdTrack](https://github.com/loseevaya/CrowdTrack) train — the long-gap recovery battleground: its ~5s occlusion-episode pool is ~19x MOT17's. Rendered by `soma-eval table` (protocol details inside `results/eval_table.json`):

| column | better | meaning |
|---|:---:|---|
| HOTA | ⏫ | Higher-Order Tracking Accuracy — geometric mean of detection and association quality; the standard single-number ranking metric |
| DetA | ⏫ | Detection Accuracy — how much of the ground truth is covered by correctly localized boxes |
| AssA | ⏫ | Association Accuracy — how consistently detections are linked into the correct identity over time |
| MOTA | ⏫ | Multi-Object Tracking Accuracy — 1 − (FN + FP + IDSW) / GT; dominated by detection errors |
| IDF1 | ⏫ | Identity F1 — rewards keeping each person under one id across their whole trajectory |
| IDSW | ⬇️ | Identity switches — absolute count of id changes on continuing tracks |
| sw/TP | ⬇️ | Coverage-fair switch rate — IDSW per tracked box (TP = GT − FN); comparable across trackers with different recall, unlike absolute IDSW |
| ~1s / ~3s / ~5s | ⏫ | **Long-gap same-id recovery — the primary KPI.** Fraction of occlusion episodes (a tracked person fully hidden for ~N seconds — ~25 / ~75 / ~125 frames at 25 fps — then re-emerging; counted only if tracked in the second before the gap) re-attached under the **same id** after re-emergence |

Concretely: SOMA-R's **44%** in the ~5s bin below means 44% of the people who vanished for ~5 seconds came back with their identity intact — every other tracker scores **0%** there; each of those people was reborn as a "new" person. Frame-to-frame metrics barely register these events; these columns are what SOMA is designed to win.

### 640x640 stretch — detector: wb28 — ReID: PersonViT ViT-S/16 aug v3 (raw)

| tracker | HOTA | DetA | AssA | MOTA | IDF1 | IDSW | sw/TP | ~1s | ~3s | ~5s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SOMA | 29.2% | 28.5% | 30.2% | 31.1% | 31.2% | 7,964 | 3.32% | 8% | 0% | 0% |
| **SOMA-R** | **37.4%** | 30.6% | **46.1%** | **33.7%** | **45.2%** | 4,376 | 1.65% | **34%** | **30%** | **44%** |
| ByteTrack | 26.4% | 27.7% | 25.4% | 31.2% | 28.6% | 7,456 | 3.08% | 6% | 0% | 0% |
| BoostTrack++ | 28.9% | 26.1% | 32.2% | 28.9% | 30.4% | 4,500 | 2.08% | 5% | 0% | 0% |
| BoostTrack++-R | 30.7% | 26.3% | 36.0% | 29.3% | 33.2% | 3,426 | 1.58% | 7% | 0% | 0% |

### 640x640 stretch — detector: wb28 — ReID: OSNet-AIN aug v3 (whitened)

| tracker | HOTA | DetA | AssA | MOTA | IDF1 | IDSW | sw/TP | ~1s | ~3s | ~5s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SOMA | 29.2% | 28.5% | 30.2% | 31.1% | 31.2% | 7,964 | 3.32% | 8% | 0% | 0% |
| **SOMA-R** | **36.7%** | 30.6% | **44.4%** | **33.5%** | **44.0%** | 4,938 | 1.87% | **32%** | **26%** | **32%** |
| ByteTrack | 26.4% | 27.7% | 25.4% | 31.2% | 28.6% | 7,456 | 3.08% | 6% | 0% | 0% |
| BoostTrack++ | 28.9% | 26.1% | 32.2% | 28.9% | 30.4% | 4,500 | 2.08% | 5% | 0% | 0% |
| BoostTrack++-R | 30.7% | 26.2% | 35.8% | 29.1% | 32.8% | 3,638 | 1.68% | 6% | 0% | 0% |

All rows share the same low-floor (0.10) wb28 detections; within a section every ReID row is fed the SAME features (fairness pairing). External baselines run official code (computed in the research repo via its shims).

## Setup

```bash
uv sync # pinned deps (numpy/opencv/ort-gpu/scipy)
# models/ : copy the three ONNX files (gitignored) —
#   yolov9_e_wholebody28_refine_Nx3HxW.onnx   (detector, dynamic res)
#   personvit_vits16_ain_unified_aug_n.onnx   (SOMA-R ReID, 384-d, raw)
#   osnet_ain_x1_0_p_unified_aug_n.onnx       (SOMA-R ReID, 512-d, whitened)
# data/CrowdTrack : the benchmark split (MOT layout)
```

Both SOMA-R ReID embedders (PersonViT / OSNet-AIN) are distributed in [PINTO_model_zoo 502_PersonViT](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/502_PersonViT). ReID preprocessing is RGB `(x/255 - 0.5) / 0.5`; both embedders run TensorRT fp16 with `batch_max=1` (one static engine shape — validated config). The detector runs TensorRT fp16 (first run per shape builds the engine; use `--backend cuda` to skip TensorRT entirely).

| embedder (`--variant`) | fine-tune | backbone | params | GFLOPs<br>@256x128 | emb |
|---|---|---|---:|---:|---:|
| `personvit_vits16_ain_unified_aug_n` | [S-ain-aug](https://github.com/PINTO0309/PersonViT/tree/uv#s-ain-aug----vit-s16--token-in---220m) | ViT-S/16 + token-IN | 22.0M | 2.94 | 384 |
| `osnet_ain_x1_0_p_unified_aug_n` | [P-ain-aug](https://github.com/PINTO0309/PersonViT/tree/uv#p-ain-aug---osnet-ain-x10---22m) | OSNet-AIN x1.0 | 2.2M | 0.98 | 512 |

Standalone ReID accuracy of the two fine-tuned embedders, quoted from the [PersonViT](https://github.com/PINTO0309/PersonViT/tree/uv) repo:

**[S-ain-aug](https://github.com/PINTO0309/PersonViT/tree/uv#s-ain-aug----vit-s16--token-in---220m) — ViT-S/16 + token-IN, 384-d — `personvit_vits16_ain_unified_aug_n.onnx` (`--variant pv`)**

- official dataset eval

  | dataset | queries | gallery | mAP | R1 | R5 | R10 |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: |
  | market | 3,368 | 15,913 | 0.9872 | 0.9911 | 0.9976 | 0.9991 |
  | msmt17 | 11,659 | 82,161 | 0.9397 | 0.9697 | 0.9860 | 0.9882 |
  | duke_occ | 2,210 | 17,661 | 0.9527 | 0.9674 | 0.9819 | 0.9855 |
  | cuhk03np | 1,400 | 5,332 | 0.9875 | 0.9900 | 0.9950 | 0.9986 |
  | occ_reid | 1,000 | 1,000 | 0.9955 | 0.9960 | 0.9980 | 0.9990 |

- official dataset style-shift eval - query only shifted

  | condition | mAP | R1 | dmAP | dR1 |
  | --- | ---: | ---: | ---: | ---: |
  | clean | 0.9555 | 0.9759 | — | — |
  | bright+30% | 0.9547 | 0.9753 | -0.0008 | -0.0006 |
  | dark-30% | 0.9555 | 0.9760 | +0.0000 | +0.0001 |
  | contrast-40% | 0.9555 | 0.9760 | +0.0000 | +0.0001 |
  | contrast+40% | 0.9090 | 0.9400 | -0.0465 | -0.0360 |
  | warm | 0.9192 | 0.9477 | -0.0363 | -0.0282 |
  | cool | 0.9411 | 0.9664 | -0.0145 | -0.0095 |
  | gamma0.6 | 0.9481 | 0.9717 | -0.0074 | -0.0042 |
  | gamma1.6 | 0.9448 | 0.9705 | -0.0107 | -0.0054 |
  | jpeg-q40 | 0.9523 | 0.9748 | -0.0032 | -0.0011 |
  | jpeg-q20 | 0.9430 | 0.9693 | -0.0125 | -0.0066 |

**[P-ain-aug](https://github.com/PINTO0309/PersonViT/tree/uv#p-ain-aug---osnet-ain-x10---22m) — OSNet-AIN x1.0, 512-d — `osnet_ain_x1_0_p_unified_aug_n.onnx` (`--variant os`)**

- official dataset eval

  | dataset | queries | gallery | mAP | R1 | R5 | R10 |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: |
  | market | 3,368 | 15,913 | 0.9711 | 0.9857 | 0.9964 | 0.9976 |
  | msmt17 | 11,659 | 82,161 | 0.8711 | 0.9472 | 0.9780 | 0.9828 |
  | duke_occ | 2,210 | 17,661 | 0.8995 | 0.9362 | 0.9733 | 0.9801 |
  | cuhk03np | 1,400 | 5,332 | 0.9800 | 0.9857 | 0.9936 | 0.9979 |
  | occ_reid | 1,000 | 1,000 | 0.9878 | 0.9900 | 0.9940 | 0.9980 |

- official dataset style-shift eval - query only shifted

  | condition | mAP | R1 | dmAP | dR1 |
  | --- | ---: | ---: | ---: | ---: |
  | clean | 0.9051 | 0.9575 | — | — |
  | bright+30% | 0.9014 | 0.9547 | -0.0037 | -0.0027 |
  | dark-30% | 0.9042 | 0.9566 | -0.0009 | -0.0009 |
  | contrast-40% | 0.9051 | 0.9575 | -0.0000 | +0.0001 |
  | contrast+40% | 0.8172 | 0.8775 | -0.0879 | -0.0800 |
  | warm | 0.8159 | 0.8849 | -0.0892 | -0.0726 |
  | cool | 0.8447 | 0.9107 | -0.0605 | -0.0467 |
  | gamma0.6 | 0.8865 | 0.9451 | -0.0186 | -0.0124 |
  | gamma1.6 | 0.8713 | 0.9345 | -0.0338 | -0.0230 |
  | jpeg-q40 | 0.8982 | 0.9529 | -0.0070 | -0.0046 |
  | jpeg-q20 | 0.8776 | 0.9367 | -0.0275 | -0.0208 |

**[osnet_ain_ms_d_c](https://github.com/PINTO0309/PersonViT/tree/uv#osnet_ain_ms_d_c---22m) — official multi-source OSNet-AIN x1.0, 512-d — untuned reference (not shipped)**

For comparison under the same criteria: the official pre-trained OSNet-AIN weights — the same 2.2M architecture as P-ain-aug, without the unified fine-tuning:

- official dataset eval

  | dataset | queries | gallery | mAP | R1 | R5 | R10 |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: |
  | market | 3,368 | 15,913 | 0.4580 | 0.7304 | 0.8655 | 0.9047 |
  | msmt17 | 11,659 | 82,161 | 0.4869 | 0.7613 | 0.8662 | 0.8965 |
  | duke_occ | 2,210 | 17,661 | 0.4757 | 0.6167 | 0.7670 | 0.8163 |
  | cuhk03np | 1,400 | 5,332 | 0.5776 | 0.6079 | 0.7779 | 0.8543 |
  | occ_reid | 1,000 | 1,000 | 0.7407 | 0.8040 | 0.8970 | 0.9320 |

- official dataset style-shift eval - query only shifted

  | condition | mAP | R1 | dmAP | dR1 |
  | --- | ---: | ---: | ---: | ---: |
  | clean | 0.5001 | 0.7310 | — | — |
  | bright+30% | 0.4945 | 0.7236 | -0.0055 | -0.0074 |
  | dark-30% | 0.4992 | 0.7307 | -0.0009 | -0.0003 |
  | contrast-40% | 0.4903 | 0.7220 | -0.0097 | -0.0090 |
  | contrast+40% | 0.4101 | 0.6127 | -0.0900 | -0.1183 |
  | warm | 0.4159 | 0.6370 | -0.0841 | -0.0940 |
  | cool | 0.4340 | 0.6668 | -0.0661 | -0.0642 |
  | gamma0.6 | 0.4791 | 0.7068 | -0.0210 | -0.0242 |
  | gamma1.6 | 0.4492 | 0.6788 | -0.0508 | -0.0521 |
  | jpeg-q40 | 0.4819 | 0.7087 | -0.0182 | -0.0223 |
  | jpeg-q20 | 0.4379 | 0.6571 | -0.0621 | -0.0738 |

## Fine-tuning ReID Models Using Synthetic Datasets

See: https://github.com/PINTO0309/PersonViT

A synthetic person re-identification (ReID) dataset of 500 fictional adults rendered under 33 fixed camera conditions, generated with the gpt-image-2 model family (quality locked to low).

### Scale and splits

| Item | Value |
| --- | --- |
| Total images | 20,000 JPEG |
| train | 16,000 images (400 ids) |
| query | 400 images (100 ids, **occluded**: occlusion ratio 0.20–0.50) |
| gallery | 3,600 images (100 ids, clean) |
| train/test id overlap | none (p000–p399 / p400–p499) |
| Image size | 128×256 RGB JPEG (the unified-pipeline standard) |

### Protocol structure (uniform across all ids, verified)

- **40 images/id over 8 cameras/id** (train includes 12 occluded images per id)
- Eval-id split shape: 4 query + 36 gallery images per id
- **Exactly 32 cross-camera positives per query** — cross-camera matching is always possible
- Filenames follow the unified reid convention `p{pid:05d}_d{domain:02d}_c{camera:03d}_{seq:06d}.jpg` (domain `d05`, cameras in the reserved global range `c033`–`c065`)

### Camera system (33 cameras, seed-locked geometry)

Every camera has a seed-locked mounting height, downward pitch, focal length, and derived horizon/eye-level vanishing-line position, grouped　into three view families:

| Family | Downward pitch | Intent |
| --- | --- | --- |
| high-wide | ~20–28° | elevated wide-angle (surveillance overhead) |
| diagonal-medium | ~7–12° | diagonal mid-range |
| telephoto-exit | ~1.5–4.5° | doorway telephoto (near-horizontal) |

The geometry is recorded in `state/cameras.jsonl`, in the generation　prompts, and in every manifest row (e.g. mounting height 5.44 m / pitch −24.17° / focal 20 mm). Body yaw is also generated in labeled steps (front / front_right / right, ...).

### Generation-pipeline characteristics

- Model family locked to gpt-image-2 (no automatic fallback to another family), quality locked to low, seed 20260815
- Full generation proceeds only after a 96-image quality pilot and a 96-image body-rotation pilot pass automatic and manual QA
- Per-camera JPEG quality varies (e.g. 80/82/86) to mimic real-world compression diversity

  <img width="680" height="574" alt="image" src="https://github.com/user-attachments/assets/2f86f3b0-37e6-4232-b2d9-a2aa557a397d" />

## Reproduce

```bash
# token caches (one per variant; ~15-20 min each on an RTX 3070)
soma-eval cache data/CrowdTrack/train \
--variant det \
--out data/cache/l028b_crowdtrack_train

soma-eval cache data/CrowdTrack/train \
--variant pv  \
--out data/cache/l028pv3rb_crowdtrack_train

soma-eval cache data/CrowdTrack/train \
--variant os  \
--out data/cache/l028oaug3Wrb_crowdtrack_train

soma-eval bench --refresh # recompute the SOMA rows
soma-eval table           # render the table above

# operation check: live tracking on any video (head mosaic included)
soma-eval video \
--video path/to/video.mp4 \
--variant pv
```

`soma-eval run` writes per-sequence MOT rows for external evaluation.

## Package layout

The repository is split at the top level: `soma/` is the python package
(ONNX / TensorRT, benchmark + evaluation), `web/` is the standalone web
runtime (Electron + LiteRT) — neither imports the other.

```
soma/                 python package (benchmark & evaluation stack)
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

web/                  web runtime (Electron + Vite + TypeScript + React + LiteRT)
  electron/         main process (WebGPU command-line switches, COOP/COEP)
  src/runtime/      LiteRT load/compile/run + wasm/webgpu exception handling,
                    webcam helper
  src/soma/         TypeScript port of the tracking stack (constants, assembly,
                    tokens, kalman, matching, tracker, presets, detector/reid
                    inference, pipeline, canvas overlay)
  scripts/          asset staging (models + LiteRT wasm), zero-dependency dev runner
```

## Testing web runtime kernel efficiency (Electron + LiteRT.js/onnxruntime-web)

Live SOMA / SOMA-R tracking on a webcam or a video file, fully in-browser inference: LiteRT.js with the **WebGPU** accelerator (or WASM fallback), no python and no server. The tracking core is a line-by-line TypeScript port of `soma/` — same presets (`soma`, `somar-pv`, `somar-os`), same association stack (stage-1 fusion, identity memory, embedding-only revival, ghost coasting, head-mosaic privacy overlay).

<img width="1000" alt="image" src="https://github.com/user-attachments/assets/6a48521c-9fa6-4907-9283-7992579f2629" />

```bash
cd web
pnpm install           # pnpm >= 10.16 REQUIRED (npm/yarn installs are blocked)
# models: drop model exports into web/models/ or ../models/
#   LiteRT runtime (default): fixed-resolution float32 .tflite exports
#     yolov9_n_wholebody28_refine_1x3x640x640_float32.tflite (detector; N/T/S = real-time)
#     personvit_vits16_ain_unified_aug_float32.tflite        (SOMA-R ReID, 384-d)
#     osnet_ain_x1_0_p_unified_aug_float32.tflite            (SOMA-R ReID, 512-d)
#     ("_aug_n_" exports and float16/quantized files are out of scope and filtered out)
#   onnxruntime-web runtime (--runtime=ort): .onnx exports (dynamic shapes allowed)
pnpm run dev                          # vite + electron (dev)
pnpm run start                        # production build + launch
# runtime is switchable in the GUI (Runtime selector, top of the left pane);
# --runtime=ort merely sets the initial value. Inference runs in a dedicated
# worker by default; opt out with:
pnpm run start -- --web-inference-worker main
```

Supply-chain hardening: dependencies are pinned to exact versions with the lockfile committed (`pnpm audit` clean), dependency build scripts are blocked except electron/esbuild, and `minimumReleaseAge: 10080` in `pnpm-workspace.yaml` quarantines any package version published less than **7 days** ago — a freshly compromised release cannot enter the project (this is why electron is pinned to 43.3.0 rather than the 6-day-old 43.4.0; both are outside the advisory range of GHSA-9f4c-93c8-jc8g).

- **Model switching is user-driven**: the detection model and the ReID model are selected independently in the UI from the staged catalog (`web/models/` and `models/`); the ReID list follows the selected variant (PersonViT / OSNet-AIN).
- **Two inference runtimes**: LiteRT.js (`.tflite`, default) and onnxruntime-web (`.onnx`) behind one engine interface, switchable from the GUI's Runtime selector (or the `--runtime=ort` startup option). The ort runtime accepts dynamic-shape exports and runs single-threaded wasm orchestration under its WebGPU EP (measured: YOLOv9-S ~31 ms/frame, PersonViT ~13 ms/crop, SOMA-R ~6.4 fps end-to-end).
- **Dedicated inference worker by default** (the `--web-inference-worker dedicated` design of PINTO0309/screen-eye-tracking): the whole perception + tracking pipeline runs in a worker, keeping the UI thread free — SOMA-R (LiteRT/PersonViT) improves from ~7.4 to **~9.2 fps** end-to-end. `--web-inference-worker main` runs the engines on the UI thread instead.
- **Premises**: LiteRT runs **fixed-resolution float32** exports. Dynamic spatial shapes (`Nx3HxW`) and non-float32 inputs are rejected up front with a readable message (float16/quantized files are filtered out of the catalog); a dynamic batch dim is fine.
- The WebGPU chromium switches in `web/electron/main.ts` (`enable-unsafe-webgpu`, `Vulkan` feature on Linux, ...) are required — without them the GPU is not recognized by LiteRT's WebGPU accelerator. wasm/webgpu error classification follows PINTO0309/litertjs-test.
- Measured on the CrowdTrack test footage (RTX 3070, LiteRT WebGPU, detector inference per frame): YOLOv9-**N** wb28 ~25 ms (~39 fps end-to-end), **T** wb25 ~26 ms, **S** wb28 ~29 ms, **E** wb28 ~2.7 s — use the N/T/S exports for real-time; the detector slot accepts both the wb28 and wb25 vocabularies (class ids are remapped automatically).
- SOMA-R note: the shipped PersonViT `.tflite` exports are graph-optimized for full WebGPU delegation — the original exports carried rank-5 GATHER/RESHAPE in every attention block, which the GPU delegate rejects ("Tensor dimensions must be less than 5"), silently pushing the whole transformer onto the CPU (~130 ms/crop). [tools/optimize_personvit_tflite.py](tools/optimize_personvit_tflite.py) rewrites the qkv unpack to rank-4 Split-style gathers (bit-identical outputs, batch-1 only); with it PersonViT runs **~11 ms/crop** on WebGPU and SOMA-R reaches **~7 fps** end-to-end with the YOLOv9-S detector. OSNet remains impractical on the current LiteRT WebGPU (~1.4 s/crop, slow depthwise convolutions) — prefer the PersonViT embedder in the web runtime.
- **Few-person whitening guard** (web-only deviation from the python package): OSNet whitening statistics are computed across the people in a frame, so sparse scenes degenerate them — with 1-2 people the frame mean absorbs the present identities and same-person whitened cosine collapses to ~0-0.2, mis-firing the calibrated `somar-os` gates. The web runtime therefore updates the whitening statistics only on frames with **>= 4** valid embeddings and whitens sparse frames with the frozen statistics (simulated: same-person cosine holds the dense-regime ~0.42 at K=1-2 instead of collapsing); until a dense-enough frame has been seen, the `os` variant withholds embeddings and tracks on geometry alone, because raw OSNet cosines (~0.94 between different people) must never meet the whitened-space thresholds.

## Models

- ONNX
- LiteRT (TFLite)

  https://github.com/PINTO0309/soma/releases/tag/models

## Cited

- Teng Fu, Yuwen Chen, Zhuofan Chen, Mengyang Zhao, Bin Li, Xiangyang Xue. *CrowdTrack: A Benchmark for Difficult Multiple Pedestrian Tracking in Real Scenarios.* arXiv:2507.02479, 2025. [[paper](https://arxiv.org/abs/2507.02479)] [[dataset](https://github.com/loseevaya/CrowdTrack)]

  ```bibtex
  @article{fu2025crowdtrack,
    title   = {CrowdTrack: A Benchmark for Difficult Multiple Pedestrian Tracking in Real Scenarios},
    author  = {Fu, Teng and Chen, Yuwen and Chen, Zhuofan and Zhao, Mengyang and Li, Bin and Xue, Xiangyang},
    journal = {arXiv preprint arXiv:2507.02479},
    year    = {2025}
  }
  ```

- Yifu Zhang, Peize Sun, Yi Jiang, Dongdong Yu, Fucheng Weng, Zehuan Yuan, Ping Luo, Wenyu Liu, Xinggang Wang. *ByteTrack: Multi-Object Tracking by Associating Every Detection Box.* ECCV 2022. [[paper](https://arxiv.org/abs/2110.06864)] [[code](https://github.com/FoundationVision/ByteTrack)]

  ```bibtex
  @article{zhang2022bytetrack,
    title     = {ByteTrack: Multi-Object Tracking by Associating Every Detection Box},
    author    = {Zhang, Yifu and Sun, Peize and Jiang, Yi and Yu, Dongdong and Weng, Fucheng and Yuan, Zehuan and Luo, Ping and Liu, Wenyu and Wang, Xinggang},
    booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
    year      = {2022}
  }
  ```

- Vukašin Stanojević, Branimir Todorović. *BoostTrack: boosting the similarity measure and detection confidence for improved multiple object tracking.* Machine Vision and Applications, 2024. [[paper](https://doi.org/10.1007/s00138-024-01531-5)] [[code](https://github.com/vukasin-stanojevic/BoostTrack)]

  ```bibtex
  @article{stanojevic2024boostTrack,
    title   = {BoostTrack: boosting the similarity measure and detection confidence for improved multiple object tracking},
    author  = {Stanojevic, Vukasin D and Todorovic, Branimir T},
    journal = {Machine Vision and Applications},
    issn    = {0932-8092},
    year    = {2024},
    volume  = {35},
    number  = {3},
    doi     = {10.1007/s00138-024-01531-5}
  }
  ```

- Vukašin Stanojević, Branimir Todorović. *BoostTrack++: using tracklet information to detect more objects in multiple object tracking.* Filomat, 2025. [[paper](https://arxiv.org/abs/2408.13003)] [[code](https://github.com/vukasin-stanojevic/BoostTrack)]

  ```bibtex
  @article{stanojevic2024btpp,
    title   = {BoostTrack++: using tracklet information to detect more objects in multiple object tracking},
    author  = {Stanojevi{\'c}, Vuka{\v s}in and Todorovi{\'c}, Branimir},
    journal = {Filomat},
    volume  = {39},
    number  = {16},
    pages   = {5685--5702},
    year    = {2025},
    doi     = {https://doi.org/10.2298/FIL2516685S}
  }
  ```

- Bin Hu, Xinggang Wang, Wenyu Liu. *PersonViT: Large-scale Self-supervised Vision Transformer for Person Re-Identification.* arXiv:2408.05398, 2024. [[paper](https://arxiv.org/abs/2408.05398)] [[code](https://github.com/hustvl/PersonViT)]

  ```bibtex
  @article{hu2024personvit,
    title   = {PersonViT: Large-scale Self-supervised Vision Transformer for Person Re-Identification},
    author  = {Hu, Bin and Wang, Xinggang and Liu, Wenyu},
    journal = {arXiv preprint arXiv:2408.05398},
    year    = {2024}
  }
  ```

- Kaiyang Zhou, Yongxin Yang, Andrea Cavallaro, Tao Xiang. *Omni-Scale Feature Learning for Person Re-Identification.* ICCV 2019. [[paper](https://arxiv.org/abs/1905.00953)] [[code](https://github.com/KaiyangZhou/deep-person-reid)]

  ```bibtex
  @inproceedings{zhou2019osnet,
    title     = {Omni-Scale Feature Learning for Person Re-Identification},
    author    = {Zhou, Kaiyang and Yang, Yongxin and Cavallaro, Andrea and Xiang, Tao},
    booktitle = {ICCV},
    year      = {2019}
  }
  ```

- Kaiyang Zhou, Yongxin Yang, Andrea Cavallaro, Tao Xiang. *Learning Generalisable Omni-Scale Representations for Person Re-Identification.* TPAMI 2021. [[paper](https://arxiv.org/abs/1910.06827)] [[code](https://github.com/KaiyangZhou/deep-person-reid)]

  ```bibtex
  @article{zhou2021osnet,
    title   = {Learning Generalisable Omni-Scale Representations for Person Re-Identification},
    author  = {Zhou, Kaiyang and Yang, Yongxin and Cavallaro, Andrea and Xiang, Tao},
    journal = {TPAMI},
    year    = {2021}
  }
  ```

## Citation

If you find SOMA useful in your research or products, please cite:

```bibtex
@software{hyodo2026soma,
  title     = {SOMA: Structured Output Matching \& Association},
  author    = {Hyodo, Katsuya},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.21986816},
  url       = {https://github.com/PINTO0309/soma}
}
```

## License

[MIT License](./LICENSE)
