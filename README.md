# SOMA

**S**tructured **O**utput **M**atching &amp; **A**ssociation: online multi-person tracking built on the structured output of a multi-task wholebody detector, one whole-frame pass per frame (plus per-crop ReID embeddings in SOMA-R) — no tiling, no offline post-processing.

A person is not "one box": it is an **anatomical token** — body box, body parts (head/shoulders/elbows/hands/knees/feet), head-orientation ring and person attributes — assembled from one [YOLOv9-E Wholebody28-Refine](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/468_YOLOv9-Wholebody28-Refine) forward pass. Association fuses cheap identity channels (box IoU, part OKS, orientation continuity, attribute vetoes) with anatomical amodal box synthesis and an online scene-geometry size prior. **SOMA-R** adds an external ReID embedding as the dominant stage-1 channel plus a post-death identity memory, an embedding-only revival stage, appearance-locked track extension and short KF ghost coasting — the long-gap same-id recovery stack.

The detector slot is deliberately interchangeable, not wedded to that one checkpoint: the research lineage walked DEIMv2-Wholebody49 → Wholebody34 → wb28-Refine, and any wholebody detector that emits the same anatomical output family can take its place — [DEIM-Wholebody28](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/465_DEIM-Wholebody28), [YOLO-Wholebody34](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/471_YOLO-Wholebody34), [DEIMv2-Wholebody34](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/472_DEIMv2-Wholebody34), [DEIMv2-Wholebody40](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/485_DEIMv2-Wholebody40) and [DEIMv2-Wholebody49](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/488_DEIMv2-Wholebody49) are all drop-in candidates.

SOMA's center of gravity is **recovery from long occlusion loss**: when a person disappears behind an occluder for seconds and re-emerges, they should come back under the *same id*. Frame-to-frame matching is a largely solved problem that every modern tracker handles well; surviving a multi-second gap is not — and it is exactly what the SOMA-R stack above (identity memory, embedding-only revival, appearance-locked extension, ghost coasting) is built for, and what the primary KPI below (`same-id ~1s/~3s/~5s`) scores directly.

## Why SOMA?

The obvious alternative is the classic recipe — ByteTrack-style association on a YOLOX-X detector, plus BoostTrack++ — which looks overwhelmingly strong on the MOT17/MOT20 leaderboards. The research repo behind SOMA measured what those numbers are actually made of, ingredient by ingredient, and found that much of the margin is benchmark artifact, not deployment quality:

- **Train leakage** — the published `bytetrack_x_mot17` detector is trained on the *full* MOT17 train set, so any MOT17-train-derived evaluation partly re-detects its own training data (MOTA ~88-90).
- **Detector-benchmark co-adaptation** — that ByteTrack YOLOX-X is trained on a dataset recipe assembled *for this benchmark family*, and it regresses MOT's **amodal full-body boxes**: a person standing behind an occluder still gets a box stretched over their invisible legs. That convention serves exactly one consumer — MOT-style association — and is the wrong primitive for every other use of a person detector: crop such a box and a ReID embedder, pose estimator or privacy mask mostly sees the occluder; gate a region-entry counter with it and people walk through walls. A generic visible-extent detector is systematically IoU-penalized by MOT GT for refusing to hallucinate — a penalty that says nothing about real-world quality. SOMA keeps the detector generic and reusable beyond tracking (visible-extent boxes + parts) and synthesizes the amodal box *inside the tracker* instead.

  <img width="500" alt="amodal_bbox" src="https://github.com/user-attachments/assets/bef1de7e-3c12-43be-9531-7bf04c538431" />

- **The literature is detector-bound** — for well over five years the tracking-by-detection line of papers has presupposed the output of a specialized detection model (benchmark-adapted, amodal, high-resolution), so a "better tracker" result holds only inside that detector's output distribution. The association step moves single HOTA points while the detector choice moves tens — the research repo measured +14 HOTA / +19 MOTA on MOT20 from swapping the detector alone under the *same* SOMA-R tracker. To first order, tracker rankings are detector rankings.
- **Tuning and post-processing** — official thresholds were tuned on the very sequences being scored (official repos even carry per-sequence overrides), and published numbers include offline interpolation — unusable in a live system.
- **The compute budget is off-screen** — most published configurations assume high-resolution inputs (well above 720p; ByteTrack's official test size is 1440x800), where detector inference — not the tracker's few milliseconds of association — is nearly the entire end-to-end cost. A ranking bought at that input size says nothing about what holds up on a real deployment budget.
- **The metric barely sees the hardest failure** — losing a person for *seconds* and re-acquiring them with the same id is the failure that hurts in production, yet MOT17 val contains exactly **7** ground-truth occlusion episodes in the 4-6 s range.

SOMA is the counter-design, built for the question those leaderboards don't answer — *what tracks well behind a generic detector in a live system?*

1. **Structure over scale.** A multi-task wholebody detector that was never trained on MOT data already emits rich per-person structure in one whole-frame pass; SOMA turns it into an anatomical token and fuses many weak, cheap identity channels instead of relying on one strong, expensive one.
2. **Online only, end to end.** No offline interpolation ever, no per-sequence tuning, TensorRT throughout — every number in the table below is producible by the live pipeline.
3. **Low resolution by design.** SOMA assumes VGA-class inputs and below — the shipped presets run the detector at 640x640 stretch even on 720p footage — so the dominant cost term stays small enough for real-time edge deployment, and the detection-recall hit is taken openly instead of being hidden behind a high-resolution detector pass.
4. **Score what hurts in production.** The primary KPI is long-gap same-id recovery (`same-id ~1s/~3s/~5s`) plus the coverage-fair switch rate `sw/TP` — hence the [CrowdTrack](https://github.com/loseevaya/CrowdTrack) benchmark, whose ~5s occlusion-episode pool is ~19x MOT17's.

Two honest caveats carry over from the research repo: SOMA's edge is *co-designed with the anatomical detector* — on body-only detections its extra channels have nothing to read and the ID lead disappears; and the classic trackers are not dismissed — the BoostTrack++ ingredients that survived ablation (always-on appearance, Mahalanobis channel, KF-posterior output) are transplanted into SOMA, while the ones that failed are documented as rejected, with numbers.

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

Standalone ReID accuracy of the two fine-tuned embedders, from the [PersonViT](https://github.com/PINTO0309/PersonViT/tree/uv) repo (identity-disjoint unified test split spanning five public ReID benchmarks):

| embedder (`--variant`) | fine-tune | backbone | params | GFLOPs<br>@256x128 | emb | mAP | Rank-1 | Rank-5 | Rank-10 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `personvit_vits16_ain_unified_aug_n.onnx` (`pv`) | [S-ain-aug](https://github.com/PINTO0309/PersonViT/tree/uv#s-ain-aug----vit-s16--token-in---220m) | ViT-S/16 + token-IN | 22.0M | 2.94 | 384 | 93.1 | 97.2 | 98.2 | 98.4 |
| `osnet_ain_x1_0_p_unified_aug_n.onnx` (`os`) | [P-ain-aug](https://github.com/PINTO0309/PersonViT/tree/uv#p-ain-aug---osnet-ain-x10---22m) | OSNet-AIN x1.0 | 2.2M | 0.98 | 512 | 89.1 | 95.4 | 97.7 | 98.1 |

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

## License

[MIT License](./LICENSE)
