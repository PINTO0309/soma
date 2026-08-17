"""MOTChallenge sequence utilities (gt loading, seqinfo)."""
from __future__ import annotations

import configparser
import os
from dataclasses import dataclass

import numpy as np

DISTRACTOR_CLASSES = (2, 7, 8, 12)   # person_on_vehicle, static_person, distractor, reflection


@dataclass
class SeqInfo:
    name: str
    path: str
    fps: int
    length: int
    width: int
    height: int
    ext: str = ".jpg"


def find_sequences(split_dir: str, variant: str | None = "FRCNN") -> list[SeqInfo]:
    """MOT17 dirs come in 3 detector variants sharing frames; keep one."""
    out = []
    for name in sorted(os.listdir(split_dir)):
        path = os.path.join(split_dir, name)
        if not os.path.isdir(path) or not os.path.exists(os.path.join(path, "img1")):
            continue
        if "MOT17" in name and variant and not name.endswith(variant):
            continue
        out.append(read_seqinfo(path))
    return out


def read_seqinfo(path: str) -> SeqInfo:
    ini = configparser.ConfigParser()
    ini.read(os.path.join(path, "seqinfo.ini"))
    s = ini["Sequence"]
    return SeqInfo(name=s.get("name", os.path.basename(path)), path=path,
                   fps=s.getint("frameRate", 30), length=s.getint("seqLength", 0),
                   width=s.getint("imWidth", 1920), height=s.getint("imHeight", 1080),
                   ext=s.get("imExt", ".jpg"))


def frame_path(seq: SeqInfo, frame_id: int) -> str:
    return os.path.join(seq.path, "img1", f"{frame_id:06d}{seq.ext}")


def load_gt(seq_path: str) -> tuple[dict[int, dict], int]:
    """-> ({frame: {ids, boxes, distractors}}, n_considered_boxes)."""
    gt = np.loadtxt(os.path.join(seq_path, "gt", "gt.txt"), delimiter=",")
    considered = gt[(gt[:, 6] == 1) & (gt[:, 7] == 1)]
    distractor = gt[np.isin(gt[:, 7], DISTRACTOR_CLASSES)]
    frames: dict[int, dict] = {}
    for fid in np.unique(gt[:, 0]).astype(int):
        c = considered[considered[:, 0] == fid]
        d = distractor[distractor[:, 0] == fid]
        frames[fid] = {
            "ids": c[:, 1].astype(np.int64),
            "boxes": np.stack([c[:, 2], c[:, 3], c[:, 2] + c[:, 4], c[:, 3] + c[:, 5]], 1)
            if len(c) else np.zeros((0, 4)),
            "distractors": np.stack([d[:, 2], d[:, 3], d[:, 2] + d[:, 4], d[:, 3] + d[:, 5]], 1)
            if len(d) else np.zeros((0, 4)),
        }
    return frames, len(considered)


def load_gt_envelope(seq_path: str, cap: int = 20,
                     vis_min: float = 0.0, min_height: float = 0.0) -> tuple[dict[int, dict], int]:
    """Design-envelope GT: per frame, keep the `cap` most perceivable
    pedestrians (ranked by visibility x box area) as considered; everyone
    else — and anyone below the optional vis/height floors — is moved to the
    ignore set (treated exactly like distractor classes: predictions matching
    them are not FP, missing them is not FN).

    Rationale: DEIMv2-Wholebody49's 1240 queries are designed for ~20 people
    (~62 queries/person); beyond that the model is out of spec (plan §2.6).
    """
    gt = np.loadtxt(os.path.join(seq_path, "gt", "gt.txt"), delimiter=",")
    considered = gt[(gt[:, 6] == 1) & (gt[:, 7] == 1)]
    distractor = gt[np.isin(gt[:, 7], DISTRACTOR_CLASSES)]
    frames: dict[int, dict] = {}
    n_considered = 0
    for fid in np.unique(gt[:, 0]).astype(int):
        c = considered[considered[:, 0] == fid]
        d = distractor[distractor[:, 0] == fid]
        ignore = [np.stack([d[:, 2], d[:, 3], d[:, 2] + d[:, 4], d[:, 3] + d[:, 5]], 1)
                  if len(d) else np.zeros((0, 4))]
        if len(c):
            ok = (c[:, 8] >= vis_min) & (c[:, 5] >= min_height)
            floor_out = c[~ok]
            c = c[ok]
            if len(c) > cap:
                salience = c[:, 8] * c[:, 4] * c[:, 5]      # visibility x area
                order = np.argsort(-salience)
                cap_out = c[order[cap:]]
                c = c[order[:cap]]
            else:
                cap_out = np.zeros((0, gt.shape[1]))
            for out in (floor_out, cap_out):
                if len(out):
                    ignore.append(np.stack([out[:, 2], out[:, 3],
                                            out[:, 2] + out[:, 4], out[:, 3] + out[:, 5]], 1))
        n_considered += len(c)
        frames[fid] = {
            "ids": c[:, 1].astype(np.int64) if len(c) else np.zeros(0, np.int64),
            "boxes": np.stack([c[:, 2], c[:, 3], c[:, 2] + c[:, 4], c[:, 3] + c[:, 5]], 1)
            if len(c) else np.zeros((0, 4)),
            "distractors": np.concatenate(ignore, axis=0),
        }
    return frames, n_considered


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))[:, None]
    bb = ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))[None, :]
    return inter / (aa + bb - inter + 1e-9)
