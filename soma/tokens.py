"""Anatomical tokens: fixed-layout numeric encoding of an assembled person.

The token is the interchange format between the perception layer (L0) and
every association layer above it (SOMA-Zero cost channels, SOMA-Net input).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import constants as C
from .assembly import Person
from .detector import Detections

# Slot layout: sided parents get L/R slots, unsided parents one slot, +head.
SLOT_LAYOUT: list[tuple[int, int]] = []   # (class_id, side) ; side -1 = unsided
for _cid in C.KEYPOINT_PARENT_CLASS_IDS:
    if _cid in C.SIDE_PARENT_TO_CHILDREN:
        SLOT_LAYOUT.append((_cid, 0))
        SLOT_LAYOUT.append((_cid, 1))
    else:
        SLOT_LAYOUT.append((_cid, -1))
HEAD_SLOT = len(SLOT_LAYOUT)
N_SLOTS = len(SLOT_LAYOUT) + 1           # keypoint slots + head

_SLOT_INDEX = {key: i for i, key in enumerate(SLOT_LAYOUT)}



@dataclass
class AnatomicalToken:
    anchor: str                     # body | wheelchair | crutches | orphan
    body_box: np.ndarray | None     # (4,) frame coords, None for orphan groups
    body_score: float
    box_proxy: np.ndarray           # (4,) always present: body box or part-group extent
    presence: np.ndarray            # (N_SLOTS,) bool
    points: np.ndarray              # (N_SLOTS, 2) normalized to box_proxy, NaN if absent
    generation: int
    gender: int
    embedding: np.ndarray | None = None   # SOMA-R: external ReID embedding, L2-normalized
    head_box: np.ndarray | None = None    # (4,) frame coords (amodal synthesis input)
    head_dir: np.ndarray | None = None    # (2,) unit (cos, sin) soft head orientation
    head_dir_conf: float = 0.0            # circular resultant length in [0,1]
    crowding: float = 0.0                 # max IoU with another body this frame


def _group_extent(person: Person) -> np.ndarray | None:
    xs = [kp.x for kp in person.keypoints]
    ys = [kp.y for kp in person.keypoints]
    if person.head_box is not None:
        xs += [person.head_box[0], person.head_box[2]]
        ys += [person.head_box[1], person.head_box[3]]
    if not xs:
        return None
    return np.array([min(xs), min(ys), max(xs), max(ys)], dtype=np.float32)


def build_token(person: Person, det: Detections) -> AnatomicalToken | None:
    box_proxy = person.body_box if person.body_box is not None else _group_extent(person)
    if box_proxy is None:
        return None
    w = max(float(box_proxy[2] - box_proxy[0]), 1.0)
    h = max(float(box_proxy[3] - box_proxy[1]), 1.0)

    presence = np.zeros(N_SLOTS, dtype=bool)
    points = np.full((N_SLOTS, 2), np.nan, dtype=np.float32)

    # best-scoring keypoint per slot; unknown-side keypoints take a free side slot
    for kp in sorted(person.keypoints, key=lambda k: -k.score):
        cid = kp.class_id
        if cid in C.SIDE_PARENT_TO_CHILDREN:
            cand = [(cid, kp.side)] if kp.side in (0, 1) else [(cid, 0), (cid, 1)]
        else:
            cand = [(cid, -1)]
        for key in cand:
            si = _SLOT_INDEX[key]
            if not presence[si]:
                presence[si] = True
                points[si] = ((kp.x - box_proxy[0]) / w, (kp.y - box_proxy[1]) / h)
                break
    if person.head_box is not None:
        hc = person.head_box
        presence[HEAD_SLOT] = True
        points[HEAD_SLOT] = (((hc[0] + hc[2]) / 2 - box_proxy[0]) / w,
                             ((hc[1] + hc[3]) / 2 - box_proxy[1]) / h)

    return AnatomicalToken(
        anchor=person.anchor,
        body_box=None if person.body_box is None else person.body_box.astype(np.float32),
        body_score=person.body_score,
        box_proxy=box_proxy.astype(np.float32),
        presence=presence,
        points=points,
        generation=person.generation,
        gender=person.gender,
        embedding=None,          # SOMA-R: injected by the perception layer
        head_box=None if person.head_box is None else person.head_box.astype(np.float32),
        head_dir=person.head_dir,
        head_dir_conf=person.head_dir_conf,
    )




# Amodal full-body synthesis (v3): MOT GT is amodal — occluded people keep
# their full-extent box. Our detector boxes hug the VISIBLE extent (exp007:
# det_h/gt_h p25 = 0.58 for vis<0.6), so we reconstruct the standing extent
# from anatomy: full height ~= alpha * head height. Applied at tracking time
# (sweepable without cache rebuilds); only ever EXTENDS the box downward.
_ANKLE_SLOTS = tuple(i for i, (cid, _s) in enumerate(SLOT_LAYOUT) if cid == 42)


def apply_amodal(tok: AnatomicalToken, alpha: float, gamma: float = 0.90,
                 overshoot: float = 1.0) -> None:
    if tok.body_box is None or tok.head_box is None:
        return
    head_h = float(tok.head_box[3] - tok.head_box[1])
    if head_h <= 2.0:
        return
    if any(tok.presence[s] for s in _ANKLE_SLOTS):
        return                       # visible ankles: the bottom edge is real
    y1 = min(float(tok.body_box[1]), float(tok.head_box[1]))
    h_est = alpha * head_h
    if (float(tok.body_box[3]) - y1) >= gamma * h_est:
        return                       # tall enough: no truncation evidence
    new_y2 = y1 + overshoot * h_est
    tok.body_box = tok.body_box.copy()
    tok.body_box[1] = y1
    tok.body_box[3] = max(float(tok.body_box[3]), new_y2)
    tok.box_proxy = tok.body_box

