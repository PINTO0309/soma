"""Person assembly: structured 49-class detections -> per-person part groups.

Ports the joining rules of the reference demo
(docs/DEIMv2/demo/wholebody49/demo_deimv2_torch_wholebody49_ins.py) with one
deliberate change: keypoint->instance assignment samples the 80x80 mask grid
bilinearly at the keypoint center instead of rasterizing full-resolution
masks. This keeps assembly O(#keypoints x #bodies) with no mask resize.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import constants as C
from .detector import Detections

# vocab id used in BONE_EDGE_PAIRS -> (parent keypoint class, side or None)
_VOCAB_TO_PARENT_SIDE: dict[int, tuple[int, int | None]] = {}
for _pid in (21, 25, 35, 39, 42):
    _VOCAB_TO_PARENT_SIDE[_pid] = (_pid, None)
for _parent, (_l, _r) in C.SIDE_PARENT_TO_CHILDREN.items():
    _VOCAB_TO_PARENT_SIDE[_l] = (_parent, 0)
    _VOCAB_TO_PARENT_SIDE[_r] = (_parent, 1)


@dataclass
class Keypoint:
    class_id: int
    side: int              # 0 left, 1 right, -1 unknown
    x: float
    y: float
    score: float
    row: int
    body_prob: float = 0.0
    mixed: bool = False
    person: int = -1       # index into AssemblyResult.persons, -1 = orphan


@dataclass
class Person:
    anchor: str                        # "body" | "wheelchair" | "crutches" | "orphan"
    body_row: int = -1                 # detection row of the body box (mask row too)
    body_box: np.ndarray | None = None
    body_score: float = 0.0
    generation: int = -1               # 0 adult, 1 child
    gender: int = -1                   # 0 male, 1 female
    head_pose: int = -1                # 0..7
    head_row: int = -1
    head_box: np.ndarray | None = None
    # soft continuous head orientation (BiternionNet-style score-weighted
    # circular mean over the 8 orientation classes); conf = resultant length
    head_dir: np.ndarray | None = None   # (2,) unit (cos, sin)
    head_dir_conf: float = 0.0
    keypoints: list[Keypoint] = field(default_factory=list)
    bone_edges: list[tuple[int, int]] = field(default_factory=list)  # kp indices (global)
    mixed_count: int = 0


@dataclass
class AssemblyResult:
    persons: list[Person]
    keypoints: list[Keypoint]          # all kept keypoint instances (global indices)
    stats: dict


def _nms_class(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    # Greedy NMS with areas precomputed once (the per-round recompute was the
    # hotspot on dense YOLO frames — exp024).
    order = np.argsort(-scores)
    bx = boxes[order]
    areas = (bx[:, 2] - bx[:, 0]) * (bx[:, 3] - bx[:, 1])
    n = len(order)
    alive = np.ones(n, dtype=bool)
    keep = []
    for k in range(n):
        if not alive[k]:
            continue
        keep.append(int(order[k]))
        rest = np.nonzero(alive[k + 1:])[0] + k + 1
        if not len(rest):
            continue
        x1 = np.maximum(bx[k, 0], bx[rest, 0])
        y1 = np.maximum(bx[k, 1], bx[rest, 1])
        x2 = np.minimum(bx[k, 2], bx[rest, 2])
        y2 = np.minimum(bx[k, 3], bx[rest, 3])
        inter = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
        iou = inter / (areas[k] + areas[rest] - inter + 1e-9)
        alive[rest[iou >= iou_thr]] = False
    return np.array(keep, dtype=np.int64)


def _best_shared_box_match(
    center: np.ndarray, box: np.ndarray,
    cand_centers: np.ndarray, cand_boxes: np.ndarray, cand_scores: np.ndarray,
    max_dist: float,
) -> int:
    """Index of best candidate sharing coordinates with (center, box), else -1."""
    if not len(cand_centers):
        return -1
    d = np.hypot(cand_centers[:, 0] - center[0], cand_centers[:, 1] - center[1])
    ok = d <= max_dist
    if not ok.any():
        return -1
    x1 = np.maximum(box[0], cand_boxes[ok, 0])
    y1 = np.maximum(box[1], cand_boxes[ok, 1])
    x2 = np.minimum(box[2], cand_boxes[ok, 2])
    y2 = np.minimum(box[3], cand_boxes[ok, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a = (box[2] - box[0]) * (box[3] - box[1])
    b = (cand_boxes[ok, 2] - cand_boxes[ok, 0]) * (cand_boxes[ok, 3] - cand_boxes[ok, 1])
    iou = inter / (a + b - inter + 1e-9)
    cand_idx = np.where(ok)[0]
    good = iou > 0.0
    if not good.any():
        return -1
    return int(cand_idx[good][np.argmax(cand_scores[cand_idx[good]] + iou[good])])


def _batch_shared_box_match(
    q_centers: np.ndarray, q_boxes: np.ndarray,
    cand_centers: np.ndarray, cand_boxes: np.ndarray, cand_scores: np.ndarray,
    max_dist: float,
) -> np.ndarray:
    """Vectorized _best_shared_box_match over P queries -> (P,) indices / -1.

    Same gate (center distance), same pick (first argmax of score+IoU among
    IoU>0 candidates in candidate order) — exp024 hotspot batching.
    """
    out = np.full(len(q_centers), -1, dtype=np.int64)
    if not len(q_centers) or not len(cand_centers):
        return out
    d = np.hypot(q_centers[:, None, 0] - cand_centers[None, :, 0],
                 q_centers[:, None, 1] - cand_centers[None, :, 1])
    ok = d <= max_dist
    x1 = np.maximum(q_boxes[:, None, 0], cand_boxes[None, :, 0])
    y1 = np.maximum(q_boxes[:, None, 1], cand_boxes[None, :, 1])
    x2 = np.minimum(q_boxes[:, None, 2], cand_boxes[None, :, 2])
    y2 = np.minimum(q_boxes[:, None, 3], cand_boxes[None, :, 3])
    inter = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    qa = (q_boxes[:, 2] - q_boxes[:, 0]) * (q_boxes[:, 3] - q_boxes[:, 1])
    ca = (cand_boxes[:, 2] - cand_boxes[:, 0]) * (cand_boxes[:, 3] - cand_boxes[:, 1])
    iou = inter / (qa[:, None] + ca[None, :] - inter + 1e-9)
    val = np.where(ok & (iou > 0.0), cand_scores[None, :] + iou, -np.inf)
    best = val.argmax(axis=1)
    hit = np.take_along_axis(val, best[:, None], axis=1)[:, 0] > -np.inf
    out[hit] = best[hit]
    return out


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def assemble(
    det: Detections,
    score_threshold: float = C.DEFAULT_SCORE_THRESHOLD,
    attr_score_threshold: float = C.DEFAULT_ATTR_SCORE_THRESHOLD,
    nms_iou: float = C.DEFAULT_NMS_IOU,
    mask_assign_prob: float = 0.5,
    mixed_runner_up_prob: float = 0.35,
    body_score_floor: float | None = None,
) -> AssemblyResult:
    """body_score_floor: when set (< score_threshold), body-anchored persons
    are created down to this score (BYTE-style low-score pool) while parts,
    bones and attributes keep the regular threshold — the low-score band is
    saturated with junk part detections (exp006), bodies are not (exp007)."""
    labels, boxes, scores = det.labels, det.boxes, det.scores
    centers = det.centers()

    def kept_rows(class_ids, thr) -> np.ndarray:
        m = np.isin(labels, np.asarray(list(class_ids))) & (scores >= thr)
        return np.where(m)[0]

    def nms_rows(class_id: int, thr: float) -> np.ndarray:
        rows = kept_rows((class_id,), thr)
        if not len(rows):
            return rows
        return rows[_nms_class(boxes[rows], scores[rows], nms_iou)]

    # ---- anchors (body variants), heads --------------------------------
    body_thr = body_score_floor if body_score_floor is not None else score_threshold
    body_rows = np.concatenate([nms_rows(c, body_thr) for c in C.BODY_ANCHOR_CLASS_IDS])
    if len(body_rows) > 1:  # cross-class NMS between body variants
        body_rows = body_rows[_nms_class(boxes[body_rows], scores[body_rows], nms_iou)]
    head_rows = nms_rows(C.HEAD, score_threshold)

    # ---- attribute lookup tables ---------------------------------------
    def attr_table(attr_ids: dict[int, int]):
        rows = kept_rows(attr_ids.keys(), attr_score_threshold)
        vals = np.array([attr_ids[int(labels[r])] for r in rows], dtype=np.int64)
        return rows, vals

    gen_rows, gen_vals = attr_table(C.GENERATION_ATTRS)
    gender_rows, gender_vals = attr_table(C.GENDER_ATTRS)
    pose_rows, pose_vals = attr_table(C.HEAD_POSE_ATTRS)

    persons: list[Person] = []
    for r in body_rows:
        anchor = {C.BODY: "body", C.BODY_WHEELCHAIR: "wheelchair", C.BODY_CRUTCHES: "crutches"}[int(labels[r])]
        persons.append(Person(anchor=anchor, body_row=int(r), body_box=boxes[r].copy(),
                              body_score=float(scores[r])))
    body_pi = [i for i, p in enumerate(persons) if p.anchor == "body"]
    if body_pi:
        q_rows = np.array([persons[i].body_row for i in body_pi], dtype=np.int64)
        for rows, vals, field in ((gen_rows, gen_vals, "generation"),
                                  (gender_rows, gender_vals, "gender")):
            match = _batch_shared_box_match(centers[q_rows], boxes[q_rows], centers[rows],
                                            boxes[rows], scores[rows], C.ATTR_MERGE_CENTER_DIST)
            for pi, j in zip(body_pi, match):
                if j >= 0:
                    setattr(persons[pi], field, int(vals[j]))

    # ---- keypoints (parent classes, with L/R side from child rows) --------
    # Only parents the keypoint loop below actually iterates: hand(32) and
    # foot(45) are objects, not keypoint slots — their side tables used to be
    # computed here and then never read.
    side_tables = {}
    for parent, (lc, rc) in C.SIDE_PARENT_TO_CHILDREN.items():
        if parent not in C.KEYPOINT_PARENT_CLASS_IDS:
            continue
        rows_l = kept_rows((lc,), attr_score_threshold)
        rows_r = kept_rows((rc,), attr_score_threshold)
        side_tables[parent] = (rows_l, rows_r)

    keypoints: list[Keypoint] = []
    for class_id in C.KEYPOINT_PARENT_CLASS_IDS:
        rows_p = nms_rows(class_id, score_threshold)
        if not len(rows_p):
            continue
        sides = np.full(len(rows_p), -1, dtype=np.int64)
        if class_id in side_tables:
            rows_l, rows_r = side_tables[class_id]
            jl = _batch_shared_box_match(centers[rows_p], boxes[rows_p], centers[rows_l],
                                         boxes[rows_l], scores[rows_l], C.ATTR_MERGE_CENTER_DIST)
            jr = _batch_shared_box_match(centers[rows_p], boxes[rows_p], centers[rows_r],
                                         boxes[rows_r], scores[rows_r], C.ATTR_MERGE_CENTER_DIST)
            both = (jl >= 0) & (jr >= 0)
            sides[both] = np.where(scores[rows_l[jl[both]]] >= scores[rows_r[jr[both]]], 0, 1)
            sides[(jl >= 0) & ~both] = 0
            sides[(jr >= 0) & ~both] = 1
        for k, r in enumerate(rows_p):
            keypoints.append(Keypoint(class_id=class_id, side=int(sides[k]),
                                      x=float(centers[r][0]), y=float(centers[r][1]),
                                      score=float(scores[r]), row=int(r)))

    # ---- mask-based instance assignment (vectorized over all query points) --
    # Persons with a usable instance mask (classid 0 only; wheelchair/crutches
    # anchors fall back to box containment).
    mask_person_idx = np.array(
        [] if det.mask_probs is None else
        [i for i, p in enumerate(persons) if labels[p.body_row] == C.BODY],
        dtype=np.int64)
    query_pts = np.array(
        [[kp.x, kp.y] for kp in keypoints] + [centers[r].tolist() for r in head_rows],
        dtype=np.float32).reshape(-1, 2)
    n_q = len(query_pts)
    assign_pi = np.full(n_q, -1, dtype=np.int64)
    assign_prob = np.zeros(n_q, dtype=np.float32)
    assign_mixed = np.zeros(n_q, dtype=bool)
    if n_q and len(mask_person_idx):
        probs = det.mask_probs_matrix(
            np.array([persons[i].body_row for i in mask_person_idx]), query_pts)  # (B, Q)
        order = np.argsort(-probs, axis=0)
        best = probs[order[0], np.arange(n_q)]
        second = probs[order[1], np.arange(n_q)] if len(mask_person_idx) > 1 else np.zeros(n_q)
        hit = best >= mask_assign_prob
        assign_pi[hit] = mask_person_idx[order[0][hit]]
        assign_prob = best.astype(np.float32)
        assign_mixed = hit & (second >= mixed_runner_up_prob)
    if n_q and len(persons):
        # containment fallback for points no mask claimed
        pboxes = np.array([p.body_box for p in persons], dtype=np.float32)  # (P,4)
        inside = ((query_pts[:, None, 0] >= pboxes[None, :, 0])
                  & (query_pts[:, None, 0] <= pboxes[None, :, 2])
                  & (query_pts[:, None, 1] >= pboxes[None, :, 1])
                  & (query_pts[:, None, 1] <= pboxes[None, :, 3]))       # (Q,P)
        n_inside = inside.sum(axis=1)
        fb = (assign_pi < 0) & (n_inside == 1)
        assign_pi[fb] = inside[fb].argmax(axis=1)

    for qi, kp in enumerate(keypoints):
        kp.person = int(assign_pi[qi])
        kp.body_prob = float(assign_prob[qi])
        kp.mixed = bool(assign_mixed[qi])

    # ---- heads -----------------------------------------------------------
    _bin_angles = np.deg2rad(np.arange(8) * 45.0)   # classes 8..15: 45-degree ring
    pose_match = _batch_shared_box_match(centers[head_rows], boxes[head_rows],
                                         centers[pose_rows], boxes[pose_rows],
                                         scores[pose_rows], C.ATTR_MERGE_CENTER_DIST)
    for hi, r in enumerate(head_rows):
        pi = int(assign_pi[len(keypoints) + hi])
        pose = int(pose_vals[pose_match[hi]]) if pose_match[hi] >= 0 else -1
        if pi >= 0 and persons[pi].head_row < 0:
            persons[pi].head_row = int(r)
            persons[pi].head_box = boxes[r].copy()
            persons[pi].head_pose = pose
            if len(pose_rows):
                hb = boxes[r]
                hdiag = max(float(np.hypot(hb[2] - hb[0], hb[3] - hb[1])), 1.0)
                d = np.hypot(centers[pose_rows, 0] - centers[r][0],
                             centers[pose_rows, 1] - centers[r][1])
                near = pose_rows[d <= 0.5 * hdiag]
                if len(near):
                    w = scores[near]
                    th = _bin_angles[(labels[near] - 8)]
                    vx = float((w * np.cos(th)).sum())
                    vy = float((w * np.sin(th)).sum())
                    n = float(np.hypot(vx, vy))
                    if n > 1e-6:
                        persons[pi].head_dir = np.array([vx / n, vy / n], dtype=np.float32)
                        persons[pi].head_dir_conf = float(n / max(w.sum(), 1e-9))

    # ---- bone edges --------------------------------------------------------
    # A bone box's diagonal joins exactly two joint centers (unique diagonal by
    # dataset design). For each vocab pair and each of the 4 corner
    # orientations the two endpoints decouple, so everything vectorizes over
    # ALL bones at once: 15 vocab x 4 orientations distance-matrix sweeps.
    bone_rows = nms_rows(C.BONE, score_threshold)
    bone_edges: list[tuple[int, int]] = []
    if len(bone_rows) and keypoints:
        kp_pos_by_class: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for cid in C.KEYPOINT_PARENT_CLASS_IDS:
            gis = np.array([gi for gi, kp in enumerate(keypoints) if kp.class_id == cid], dtype=np.int64)
            pos = np.array([[keypoints[gi].x, keypoints[gi].y] for gi in gis],
                           dtype=np.float32).reshape(-1, 2)
            sides = np.array([keypoints[gi].side for gi in gis], dtype=np.int64)
            kp_pos_by_class[cid] = (gis, pos, sides)

        bb = boxes[bone_rows]                                    # (M,4)
        n_bones = len(bone_rows)
        tol = np.maximum(4.0, 0.25 * np.hypot(bb[:, 2] - bb[:, 0], bb[:, 3] - bb[:, 1]))  # (M,)
        c00 = bb[:, [0, 1]]
        c11 = bb[:, [2, 3]]
        c01 = bb[:, [0, 3]]
        c10 = bb[:, [2, 1]]
        orients = ((c00, c11), (c11, c00), (c01, c10), (c10, c01))

        def nearest_all(cid: int, want_side: int | None, corners: np.ndarray):
            """Per-bone (dist, kp_gi) of nearest side-compatible kp of class cid."""
            gis, pos, sides = kp_pos_by_class[cid]
            if not len(gis):
                return None
            ok = np.ones(len(gis), bool) if want_side is None else (sides < 0) | (sides == want_side)
            if not ok.any():
                return None
            p = pos[ok]
            d = np.hypot(corners[:, None, 0] - p[None, :, 0],
                         corners[:, None, 1] - p[None, :, 1])   # (M, K)
            j = np.argmin(d, axis=1)
            return d[np.arange(n_bones), j], gis[np.where(ok)[0][j]]

        best_total = np.full(n_bones, np.inf, dtype=np.float32)
        best_a = np.full(n_bones, -1, dtype=np.int64)
        best_b = np.full(n_bones, -1, dtype=np.int64)
        for va, vb in C.BONE_EDGE_PAIRS:
            (ca, sa), (cb, sb) = _VOCAB_TO_PARENT_SIDE[va], _VOCAB_TO_PARENT_SIDE[vb]
            for p_a, p_b in orients:
                ra = nearest_all(ca, sa, p_a)
                rb = nearest_all(cb, sb, p_b)
                if ra is None or rb is None:
                    continue
                total = ra[0] + rb[0]
                better = (ra[0] <= tol) & (rb[0] <= tol) & (total < best_total)
                best_total[better] = total[better]
                best_a[better] = ra[1][better]
                best_b[better] = rb[1][better]

        used_bone_pairs: set[tuple[int, int]] = set()
        for bi in np.argsort(best_total):
            if best_a[bi] < 0 or not np.isfinite(best_total[bi]):
                continue
            key = (min(best_a[bi], best_b[bi]), max(best_a[bi], best_b[bi]))
            if key not in used_bone_pairs:
                used_bone_pairs.add(key)
                bone_edges.append((int(best_a[bi]), int(best_b[bi])))

    # ---- propagate person assignment along bone edges; orphan grouping -----
    uf = _UnionFind(len(keypoints))
    for a, b in bone_edges:
        uf.union(a, b)
    groups: dict[int, list[int]] = {}
    for gi in range(len(keypoints)):
        groups.setdefault(uf.find(gi), []).append(gi)

    n_propagated = 0
    for members in groups.values():
        owners = {keypoints[m].person for m in members if keypoints[m].person >= 0}
        if len(owners) == 1:
            owner = owners.pop()
            for m in members:
                if keypoints[m].person < 0:
                    keypoints[m].person = owner
                    n_propagated += 1

    # orphan groups (connected, but no member belongs to any body): PARTIAL persons
    n_orphan_groups = 0
    for members in groups.values():
        if len(members) < 2:
            continue
        if all(keypoints[m].person < 0 for m in members):
            p = Person(anchor="orphan")
            persons.append(p)
            pi = len(persons) - 1
            for m in members:
                keypoints[m].person = pi
            n_orphan_groups += 1

    for gi, kp in enumerate(keypoints):
        if kp.person >= 0:
            persons[kp.person].keypoints.append(kp)
            if kp.mixed:
                persons[kp.person].mixed_count += 1
    for a, b in bone_edges:
        pa, pb = keypoints[a].person, keypoints[b].person
        if pa >= 0 and pa == pb:
            persons[pa].bone_edges.append((a, b))

    n_unassigned = sum(1 for kp in keypoints if kp.person < 0)
    stats = {
        "n_persons": len(persons),
        "n_body_anchored": len(body_rows),
        "n_orphan_groups": n_orphan_groups,
        "n_heads": len(head_rows),
        "n_keypoints": len(keypoints),
        "n_kp_unassigned": n_unassigned,
        "n_kp_propagated_by_bone": n_propagated,
        "n_kp_mixed": sum(1 for kp in keypoints if kp.mixed),
        "n_bone_edges": len(bone_edges),
        "n_bones_detected": len(bone_rows),
    }
    return AssemblyResult(persons=persons, keypoints=keypoints, stats=stats)
