"""CLEAR-MOT (MOTA/FP/FN/IDSW/MOTP) and ID metrics (IDF1/IDP/IDR).

Follows the py-motmetrics MOTChallenge protocol: persistent frame matching at
IoU>=0.5, distractor-class suppression of unmatched predictions, ID metrics
via a global bipartite matching over per-frame IoU>=0.5 co-occurrences.
Requires scipy (dev dependency) for linear_sum_assignment.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Literal, overload

import numpy as np
from scipy.optimize import linear_sum_assignment

from .mot import iou_matrix


@overload
def evaluate_sequence(
    gt_frames: dict[int, dict], n_gt_boxes: int,
    results: list[tuple], iou_thr: float = ...,
    return_matches: Literal[False] = ...,
) -> dict[str, float]: ...
@overload
def evaluate_sequence(
    gt_frames: dict[int, dict], n_gt_boxes: int,
    results: list[tuple], iou_thr: float = ...,
    *, return_matches: Literal[True],
) -> tuple[dict[str, float], dict[int, dict[int, int]]]: ...


def evaluate_sequence(
    gt_frames: dict[int, dict], n_gt_boxes: int,
    results: list[tuple], iou_thr: float = 0.5,
    return_matches: bool = False,
) -> "dict[str, float] | tuple[dict[str, float], dict[int, dict[int, int]]]":
    """return_matches=True -> (metrics, {fid: {gt_id: pred_id}}) for the
    gap-recovery evaluation (same persistent matching as the CLEAR pass)."""
    preds: dict[int, dict] = defaultdict(lambda: {"ids": [], "boxes": []})
    for r in results:
        preds[int(r[0])]["ids"].append(int(r[1]))
        preds[int(r[0])]["boxes"].append([r[2], r[3], r[2] + r[4], r[3] + r[5]])

    fp = fn = idsw = tp = 0
    motp_sum = 0.0
    last_pred_of_gt: dict[int, int] = {}
    prev_match: dict[int, int] = {}
    cooc: dict[tuple[int, int], int] = defaultdict(int)
    n_pred_boxes = 0
    match_history: dict[int, dict[int, int]] = {}

    for fid in sorted(gt_frames):
        g = gt_frames[fid]
        g_ids, g_boxes = g["ids"], g["boxes"]
        p = preds.get(fid, {"ids": [], "boxes": []})
        p_ids = np.array(p["ids"], dtype=np.int64)
        p_boxes = np.array(p["boxes"], dtype=np.float64).reshape(-1, 4)

        iou = iou_matrix(g_boxes, p_boxes)
        matches: dict[int, int] = {}       # gt index -> pred index
        used_p = np.zeros(len(p_ids), dtype=bool)

        # persistence: keep last frame's (gt id, pred id) pairs when still valid
        pid_to_idx = {int(pid): j for j, pid in enumerate(p_ids)}
        for gi, gid in enumerate(g_ids):
            pj = prev_match.get(int(gid))
            if pj is not None and pj in pid_to_idx:
                j = pid_to_idx[pj]
                if not used_p[j] and iou[gi, j] >= iou_thr:
                    matches[gi] = j
                    used_p[j] = True

        # Hungarian on the rest
        rem_g = [gi for gi in range(len(g_ids)) if gi not in matches]
        rem_p = [j for j in range(len(p_ids)) if not used_p[j]]
        if rem_g and rem_p:
            sub = 1.0 - iou[np.ix_(rem_g, rem_p)]
            ri, ci = linear_sum_assignment(sub)
            for a, b in zip(ri, ci):
                if sub[a, b] <= 1.0 - iou_thr:
                    matches[rem_g[a]] = rem_p[b]
                    used_p[rem_p[b]] = True

        # distractor suppression of unmatched predictions
        keep_p = np.ones(len(p_ids), dtype=bool)
        if len(g["distractors"]) and len(p_boxes):
            diou = iou_matrix(p_boxes, g["distractors"])
            for j in range(len(p_ids)):
                if not used_p[j] and diou[j].max(initial=0.0) >= iou_thr:
                    keep_p[j] = False

        tp += len(matches)
        fp += int((~used_p & keep_p).sum())
        fn += len(g_ids) - len(matches)
        prev_match = {}
        for gi, j in matches.items():
            gid, pid = int(g_ids[gi]), int(p_ids[j])
            motp_sum += iou[gi, j]
            if gid in last_pred_of_gt and last_pred_of_gt[gid] != pid:
                idsw += 1
            last_pred_of_gt[gid] = pid
            prev_match[gid] = pid
        match_history[fid] = prev_match

        # ID-metric co-occurrences over kept predictions
        n_pred_boxes += int(keep_p.sum())
        if len(g_ids) and keep_p.any():
            kj = np.where(keep_p)[0]
            over = iou[:, kj] >= iou_thr
            for gi, jj in zip(*np.where(over)):
                cooc[(int(g_ids[gi]), int(p_ids[kj[jj]]))] += 1

    # global ID matching
    g_ids_all = sorted({k[0] for k in cooc})
    p_ids_all = sorted({k[1] for k in cooc})
    idtp = 0
    if g_ids_all and p_ids_all:
        gi_idx = {g: i for i, g in enumerate(g_ids_all)}
        pi_idx = {p: i for i, p in enumerate(p_ids_all)}
        m = np.zeros((len(g_ids_all), len(p_ids_all)))
        for (gid, pid), c in cooc.items():
            m[gi_idx[gid], pi_idx[pid]] = c
        ri, ci = linear_sum_assignment(-m)
        idtp = int(m[ri, ci].sum())

    idfp = n_pred_boxes - idtp
    idfn = n_gt_boxes - idtp
    metrics = {
        "MOTA": 1.0 - (fp + fn + idsw) / max(n_gt_boxes, 1),
        "MOTP": motp_sum / max(tp, 1),
        "IDF1": 2 * idtp / max(2 * idtp + idfp + idfn, 1),
        "IDP": idtp / max(idtp + idfp, 1),
        "IDR": idtp / max(idtp + idfn, 1),
        "TP": tp, "FP": fp, "FN": fn, "IDSW": idsw,
        "GT": n_gt_boxes, "PRED": n_pred_boxes,
    }
    if return_matches:
        return metrics, match_history
    return metrics


# ---------------------------------------------------------------- HOTA
# TrackEval-equivalent math (Luiten et al.): 19 alphas, two-pass matching —
# pass 1 accumulates soft IoU-Jaccard "global alignment scores" per
# (gt id, pred id) pair, pass 2 runs one Hungarian per frame on gas*sim and
# thresholds the matched similarities per alpha. Sparse accumulators so
# MOT20-sized id spaces stay cheap. Distractor handling follows the house
# style: a prediction overlapping a distractor >=0.5 with no considered GT
# overlap >=0.5 is dropped up front (static rule, no assignment dependency).
HOTA_ALPHAS = np.round(np.arange(0.05, 0.96, 0.05), 2)


def hota_accumulate(gt_frames: dict[int, dict], results: list[tuple]) -> dict:
    preds: dict[int, dict] = defaultdict(lambda: {"ids": [], "boxes": []})
    for r in results:
        preds[int(r[0])]["ids"].append(int(r[1]))
        preds[int(r[0])]["boxes"].append([r[2], r[3], r[2] + r[4], r[3] + r[5]])

    # dense id spaces for the whole sequence
    all_g = sorted({int(i) for g in gt_frames.values() for i in g["ids"]})
    all_p = sorted({int(r[1]) for r in results})
    gi_of = {g: i for i, g in enumerate(all_g)}
    pi_of = {p: i for i, p in enumerate(all_p)}
    G, P = len(all_g), len(all_p)

    frames = []                     # (g_dense, p_dense, iou) after prefilter
    gt_cnt = np.zeros(G, dtype=np.float64)
    pr_cnt = np.zeros(P, dtype=np.float64)
    pot = np.zeros((G, P), dtype=np.float64)
    n_gt_dets = n_pr_dets = 0
    for fid in sorted(gt_frames):
        g = gt_frames[fid]
        g_ids = np.asarray(g["ids"], dtype=np.int64)
        p = preds.get(fid, {"ids": [], "boxes": []})
        p_ids = np.array(p["ids"], dtype=np.int64)
        p_boxes = np.array(p["boxes"], dtype=np.float64).reshape(-1, 4)
        iou = iou_matrix(g["boxes"], p_boxes)
        if len(g["distractors"]) and len(p_boxes):
            best_g = iou.max(axis=0) if len(g_ids) else np.zeros(len(p_ids))
            best_d = iou_matrix(p_boxes, g["distractors"]).max(axis=1)
            keep = ~((best_d >= 0.5) & (best_g < 0.5))
            p_ids, iou = p_ids[keep], iou[:, keep]
        gd = np.array([gi_of[int(i)] for i in g_ids], dtype=np.int64)
        pd = np.array([pi_of[int(i)] for i in p_ids], dtype=np.int64)
        frames.append((gd, pd, iou))
        n_gt_dets += len(gd)
        n_pr_dets += len(pd)
        np.add.at(gt_cnt, gd, 1.0)
        np.add.at(pr_cnt, pd, 1.0)
        if len(gd) and len(pd):
            denom = iou.sum(0)[None, :] + iou.sum(1)[:, None] - iou
            soft = np.where(denom > 1e-9, iou / np.maximum(denom, 1e-9), 0.0)
            pot[np.ix_(gd, pd)] += soft

    # pass 2: one Hungarian per frame on gas*sim; matched sims deferred so all
    # 19 alphas are counted at once afterwards.
    pair_sims: dict[tuple[int, int], list[float]] = defaultdict(list)
    for gd, pd, iou in frames:
        if not len(gd) or not len(pd):
            continue
        pm = pot[np.ix_(gd, pd)]
        gas = pm / np.maximum(gt_cnt[gd][:, None] + pr_cnt[pd][None, :] - pm, 1e-9)
        ri, ci = linear_sum_assignment(-(gas * iou))
        for a, b in zip(ri, ci):
            s = iou[a, b]
            if s > 0:
                pair_sims[(int(gd[a]), int(pd[b]))].append(float(s))

    n_a = len(HOTA_ALPHAS)
    tp = np.zeros(n_a)
    sum_ass = np.zeros(n_a)
    for (gk, pk), sims in pair_sims.items():
        s = np.sort(np.asarray(sims))
        c = len(s) - np.searchsorted(s, HOTA_ALPHAS)      # matches per alpha
        tp += c
        denom = gt_cnt[gk] + pr_cnt[pk] - c
        nz = c > 0
        sum_ass[nz] += (c[nz] * c[nz]) / np.maximum(denom[nz], 1e-9)
    fn = n_gt_dets - tp
    fp = n_pr_dets - tp
    return {"tp": tp, "fn": fn, "fp": fp, "sum_ass": sum_ass}


def hota_merge(accs: list[dict]) -> dict:
    return {k: np.sum([a[k] for a in accs], axis=0)
            for k in ("tp", "fn", "fp", "sum_ass")}


def hota_final(acc: dict) -> dict:
    det_a = acc["tp"] / np.maximum(acc["tp"] + acc["fn"] + acc["fp"], 1)
    ass_a = acc["sum_ass"] / np.maximum(acc["tp"], 1)
    hota_a = np.sqrt(det_a * ass_a)
    return {"HOTA": float(hota_a.mean()), "DetA": float(det_a.mean()),
            "AssA": float(ass_a.mean())}


# ------------------------------------------------------- gap recovery (exp009)
RECOVERY_BUCKETS = (
    ("<0.5s", 0.0, 0.5), ("~1s", 0.5, 2.0), ("~3s", 2.0, 4.0), ("~5s", 4.0, 6.0),
    ("~7s", 6.0, 8.5), ("~10s", 8.5, 12.0), (">12s", 12.0, 1e9),
)


def recovery_evaluate(seq_path: str, fps: int, match_history: dict[int, dict[int, int]],
                      vis_thr: float = 0.25, window_s: float = 1.0) -> dict:
    """Same-id re-attachment across exp009 lost episodes.

    Episode = trackable GT (conf1 / pedestrian / vis>=0.25) -> gap ->
    trackable again; duration = gap frames / fps. pre id = tracker id
    matched to the GT at the latest trackable frame within window_s before
    the gap; post id = first match within window_s after reappearance.
    Per bucket: [episodes-tracked-before, same-id, different-id,
    not-reacquired, untracked-before]."""
    import os
    gt = np.loadtxt(os.path.join(seq_path, "gt", "gt.txt"), delimiter=",")
    track = gt[(gt[:, 6] == 1) & (gt[:, 7] == 1) & (gt[:, 8] >= vis_thr)]
    win = max(int(round(window_s * fps)), 1)
    out = {name: np.zeros(5, dtype=np.int64) for name, _, _ in RECOVERY_BUCKETS}
    for tid in np.unique(track[:, 1]).astype(int):
        f = np.unique(track[track[:, 1] == tid, 0]).astype(int)
        if len(f) < 2:
            continue
        for gi in np.where(np.diff(f) > 1)[0]:
            f1, f2 = int(f[gi]), int(f[gi + 1])
            dur = (f2 - f1 - 1) / fps
            name = next(n for n, lo, hi in RECOVERY_BUCKETS if lo <= dur < hi)
            pre = None
            for fr in f[(f <= f1) & (f > f1 - win)][::-1]:
                pre = match_history.get(int(fr), {}).get(tid)
                if pre is not None:
                    break
            if pre is None:
                out[name][4] += 1
                continue
            post = None
            for fr in f[(f >= f2) & (f < f2 + win)]:
                post = match_history.get(int(fr), {}).get(tid)
                if post is not None:
                    break
            out[name][0] += 1
            if post is None:
                out[name][3] += 1
            elif post == pre:
                out[name][1] += 1
            else:
                out[name][2] += 1
    return out


def recovery_merge(parts: list[dict]) -> dict:
    out = {name: np.zeros(5, dtype=np.int64) for name, _, _ in RECOVERY_BUCKETS}
    for p in parts:
        for name in out:
            out[name] += p[name]
    return out


def combine(metrics: list[dict]) -> dict:
    tot = {k: sum(m[k] for m in metrics) for k in ("TP", "FP", "FN", "IDSW", "GT", "PRED")}
    idtp_terms = [m["IDF1"] * (m["GT"] + m["PRED"]) / 2 for m in metrics]
    idtp = sum(idtp_terms)
    return {
        "MOTA": 1.0 - (tot["FP"] + tot["FN"] + tot["IDSW"]) / max(tot["GT"], 1),
        "MOTP": float(np.mean([m["MOTP"] for m in metrics])),
        "IDF1": 2 * idtp / max(tot["GT"] + tot["PRED"], 1),
        **tot,
    }
