"""SOMA: training-free multi-channel tracker over anatomical tokens.

Identity lives in the part ensemble, not the body box: a track keeps a box
Kalman filter, per-slot part predictors and attribute votes. Stage-1
association fuses box IoU, part OKS, head-orientation continuity, attribute
penalties and (SOMA-R) an external ReID embedding cosine; post-death memory
rebinds identities across long occlusions. Strictly online: every decision
uses only past and current frames.

This is the cleaned deployment core of the ptrack research tracker — every
mechanism here is exercised by the shipped presets; rejected/experimental
knobs were removed (their record lives in the research repo).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .kalman import BoxKalman, PartPoints, batched_predict
from .matching import linear_assignment
from .tokens import N_SLOTS, AnatomicalToken, apply_amodal


@dataclass
class TrackerConfig:
    det_thresh: float = 0.45        # stage-1 pool
    init_thresh: float = 0.55       # new-track creation
    min_hits: int = 3
    max_age: int = 60               # frames a lost track survives (2 x fps typical)
    max_age_sec: float = 0.0        # overrides max_age with sec * fps (set by driver)
    sim_gate: float = 0.30          # stage-1 combined-similarity gate
    part_gate: float = 0.45         # stage-2 part-only OKS gate (>= 1 disables)
    fps: int = 30                   # set by driver (seconds -> frames)
    # frame->detector-input mapping; set per sequence by the driver
    scale: tuple[float, float] = (1.0, 1.0)
    # ---- similarity channels ----
    w_iou: float = 1.0
    w_oks: float = 0.8              # part-OKS channel (0 keeps the availability floor)
    oks_kappa: float = 0.15
    w_emb: float = 1.0              # ReID embedding cosine channel (SOMA-R)
    emb_lo: float = 0.60            # cosine rescale range
    emb_hi: float = 0.98
    emb_veto: float = 0.50          # hard penalty below this cosine (crossing guard)
    emb_alpha: float = 0.9          # track embedding EMA keep-rate
    w_dir: float = 0.0              # head-orientation similarity channel
    dir_min_conf: float = 0.60      # min resultant length R on both sides
    dir_alpha: float = 0.8          # circular EMA keep-rate for track direction
    dir_veto: float = 0.0           # penalty when |dtheta| exceeds the physical cap
    dir_max_step_deg: float = 30.0  # allowed head turn per frame (cap grows with gap)
    attr_penalty: float = 0.10      # gender/generation disagreement (each)
    center_gate_mult: float = 2.0   # hard gate: center dist <= mult * pred diag
    gate_grow: float = 0.0          # per-second growth of the lost-track center gate
    # ---- anatomical amodal synthesis (visible box -> full extent) ----
    amodal_alpha: float = 0.0       # full height = alpha * head height; 0 disables
    amodal_gamma: float = 0.90
    # ---- DLO-lite confidence boost ----
    dlo_beta: float = 0.0           # s_hat = max(s, beta * max IoU(pred, det)); 0 off
    token_floor: float = 0.0        # drop body tokens below this score before tracking
    # ---- online scene-geometry size prior h(y) ----
    size_prior_thr: float = 0.0     # 0 disables
    size_prior_quality: float = 0.12
    size_prior_scale: float = 0.5   # pool-score multiplier for violators
    # ---- lost-track handling ----
    lost_vel_decay: float = 1.0     # per-miss velocity decay of unmatched tracks
    # appearance-locked extension ("no look, no re-latch"): a track lost
    # longer than gate_sec may only stage-1 match a det whose embedding
    # confirms it — makes a long max_age theft-safe
    lost_emb_gate_sec: float = 0.0  # 0 disables
    lost_emb_gate_cos: float = 0.55
    # revival stage: long-lost tracks x unmatched strong tokens, embedding-only
    revive_cos: float = 0.0         # 0 disables; else min cosine to reclaim the ID
    revive_min_gap: int = 15        # frames a track must have been lost
    revive_dist_mult: float = 3.0   # spatial gate: dist <= mult * pred diag
    # ---- post-death identity memory (rebinding at birth time) ----
    mem_ttl_sec: float = 0.0        # 0 disables; memory horizon after death
    mem_dist_mult: float = 1.6      # dist <= mult * h * (1 + mem_grow * gap_s)
    mem_grow: float = 0.5
    mem_size_tol: float = 0.35      # |log(h_det/h_mem)| <= log(1+tol)
    mem_margin: float = 0.08        # winner must beat the runner-up by this
    mem_cos: float = 0.0            # SOMA-R: rebinding requires bank cosine >= this
    # ---- embedding hygiene ----
    emb_update_crowd_max: float = 1.0   # skip EMA update above this det-overlap IoU
    emb_center_lambda: float = 0.0  # subtract lambda * frame-mean embedding, renorm
    # ---- output ----
    emit_kf: bool = False           # emit the KF posterior box instead of the raw det
    ghost_emit_max_s: float = 0.0   # emit the KF prediction for lost confirmed
    #                                 tracks up to this long (online coasting)
    ghost_score_mult: float = 0.5
    ghost_crowd_max: float = 1.0    # ghosts only for tracks whose last observation
    #                                 had crowding <= this (1 = all)


@dataclass
class Track:
    tid: int
    kf: BoxKalman
    parts: PartPoints
    cfg: TrackerConfig
    hits: int = 1
    age: int = 1
    time_since_update: int = 0
    last_frame: int = 0
    last_box: np.ndarray | None = None
    gender_votes: np.ndarray = field(default_factory=lambda: np.zeros(2))
    gen_votes: np.ndarray = field(default_factory=lambda: np.zeros(2))
    anchor: str = "body"
    score: float = 0.0
    embedding: np.ndarray | None = None   # EMA of token embeddings, L2-normalized
    emb_pregap: np.ndarray | None = None  # EMA snapshot at the last frame before a gap
    emb_best: np.ndarray | None = None    # embedding of the highest-score observation
    best_score: float = 0.0
    head_dir: np.ndarray | None = None    # circular EMA of soft head orientation (unit)
    head_dir_conf: float = 0.0
    last_crowding: float = 0.0            # crowding of the last absorbed observation
    _gender_c: int | None = None          # memoized vote argmaxes
    _gen_c: int | None = None

    def emb_cos(self, e: np.ndarray | None) -> float | None:
        if e is None or self.embedding is None:
            return None
        return float(self.embedding @ e)

    def gender(self) -> int:
        if self._gender_c is None:
            self._gender_c = (int(np.argmax(self.gender_votes))
                              if self.gender_votes.sum() >= 3 else -1)
        return self._gender_c

    def generation(self) -> int:
        if self._gen_c is None:
            self._gen_c = (int(np.argmax(self.gen_votes))
                           if self.gen_votes.sum() >= 3 else -1)
        return self._gen_c

    def absorb(self, tok: AnatomicalToken, cfg: TrackerConfig,
               full: bool = True) -> None:
        pts = _abs_points(tok)
        if full:
            self.kf.update(tok.body_box)
            self.last_box = tok.body_box.copy()
            self.score = tok.body_score
            self.anchor = tok.anchor
        else:
            # stage-2 partial continuation: translation-only observation from
            # the common part slots ("a head alone can carry the ID")
            pred = self.parts.predict()
            common = tok.presence & self.parts.seen
            if common.any():
                delta = np.nanmedian(pts[common] - pred[common], axis=0)
                if np.isfinite(delta).all():
                    self.kf.shift(float(delta[0]), float(delta[1]))
        crowding = getattr(tok, "crowding", 0.0)
        if full:
            self.last_crowding = crowding
        if tok.embedding is not None and crowding <= cfg.emb_update_crowd_max:
            if self.embedding is None:
                self.embedding = tok.embedding.copy()
            else:
                e = cfg.emb_alpha * self.embedding + (1 - cfg.emb_alpha) * tok.embedding
                self.embedding = e / (np.linalg.norm(e) + 1e-9)
            if tok.body_score >= self.best_score:
                self.best_score = tok.body_score
                self.emb_best = tok.embedding.copy()
        if tok.head_dir is not None and tok.head_dir_conf >= cfg.dir_min_conf:
            if self.head_dir is None:
                self.head_dir = tok.head_dir.copy()
            else:
                v = cfg.dir_alpha * self.head_dir + (1 - cfg.dir_alpha) * tok.head_dir
                n = float(np.hypot(v[0], v[1]))
                if n > 1e-6:
                    self.head_dir = (v / n).astype(np.float32)
            self.head_dir_conf = tok.head_dir_conf
        self.parts.update(pts, tok.presence)
        if tok.gender >= 0:
            self.gender_votes[tok.gender] += 1
            self._gender_c = None
        if tok.generation >= 0:
            self.gen_votes[tok.generation] += 1
            self._gen_c = None
        self.hits += 1
        self.time_since_update = 0


def _center(box: np.ndarray) -> np.ndarray:
    return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2], dtype=np.float32)


def _abs_points(tok: AnatomicalToken) -> np.ndarray:
    bp = tok.box_proxy
    w, h = max(bp[2] - bp[0], 1.0), max(bp[3] - bp[1], 1.0)
    return bp[:2] + tok.points * (w, h)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    bb = (b[2] - b[0]) * (b[3] - b[1])
    return float(inter / (aa + bb - inter + 1e-9))


def _iou_mat(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(N,4) x (M,4) -> (N,M), same formula as _iou."""
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + bb[None, :] - inter + 1e-9)


class SomaTracker:
    def __init__(self, cfg: TrackerConfig | None = None):
        self.cfg = cfg or TrackerConfig()
        self.tracks: list[Track] = []
        self.next_id = 1
        self.rows: list[tuple[int, int, float, float, float, float, float]] = []
        self._memory: list[dict] = []                        # post-death identity memory
        self._geom_samples: list[tuple[float, float]] = []   # (y_bottom, h) reservoir
        self._geom_fit: tuple[float, float] | None = None    # (a, b), None = disabled
        self._geom_next_fit = 100

    # ---- online scene-geometry size prior ----------------------------------
    def _geom_update(self, tokens, matched_tokens) -> None:
        for di in matched_tokens:
            tok = tokens[di]
            if tok.body_box is not None and tok.body_score >= 0.5:
                self._geom_samples.append((float(tok.body_box[3]),
                                           float(tok.body_box[3] - tok.body_box[1])))
        if len(self._geom_samples) > 4000:
            self._geom_samples = self._geom_samples[-2000:]
        if len(self._geom_samples) >= self._geom_next_fit:
            self._geom_next_fit = len(self._geom_samples) + 300
            arr = np.array(self._geom_samples)
            y, h = arr[:, 0], arr[:, 1]
            rng = np.random.default_rng(0)
            idx = rng.integers(0, len(y), size=(2000, 2))
            dy = y[idx[:, 0]] - y[idx[:, 1]]
            ok = np.abs(dy) > 5
            if ok.sum() < 50:
                return
            a = float(np.median((h[idx[ok, 0]] - h[idx[ok, 1]]) / dy[ok]))
            b = float(np.median(h - a * y))
            pred = np.maximum(a * y + b, 1.0)
            quality = float(np.median(np.abs(h - pred) / pred))
            self._geom_fit = (a, b) if quality <= self.cfg.size_prior_quality else None

    def _size_residual(self, box: np.ndarray) -> float:
        if self._geom_fit is None:
            return 0.0
        a, b = self._geom_fit
        pred = max(a * float(box[3]) + b, 1.0)
        return abs(float(box[3] - box[1]) - pred) / pred

    def _oks(self, tr: Track, tok: AnatomicalToken, pred_h: float) -> float | None:
        common = tok.presence & tr.parts.seen
        if not common.any():
            return None
        pred = tr.parts.predict()[common]
        obs = _abs_points(tok)[common]
        d2 = ((pred - obs) ** 2).sum(axis=1)
        s2 = (self.cfg.oks_kappa * max(pred_h, 8.0)) ** 2
        return float(np.exp(-d2 / (2 * s2)).mean())

    # ---- stage-1 similarity (vectorized) ------------------------------------
    def _similarity_matrix(self, tokens, full_idx, pred_boxes) -> np.ndarray:
        cfg = self.cfg
        T, D = len(self.tracks), len(full_idx)
        pb = np.stack(pred_boxes)                                   # (T,4)
        pc = np.stack([(pb[:, 0] + pb[:, 2]) * 0.5, (pb[:, 1] + pb[:, 3]) * 0.5], 1)
        ph = pb[:, 3] - pb[:, 1]
        diag = np.hypot(pb[:, 2] - pb[:, 0], ph)
        gap = np.array([max(tr.time_since_update - 1, 0) for tr in self.tracks], float)
        gap_s = gap / max(cfg.fps, 1)
        gate_lim = cfg.center_gate_mult * (1.0 + cfg.gate_grow * gap_s) * diag

        toks = [tokens[di] for di in full_idx]
        tb = np.stack([t.body_box if t.body_box is not None else t.box_proxy
                       for t in toks])
        tc = np.stack([[(t.box_proxy[0] + t.box_proxy[2]) * 0.5,
                        (t.box_proxy[1] + t.box_proxy[3]) * 0.5] for t in toks])
        dist = np.hypot(tc[None, :, 0] - pc[:, None, 0], tc[None, :, 1] - pc[:, None, 1])
        in_gate = dist <= gate_lim[:, None]

        # IoU (pred boxes x token body/proxy boxes)
        x1 = np.maximum(pb[:, None, 0], tb[None, :, 0])
        y1 = np.maximum(pb[:, None, 1], tb[None, :, 1])
        x2 = np.minimum(pb[:, None, 2], tb[None, :, 2])
        y2 = np.minimum(pb[:, None, 3], tb[None, :, 3])
        inter = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
        pa = np.maximum(pb[:, 2] - pb[:, 0], 0) * np.maximum(ph, 0)
        ta = np.maximum(tb[:, 2] - tb[:, 0], 0) * np.maximum(tb[:, 3] - tb[:, 1], 0)
        iou = inter / np.maximum(pa[:, None] + ta[None, :] - inter, 1e-9)

        # part OKS (parts predicted once per track). NOTE: even at w_oks 0 the
        # OKS values feed the availability floor (oks >= .35 keeps a pair
        # alive) — the tensor must stay.
        seen = np.stack([tr.parts.seen for tr in self.tracks])       # (T,S)
        pres = np.stack([t.presence for t in toks])                  # (D,S)
        pred_pts = np.stack([tr.parts.predict() for tr in self.tracks])
        obs = np.stack([_abs_points(t) for t in toks])               # (D,S,2)
        common = seen[:, None, :] & pres[None, :, :]
        d2 = ((pred_pts[:, None, :, :] - obs[None, :, :, :]) ** 2).sum(-1)
        s2 = (cfg.oks_kappa * np.maximum(ph, 8.0)) ** 2
        with np.errstate(over="ignore", invalid="ignore"):
            ek = np.exp(-d2 / (2 * s2[:, None, None]))
        cnt = common.sum(-1)
        has_oks = cnt > 0
        oks = np.where(has_oks, (np.nan_to_num(ek) * common).sum(-1)
                       / np.maximum(cnt, 1), 0.0)

        # ReID embedding cosine (EMA)
        e_dim = next((tr.embedding.shape[0] for tr in self.tracks
                      if tr.embedding is not None), 0)
        cos = None
        if e_dim:
            trv = np.array([tr.embedding is not None for tr in self.tracks])
            dv = np.array([t.embedding is not None for t in toks])
            if trv.any() and dv.any():
                trE = np.stack([tr.embedding if tr.embedding is not None
                                else np.zeros(e_dim, np.float32) for tr in self.tracks])
                dE = np.stack([t.embedding if t.embedding is not None
                               else np.zeros(e_dim, np.float32) for t in toks])
                cos = trE @ dE.T
                cos_valid = trv[:, None] & dv[None, :]
        if cos is not None:
            emb = np.clip((cos - cfg.emb_lo) / (cfg.emb_hi - cfg.emb_lo), 0.0, 1.0)
            emb_on = cos_valid & (cfg.w_emb > 0)

        # head-direction channel
        if cfg.w_dir > 0 or cfg.dir_veto > 0:
            trd_ok = np.array([tr.head_dir is not None
                               and tr.head_dir_conf >= cfg.dir_min_conf
                               for tr in self.tracks])
            dd_ok = np.array([t.head_dir is not None
                              and t.head_dir_conf >= cfg.dir_min_conf
                              for t in toks])
        else:
            trd_ok = np.zeros(T, dtype=bool)
            dd_ok = np.zeros(D, dtype=bool)
        dir_valid = trd_ok[:, None] & dd_ok[None, :]
        dcos = np.zeros((T, D))
        if trd_ok.any() and dd_ok.any():
            trD = np.stack([tr.head_dir if tr.head_dir is not None
                            else np.zeros(2, np.float32) for tr in self.tracks])
            dD = np.stack([t.head_dir if t.head_dir is not None
                           else np.zeros(2, np.float32) for t in toks])
            dcos = trD @ dD.T

        num = cfg.w_iou * iou
        den = np.full((T, D), cfg.w_iou)
        num += np.where(has_oks, cfg.w_oks * oks, 0.0)
        den += np.where(has_oks, cfg.w_oks, 0.0)
        if cos is not None:
            num += np.where(emb_on, cfg.w_emb * emb, 0.0)
            den += np.where(emb_on, cfg.w_emb, 0.0)
        if cfg.w_dir > 0:
            num += np.where(dir_valid, cfg.w_dir * 0.5 * (1.0 + dcos), 0.0)
            den += np.where(dir_valid, cfg.w_dir, 0.0)
        s = num / den
        if cos is not None:
            s -= np.where(cos_valid & (cos < cfg.emb_veto), 0.15, 0.0)
        dir_pen = np.zeros((T, D))
        if cfg.dir_veto > 0:
            tsu = np.array([max(tr.time_since_update, 1) for tr in self.tracks], float)
            cap = np.minimum(180.0, cfg.dir_max_step_deg * tsu + 45.0)
            dir_pen = np.where(dir_valid & (dcos < np.cos(np.deg2rad(cap))[:, None]),
                               cfg.dir_veto, 0.0)
            s -= dir_pen

        # availability floor: no geometric or part evidence -> no stage-1 match
        floor = (iou < 0.05) & (~has_oks | (oks < 0.35))

        # attribute penalties (skipped for floored cells, matching the floor)
        alive = ~floor
        tg = np.array([tr.gender() for tr in self.tracks])
        tt = np.array([t.gender for t in toks])
        s -= np.where(alive & (tg[:, None] >= 0) & (tt[None, :] >= 0)
                      & (tg[:, None] != tt[None, :]), cfg.attr_penalty, 0.0)
        tg = np.array([tr.generation() for tr in self.tracks])
        tt = np.array([t.generation for t in toks])
        s -= np.where(alive & (tg[:, None] >= 0) & (tt[None, :] >= 0)
                      & (tg[:, None] != tt[None, :]), cfg.attr_penalty * 0.5, 0.0)
        wc = np.array([tr.anchor in ("wheelchair", "crutches") and tr.hits > 5
                       for tr in self.tracks])
        tbody = np.array([t.anchor == "body" for t in toks])
        s -= np.where(alive & wc[:, None] & tbody[None, :], cfg.attr_penalty, 0.0)

        # "no look, no re-latch": long-lost rows require embedding confirmation
        if cfg.lost_emb_gate_sec > 0 and cos is not None:
            longlost = gap_s >= cfg.lost_emb_gate_sec
            s = np.where(longlost[:, None]
                         & (~cos_valid | (cos < cfg.lost_emb_gate_cos)), 0.0, s)

        s = np.maximum(s, 0.0)
        s[~in_gate] = 0.0
        s[floor] = 0.0
        return s

    # ---- main step ----------------------------------------------------------
    def step(self, frame_id: int, tokens: list[AnatomicalToken]) -> None:
        cfg = self.cfg
        if cfg.token_floor > 0:
            tokens = [t for t in tokens
                      if t.anchor == "orphan" or t.body_score >= cfg.token_floor]
        pred_boxes = batched_predict([tr.kf for tr in self.tracks])
        for tr in self.tracks:
            tr.age += 1
            tr.time_since_update += 1

        if cfg.amodal_alpha > 0:
            for t in tokens:
                apply_amodal(t, cfg.amodal_alpha, cfg.amodal_gamma)

        if cfg.emb_center_lambda > 0:
            embs = [t.embedding for t in tokens if t.embedding is not None]
            if len(embs) >= 3:
                mean = np.mean(embs, axis=0)
                for t in tokens:
                    if t.embedding is not None:
                        e = t.embedding - cfg.emb_center_lambda * mean
                        t.embedding = e / (np.linalg.norm(e) + 1e-9)

        if cfg.emb_update_crowd_max < 1.0:
            bodies = [t for t in tokens if t.anchor != "orphan"]
            boxes_b = np.array([t.body_box if t.body_box is not None else t.box_proxy
                                for t in bodies]).reshape(-1, 4)
            if len(bodies) > 1:
                m = _iou_mat(boxes_b, boxes_b)
                np.fill_diagonal(m, 0.0)
                crowd = m.max(axis=1)
            else:
                crowd = np.zeros(len(bodies))
            for i, t in enumerate(bodies):
                t.crowding = float(crowd[i])

        # DLO-lite: a low-score body detection overlapping a predicted track
        # is probably a real (occluded) person — lift it into the stage-1 pool.
        pool_scores = np.array([t.body_score for t in tokens], dtype=np.float64)
        if cfg.dlo_beta > 0 and pred_boxes and len(tokens):
            cand = [i for i, t in enumerate(tokens)
                    if t.anchor != "orphan" and pool_scores[i] < cfg.det_thresh]
            if cand:
                cb = np.stack([tokens[i].body_box if tokens[i].body_box is not None
                               else tokens[i].box_proxy for i in cand])
                best = _iou_mat(np.stack(pred_boxes), cb).max(axis=0)
                pool_scores[cand] = np.maximum(pool_scores[cand], cfg.dlo_beta * best)

        size_bad = np.zeros(len(tokens), dtype=bool)
        if cfg.size_prior_thr > 0 and self._geom_fit is not None:
            for i, t in enumerate(tokens):
                if t.anchor == "orphan":
                    continue
                box = t.body_box if t.body_box is not None else t.box_proxy
                if self._size_residual(box) > cfg.size_prior_thr:
                    size_bad[i] = True
                    pool_scores[i] *= cfg.size_prior_scale

        full_idx = [i for i, t in enumerate(tokens)
                    if t.anchor != "orphan" and pool_scores[i] >= cfg.det_thresh]
        full_set = set(full_idx)
        rest_idx = [i for i in range(len(tokens)) if i not in full_set]

        # stage 1
        n_t, n_d = len(self.tracks), len(full_idx)
        matched_tracks: set[int] = set()
        matched_tokens: set[int] = set()
        if n_t and n_d:
            sim = self._similarity_matrix(tokens, full_idx, pred_boxes)
            for ti, dj in linear_assignment(1.0 - sim, gate=1.0 - cfg.sim_gate):
                tr, di = self.tracks[ti], full_idx[dj]
                tr.absorb(tokens[di], cfg)
                matched_tracks.add(ti)
                matched_tokens.add(di)
                self._emit(frame_id, tr,
                           tr.kf.box() if cfg.emit_kf else tokens[di].body_box)

        # revival: long-lost tracks x unmatched strong tokens, embedding-only
        # with a wide spatial gate — stage 1's availability floor blocks
        # exactly these drifted pairs.
        if cfg.revive_cos > 0:
            rev_tracks = [ti for ti in range(n_t) if ti not in matched_tracks
                          and self.tracks[ti].time_since_update - 1 >= cfg.revive_min_gap
                          and self.tracks[ti].embedding is not None]
            rev_tokens = [di for di in full_idx if di not in matched_tokens
                          and tokens[di].embedding is not None
                          and tokens[di].body_score >= cfg.init_thresh]
            if rev_tracks and rev_tokens:
                sim = np.zeros((len(rev_tracks), len(rev_tokens)))
                for a, ti in enumerate(rev_tracks):
                    tr = self.tracks[ti]
                    pb = pred_boxes[ti]
                    diag = float(np.hypot(pb[2] - pb[0], pb[3] - pb[1]))
                    pc = _center(pb)
                    for b, di in enumerate(rev_tokens):
                        tok = tokens[di]
                        if float(np.hypot(*(_center(tok.box_proxy) - pc))) > \
                                cfg.revive_dist_mult * diag:
                            continue
                        c = tr.emb_cos(tok.embedding)
                        if c is not None and c >= cfg.revive_cos:
                            sim[a, b] = c
                for a, b in linear_assignment(1.0 - sim, gate=1.0 - cfg.revive_cos):
                    ti, di = rev_tracks[a], rev_tokens[b]
                    tr = self.tracks[ti]
                    tr.kf = BoxKalman(tokens[di].body_box)   # motion state is stale
                    tr.absorb(tokens[di], cfg)
                    matched_tracks.add(ti)
                    matched_tokens.add(di)
                    self._emit(frame_id, tr,
                               tr.kf.box() if cfg.emit_kf else tokens[di].body_box)

        # stage 2: part-only continuation (orphan part groups + low-score
        # tokens). part_gate >= 1 disables the assignment AND the OKS build.
        rem_tracks = ([ti for ti in range(n_t) if ti not in matched_tracks]
                      if cfg.part_gate < 1.0 else [])
        rem_tokens = [di for di in rest_idx if di not in matched_tokens]
        if rem_tracks and rem_tokens:
            sim = np.zeros((len(rem_tracks), len(rem_tokens)), dtype=np.float64)
            for a, ti in enumerate(rem_tracks):
                tr = self.tracks[ti]
                pb = pred_boxes[ti]
                for b, di in enumerate(rem_tokens):
                    tc = _center(tokens[di].box_proxy)
                    diag = float(np.hypot(pb[2] - pb[0], pb[3] - pb[1]))
                    if float(np.hypot(*(tc - _center(pb)))) > cfg.center_gate_mult * diag:
                        continue
                    sim[a, b] = self._oks(tr, tokens[di], pb[3] - pb[1]) or 0.0
            for a, b in linear_assignment(1.0 - sim, gate=1.0 - cfg.part_gate):
                ti, di = rem_tracks[a], rem_tokens[b]
                tr = self.tracks[ti]
                tr.absorb(tokens[di], cfg, full=False)
                matched_tracks.add(ti)
                matched_tokens.add(di)
                self._emit(frame_id, tr, tr.kf.box())    # amodal: KF box

        # post-death memory: bind dying-out identities to would-be births. The
        # bound id is still inside min_hits probation (nothing emitted yet), so
        # a wrong bind that never confirms costs nothing.
        reborn: dict[int, dict] = {}
        if cfg.mem_ttl_sec > 0:
            ttl = cfg.mem_ttl_sec * max(cfg.fps, 1)
            self._memory = [e for e in self._memory
                            if frame_id - e["last_frame"] <= ttl][-300:]
            birth_dis = [di for di in full_idx
                         if di not in matched_tokens and not size_bad[di]
                         and tokens[di].body_score >= cfg.init_thresh]
            if birth_dis and self._memory:
                M, D = len(self._memory), len(birth_dis)
                score = np.zeros((M, D))
                live_of = {t.tid: t for t in self.tracks}
                birth_geo = []
                for di in birth_dis:
                    tb0 = (tokens[di].body_box
                           if tokens[di].body_box is not None
                           else tokens[di].box_proxy)
                    birth_geo.append((tb0, _center(tb0)))
                for mi, e in enumerate(self._memory):
                    gap_f = frame_id - e["last_frame"]
                    lim = cfg.mem_dist_mult * e["h"] * (1.0 + cfg.mem_grow * gap_f
                                                        / max(cfg.fps, 1))
                    occ = live_of.get(e["occ_tid"]) if e.get("occ_tid", -1) >= 0 else None
                    oc = None
                    if occ is not None and e["occ_off"] is not None:
                        ob = occ.last_box if occ.last_box is not None else occ.kf.box()
                        oc = _center(ob) + e["occ_off"]          # riding the occluder
                    for dj, di in enumerate(birth_dis):
                        tok = tokens[di]
                        tb, tc = birth_geo[dj]
                        d = float(np.hypot(*(tc - e["center"])))
                        pc = e["center"] + e["vel"] * gap_f      # linear continuation
                        d = min(d, float(np.hypot(*(tc - pc))))
                        if oc is not None:
                            d = min(d, float(np.hypot(*(tc - oc))))
                        if d > lim:
                            continue
                        h = max(float(tb[3] - tb[1]), 1e-3)
                        dlh = abs(float(np.log(h / e["h"])))
                        if dlh > np.log1p(cfg.mem_size_tol):
                            continue
                        if tok.gender >= 0 and e["gender"] >= 0 and tok.gender != e["gender"]:
                            continue
                        if (tok.generation >= 0 and e["generation"] >= 0
                                and tok.generation != e["generation"]):
                            continue
                        pos_s = 1.0 - d / lim
                        size_s = 1.0 - dlh / np.log1p(cfg.mem_size_tol)
                        if cfg.mem_cos > 0:
                            bank = e.get("bank") or []
                            if tok.embedding is None or not bank:
                                continue    # SOMA-R: no look, no bind
                            cos = max(float(be @ tok.embedding) for be in bank)
                            if cos < cfg.mem_cos:
                                continue
                            score[mi, dj] = 0.45 * pos_s + 0.2 * size_s + 0.35 * cos
                        else:
                            score[mi, dj] = 0.6 * pos_s + 0.4 * size_s
                used_mi = set()
                for mi, dj in linear_assignment(1.0 - score, gate=1.0 - 1e-9):
                    if score[mi, dj] <= 0:
                        continue
                    run_r = np.partition(score[mi], -2)[-2] if D > 1 else 0.0
                    run_c = np.partition(score[:, dj], -2)[-2] if M > 1 else 0.0
                    if score[mi, dj] - max(run_r, run_c) < cfg.mem_margin:
                        continue
                    reborn[birth_dis[dj]] = self._memory[mi]
                    used_mi.add(mi)
                if used_mi:
                    self._memory = [e for k, e in enumerate(self._memory)
                                    if k not in used_mi]

        # births
        for di in full_idx:
            if di in matched_tokens:
                continue
            tok = tokens[di]
            if tok.body_score < cfg.init_thresh:
                continue
            if size_bad[di]:
                continue                        # scale-implausible: no birth
            e = reborn.get(di)
            tr = Track(tid=(e["tid"] if e is not None else self.next_id),
                       kf=BoxKalman(tok.body_box),
                       parts=PartPoints(N_SLOTS), cfg=cfg)
            tr.absorb(tok, cfg)
            tr.hits = 1
            if e is not None:                    # inherit identity evidence
                tr.gender_votes = e["gender_votes"].copy()
                tr.gen_votes = e["gen_votes"].copy()
                if e["head_dir"] is not None:
                    tr.head_dir = e["head_dir"].copy()
                    tr.head_dir_conf = e["head_dir_conf"]
            else:
                self.next_id += 1
            self.tracks.append(tr)
            if cfg.min_hits <= 1:
                self._emit(frame_id, tr, tok.body_box)

        # lost-track bookkeeping
        for ti in range(n_t):
            tr = self.tracks[ti]
            if ti in matched_tracks:
                continue
            if tr.time_since_update == 1 and tr.embedding is not None:
                tr.emb_pregap = tr.embedding.copy()      # last clean look before the gap
            if cfg.lost_vel_decay < 1.0:
                tr.kf.x[4:6] *= cfg.lost_vel_decay
                tr.parts.vel *= cfg.lost_vel_decay

        if cfg.size_prior_thr > 0:
            self._geom_update(tokens, matched_tokens)

        for ti in matched_tracks:
            self.tracks[ti].last_frame = frame_id

        # deaths (finished confirmed tracklets feed the identity memory)
        alive = []
        for tr in self.tracks:
            if tr.time_since_update <= cfg.max_age:
                alive.append(tr)
            elif (cfg.mem_ttl_sec > 0 and tr.hits >= cfg.min_hits
                    and tr.last_box is not None):
                self._memory.append(self._mem_entry(tr))
        self.tracks = alive

        # ghost output: keep emitting the KF prediction through SHORT losses
        # (coasting; strictly online).
        if cfg.ghost_emit_max_s > 0:
            lim = max(int(round(cfg.ghost_emit_max_s * max(cfg.fps, 1))), 1)
            for tr in self.tracks:
                if (tr.hits >= cfg.min_hits
                        and 1 <= tr.time_since_update <= lim
                        and tr.last_crowding <= cfg.ghost_crowd_max):
                    self._append(frame_id, tr.tid, tr.kf.box(),
                                 tr.score * cfg.ghost_score_mult)

    def _emit(self, frame_id: int, tr: Track, box: np.ndarray) -> None:
        if tr.hits < self.cfg.min_hits:
            return
        self._append(frame_id, tr.tid, box, tr.score)

    def _append(self, frame_id: int, tid: int, box: np.ndarray, score: float) -> None:
        self.rows.append((frame_id, tid, float(box[0]), float(box[1]),
                          float(box[2] - box[0]), float(box[3] - box[1]), score))

    def _mem_entry(self, tr: Track) -> dict:
        """Identity memory captured at death — anchored on the LAST OBSERVED
        box (the KF box has extrapolated through max_age misses). Also records
        the OCCLUDER (max-overlap live track at death): the vanished person
        tends to re-emerge near the occluder's CURRENT position."""
        b = tr.last_box
        c = _center(b)
        occ_tid, occ_off, best = -1, None, 0.15
        for o in self.tracks:
            if o.tid == tr.tid or o.time_since_update > 1:
                continue
            ob = o.last_box if o.last_box is not None else o.kf.box()
            v = _iou(b, ob)
            if v > best:
                best = v
                occ_tid = o.tid
                occ_off = c - _center(ob)
        return {"tid": tr.tid, "last_frame": tr.last_frame,
                "center": c, "h": max(float(b[3] - b[1]), 8.0),
                "vel": tr.kf.x[4:6].copy(),
                "occ_tid": occ_tid, "occ_off": occ_off,
                "bank": [e for e in (tr.embedding, tr.emb_pregap, tr.emb_best)
                         if e is not None],
                "gender": tr.gender(), "generation": tr.generation(),
                "gender_votes": tr.gender_votes.copy(), "gen_votes": tr.gen_votes.copy(),
                "head_dir": None if tr.head_dir is None else tr.head_dir.copy(),
                "head_dir_conf": tr.head_dir_conf}

    def results(self) -> list[tuple]:
        return sorted(self.rows, key=lambda r: (r[0], r[1]))
