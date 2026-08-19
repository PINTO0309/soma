"""Shipped tracker presets and ReID variants (the public configuration API).

SOMA: geometry/parts/orientation channels only (no ReID).
SOMA-R: appearance-dominant stage-1 + memory/revival/ghost stack; the
embedding thresholds are quantile-calibrated per ReID model (ptrack exp050
procedure), everything else is shared.
"""
from __future__ import annotations

from .tracker import TrackerConfig

# ReID variants of the wb28x640 CrowdTrack benchmark. PersonViT v3 runs RAW
# (its raw cosine geometry is already spread — internal domain
# normalization); OSNet-AIN aug v3 needs causal whitening (compressed cone).
VARIANTS = {
    "det": {"reid": None, "whiten": 0.0},
    "pv": {"reid": "models/personvit_vits16_ain_unified_aug_n.onnx", "whiten": 0.0},
    "os": {"reid": "models/osnet_ain_x1_0_p_unified_aug_n.onnx", "whiten": 0.98},
}

_SOMAR_BASE = {
    "det_thresh": 0.35, "init_thresh": 0.55, "sim_gate": 0.30,
    "w_emb": 4.0, "w_oks": 0.0, "amodal_alpha": 4.0, "dlo_beta": 0.65,
    "token_floor": 0.12, "w_dir": 0.0, "dir_veto": 0.0, "size_prior_thr": 0.5,
    "emb_center_lambda": 0.25, "mem_ttl_sec": 12.0, "mem_margin": 0.10,
    "lost_vel_decay": 0.85, "gate_grow": 0.6, "max_age_sec": 6.0,
    "lost_emb_gate_sec": 1.2, "emit_kf": True, "part_gate": 1.01,
    "emb_update_crowd_max": 0.60,
    "ghost_emit_max_s": 0.10, "ghost_crowd_max": 0.30,
}
PRESETS = {
    "soma": {"det_thresh": 0.45, "init_thresh": 0.55, "sim_gate": 0.20,
             "amodal_alpha": 4.0, "dlo_beta": 0.65, "token_floor": 0.25,
             "w_dir": 0.5, "dir_veto": 0.15, "size_prior_thr": 0.5},
    "somar-pv": {**_SOMAR_BASE,
                 "emb_lo": 0.398, "emb_hi": 0.945, "emb_veto": 0.479,
                 "lost_emb_gate_cos": 0.398, "mem_cos": 0.642,
                 "revive_cos": 0.560},
    # v4 (cam-branch generation, ptrack exp065): whitening kept — the raw
    # pool is healthy now but whitening still wins the cell A/B; thresholds
    # re-qmapped on the v4 whitened pool.
    "somar-os": {**_SOMAR_BASE,
                 "emb_lo": 0.251, "emb_hi": 0.901, "emb_veto": 0.314,
                 "lost_emb_gate_cos": 0.251, "mem_cos": 0.470,
                 "revive_cos": 0.383},
}


def tracker_config(preset: str = "somar-pv", fps: int = 30,
                   **overrides) -> TrackerConfig:
    """Build a ready-to-run TrackerConfig from a shipped preset.

    Resolves the frame-rate-derived horizons (``max_age`` from
    ``max_age_sec``) so callers never repeat the driver boilerplate.
    ``overrides`` are applied on top of the preset fields.
    """
    if preset not in PRESETS:
        raise ValueError(f"unknown preset: {preset!r}; expected one of: "
                         f"{', '.join(sorted(PRESETS))}")
    kw = dict(PRESETS[preset])
    kw.update(overrides)
    cfg = TrackerConfig(**kw)
    cfg.fps = int(fps)
    cfg.max_age = (int(round(cfg.max_age_sec * cfg.fps))
                   if cfg.max_age_sec > 0 else 2 * cfg.fps)
    return cfg
