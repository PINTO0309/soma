"""SOMA — Structured Output Matching & Association.

Online multi-person tracking on the structured output of a multi-task
wholebody detector, with a ReID-dominant long-gap recovery stack (SOMA-R).

Quickstart (bring your own detector, numpy-only)::

    from soma import Detection, SomaTracker, tracker_config

    trk = SomaTracker(tracker_config("somar-pv", fps=25), record_rows=False)
    tracks = trk.update([Detection(box=(x1, y1, x2, y2), score=0.9,
                                   embedding=vec)])

Quickstart (full pipeline; requires the ``perception`` extra)::

    from soma import SomaVideoTracker

    vt = SomaVideoTracker.from_variant("pv", fps=30)
    tracks = vt.update(frame_bgr)
"""
from .api import Detection, SomaVideoTracker, token_from_detection
from .presets import PRESETS, VARIANTS, tracker_config
from .tokens import AnatomicalToken
from .tracker import SomaTracker, Track, TrackerConfig, TrackResult

__version__ = "1.1.0"

__all__ = [
    "AnatomicalToken",
    "Detection",
    "PRESETS",
    "SomaTracker",
    "SomaVideoTracker",
    "Track",
    "TrackResult",
    "TrackerConfig",
    "VARIANTS",
    "token_from_detection",
    "tracker_config",
    "__version__",
]
