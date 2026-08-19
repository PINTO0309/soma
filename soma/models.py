"""Model download helper: fetch the released ONNX/TFLite exports.

All shipped models live in the GitHub release tagged ``models``
(https://github.com/PINTO0309/soma/releases/tag/models). stdlib-only.

    from soma import models
    path = models.download("yolov9_e_wholebody28_refine_Nx3HxW.onnx")
    reid = models.download("personvit_vits16_ain_unified_aug_n.onnx")
"""
from __future__ import annotations

import json
import os
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

RELEASE_TAG = "models"
RELEASE_DOWNLOAD_BASE = (
    f"https://github.com/PINTO0309/soma/releases/download/{RELEASE_TAG}/"
)
RELEASE_API_URL = (
    f"https://api.github.com/repos/PINTO0309/soma/releases/tags/{RELEASE_TAG}"
)
DEFAULT_DIR = "models"


def list_assets(timeout: float = 30.0) -> list[str]:
    """Names of every model asset published in the ``models`` release."""
    req = urllib.request.Request(
        RELEASE_API_URL, headers={"Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.load(resp)
    return sorted(asset["name"] for asset in payload.get("assets", []))


def download(name: str, dest_dir: "str | os.PathLike" = DEFAULT_DIR,
             overwrite: bool = False, timeout: float = 600.0) -> Path:
    """Download one released model into ``dest_dir`` (skips existing files).

    Returns the local path — pass it straight to ``SomaVideoTracker`` /
    ``Perception``.
    """
    dest = Path(dest_dir) / name
    if dest.exists() and not overwrite:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = RELEASE_DOWNLOAD_BASE + urllib.parse.quote(name)
    fd, tmp = tempfile.mkstemp(prefix=f".{name}.", suffix=".part",
                               dir=dest.parent)
    os.close(fd)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp, \
                open(tmp, "wb") as out:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
        os.replace(tmp, dest)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise
    return dest
