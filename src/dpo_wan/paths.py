"""Repo-relative paths used throughout the pipeline."""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_DIR = REPO_ROOT / "external"
WAN_REF_DIR = EXTERNAL_DIR / "wan"
VIDEOALIGN_DIR = EXTERNAL_DIR / "VideoAlign"

DATA_DIR = REPO_ROOT / "data"
PROMPTS_DIR = DATA_DIR / "prompts"
VIDEOS_DIR = DATA_DIR / "videos"
LATENTS_DIR = DATA_DIR / "latents"
PREFS_DIR = DATA_DIR / "preferences"

RUNS_DIR = REPO_ROOT / "runs"
CONFIGS_DIR = REPO_ROOT / "configs"

WAN_CHECKPOINT_DIR = Path(
    os.environ.get(
        "WAN_CHECKPOINT_DIR",
        "/home/kaiwen/dev/LongLive/wan_models/Wan2.1-T2V-1.3B",
    )
)

VIDEOREWARD_CHECKPOINT_DIR = Path(
    os.environ.get(
        "VIDEOREWARD_CHECKPOINT_DIR",
        str(EXTERNAL_DIR / "VideoAlign" / "checkpoints" / "VideoReward"),
    )
)


def add_external_to_sys_path() -> None:
    """Make the vendored Wan reference and the cloned VideoAlign repo importable."""
    if str(EXTERNAL_DIR) not in sys.path:
        sys.path.insert(0, str(EXTERNAL_DIR))
    if str(VIDEOALIGN_DIR) not in sys.path:
        sys.path.insert(0, str(VIDEOALIGN_DIR))


def ensure_dirs() -> None:
    for p in [
        DATA_DIR, PROMPTS_DIR, VIDEOS_DIR, LATENTS_DIR, PREFS_DIR,
        RUNS_DIR, CONFIGS_DIR,
    ]:
        p.mkdir(parents=True, exist_ok=True)
