"""Build per-reward preference datasets from generated candidates + scores.

For each reward dim (MQ, VQ, TA) emit a directory under data/preferences/<split>/<dim>/
with chosen/rejected latent symlinks and a Parquet index.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from dpo_wan import paths, preferences
from dpo_wan.utils import setup_logging

log = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--reward-dims", nargs="+", default=["MQ", "VQ", "TA"])
    args = ap.parse_args()

    setup_logging()
    paths.ensure_dirs()

    df_prompts = pd.read_csv(paths.PROMPTS_DIR / f"{args.split}.csv")
    df_scores = pd.read_csv(paths.DATA_DIR / "scores" / f"{args.split}.csv")
    candidates_dir = paths.VIDEOS_DIR / args.split
    text_dir = paths.DATA_DIR / "text" / args.split
    out_root = paths.PREFS_DIR / args.split

    for dim in args.reward_dims:
        preferences.build_preference_set(
            df_prompts=df_prompts,
            df_scores=df_scores,
            candidates_dir=candidates_dir,
            text_dir=text_dir,
            out_root=out_root,
            reward_dim=dim,
            keep_videos=True,
        )


if __name__ == "__main__":
    main()
