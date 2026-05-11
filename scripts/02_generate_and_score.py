"""Generate K candidates per prompt with Wan2.1, then score with VideoReward.

Outputs:
  data/videos/<split>/<uuid>_k{k}.mp4
  data/latents/<split>/<uuid>_k{k}.latent.pt
  data/text/<split>/<uuid>.pt
  data/scores/<split>.csv  (one row per (uuid, k) with VQ, MQ, TA)
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from dpo_wan import paths, sampling, preferences, rewards
from dpo_wan.utils import setup_logging

log = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "ablation", "eval"], required=True)
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap number of prompts (handy for smoke tests)")
    ap.add_argument("--frame-num", type=int, default=33)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--score-only", action="store_true",
                    help="skip generation, just (re)compute the scores CSV")
    args = ap.parse_args()

    setup_logging()
    paths.ensure_dirs()

    df = pd.read_csv(paths.PROMPTS_DIR / f"{args.split}.csv")
    if args.limit is not None:
        df = df.head(args.limit)
    log.info("split=%s prompts=%d", args.split, len(df))

    candidates_dir = paths.VIDEOS_DIR / args.split
    text_dir = paths.DATA_DIR / "text" / args.split
    scores_csv = paths.DATA_DIR / "scores" / f"{args.split}.csv"

    spec = sampling.SampleSpec(
        size=(args.width, args.height),
        frame_num=args.frame_num,
        sampling_steps=args.steps,
    )

    if not args.score_only:
        log.info("loading Wan2.1-T2V-1.3B...")
        wan = sampling.load_wan_t2v()
        preferences.generate_candidates_for_prompts(
            wan=wan,
            df_prompts=df,
            candidates_dir=candidates_dir,
            K=args.K,
            base_seed=4242,
            spec=spec,
            text_dir=text_dir,
        )
        del wan

    log.info("loading VideoReward scorer...")
    scorer = rewards.VideoRewardScorer()
    df_scores = preferences.score_candidates(
        df_prompts=df,
        candidates_dir=candidates_dir,
        K=args.K,
        scorer=scorer,
        out_csv=scores_csv,
    )
    log.info("scores written: %s (%d rows)", scores_csv, len(df_scores))


if __name__ == "__main__":
    main()
