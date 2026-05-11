"""End-to-end smoke test on a tiny number of prompts.

Walks the full pipeline:
  prompts -> 2 candidates each -> score -> build pair -> 1 DPO step -> sanity.

Useful for catching shape / dtype / device errors before launching the real run.
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

import torch
import pandas as pd

from dpo_wan import paths, sampling, rewards, preferences, training
from dpo_wan.utils import set_seed, setup_logging

log = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-prompts", type=int, default=3)
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--frame-num", type=int, default=17)  # smallest legal: 4n+1
    ap.add_argument("--steps", type=int, default=10)
    args = ap.parse_args()

    setup_logging()
    set_seed(0)
    paths.ensure_dirs()

    smoke_dir = paths.DATA_DIR / "smoke"
    if smoke_dir.exists():
        shutil.rmtree(smoke_dir)
    smoke_dir.mkdir(parents=True, exist_ok=True)
    cand_dir = smoke_dir / "videos"
    text_dir = smoke_dir / "text"
    pref_dir = smoke_dir / "prefs"

    df = pd.DataFrame({
        "uuid": [f"smoke{i:02d}" for i in range(args.n_prompts)],
        "prompt": [
            "A golden retriever puppy chasing a butterfly in a sunlit meadow",
            "A drone shot of a red Ferrari driving down a coastal mountain road",
            "Time-lapse of cherry blossoms blooming in a Japanese garden",
        ][: args.n_prompts],
    })
    spec = sampling.SampleSpec(frame_num=args.frame_num, sampling_steps=args.steps)

    log.info("=== loading Wan ===")
    wan = sampling.load_wan_t2v(t5_cpu=True)
    log.info("=== generating %d * K=%d candidates ===", args.n_prompts, args.K)
    preferences.generate_candidates_for_prompts(
        wan=wan, df_prompts=df, candidates_dir=cand_dir, K=args.K,
        base_seed=4242, spec=spec, text_dir=text_dir,
    )

    scorer = rewards.VideoRewardScorer()
    df_scores = preferences.score_candidates(
        df_prompts=df, candidates_dir=cand_dir, K=args.K,
        scorer=scorer, out_csv=smoke_dir / "scores.csv",
    )
    log.info("scores:\n%s", df_scores)

    out_dir = preferences.build_preference_set(
        df_prompts=df, df_scores=df_scores,
        candidates_dir=cand_dir, text_dir=text_dir,
        out_root=pref_dir, reward_dim="MQ", keep_videos=True,
    )

    # Free VideoReward GPU memory before DPO training
    del scorer
    torch.cuda.empty_cache()
    import gc; gc.collect()
    torch.cuda.empty_cache()

    log.info("=== running 2 DPO steps ===")
    cfg = training.TrainConfig(
        pref_root=str(out_dir),
        output_dir=str(smoke_dir / "run"),
        reward_dim="MQ",
        learning_rate=1e-4,
        max_steps=2,
        gradient_accumulation_steps=1,
        log_every=1,
        save_every=2,
    )
    final = training.train_dpo(cfg, wan)
    log.info("smoke OK -> %s", final)


if __name__ == "__main__":
    main()
