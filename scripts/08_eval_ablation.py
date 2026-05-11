"""Evaluate hyperparameter-ablation LoRAs on the small holdout split.

After ``06_run_ablation.py`` produces a directory tree under runs/abl_*/final, this
script generates ${eval} videos with each LoRA, scores them, and writes a
single summary CSV used by 07_render_paper.py.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from dpo_wan import paths, sampling, rewards
from dpo_wan import eval as dpo_eval
from dpo_wan.utils import setup_logging

log = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(paths.RUNS_DIR / "ablation_eval"))
    ap.add_argument("--split", default="eval")
    ap.add_argument("--frame-num", type=int, default=21)
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap number of ablation runs to evaluate")
    args = ap.parse_args()

    setup_logging()
    paths.ensure_dirs()

    runs: dict[str, str | None] = {"baseline": None}
    for ckpt in sorted(paths.RUNS_DIR.glob("abl_*/final")):
        run = ckpt.parts[-2]
        runs[run] = str(ckpt)
        if args.limit and len(runs) > args.limit:
            break
    log.info("ablation runs to evaluate: %d", len(runs))

    spec = sampling.SampleSpec(frame_num=args.frame_num, sampling_steps=args.steps)
    eval_spec = dpo_eval.EvalSpec(
        holdout_csv=str(paths.PROMPTS_DIR / f"{args.split}.csv"),
        output_dir=args.out,
        runs=runs,
        sample=spec,
    )

    log.info("loading Wan2.1-T2V-1.3B...")
    wan = sampling.load_wan_t2v(t5_cpu=True)
    log.info("loading VideoReward scorer...")
    scorer = rewards.VideoRewardScorer()
    out_dir = dpo_eval.run_eval(eval_spec, wan, scorer)
    log.info("done: %s", out_dir)


if __name__ == "__main__":
    main()
