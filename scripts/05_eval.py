"""Evaluate baseline + LoRA-DPO checkpoints on the holdout split."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from dpo_wan import eval as dpo_eval
from dpo_wan import paths, sampling, rewards
from dpo_wan.utils import setup_logging

log = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--runs", nargs="+",
        help="space-separated NAME=PATH pairs.  PATH=baseline means no LoRA.",
        required=True,
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="eval")
    ap.add_argument("--frame-num", type=int, default=33)
    ap.add_argument("--steps", type=int, default=25)
    args = ap.parse_args()

    setup_logging()
    paths.ensure_dirs()

    runs: dict[str, str | None] = {}
    for r in args.runs:
        name, path = r.split("=", 1)
        runs[name] = None if path.lower() == "baseline" else path

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
