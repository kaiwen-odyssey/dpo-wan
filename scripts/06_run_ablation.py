"""Hyperparameter ablation on the small ablation split.

Sweeps batch-size, learning-rate and DPO-beta on a single reward dim, then
ranks configurations by *win-rate vs. baseline* on the eval holdout.
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
from pathlib import Path

import pandas as pd

from dpo_wan import paths, sampling, training, rewards
from dpo_wan import eval as dpo_eval
from dpo_wan.utils import setup_logging

log = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reward-dim", default="MQ")
    ap.add_argument("--ablation-split", default="ablation")
    ap.add_argument("--eval-split", default="eval")
    ap.add_argument("--max-steps", type=int, default=120)
    ap.add_argument("--lrs",   nargs="+", type=float, default=[1e-5, 5e-5, 1e-4])
    ap.add_argument("--betas", nargs="+", type=float, default=[100.0, 500.0, 2500.0])
    ap.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2])
    args = ap.parse_args()

    setup_logging()
    paths.ensure_dirs()

    pref_root = paths.PREFS_DIR / args.ablation_split / args.reward_dim
    if not pref_root.exists():
        raise FileNotFoundError(f"missing {pref_root}; run 03_build_preferences.py")

    log.info("loading Wan2.1-T2V-1.3B...")
    wan = sampling.load_wan_t2v(t5_cpu=True)

    rows: list[dict] = []
    for lr, beta, bs in itertools.product(args.lrs, args.betas, args.batch_sizes):
        run_name = f"abl_{args.reward_dim}_lr{lr:g}_b{beta:g}_bs{bs}"
        cfg = training.TrainConfig(
            pref_root=str(pref_root),
            output_dir=str(paths.RUNS_DIR / run_name),
            reward_dim=args.reward_dim,
            learning_rate=lr,
            dpo_beta=beta,
            batch_size=bs,
            gradient_accumulation_steps=1,           # fast path: no accumulation
            max_steps=args.max_steps,
            grad_clip=0.1,                           # tighter to stop spikes
            warmup_steps=10,                         # let optimizer settle
            wandb_project="dpo-wan-ablation",
            wandb_run_name=run_name,
            wandb_mode="offline",
            wandb_tags=("stability",),
        )
        log.info("=== ablation cfg %s ===", run_name)
        ckpt = training.train_dpo(cfg, wan)
        rows.append({"run": run_name, "lr": lr, "beta": beta, "batch_size": bs, "ckpt": str(ckpt)})

    out_df = pd.DataFrame(rows)
    abl_dir = paths.RUNS_DIR / "ablation_summary"
    abl_dir.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(abl_dir / "configs.csv", index=False)
    log.info("ablation runs written to %s", abl_dir)


if __name__ == "__main__":
    main()
