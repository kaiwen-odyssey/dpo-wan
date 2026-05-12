"""Plot per-checkpoint eval reward + moving train-bucket reward + reference baseline.

Reads `runs/<run>/checkpoint_eval/summary.json` (produced by
`checkpoint_reward_eval.py`) and writes a PNG with:
  - blue curve: mean eval reward at each checkpoint (online policy)
  - orange curve: mean train-bucket reward at each checkpoint (online policy
    on the 20 prompts seen in the preceding 20-step window)
  - dashed horizontal: reference-model (no LoRA) eval mean reward
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dpo_wan import paths
from dpo_wan.utils import setup_logging

log = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--reward-dim", default="MQ")
    ap.add_argument("--out", default=None,
                    help="output PNG path; default = runs/<run>/checkpoint_eval/curve.png")
    args = ap.parse_args()

    setup_logging()
    run_dir = paths.RUNS_DIR / args.run_name
    summary_path = run_dir / "checkpoint_eval" / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"no summary at {summary_path}; run checkpoint_reward_eval.py first")

    data = json.loads(summary_path.read_text())
    per_ckpt = data["per_checkpoint"]
    baseline = data.get("baseline") or {}

    steps = [r["step"] for r in per_ckpt]
    eval_means = [r.get(f"eval_{args.reward_dim}_mean") for r in per_ckpt]
    train_bucket_means = [r.get(f"train_{args.reward_dim}_mean_this_bucket") for r in per_ckpt]
    baseline_mean = baseline.get(f"baseline_{args.reward_dim}_mean")

    fig, ax = plt.subplots(figsize=(8, 5))
    if any(v is not None for v in eval_means):
        ax.plot(steps, eval_means, "o-", color="C0",
                label=f"eval mean {args.reward_dim} (online policy, 20 prompts)")
    if any(v is not None for v in train_bucket_means):
        ax.plot(steps, train_bucket_means, "s--", color="C1",
                label=f"train-bucket mean {args.reward_dim} (last 20 prompts, online policy)")
    if baseline_mean is not None:
        ax.axhline(baseline_mean, linestyle=":", color="gray",
                   label=f"reference model (no LoRA) eval mean = {baseline_mean:.3f}")
    ax.set_xlabel("training step")
    ax.set_ylabel(f"mean {args.reward_dim} reward")
    ax.set_title(f"{args.run_name}: per-checkpoint reward")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out_path = Path(args.out) if args.out else (run_dir / "checkpoint_eval" / "curve.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    log.info("wrote %s", out_path)


if __name__ == "__main__":
    main()
