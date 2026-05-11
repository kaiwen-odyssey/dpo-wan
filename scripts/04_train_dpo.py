"""Train one Diffusion-DPO LoRA on the chosen reward dim."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from dpo_wan import paths, sampling, training
from dpo_wan.utils import set_seed, setup_logging

log = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reward-dim", required=True, choices=["MQ", "VQ", "TA"])
    ap.add_argument("--split", default="train")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--beta", type=float, default=500.0)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wandb-project", default="dpo-wan")
    ap.add_argument("--wandb-mode", default="offline",
                    choices=["online", "offline", "disabled"])
    ap.add_argument("--pref-root", default=None,
                    help="override preference dir (e.g. for margin-filtered set)")
    ap.add_argument("--lambda-sft", type=float, default=0.0,
                    help="SFT anchor weight on chosen (DPO+SFT)")
    args = ap.parse_args()

    setup_logging()
    set_seed(args.seed)
    paths.ensure_dirs()

    if args.pref_root:
        from pathlib import Path
        pref_root = Path(args.pref_root)
    else:
        pref_root = paths.PREFS_DIR / args.split / args.reward_dim
    if not pref_root.exists():
        raise FileNotFoundError(
            f"preference dir missing: {pref_root}\n"
            "run scripts/03_build_preferences.py first"
        )

    output_dir = paths.RUNS_DIR / args.run_name
    cfg = training.TrainConfig(
        pref_root=str(pref_root),
        output_dir=str(output_dir),
        reward_dim=args.reward_dim,
        learning_rate=args.lr,
        dpo_beta=args.beta,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_steps=args.max_steps,
        lora_rank=args.lora_rank,
        seed=args.seed,
        lambda_sft=args.lambda_sft,
        wandb_project=args.wandb_project,
        wandb_run_name=args.run_name,
        wandb_mode=args.wandb_mode,
    )
    log.info("loading Wan2.1-T2V-1.3B (T5 on CPU)...")
    wan = sampling.load_wan_t2v(t5_cpu=True)

    final = training.train_dpo(cfg, wan)
    log.info("final LoRA: %s", final)


if __name__ == "__main__":
    main()
