"""Re-generate training-set videos with a trained LoRA and score them.

Used to answer "did the trained policy actually fit the training set?" --
the canonical sanity check that's distinct from holdout eval.

For each (sub-sampled) training prompt we:
  1. Generate one video with the LoRA attached, at the *same* spec used for
     the original baseline rollouts (832x480 / 21 frames / 15 steps).
  2. Score it with VideoReward.
  3. Pair it against the baseline K=2 candidates we already have on disk
     (`data/scores/train.csv`) for that prompt and compute three "before /
     after" deltas: vs the mean, min, max of the baseline pair.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import torch

from dpo_wan import paths, rewards, sampling
from dpo_wan.utils import save_video_tensor, setup_logging

log = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", default=None,
                    help="path to LoRA final dir (omit / set 'baseline' to use the bare Wan model)")
    ap.add_argument("--name", required=True, help="run name for output dirs")
    ap.add_argument("--n-prompts", type=int, default=100)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--uuids-csv", default=None,
                    help="optional CSV of UUIDs to use; if set, take first n-prompts of these")
    ap.add_argument("--frame-num", type=int, default=21)
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--reward-dim", default="MQ")
    args = ap.parse_args()

    setup_logging()
    out_dir = paths.RUNS_DIR / "train_set_rerun" / args.name
    (out_dir / "videos").mkdir(parents=True, exist_ok=True)

    prompts = pd.read_csv(paths.PROMPTS_DIR / "train.csv")
    if args.uuids_csv:
        seen = pd.read_csv(args.uuids_csv)["uuid"].tolist()
        sub = prompts[prompts["uuid"].isin(seen)].copy()
        # preserve "seen" order
        order = {u: i for i, u in enumerate(seen)}
        sub["__order"] = sub["uuid"].map(order)
        sub = sub.sort_values("__order").drop(columns="__order").head(args.n_prompts).reset_index(drop=True)
        log.info("using first %d uuids from seen-during-training list (%d unique seen total)",
                 len(sub), len(seen))
    else:
        sub = prompts.sample(n=args.n_prompts, random_state=args.seed).reset_index(drop=True)
        log.info("uniformly subsampled %d/%d training prompts", len(sub), len(prompts))

    # Load baseline scores (cached) for the same prompts.
    train_scores = pd.read_csv(paths.DATA_DIR / "scores" / "train.csv")
    train_scores = train_scores[train_scores["uuid"].isin(sub["uuid"])].copy()
    baseline_per_prompt = train_scores.groupby("uuid")[args.reward_dim].agg(
        ["mean", "min", "max"]
    ).rename(columns={"mean": "base_mean", "min": "base_min", "max": "base_max"})

    use_lora = args.lora and args.lora.lower() != "baseline"
    log.info("loading Wan2.1-T2V-1.3B%s...", " + LoRA" if use_lora else " (baseline)")
    wan = sampling.load_wan_t2v(t5_cpu=True)
    if use_lora:
        from peft import PeftModel
        wan.model.requires_grad_(False)
        wan.model = PeftModel.from_pretrained(wan.model, args.lora)
        wan.model.eval()

    spec = sampling.SampleSpec(frame_num=args.frame_num, sampling_steps=args.steps)
    rows = []
    for _, row in sub.iterrows():
        uid = str(row["uuid"]); prompt = str(row["prompt"])
        v_path = (out_dir / "videos" / f"{uid}.mp4").resolve()
        if not v_path.exists():
            video = sampling.generate_video(wan, prompt, seed=args.seed, spec=spec)
            save_video_tensor(video, v_path, fps=spec.fps)
        rows.append({"uuid": uid, "prompt": prompt, "video": str(v_path)})

    log.info("loading VideoReward...")
    scorer = rewards.VideoRewardScorer()
    score_rows = []
    BATCH = 4
    for i in range(0, len(rows), BATCH):
        chunk = rows[i:i+BATCH]
        results = scorer.score([r["video"] for r in chunk],
                                [r["prompt"] for r in chunk],
                                num_frames=16, use_norm=True)
        for r, s in zip(chunk, results):
            score_rows.append({"uuid": r["uuid"], "VQ": s.VQ, "MQ": s.MQ, "TA": s.TA})

    df = pd.DataFrame(score_rows).set_index("uuid")
    df = df.join(baseline_per_prompt, how="inner")
    df[f"delta_{args.reward_dim}_vs_base_mean"] = df[args.reward_dim] - df["base_mean"]
    df[f"delta_{args.reward_dim}_vs_base_max"]  = df[args.reward_dim] - df["base_max"]
    df[f"delta_{args.reward_dim}_vs_base_min"]  = df[args.reward_dim] - df["base_min"]

    df.to_csv(out_dir / "scores.csv")
    summary = {
        "n_prompts": int(len(df)),
        "reward_dim": args.reward_dim,
        f"lora_mean_{args.reward_dim}": float(df[args.reward_dim].mean()),
        f"baseline_pairmean_{args.reward_dim}": float(df["base_mean"].mean()),
        f"baseline_chosen_{args.reward_dim}":   float(df["base_max"].mean()),
        f"baseline_rejected_{args.reward_dim}": float(df["base_min"].mean()),
        f"delta_vs_base_mean":   float(df[f"delta_{args.reward_dim}_vs_base_mean"].mean()),
        f"delta_vs_base_chosen": float(df[f"delta_{args.reward_dim}_vs_base_max"].mean()),
        f"win_vs_base_mean":   float((df[f"delta_{args.reward_dim}_vs_base_mean"] > 0).mean()),
        f"win_vs_base_chosen": float((df[f"delta_{args.reward_dim}_vs_base_max"] > 0).mean()),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info("summary: %s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
