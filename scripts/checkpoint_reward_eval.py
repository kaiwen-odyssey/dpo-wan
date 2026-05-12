"""Per-checkpoint reward evaluation for a DPO training run.

Given a training run that saved checkpoints every `interval` steps and a
sidecar `uuid_log.json` recording per-step seen UUIDs:

For each checkpoint:
  1. Load the LoRA at that checkpoint.
  2. Generate one video per unique prompt seen in the last `interval` steps
     (using the *training* split's prompts.csv to map uuid -> text).
  3. Generate one video per eval-set prompt (held-out).
  4. Score all videos with VideoReward on the chosen reward dim.

Outputs (under `runs/<run-name>/checkpoint_eval/`):
  bucket_<step>/scores.csv         - train-set rewards for that bucket
  bucket_<step>/eval_scores.csv    - eval-set rewards at this checkpoint
  summary.json                     - aggregated per-checkpoint means

Skips already-completed buckets so the script can resume after interruption.
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics as stat
from pathlib import Path

import pandas as pd
import torch

from dpo_wan import paths, rewards, sampling
from dpo_wan.utils import save_video_tensor, setup_logging

log = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True, help="e.g. main_MQ_ga4_2ep")
    ap.add_argument("--interval", type=int, default=50)
    ap.add_argument("--max-step", type=int, default=500)
    ap.add_argument("--reward-dim", default="MQ")
    ap.add_argument("--train-split", default="train",
                    help="split name whose prompts.csv maps uuids back to text")
    ap.add_argument("--eval-split", default="eval")
    ap.add_argument("--eval-limit", type=int, default=None,
                    help="cap eval-split rows (e.g. 20 to match current eval)")
    ap.add_argument("--seed", type=int, default=4242,
                    help="seed for *train-side* generation (one rollout/prompt)")
    ap.add_argument("--eval-seeds", type=int, nargs="+", default=[1234, 2345, 3456],
                    help="seeds used for eval-side generation; same across all checkpoints + baseline")
    ap.add_argument("--frame-num", type=int, default=21)
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--reward-num-frames", type=int, default=None,
                    help="frames fed to VideoReward (None => fps=2.0 default)")
    ap.add_argument("--skip-train", action="store_true",
                    help="only eval the holdout set, not the train-bucket prompts")
    ap.add_argument("--include-baseline", action="store_true",
                    help="also generate baseline (no LoRA) at the eval seeds")
    args = ap.parse_args()

    setup_logging()
    run_dir = paths.RUNS_DIR / args.run_name
    out_dir = run_dir / "checkpoint_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load uuid log + prompts
    uuid_log = json.loads((run_dir / "uuid_log.json").read_text())
    step_to_uuids = {entry["step"]: entry["uuids"] for entry in uuid_log}
    prompts_train = pd.read_csv(paths.PROMPTS_DIR / f"{args.train_split}.csv").set_index("uuid")
    eval_df = pd.read_csv(paths.PROMPTS_DIR / f"{args.eval_split}.csv")
    if args.eval_limit is not None:
        eval_df = eval_df.head(args.eval_limit)
    prompts_eval = eval_df.set_index("uuid")

    spec = sampling.SampleSpec(frame_num=args.frame_num, sampling_steps=args.steps)

    log.info("loading Wan2.1-T2V-1.3B (base, no LoRA yet)...")
    wan = sampling.load_wan_t2v(t5_cpu=True)

    # === phase 0: baseline (no LoRA) on eval prompts at the eval seeds ===
    baseline_dir = out_dir / "baseline_eval"
    if args.include_baseline:
        baseline_dir.mkdir(parents=True, exist_ok=True)
        log.info("=== baseline (no LoRA) generation on eval set ===")
        for uid, row in prompts_eval.iterrows():
            for sd in args.eval_seeds:
                v_path = (baseline_dir / f"{uid}__seed{sd}.mp4").resolve()
                if v_path.exists():
                    continue
                video = sampling.generate_video(wan, row["prompt"], seed=sd, spec=spec)
                save_video_tensor(video, v_path, fps=spec.fps)

    # Cache loader for per-checkpoint LoRA swap.
    from peft import PeftModel

    def _detach_lora():
        if isinstance(wan.model, PeftModel):
            wan.model = wan.model.get_base_model()

    def _attach_lora(ck_dir: Path) -> None:
        _detach_lora()
        wan.model.requires_grad_(False)
        wan.model = PeftModel.from_pretrained(wan.model, str(ck_dir))
        wan.model.eval()

    # === phase 1: generation ===
    for step in range(args.interval, args.max_step + 1, args.interval):
        ck_dir = run_dir / f"step_{step:06d}"
        if not ck_dir.exists():
            log.warning("missing checkpoint %s; skipping", ck_dir)
            continue
        bucket_dir = out_dir / f"bucket_{step:06d}"
        bucket_dir.mkdir(parents=True, exist_ok=True)

        # collect prompts in this 50-step bucket
        bucket_uuids = []
        for s in range(step - args.interval + 1, step + 1):
            bucket_uuids.extend(step_to_uuids.get(s, []))
        seen_unique = list(dict.fromkeys(bucket_uuids))  # preserve order, dedupe
        log.info("=== bucket %d (%d steps) — %d uuids, %d unique ===",
                 step, args.interval, len(bucket_uuids), len(seen_unique))

        _attach_lora(ck_dir)

        # train-side generation
        if not args.skip_train:
            train_video_dir = bucket_dir / "videos_train"
            train_video_dir.mkdir(parents=True, exist_ok=True)
            for uid in seen_unique:
                if uid not in prompts_train.index:
                    continue
                v_path = (train_video_dir / f"{uid}.mp4").resolve()
                if v_path.exists():
                    continue
                video = sampling.generate_video(
                    wan, prompts_train.loc[uid, "prompt"], seed=args.seed, spec=spec,
                )
                save_video_tensor(video, v_path, fps=spec.fps)

        # eval-side generation (always, multi-seed)
        eval_video_dir = bucket_dir / "videos_eval"
        eval_video_dir.mkdir(parents=True, exist_ok=True)
        for uid, row in prompts_eval.iterrows():
            for sd in args.eval_seeds:
                v_path = (eval_video_dir / f"{uid}__seed{sd}.mp4").resolve()
                if v_path.exists():
                    continue
                video = sampling.generate_video(wan, row["prompt"], seed=sd, spec=spec)
                save_video_tensor(video, v_path, fps=spec.fps)

    # release Wan, load VideoReward
    _detach_lora()
    del wan
    torch.cuda.empty_cache()

    log.info("loading VideoReward scorer...")
    scorer = rewards.VideoRewardScorer()

    # === phase 2: scoring ===
    summary: list[dict] = []
    train_running_scores: list[float] = []  # for cumulative moving mean
    for step in range(args.interval, args.max_step + 1, args.interval):
        bucket_dir = out_dir / f"bucket_{step:06d}"
        if not bucket_dir.exists():
            continue
        train_vd = bucket_dir / "videos_train"
        eval_vd  = bucket_dir / "videos_eval"

        # score train-side
        train_scores_path = bucket_dir / "scores_train.csv"
        if not args.skip_train and train_vd.exists() and not train_scores_path.exists():
            mp4s = sorted(train_vd.glob("*.mp4"))
            ps = [(p.stem, prompts_train.loc[p.stem, "prompt"]) for p in mp4s
                  if p.stem in prompts_train.index]
            rows = []
            BATCH = 4
            for i in range(0, len(ps), BATCH):
                chunk = ps[i:i+BATCH]
                paths_ = [str(train_vd / f"{u}.mp4") for u, _ in chunk]
                results = scorer.score(paths_, [pr for _, pr in chunk],
                                        num_frames=args.reward_num_frames, use_norm=True)
                for (u, _), r in zip(chunk, results):
                    rows.append({"uuid": u, "VQ": r.VQ, "MQ": r.MQ, "TA": r.TA})
            pd.DataFrame(rows).to_csv(train_scores_path, index=False)
        train_rewards_this_bucket = []
        if train_scores_path.exists():
            df = pd.read_csv(train_scores_path)
            train_rewards_this_bucket = df[args.reward_dim].tolist()

        # score eval-side (multi-seed: filename is "<uuid>__seed<n>.mp4")
        eval_scores_path = bucket_dir / "scores_eval.csv"
        if not eval_scores_path.exists() and eval_vd.exists():
            mp4s = sorted(eval_vd.glob("*.mp4"))
            rows = []
            todo = []
            for p in mp4s:
                stem = p.stem
                if "__seed" in stem:
                    uid, _, sd_str = stem.partition("__seed")
                    sd = int(sd_str)
                else:
                    uid, sd = stem, 0
                if uid not in prompts_eval.index:
                    continue
                todo.append((p, uid, sd, prompts_eval.loc[uid, "prompt"]))
            BATCH = 4
            for i in range(0, len(todo), BATCH):
                chunk = todo[i:i+BATCH]
                results = scorer.score([str(p) for p, _, _, _ in chunk],
                                        [pr for _, _, _, pr in chunk],
                                        num_frames=args.reward_num_frames, use_norm=True)
                for (_, u, sd, _), r in zip(chunk, results):
                    rows.append({"uuid": u, "seed": sd, "VQ": r.VQ, "MQ": r.MQ, "TA": r.TA})
            pd.DataFrame(rows).to_csv(eval_scores_path, index=False)
        eval_df = pd.read_csv(eval_scores_path) if eval_scores_path.exists() else pd.DataFrame()
        # per-prompt mean across seeds, then mean across prompts
        if not eval_df.empty:
            per_prompt = eval_df.groupby("uuid")[args.reward_dim].mean()
            eval_mean = float(per_prompt.mean())
            eval_n_prompts = int(per_prompt.shape[0])
        else:
            eval_mean = None
            eval_n_prompts = 0

        # extend running train-side list
        train_running_scores.extend(train_rewards_this_bucket)

        summary.append({
            "step": step,
            "n_train_uuids_this_bucket": len(train_rewards_this_bucket),
            f"train_{args.reward_dim}_mean_this_bucket":
                stat.mean(train_rewards_this_bucket) if train_rewards_this_bucket else None,
            f"train_{args.reward_dim}_mean_cumulative":
                stat.mean(train_running_scores) if train_running_scores else None,
            f"eval_{args.reward_dim}_mean":   eval_mean,
            "n_eval_prompts":                 eval_n_prompts,
            "n_eval_seeds":                   len(args.eval_seeds),
        })
        log.info("step %d | bucket_mean=%.3f | cumul_mean=%.3f | eval_mean=%.3f",
                 step,
                 summary[-1][f"train_{args.reward_dim}_mean_this_bucket"] or float('nan'),
                 summary[-1][f"train_{args.reward_dim}_mean_cumulative"] or float('nan'),
                 summary[-1][f"eval_{args.reward_dim}_mean"]              or float('nan'))

    # === phase 3: baseline scoring (matched seeds) ===
    baseline_summary = None
    if args.include_baseline and baseline_dir.exists():
        baseline_scores_path = out_dir / "baseline_eval_scores.csv"
        if not baseline_scores_path.exists():
            mp4s = sorted(baseline_dir.glob("*.mp4"))
            rows = []
            todo = []
            for p in mp4s:
                stem = p.stem
                uid, _, sd_str = stem.partition("__seed")
                if not sd_str:
                    continue
                sd = int(sd_str)
                if uid not in prompts_eval.index:
                    continue
                todo.append((p, uid, sd, prompts_eval.loc[uid, "prompt"]))
            BATCH = 4
            for i in range(0, len(todo), BATCH):
                chunk = todo[i:i+BATCH]
                results = scorer.score([str(p) for p, _, _, _ in chunk],
                                        [pr for _, _, _, pr in chunk],
                                        num_frames=args.reward_num_frames, use_norm=True)
                for (_, u, sd, _), r in zip(chunk, results):
                    rows.append({"uuid": u, "seed": sd, "VQ": r.VQ, "MQ": r.MQ, "TA": r.TA})
            pd.DataFrame(rows).to_csv(baseline_scores_path, index=False)
        bdf = pd.read_csv(baseline_scores_path)
        per_prompt = bdf.groupby("uuid")[args.reward_dim].mean()
        baseline_summary = {
            f"baseline_{args.reward_dim}_mean": float(per_prompt.mean()),
            "n_prompts": int(per_prompt.shape[0]),
            "n_seeds":   int(bdf["seed"].nunique()),
            "seeds":     sorted(bdf["seed"].unique().tolist()),
        }
        log.info("baseline (no LoRA) eval mean %s = %.3f  (over %d prompts, %d seeds)",
                 args.reward_dim, baseline_summary[f"baseline_{args.reward_dim}_mean"],
                 baseline_summary["n_prompts"], baseline_summary["n_seeds"])

    out_payload = {"per_checkpoint": summary, "baseline": baseline_summary,
                   "eval_seeds": args.eval_seeds}
    (out_dir / "summary.json").write_text(json.dumps(out_payload, indent=2))
    log.info("written summary: %s", out_dir / "summary.json")


if __name__ == "__main__":
    main()
