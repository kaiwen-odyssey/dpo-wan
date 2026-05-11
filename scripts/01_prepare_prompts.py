"""Sample VidProM into train / ablation / eval splits.

Tries (in order):
  1. Local cached parquet under data/prompts/VidProM_unique.csv
  2. `datasets.load_dataset('WenhaoWang/VidProM', streaming=True)`
  3. Falls back to a small built-in seed list (so smoke tests still run offline).

The output `prompts/{train,ablation,eval}.csv` files contain `uuid`, `prompt`.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from dpo_wan import paths
from dpo_wan.utils import setup_logging

log = logging.getLogger(__name__)


SEED_FALLBACK = [
    "A golden retriever puppy chasing a butterfly in a sunlit meadow",
    "A vintage red sailboat gliding across a calm turquoise lagoon at sunrise",
    "Time-lapse of cherry blossoms blooming in a Japanese garden",
    "A chef plating a colorful sushi roll on a black slate platter",
    "A SpaceX rocket lifting off from a launchpad with smoke billowing below",
    "An astronaut bouncing across the lunar surface, Earth visible in the sky",
    "A snow leopard prowling across a rocky Himalayan ridge at dusk",
    "A street violinist performing under a glowing streetlamp on a rainy Paris night",
    "A drone shot tracking a red Ferrari driving down a coastal mountain road",
    "Macro footage of a honeybee collecting pollen from a sunflower",
    "A wizard casting a glowing blue spell in a candlelit medieval library",
    "An origami crane unfolding itself against a black background",
    "A bullet train speeding across a snowy Japanese countryside",
    "Underwater shot of a coral reef teeming with tropical fish",
    "A cat knocking a crystal glass off a table in slow motion",
    "Volumetric clouds rolling over a green mountain valley, time-lapse",
    "A blacksmith hammering glowing red steel into a sword",
    "Hot air balloons floating above the canyons of Cappadocia at dawn",
    "A child blowing a giant soap bubble in a city park",
    "A fluffy white puppy splashing in a puddle after the rain",
]


def load_vidprom(n_total: int) -> pd.DataFrame:
    cache_csv = paths.PROMPTS_DIR / "VidProM_unique.csv"
    if cache_csv.exists():
        log.info("loading cached VidProM CSV: %s", cache_csv)
        df = pd.read_csv(cache_csv, usecols=["uuid", "prompt"]).dropna()
        return df

    log.info("attempting to stream VidProM via datasets...")
    try:
        from datasets import load_dataset
        stream = load_dataset("WenhaoWang/VidProM", split="train", streaming=True)
        rows: list[dict] = []
        for i, ex in enumerate(stream):
            if i >= n_total:
                break
            rows.append({"uuid": ex.get("uuid", f"vp{i:08d}"), "prompt": ex["prompt"]})
        if rows:
            df = pd.DataFrame(rows).dropna()
            cache_csv.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(cache_csv, index=False)
            return df
    except Exception as exc:                                                  # noqa: BLE001
        log.warning("VidProM streaming failed: %s", exc)

    log.warning("falling back to built-in 20-prompt seed list (offline mode)")
    rows = [{"uuid": f"seed{i:04d}", "prompt": p} for i, p in enumerate(SEED_FALLBACK)]
    return pd.DataFrame(rows)


def stratified_split(
    df: pd.DataFrame, n_train: int, n_ablation: int, n_eval: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    total = n_train + n_ablation + n_eval
    if len(df) < total:
        log.warning(
            "only %d prompts available; downscaling: train=%d ablation=%d eval=%d",
            len(df), n_train, n_ablation, n_eval,
        )
        scale = len(df) / total
        n_train    = max(1, int(n_train    * scale))
        n_ablation = max(1, int(n_ablation * scale))
        n_eval     = max(1, int(n_eval     * scale))

    eval_df     = df.iloc[:n_eval].reset_index(drop=True)
    ablation_df = df.iloc[n_eval : n_eval + n_ablation].reset_index(drop=True)
    train_df    = df.iloc[n_eval + n_ablation : n_eval + n_ablation + n_train].reset_index(drop=True)
    return train_df, ablation_df, eval_df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train",    type=int, default=10000)
    ap.add_argument("--n-ablation", type=int, default=1000)
    ap.add_argument("--n-eval",     type=int, default=100)
    ap.add_argument("--seed",       type=int, default=0)
    args = ap.parse_args()

    setup_logging()
    paths.ensure_dirs()

    df = load_vidprom(n_total=args.n_train + args.n_ablation + args.n_eval)
    log.info("loaded %d total prompts", len(df))

    train_df, ablation_df, eval_df = stratified_split(
        df, args.n_train, args.n_ablation, args.n_eval, args.seed
    )

    out_train    = paths.PROMPTS_DIR / "train.csv"
    out_ablation = paths.PROMPTS_DIR / "ablation.csv"
    out_eval     = paths.PROMPTS_DIR / "eval.csv"
    train_df.to_csv(out_train, index=False)
    ablation_df.to_csv(out_ablation, index=False)
    eval_df.to_csv(out_eval, index=False)

    log.info("train=%d ablation=%d eval=%d", len(train_df), len(ablation_df), len(eval_df))
    log.info("written to %s", paths.PROMPTS_DIR)


if __name__ == "__main__":
    main()
