"""Evaluation harness on the holdout split.

For each prompt:
  * generate one video with the baseline (no LoRA) and one with each LoRA
  * score all videos with the VideoReward model on all 3 dimensions (VQ, MQ, TA)
  * compute per-dimension reward delta and pairwise win rate against baseline
"""
from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import torch

from . import paths, sampling
from .rewards import VideoRewardScorer
from .utils import save_video_tensor

log = logging.getLogger(__name__)


@dataclasses.dataclass
class EvalSpec:
    holdout_csv: str
    output_dir: str
    runs: dict[str, str | None]               # name -> LoRA path (None = baseline)
    sample: sampling.SampleSpec = dataclasses.field(default_factory=sampling.SampleSpec)
    seed: int = 1234


def _load_lora(wan: sampling.WanT2V, lora_path: str | None):
    """If `lora_path` is None, return the bare model.  Otherwise attach LoRA weights."""
    from peft import PeftModel

    if lora_path is None:
        return wan
    log.info("loading LoRA: %s", lora_path)
    base = wan.model
    base.requires_grad_(False)
    wan.model = PeftModel.from_pretrained(base, lora_path)
    wan.model.eval()
    return wan


def _detach_lora(wan: sampling.WanT2V):
    """Pop the PEFT wrapper, returning the underlying base model."""
    from peft import PeftModel
    if isinstance(wan.model, PeftModel):
        wan.model = wan.model.get_base_model()


def run_eval(
    spec: EvalSpec,
    wan: sampling.WanT2V,
    scorer: VideoRewardScorer,
) -> Path:
    out_dir = Path(spec.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_holdout = pd.read_csv(spec.holdout_csv)
    log.info("holdout: %d prompts", len(df_holdout))

    rows: list[dict] = []
    for run_name, lora_path in spec.runs.items():
        log.info("=== run %s ===", run_name)
        run_video_dir = out_dir / "videos" / run_name
        run_video_dir.mkdir(parents=True, exist_ok=True)

        _detach_lora(wan)
        _load_lora(wan, lora_path)

        for _, row in df_holdout.iterrows():
            uid = str(row["uuid"]) if "uuid" in row else str(row.name)
            prompt = str(row["prompt"])
            v_path = run_video_dir / f"{uid}.mp4"
            if not v_path.exists():
                video = sampling.generate_video(
                    wan, prompt, seed=spec.seed, spec=spec.sample
                )
                save_video_tensor(video, v_path, fps=spec.sample.fps)
            # VideoReward / decord need absolute paths (it prefixes file://).
            rows.append({"run": run_name, "uuid": uid, "prompt": prompt,
                         "video": str(v_path.resolve())})

    log.info("=== scoring %d videos ===", len(rows))
    score_rows = []
    for chunk_start in range(0, len(rows), 4):
        chunk = rows[chunk_start : chunk_start + 4]
        results = scorer.score(
            [r["video"] for r in chunk],
            [r["prompt"] for r in chunk],
            num_frames=16,
            use_norm=True,
        )
        for r, s in zip(chunk, results):
            score_rows.append({**r, **s.to_dict()})

    df = pd.DataFrame(score_rows)
    df.to_csv(out_dir / "scores.csv", index=False)

    # Aggregates: mean reward + win rate vs baseline.
    summary: list[dict] = []
    if "baseline" in df["run"].unique():
        base_df = df[df["run"] == "baseline"].set_index("uuid")
        for run_name in df["run"].unique():
            sub = df[df["run"] == run_name].set_index("uuid")
            cmn = sub.index.intersection(base_df.index)
            for dim in ("VQ", "MQ", "TA"):
                wins = (sub.loc[cmn, dim] > base_df.loc[cmn, dim]).mean()
                ties = (sub.loc[cmn, dim] == base_df.loc[cmn, dim]).mean()
                summary.append({
                    "run": run_name,
                    "dim": dim,
                    "mean": float(sub[dim].mean()),
                    "delta_vs_baseline": float(sub.loc[cmn, dim].mean() - base_df.loc[cmn, dim].mean()),
                    "win_rate_vs_baseline": float(wins),
                    "tie_rate_vs_baseline": float(ties),
                })
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(out_dir / "summary.csv", index=False)

    log.info("eval written to %s", out_dir)
    return out_dir
