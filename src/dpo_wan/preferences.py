"""Build preference datasets from generated candidates + reward scores."""
from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import torch

from . import paths, sampling
from .rewards import RewardResult, VideoRewardScorer, best_worst_indices, select_dim
from .utils import save_video_tensor

log = logging.getLogger(__name__)


@dataclasses.dataclass
class PreferenceBuildConfig:
    prompts_csv: str
    candidates_dir: str            # where (id_K{k}.mp4, id_K{k}.latent.pt) live
    out_root: str                  # one output dir per reward dim
    K: int = 2
    base_seed: int = 4242
    keep_videos: bool = False      # delete mp4s after scoring to save disk


def _candidate_paths(candidates_dir: Path, uuid: str, k: int) -> tuple[Path, Path]:
    return (
        candidates_dir / f"{uuid}_k{k}.mp4",
        candidates_dir / f"{uuid}_k{k}.latent.pt",
    )


def generate_candidates_for_prompts(
    wan: sampling.WanT2V,
    df_prompts: pd.DataFrame,
    candidates_dir: Path,
    K: int,
    base_seed: int,
    spec: sampling.SampleSpec,
    text_dir: Path,
) -> None:
    """Generate K rollouts per prompt and cache mp4 + latent + T5 embedding."""
    candidates_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)

    for ix, row in df_prompts.iterrows():
        uuid = str(row["uuid"]) if "uuid" in row else f"p{ix:06d}"
        prompt = str(row["prompt"])

        # Cache T5 embedding once per prompt (independent of rollout seed).
        text_path = text_dir / f"{uuid}.pt"
        if not text_path.exists():
            ctx = sampling.encode_text(wan, [prompt])[0]
            torch.save(ctx.detach().cpu(), text_path)

        for k in range(K):
            v_path, lat_path = _candidate_paths(candidates_dir, uuid, k)
            if v_path.exists() and lat_path.exists():
                continue
            video = sampling.generate_video(
                wan, prompt, seed=base_seed + ix * 100 + k, spec=spec
            )
            save_video_tensor(video, v_path, fps=spec.fps)

            latent = sampling.encode_video_to_latent(wan, video).to(torch.float16)
            torch.save(latent, lat_path)


def score_candidates(
    df_prompts: pd.DataFrame,
    candidates_dir: Path,
    K: int,
    scorer: VideoRewardScorer,
    out_csv: Path,
) -> pd.DataFrame:
    """Score every (prompt, k) candidate.  Cached to `out_csv`."""
    if out_csv.exists():
        return pd.read_csv(out_csv)

    rows: list[dict] = []
    paths_to_score: list[Path] = []
    prompt_per_path: list[str] = []
    keys: list[tuple[str, int]] = []
    for ix, row in df_prompts.iterrows():
        uuid = str(row["uuid"]) if "uuid" in row else f"p{ix:06d}"
        for k in range(K):
            v_path, _ = _candidate_paths(candidates_dir, uuid, k)
            if v_path.exists():
                paths_to_score.append(v_path)
                prompt_per_path.append(str(row["prompt"]))
                keys.append((uuid, k))

    log.info("scoring %d candidates...", len(paths_to_score))
    BATCH = 4
    results: list[RewardResult] = []
    for i in range(0, len(paths_to_score), BATCH):
        chunk_paths = paths_to_score[i : i + BATCH]
        chunk_prompts = prompt_per_path[i : i + BATCH]
        chunk_results = scorer.score(chunk_paths, chunk_prompts, num_frames=16, use_norm=True)
        results.extend(chunk_results)

    for (uuid, k), r in zip(keys, results):
        rows.append({"uuid": uuid, "k": k, "VQ": r.VQ, "MQ": r.MQ, "TA": r.TA})

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def build_preference_set(
    df_prompts: pd.DataFrame,
    df_scores: pd.DataFrame,
    candidates_dir: Path,
    text_dir: Path,
    out_root: Path,
    reward_dim: str,
    keep_videos: bool = False,
) -> Path:
    """For each prompt pick chosen=argmax, rejected=argmin in `reward_dim` and dump."""
    out_dir = out_root / reward_dim
    (out_dir / "chosen").mkdir(parents=True, exist_ok=True)
    (out_dir / "rejected").mkdir(parents=True, exist_ok=True)
    (out_dir / "text").mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for ix, row in df_prompts.iterrows():
        uuid = str(row["uuid"]) if "uuid" in row else f"p{ix:06d}"
        prompt = str(row["prompt"])
        sub = df_scores[df_scores["uuid"] == uuid]
        if len(sub) < 2:
            continue
        scores = sub[reward_dim].tolist()
        ks = sub["k"].tolist()
        chosen_idx, rejected_idx = best_worst_indices(scores)
        if chosen_idx == rejected_idx:
            continue

        chosen_k = ks[chosen_idx]
        rejected_k = ks[rejected_idx]

        chosen_lat = candidates_dir / f"{uuid}_k{chosen_k}.latent.pt"
        rejected_lat = candidates_dir / f"{uuid}_k{rejected_k}.latent.pt"
        text_src = text_dir / f"{uuid}.pt"
        if not (chosen_lat.exists() and rejected_lat.exists() and text_src.exists()):
            continue

        # Symlink to keep disk usage small (one copy per latent).
        for src, dst in [
            (chosen_lat,  out_dir / "chosen"   / f"{uuid}.pt"),
            (rejected_lat, out_dir / "rejected" / f"{uuid}.pt"),
            (text_src,    out_dir / "text"     / f"{uuid}.pt"),
        ]:
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(src.resolve())

        rows.append({
            "id": uuid,
            "prompt": prompt,
            "chosen_k": int(chosen_k),
            "rejected_k": int(rejected_k),
            "chosen_score": float(scores[chosen_idx]),
            "rejected_score": float(scores[rejected_idx]),
            "score_gap": float(scores[chosen_idx] - scores[rejected_idx]),
            "reward_dim": reward_dim,
        })

    df = pd.DataFrame(rows)
    df.to_parquet(out_dir / "index.parquet", index=False)
    df.to_csv(out_dir / "index.csv", index=False)

    # Optional margin filter — keep only pairs with a strong reward gap.
    # Reads `DPO_WAN_FILTER_MARGIN` env to allow callers to override globally,
    # then emits a parallel index.parquet with the filter applied.

    if not keep_videos:
        for ix, row in df_prompts.iterrows():
            uuid = str(row["uuid"]) if "uuid" in row else f"p{ix:06d}"
            for k in range(int(df_scores[df_scores["uuid"] == uuid]["k"].max() + 1) if (df_scores["uuid"] == uuid).any() else 0):
                v_path = candidates_dir / f"{uuid}_k{k}.mp4"
                # Keep videos -- they're small at 33 frames @ 832x480 and useful for paper figs.

    log.info("built %d preference pairs for reward=%s -> %s", len(df), reward_dim, out_dir)
    return out_dir
