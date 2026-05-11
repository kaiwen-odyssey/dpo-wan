"""Preference dataset: latent pairs + cached text embeddings."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import pandas as pd
import torch
from torch.utils.data import Dataset


class LatentPreferenceDataset(Dataset):
    """Loads pre-computed (chosen_latent, rejected_latent, text_embed) triples.

    The on-disk layout (built by `scripts/02_build_preferences.py`) is:

        preferences/<reward>/
            chosen/<id>.pt          # latent  (Z, F, H, W)  fp16
            rejected/<id>.pt        # latent  (Z, F, H, W)  fp16
            text/<id>.pt            # T5 embed (L, C)       bf16
            index.parquet           # columns: id, prompt, chosen_score, rejected_score, ...
    """

    def __init__(self, root: Path | str):
        self.root = Path(root)
        idx_path = self.root / "index.parquet"
        if not idx_path.exists():
            raise FileNotFoundError(f"No preference index at {idx_path}")
        self.index = pd.read_parquet(idx_path)
        self.chosen_dir = self.root / "chosen"
        self.rejected_dir = self.root / "rejected"
        self.text_dir = self.root / "text"

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict:
        row = self.index.iloc[i]
        item_id = str(row["id"])
        chosen = torch.load(self.chosen_dir / f"{item_id}.pt", map_location="cpu", weights_only=True)
        rejected = torch.load(self.rejected_dir / f"{item_id}.pt", map_location="cpu", weights_only=True)
        text = torch.load(self.text_dir / f"{item_id}.pt", map_location="cpu", weights_only=True)
        return {
            "id": item_id,
            "chosen": chosen.float(),
            "rejected": rejected.float(),
            "text": text.float(),
            "prompt": row.get("prompt", ""),
            "chosen_score": float(row.get("chosen_score", 0.0)),
            "rejected_score": float(row.get("rejected_score", 0.0)),
        }


def collate_preferences(batch: list[dict]) -> dict:
    """Stack latents to (B, Z, F, H, W); keep text embeds as a list (variable length).

    The reward-side preference gap is preserved per-sample so the trainer
    can correlate implicit DPO margins against the ground-truth VideoReward
    score gap (a key diagnostic for "is DPO actually learning the signal").
    """
    chosen   = torch.stack([b["chosen"]   for b in batch], dim=0)
    rejected = torch.stack([b["rejected"] for b in batch], dim=0)
    text     = [b["text"] for b in batch]
    prompts  = [b["prompt"] for b in batch]
    ids      = [b["id"]    for b in batch]
    chosen_scores   = [float(b.get("chosen_score",   0.0)) for b in batch]
    rejected_scores = [float(b.get("rejected_score", 0.0)) for b in batch]
    return {
        "chosen": chosen,
        "rejected": rejected,
        "text": text,
        "prompts": prompts,
        "ids": ids,
        "chosen_scores": chosen_scores,
        "rejected_scores": rejected_scores,
    }


def write_preference_index(rows: list[dict], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def read_preference_index(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)
