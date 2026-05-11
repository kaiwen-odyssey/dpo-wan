"""VideoAlign / VideoReward inference wrapper.

Loads `KwaiVGI/VideoReward` and exposes a single function returning all 3
sub-rewards (Visual Quality, Motion Quality, Text Alignment).
"""
from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Sequence

import torch

from . import paths

paths.add_external_to_sys_path()

log = logging.getLogger(__name__)


# Names exposed by the underlying VideoReward model
_DIMS = ("VQ", "MQ", "TA")


@dataclasses.dataclass
class RewardResult:
    VQ: float
    MQ: float
    TA: float

    def get(self, name: str) -> float:
        return getattr(self, name)

    def to_dict(self) -> dict[str, float]:
        return {"VQ": self.VQ, "MQ": self.MQ, "TA": self.TA}


class VideoRewardScorer:
    """Thin wrapper around `inference.VideoVLMRewardInference`.

    Lazy-imports VideoAlign so the rest of the pipeline (data prep, training
    bookkeeping) stays import-cheap.
    """

    def __init__(
        self,
        checkpoint_path: Path | str | None = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        disable_flash_attn2: bool = True,
    ):
        ckpt = Path(checkpoint_path or paths.VIDEOREWARD_CHECKPOINT_DIR)
        if not ckpt.exists():
            raise FileNotFoundError(
                f"VideoReward checkpoint not found at {ckpt}; "
                "see scripts/00_setup_videoalign.sh"
            )
        log.info("Loading VideoReward from %s", ckpt)

        from inference import VideoVLMRewardInference  # noqa: WPS433

        self._infer = VideoVLMRewardInference(
            load_from_pretrained=str(ckpt),
            device=device,
            dtype=dtype,
            disable_flash_attn2=disable_flash_attn2,
        )

    @torch.no_grad()
    def score(
        self,
        video_paths: Sequence[Path | str],
        prompts: Sequence[str],
        num_frames: int | None = 16,
        max_pixels: int | None = None,
        use_norm: bool = True,
    ) -> list[RewardResult]:
        results = self._infer.reward(
            video_paths=[str(p) for p in video_paths],
            prompts=list(prompts),
            num_frames=num_frames,
            max_pixels=max_pixels,
            use_norm=use_norm,
        )
        # The upstream API returns the list inside a function whose last line in
        # the version we vendored was clipped by GitHub paging; defensively
        # support both "raw list" and "list with overall key" payloads.
        out: list[RewardResult] = []
        for r in results:
            out.append(RewardResult(VQ=float(r["VQ"]), MQ=float(r["MQ"]), TA=float(r["TA"])))
        return out


def select_dim(result: RewardResult, dim: str) -> float:
    if dim not in _DIMS:
        raise ValueError(f"unknown reward dim {dim!r}; expected one of {_DIMS}")
    return result.get(dim)


def best_worst_indices(scores: Sequence[float]) -> tuple[int, int]:
    """Return (chosen_idx, rejected_idx) by argmax / argmin of scores."""
    best = int(max(range(len(scores)), key=lambda i: scores[i]))
    worst = int(min(range(len(scores)), key=lambda i: scores[i]))
    if best == worst:
        # Degenerate case (all equal) — fall back to first / last.
        return 0, len(scores) - 1
    return best, worst
