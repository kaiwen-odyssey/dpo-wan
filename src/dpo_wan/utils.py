"""Misc utilities: video I/O, seeding, logging."""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_video_tensor(video: torch.Tensor, path: Path, fps: int = 16) -> Path:
    """Save (C, F, H, W) tensor in [-1, 1] range as mp4.

    Wan2.1 returns videos in [-1, 1] range with shape (C, N, H, W).
    """
    import imageio.v3 as iio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    v = video.detach().cpu().clamp(-1, 1)
    v = ((v + 1.0) * 127.5).to(torch.uint8)
    v = v.permute(1, 2, 3, 0).numpy()
    iio.imwrite(path, v, fps=fps, codec="libx264", quality=8)
    return path


def load_video_tensor(path: Path, num_frames: int | None = None) -> torch.Tensor:
    """Load mp4 -> (C, F, H, W) tensor in [-1, 1]."""
    import imageio.v3 as iio
    arr = iio.imread(path, plugin="pyav")
    if num_frames is not None and arr.shape[0] >= num_frames:
        idx = np.linspace(0, arr.shape[0] - 1, num_frames).round().astype(int)
        arr = arr[idx]
    t = torch.from_numpy(arr).float() / 127.5 - 1.0
    return t.permute(3, 0, 1, 2)


def write_json(obj, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def chunked(iterable: Iterable, n: int):
    buf = []
    for x in iterable:
        buf.append(x)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf
