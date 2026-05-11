"""Build a margin-filtered preference set from an existing preference dir.

Reads `data/preferences/<split>/<dim>/index.parquet`, drops rows whose
`score_gap` < `--margin`, and writes a new sibling dir with symlinks to the
same chosen/rejected/text artefacts.  Lets us re-use all on-disk videos and
latents with no regeneration.

Usage:
    python scripts/build_filtered_prefs.py \\
        --split train --dim MQ --margin 0.2 \\
        --out-suffix m0.2
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from dpo_wan import paths
from dpo_wan.utils import setup_logging

log = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--dim", required=True)
    ap.add_argument("--margin", type=float, default=0.2)
    ap.add_argument("--out-suffix", default=None,
                    help="appended to dim name; defaults to 'm<margin>'")
    args = ap.parse_args()

    setup_logging()
    src = paths.PREFS_DIR / args.split / args.dim
    if not (src / "index.parquet").exists():
        raise FileNotFoundError(f"missing {src/'index.parquet'}")

    suffix = args.out_suffix or f"m{args.margin}"
    dst = paths.PREFS_DIR / args.split / f"{args.dim}_{suffix}"
    (dst / "chosen").mkdir(parents=True, exist_ok=True)
    (dst / "rejected").mkdir(parents=True, exist_ok=True)
    (dst / "text").mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(src / "index.parquet")
    before = len(df)
    df = df[df["score_gap"] >= args.margin].reset_index(drop=True)
    after = len(df)
    log.info(
        "%s -> %s : kept %d/%d (%.0f%%) at margin>=%g, "
        "mean gap = %.3f (was %.3f)",
        src.name, dst.name, after, before, 100 * after / before, args.margin,
        df["score_gap"].mean(),
        pd.read_parquet(src / "index.parquet")["score_gap"].mean(),
    )

    df.to_parquet(dst / "index.parquet", index=False)
    df.to_csv(dst / "index.csv", index=False)

    for sub in ("chosen", "rejected", "text"):
        for uid in df["id"]:
            uid = str(uid)
            link = dst / sub / f"{uid}.pt"
            target = (src / sub / f"{uid}.pt").resolve()
            if link.exists() or link.is_symlink():
                link.unlink()
            link.symlink_to(target)
    log.info("filtered preference dir: %s", dst)


if __name__ == "__main__":
    main()
