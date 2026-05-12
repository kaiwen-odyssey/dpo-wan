"""Replay the trainer's DataLoader to recover the exact UUIDs seen by DPO.

We re-create the same dataset + dataloader configuration the trainer uses
(`shuffle=True`, `batch_size=1`, `num_workers=2`, `seed=0`), and iterate up
to `--max-steps` batches.  The collected UUIDs are written to a CSV.

This works because `set_seed(0)` is called by both the trainer and this
script *before* the DataLoader's RandomSampler is constructed, so the
shuffle order is deterministic.
"""
from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

from torch.utils.data import DataLoader

from dpo_wan import paths
from dpo_wan.data import LatentPreferenceDataset, collate_preferences
from dpo_wan.utils import set_seed, setup_logging

log = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pref-root", default="data/preferences/train/MQ")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--out", default="runs/train_set_rerun/seen_uuids.csv")
    args = ap.parse_args()

    setup_logging()
    set_seed(args.seed)

    ds = LatentPreferenceDataset(Path(args.pref_root))
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_preferences,
        drop_last=True, persistent_workers=args.num_workers > 0,
    )
    seen: list[str] = []
    seen_set: set[str] = set()
    for batch in loader:
        for uid in batch["ids"]:
            seen.append(uid)
            seen_set.add(uid)
        if len(seen) >= args.max_steps:
            break

    log.info("collected %d UUIDs (%d unique) over the first %d batches",
             len(seen), len(seen_set), args.max_steps)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seen_idx", "uuid"])
        for i, uid in enumerate(seen):
            w.writerow([i, uid])
    log.info("written: %s", out)


if __name__ == "__main__":
    main()
