"""Render the MQ-focused paper's main results table and loss-curves figure.

This is the trimmed renderer used after we restructured the paper to focus
exclusively on the motion-quality (MQ) sub-reward.  It writes:

    paper/tables/main.tex          (MQ holdout: baseline + 3 variants)
    paper/figures/loss_curves.pdf  (loss EMA / acc EMA / LoRA drift, 3 MQ variants)
    paper/figures/score_gaps.pdf   (regenerated, unchanged)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dpo_wan import paths
from dpo_wan.utils import setup_logging

setup_logging()

PAPER = paths.REPO_ROOT / "paper"
TBL = PAPER / "tables"
FIG = PAPER / "figures"
TBL.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Main MQ results table (rows = variants, cols = baseline mean | LoRA mean, Δ, win%)
# Reads runs/eval_mq_sft/summary.csv for the 3-variant MQ holdout.
# ---------------------------------------------------------------------------
def render_main() -> None:
    csv = paths.RUNS_DIR / "eval_mq_sft" / "summary.csv"
    if not csv.exists():
        print(f"[main] missing {csv} — skipping")
        return
    df = pd.read_csv(csv)
    mq = df[df["dim"] == "MQ"]
    base_mean = mq[mq["run"] == "baseline"]["mean"].iloc[0]
    rows = []
    label_map = {
        "MQ_noSFT":     r"MQ\_unfilt  ($\lambda{=}0$, 1000 pairs)",
        "MQ_filtered":  r"MQ\_filt   ($\lambda{=}0$, 767 pairs, m$\geq$0.2)",
        "MQ_SFT_lam10": r"MQ\_sft    ($\lambda{=}10$, 1000 pairs)",
    }
    for run in ("MQ_noSFT", "MQ_filtered", "MQ_SFT_lam10"):
        r = mq[mq["run"] == run]
        if r.empty:
            continue
        mean = r["mean"].iloc[0]
        delta = r["delta_vs_baseline"].iloc[0]
        win = 100 * r["win_rate_vs_baseline"].iloc[0]
        tie = 100 * r["tie_rate_vs_baseline"].iloc[0]
        rows.append((label_map[run], f"{mean:.3f}", f"{delta:+.3f}",
                     f"{win:.0f}\\%", f"{tie:.0f}\\%"))
    lines = [
        r"\begin{tabular}{@{}l|c|cccc@{}}",
        r"\toprule",
        r"variant & baseline mean & LoRA mean & $\Delta$ & win\% & tie\% \\",
        r"\midrule",
    ]
    for i, row in enumerate(rows):
        baseline_cell = f"{base_mean:.3f}" if i == 0 else ""
        bold = i == 0  # MQ_noSFT first; bold the best (which happens to be first)
        cells = [row[0], baseline_cell, *row[1:]]
        if bold:
            # bold the LoRA-side cells for the winner
            cells = [cells[0], cells[1]] + [r"\textbf{" + c + r"}" for c in cells[2:]]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TBL / "main.tex").write_text("\n".join(lines) + "\n")
    print(f"[main] wrote {TBL/'main.tex'} ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# Loss curves: 3-panel comparison of the 3 MQ variants.
# ---------------------------------------------------------------------------
def render_loss_curves() -> None:
    runs = {
        r"MQ_unfilt ($\lambda=0$)":      "main_MQ",
        r"MQ_filt (m≥0.2, $\lambda=0$)": "main_MQ_m0.2",
        r"MQ_sft ($\lambda=10$)":        "main_MQ_sft",
    }
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    ax_loss, ax_acc, ax_drift = axes
    for label, dirname in runs.items():
        hpath = paths.RUNS_DIR / dirname / "final" / "history.json"
        if not hpath.exists():
            print(f"  [loss] missing {hpath}")
            continue
        data = json.loads(hpath.read_text())
        if not data:
            continue
        steps = [d["step"] for d in data]
        loss_ema = [d.get("loss/ema", float("nan")) for d in data]
        acc_ema  = [d.get("accuracy/ema", float("nan")) for d in data]
        drift    = [100 * d.get("optim/relative_drift", 0.0) for d in data]
        ax_loss.plot(steps, loss_ema, label=label, linewidth=1.5)
        ax_acc.plot(steps, acc_ema, label=label, linewidth=1.5)
        ax_drift.plot(steps, drift, label=label, linewidth=1.5)

    ax_loss.set_title("Loss EMA")
    ax_loss.set_xlabel("step"); ax_loss.set_ylabel("loss")
    ax_loss.axhline(0.693, linestyle="--", color="gray", linewidth=0.8, label=r"$\ln 2$")
    ax_acc.set_title("Accuracy EMA")
    ax_acc.set_xlabel("step"); ax_acc.set_ylabel("acc")
    ax_acc.axhline(0.5, linestyle="--", color="gray", linewidth=0.8)
    ax_drift.set_title("LoRA drift (\\% of init norm)")
    ax_drift.set_xlabel("step"); ax_drift.set_ylabel("\\%")

    ax_loss.legend(fontsize=8)
    plt.tight_layout()
    out = FIG / "loss_curves.pdf"
    plt.savefig(out)
    plt.close(fig)
    print(f"[loss] wrote {out}")


# ---------------------------------------------------------------------------
# Score gap histogram - regenerate (unchanged from before).
# ---------------------------------------------------------------------------
def render_score_gaps() -> None:
    fig, ax = plt.subplots(figsize=(6, 3))
    for dim in ("MQ", "VQ", "TA"):
        idx = paths.PREFS_DIR / "train" / dim / "index.parquet"
        if not idx.exists():
            continue
        df = pd.read_parquet(idx)
        ax.hist(df["score_gap"], bins=20, alpha=0.45,
                label=f"{dim}  (mean={df['score_gap'].mean():.2f})")
    ax.axvline(0, linestyle="--", color="k", linewidth=0.8)
    ax.set_xlabel("VideoReward score gap (chosen − rejected)")
    ax.set_ylabel("# training pairs")
    ax.set_title("Data-side preference strength per reward dimension")
    ax.legend(fontsize=9)
    plt.tight_layout()
    out = FIG / "score_gaps.pdf"
    plt.savefig(out)
    plt.close(fig)
    print(f"[gaps] wrote {out}")


if __name__ == "__main__":
    render_main()
    render_loss_curves()
    render_score_gaps()
