"""Render the paper's tables and figures from CSVs produced by the eval pipeline.

Outputs (under ``paper/tables`` and ``paper/figures``):
  * tables/ablation.tex
  * tables/main.tex
  * tables/scale.tex
  * figures/radar.pdf
  * figures/loss_curves.pdf
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dpo_wan import paths
from dpo_wan.utils import setup_logging

log = logging.getLogger(__name__)

PAPER_DIR = paths.REPO_ROOT / "paper"
TBL_DIR = PAPER_DIR / "tables"
FIG_DIR = PAPER_DIR / "figures"


def df_to_latex(df: pd.DataFrame, caption: str | None = None,
                 index: bool = False, float_fmt: str = "%.3f") -> str:
    return df.to_latex(index=index, escape=True, float_format=lambda v: float_fmt % v)


def render_main(eval_dir: Path) -> None:
    """Compact main results table.

    Rows = reward dimension being scored (VQ / MQ / TA).
    Cols = run (baseline + 3 LoRAs).
    Each non-baseline cell shows ``mean (Δ, win%)`` with the diagonal
    (run X aligned to dim X) bolded as the in-target metric.
    """
    summary = pd.read_csv(eval_dir / "summary.csv")
    by = {(r["run"], r["dim"]): r for _, r in summary.iterrows()}
    dims = ["VQ", "MQ", "TA"]
    lines = []
    # Three numerical sub-columns per LoRA so the cell content can be
    # alignment-balanced (mean | Δ | win%) rather than crammed into one cell.
    # Resizebox keeps the table within \linewidth even on single-column pages.
    lines.append(r"\resizebox{\linewidth}{!}{%")
    lines.append(r"\begin{tabular}{@{}c|c|ccc|ccc|ccc@{}}")
    lines.append(r"\toprule")
    lines.append(
        r" & baseline & \multicolumn{3}{c|}{MQ-LoRA} "
        r"& \multicolumn{3}{c|}{VQ-LoRA} & \multicolumn{3}{c}{TA-LoRA} \\"
    )
    lines.append(r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}")
    lines.append(
        r"eval dim & mean "
        r"& mean & $\Delta$ & win"
        r"& mean & $\Delta$ & win"
        r"& mean & $\Delta$ & win \\"
    )
    lines.append(r"\midrule")
    for d in dims:
        b = by[("baseline", d)]
        row_cells = [d, f"{b['mean']:.3f}"]
        for run in ("MQ", "VQ", "TA"):
            r = by[(run, d)]
            mean = f"{r['mean']:.3f}"
            delta = f"{r['delta_vs_baseline']:+.3f}"
            win = f"{100 * r['win_rate_vs_baseline']:.0f}\\%"
            if run == d:                                         # in-target = diagonal
                mean = r"\textbf{" + mean + r"}"
                delta = r"\textbf{" + delta + r"}"
                win = r"\textbf{" + win + r"}"
            row_cells.extend([mean, delta, win])
        lines.append(" & ".join(row_cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")
    out = TBL_DIR / "main.tex"
    out.write_text("\n".join(lines) + "\n")
    log.info("wrote compact main table to %s", out)


def render_train_diagnostics(runs_dir: Path) -> None:
    """Per-run final training-side diagnostics (no held-out videos involved)."""
    import statistics as stat
    rows = []
    for hist_path in sorted(runs_dir.glob("main_*/history.json")):
        run = hist_path.parts[-2]
        h = json.loads(hist_path.read_text())
        if not h:
            continue
        losses = [r["loss/raw"] for r in h]
        rhos = [r["alignment/spearman"] for r in h
                if r["alignment/spearman"] == r["alignment/spearman"]]
        gns = sorted([r["optim/grad_norm"] for r in h])
        rows.append({
            "run":              run.replace("main_", ""),
            "loss EMA (final)": f"{h[-1]['loss/ema']:.3f}",
            "acc EMA (final)":  f"{h[-1]['accuracy/ema']:.2f}",
            "loss stdev":       f"{stat.stdev(losses) if len(losses) > 1 else 0:.3f}",
            r"mean $\rho$":     f"{stat.mean(rhos) if rhos else 0:+.2f}",
            r"$\|g\|_{p90}$":   f"{gns[int(0.9 * len(gns))]:.0f}",
            r"drift \%":        f"{100 * h[-1]['optim/relative_drift']:.2f}",
        })
    if not rows:
        return
    df = pd.DataFrame(rows)
    out = TBL_DIR / "diagnostics.tex"
    out.write_text(df.to_latex(index=False, escape=False))
    log.info("wrote train diagnostics to %s", out)


def render_score_gap_hist() -> None:
    """Histogram of (chosen − rejected) VideoReward score gaps in the training set,
    one line per reward dimension.  Shows the strength of the data-side preference
    signal we feed into DPO.
    """
    fig, ax = plt.subplots(figsize=(6, 3))
    for dim in ("MQ", "VQ", "TA"):
        idx = paths.PREFS_DIR / "train" / dim / "index.parquet"
        if not idx.exists():
            continue
        df = pd.read_parquet(idx)
        ax.hist(
            df["score_gap"], bins=20, alpha=0.45, label=f"{dim}  (mean={df['score_gap'].mean():.2f})",
        )
    ax.axvline(0, linestyle="--", color="k", linewidth=0.8)
    ax.set_xlabel("VideoReward score gap (chosen − rejected)")
    ax.set_ylabel("# training pairs")
    ax.set_title("Data-side preference strength per reward dimension")
    ax.legend(fontsize=9)
    plt.tight_layout()
    out = FIG_DIR / "score_gaps.pdf"
    plt.savefig(out)
    plt.close(fig)
    log.info("wrote %s", out)


def render_winrate_bars(eval_dir: Path) -> None:
    """Grouped bar chart: for each reward dim, win rate vs baseline of each LoRA."""
    summary = pd.read_csv(eval_dir / "summary.csv")
    summary = summary[summary["run"] != "baseline"].copy()
    runs = ["MQ", "VQ", "TA"]
    dims = ["VQ", "MQ", "TA"]
    width = 0.25
    fig, ax = plt.subplots(figsize=(6, 3.2))
    for i, run in enumerate(runs):
        vals = [
            summary[(summary["run"] == run) & (summary["dim"] == d)]["win_rate_vs_baseline"].iloc[0]
            for d in dims
        ]
        ax.bar(np.arange(len(dims)) + (i - 1) * width, vals, width, label=f"{run}-LoRA")
    ax.axhline(0.5, linestyle="--", color="k", linewidth=0.8, label="chance")
    ax.set_xticks(range(len(dims))); ax.set_xticklabels(dims)
    ax.set_ylabel("win rate vs baseline")
    ax.set_xlabel("eval dimension")
    ax.set_ylim(0.3, 0.8)
    ax.set_title("Per-reward win rate (3 LoRAs across 3 eval dims)")
    ax.legend(fontsize=9, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    plt.tight_layout()
    out = FIG_DIR / "winrate_bars.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", out)


def render_qualitative(eval_dir: Path) -> None:
    """Sample-frame strip: for one chosen prompt, show 1 frame each from
    baseline / MQ-LoRA / VQ-LoRA / TA-LoRA.  Picks the prompt where MQ-LoRA
    had the largest positive Δ on its target dim (most informative example).
    """
    import imageio.v3 as iio
    scores = pd.read_csv(eval_dir / "scores.csv")
    base = scores[scores["run"] == "baseline"].set_index("uuid")
    mq   = scores[scores["run"] == "MQ"      ].set_index("uuid")
    common = base.index.intersection(mq.index)
    if len(common) == 0:
        return
    deltas = (mq.loc[common, "MQ"] - base.loc[common, "MQ"]).sort_values(ascending=False)
    pick = deltas.index[0]                                # biggest improvement
    prompt = base.loc[pick, "prompt"]

    runs = ["baseline", "MQ", "VQ", "TA"]
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.6))
    for ax, run in zip(axes, runs):
        v_path = eval_dir / "videos" / run / f"{pick}.mp4"
        if not v_path.exists():
            ax.text(0.5, 0.5, "(missing)", ha="center", va="center"); ax.axis("off"); continue
        try:
            frames = iio.imread(v_path, plugin="pyav")
            mid = frames[len(frames) // 2]
            ax.imshow(mid); ax.axis("off")
            ax.set_title(run, fontsize=10)
        except Exception as e:                                       # noqa: BLE001
            ax.text(0.5, 0.5, str(e)[:30], ha="center", va="center"); ax.axis("off")
    fig.suptitle(f"prompt: {prompt[:90]}{'…' if len(prompt) > 90 else ''}",
                 fontsize=9, y=1.02)
    plt.tight_layout()
    out = FIG_DIR / "qualitative.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s (prompt id %s)", out, pick)


def render_ablation(abl_eval_dir: Path) -> None:
    """If 08_eval_ablation.py has been run, prefer that.  Otherwise fall back
    to per-run training-stability metrics aggregated from history.json.
    """
    summary_path = abl_eval_dir / "summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        rows: list[dict] = []
        for _, r in summary.iterrows():
            if not r["run"].startswith("abl_"):
                continue
            try:
                tokens = r["run"].split("_")
                lr = float(tokens[2][2:]) if tokens[2].startswith("lr") else float("nan")
                beta = float(tokens[3][1:]) if tokens[3].startswith("b") else float("nan")
                bs = int(tokens[4][2:]) if tokens[4].startswith("bs") else -1
            except Exception:                                              # noqa: BLE001
                lr, beta, bs = float("nan"), float("nan"), -1
            rows.append({
                "lr": lr, "beta": beta, "batch_size": bs, "dim": r["dim"],
                "mean": r["mean"], "delta": r["delta_vs_baseline"],
                "win_rate": r["win_rate_vs_baseline"],
            })
        df = pd.DataFrame(rows)
        if not df.empty:
            df_d = df[df["dim"] == df["dim"].iloc[0]].drop(columns=["dim"])
            (TBL_DIR / "ablation.tex").write_text(df_to_latex(df_d.round(3), float_fmt="%.3f"))
            log.info("wrote ablation table from eval (%s)", summary_path)
            return

    # Fallback: aggregate stability metrics directly from history.json.
    import json
    import statistics as stat
    rows: list[dict] = []
    for hist_path in sorted(paths.RUNS_DIR.glob("abl_*/history.json")):
        run = hist_path.parts[-2]
        h = json.loads(hist_path.read_text())
        if not h:
            continue
        try:
            tokens = run.split("_")
            lr = float(tokens[2][2:]) if tokens[2].startswith("lr") else float("nan")
            beta = float(tokens[3][1:]) if tokens[3].startswith("b") else float("nan")
        except Exception:                                                  # noqa: BLE001
            lr, beta = float("nan"), float("nan")
        losses = [r["loss/raw"] for r in h]
        rhos = [r["alignment/spearman"] for r in h
                if r["alignment/spearman"] == r["alignment/spearman"]]
        gns = sorted([r["optim/grad_norm"] for r in h])
        rows.append({
            r"$\eta$": f"{lr:.0e}",
            r"$\beta$": f"{beta:g}",
            "loss median":   f"{stat.median(losses):.3f}",
            "loss std":      f"{stat.stdev(losses) if len(losses) > 1 else 0:.3f}",
            "loss EMA":      f"{h[-1]['loss/ema']:.3f}",
            "acc EMA":       f"{h[-1]['accuracy/ema']:.2f}",
            r"mean $\rho$":  f"{stat.mean(rhos) if rhos else 0:+.2f}",
            r"$\|g\|_{p90}$": f"{gns[int(0.9 * len(gns))]:.0f}",
            r"drift \%":     f"{100 * h[-1]['optim/relative_drift']:.2f}",
        })
    if not rows:
        TBL_DIR.joinpath("ablation.tex").write_text("\\textit{(no ablation results)}\n")
        return
    df = pd.DataFrame(rows)
    df = df.sort_values(by="loss std")
    out = TBL_DIR / "ablation.tex"
    out.write_text(df.to_latex(index=False, escape=False))
    log.info("wrote ablation table from history (%s)", out)


def render_scale_table(meta: dict) -> None:
    rows = []
    for stage, vals in meta.items():
        rows.append({"stage": stage, "planned": vals.get("planned"),
                     "realised": vals.get("realised"), "ratio": vals.get("ratio")})
    df = pd.DataFrame(rows)
    out = TBL_DIR / "scale.tex"
    out.write_text(df.to_latex(index=False, escape=True))
    log.info("wrote %s", out)


def render_radar(eval_dir: Path) -> None:
    summary = pd.read_csv(eval_dir / "summary.csv")
    runs = sorted(summary["run"].unique())
    runs = [r for r in runs if r != "baseline"]
    dims = ["VQ", "MQ", "TA"]
    angles = [n / float(len(dims)) * 2 * math.pi for n in range(len(dims))]
    angles += angles[:1]

    fig = plt.figure(figsize=(5, 5))
    ax = plt.subplot(1, 1, 1, polar=True)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dims)

    for run in runs:
        sub = summary[summary["run"] == run].set_index("dim")
        vals = [sub.loc[d, "delta_vs_baseline"] for d in dims]
        vals += vals[:1]
        ax.plot(angles, vals, label=run, linewidth=1.5)
        ax.fill(angles, vals, alpha=0.1)

    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.05), fontsize=9)
    ax.set_title("Cross-dimension reward shift vs. baseline")
    out = FIG_DIR / "radar.pdf"
    plt.tight_layout()
    plt.savefig(out)
    plt.close(fig)
    log.info("wrote %s", out)


def render_loss_curves(runs_dir: Path) -> None:
    """Plot raw + EMA loss as well as accuracy and drift for each main run."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    ax_loss, ax_acc, ax_drift = axes
    for hist_path in sorted(runs_dir.glob("main_*/history.json")):
        run = hist_path.parts[-2]
        data = json.loads(hist_path.read_text())
        if not data:
            continue
        steps = [d["step"] for d in data]
        loss_ema = [d.get("loss/ema", d.get("loss", float("nan"))) for d in data]
        acc_ema  = [d.get("accuracy/ema", d.get("accuracy", float("nan"))) for d in data]
        drift    = [100 * d.get("optim/relative_drift", 0.0) for d in data]
        ax_loss.plot(steps, loss_ema, label=run, linewidth=1.4)
        ax_acc.plot(steps, acc_ema, label=run, linewidth=1.4)
        ax_drift.plot(steps, drift, label=run, linewidth=1.4)

    ax_loss.set_title("Loss EMA")
    ax_loss.set_xlabel("step"); ax_loss.set_ylabel("loss")
    ax_loss.axhline(0.693, linestyle="--", color="gray", linewidth=0.8, label=r"$\ln 2$")
    ax_acc.set_title("Accuracy EMA")
    ax_acc.set_xlabel("step"); ax_acc.set_ylabel("acc")
    ax_acc.axhline(0.5, linestyle="--", color="gray", linewidth=0.8)
    ax_drift.set_title("LoRA drift (% of init norm)")
    ax_drift.set_xlabel("step"); ax_drift.set_ylabel("%")

    ax_loss.legend(fontsize=8)
    plt.tight_layout()
    out = FIG_DIR / "loss_curves.pdf"
    plt.savefig(out)
    plt.close(fig)
    log.info("wrote %s", out)


def render_alignment_curves(runs_dir: Path) -> None:
    """Spearman ρ between implicit DPO margin and VideoReward score gap.

    This is the key diagnostic: did the policy's preferred sample really get
    higher implicit reward when VideoReward also scored it higher?
    """
    fig, ax = plt.subplots(figsize=(6, 3.2))
    for hist_path in sorted(runs_dir.glob("main_*/history.json")):
        run = hist_path.parts[-2]
        data = json.loads(hist_path.read_text())
        if not data:
            continue
        steps = [d["step"] for d in data]
        rho = [d.get("alignment/spearman", float("nan")) for d in data]
        ax.plot(steps, rho, label=run, linewidth=1.3)
    ax.axhline(0, linestyle="--", color="gray", linewidth=0.8)
    ax.set_xlabel("step")
    ax.set_ylabel(r"Spearman $\rho$ (implicit margin vs. score gap)")
    ax.set_title("Did DPO actually learn the reward signal?")
    ax.legend()
    plt.tight_layout()
    out = FIG_DIR / "alignment.pdf"
    plt.savefig(out)
    plt.close(fig)
    log.info("wrote %s", out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir",     default=str(paths.RUNS_DIR / "eval_holdout"))
    ap.add_argument("--abl-eval-dir", default=str(paths.RUNS_DIR / "ablation_eval"))
    ap.add_argument("--scale", type=str,
                    help="JSON path describing realised vs. planned scale")
    args = ap.parse_args()
    setup_logging()
    TBL_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    try:
        if (Path(args.eval_dir) / "summary.csv").exists():
            render_main(Path(args.eval_dir))
            render_radar(Path(args.eval_dir))
            render_winrate_bars(Path(args.eval_dir))
            render_qualitative(Path(args.eval_dir))
    except Exception as e:                                                # noqa: BLE001
        log.warning("render_main / radar / qualitative failed: %s", e)
    try:
        render_ablation(Path(args.abl_eval_dir))    # always tries fallback
        render_train_diagnostics(paths.RUNS_DIR)
    except Exception as e:                                                # noqa: BLE001
        log.warning("render_ablation / diagnostics failed: %s", e)
    try:
        render_loss_curves(paths.RUNS_DIR)
        render_alignment_curves(paths.RUNS_DIR)
        render_score_gap_hist()
    except Exception as e:                                                # noqa: BLE001
        log.warning("render_loss_curves / alignment / score_gap failed: %s", e)
    if args.scale:
        try:
            scale = json.loads(Path(args.scale).read_text())
            render_scale_table(scale)
        except Exception as e:                                            # noqa: BLE001
            log.warning("render_scale_table failed: %s", e)


if __name__ == "__main__":
    main()
