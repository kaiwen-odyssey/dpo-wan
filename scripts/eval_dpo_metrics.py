"""Compute DPO accuracy + implicit-reward margin on a held-out preference set.

For each LoRA checkpoint we forward both `chosen` and `rejected` latents
through the policy (LoRA enabled) and the reference (LoRA disabled) at a set
of fixed (noise, timestep) draws, then aggregate Diffusion-DPO loss
statistics per LoRA so we can compare against the training-side numbers and
detect overfitting.

Usage:
    python scripts/eval_dpo_metrics.py \\
        --pref-root data/preferences/ablation/MQ \\
        --runs MQ=runs/main_MQ/final \\
              MQ_m0.2=runs/main_MQ_m0.2/final \\
              MQ_sft=runs/main_MQ_sft/final \\
        --beta 10 --num-draws 4
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import statistics as stat
from pathlib import Path

import torch

from dpo_wan import dpo, paths, sampling
from dpo_wan.data import LatentPreferenceDataset, collate_preferences
from dpo_wan.utils import set_seed, setup_logging

log = logging.getLogger(__name__)


def _model_forward(wan, latents_list, t_scalar, ctx_list, seq_len):
    timestep = (t_scalar * wan.num_train_timesteps).clamp(min=1.0).to(latents_list[0].dtype)
    out = wan.model(latents_list, t=timestep, context=ctx_list, seq_len=seq_len)
    return torch.stack(out, dim=0)


@torch.no_grad()
def measure_one_run(wan, pref_root: Path, num_draws: int, beta: float, T: float, seed: int):
    """For one model state (already attached), iterate the pref dir and
    accumulate per-pair stats."""
    from peft import PeftModel
    ds = LatentPreferenceDataset(pref_root)
    margins = []
    rcs = []
    rrs = []
    score_gaps = []
    accs = []
    for i in range(len(ds)):
        item = ds[i]
        chosen   = item["chosen"].unsqueeze(0).to(wan.device)
        rejected = item["rejected"].unsqueeze(0).to(wan.device)
        text     = [item["text"].to(wan.device)]
        seq_len  = sampling.make_seq_len(wan, tuple(chosen.shape[1:]))
        score_gap = item["chosen_score"] - item["rejected_score"]
        # Average over `num_draws` shared-noise rollouts to denoise the metric.
        per_pair_margins = []
        per_pair_rcs = []
        per_pair_rrs = []
        for d in range(num_draws):
            torch.manual_seed(seed + 1000 * i + d)
            cfg = dpo.DPOConfig(beta=beta, timestep_T=T, shared_noise=True)
            x_w_t, x_l_t, v_w_tgt, v_l_tgt, t = dpo.dpo_step_inputs(chosen, rejected, cfg)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                v_pred_w = _model_forward(wan, list(x_w_t), t, text, seq_len)
                v_pred_l = _model_forward(wan, list(x_l_t), t, text, seq_len)
                if isinstance(wan.model, PeftModel):
                    with wan.model.disable_adapter():
                        v_ref_w = _model_forward(wan, list(x_w_t), t, text, seq_len)
                        v_ref_l = _model_forward(wan, list(x_l_t), t, text, seq_len)
                else:
                    v_ref_w = _model_forward(wan, list(x_w_t), t, text, seq_len)
                    v_ref_l = _model_forward(wan, list(x_l_t), t, text, seq_len)
                m = dpo.diffusion_dpo_loss(v_pred_w, v_pred_l, v_ref_w, v_ref_l,
                                           v_w_tgt, v_l_tgt, cfg)
            per_pair_margins.append(float(m["margin"].item()))
            per_pair_rcs.append(float(m["implicit_reward_w"].mean().item()))
            per_pair_rrs.append(float(m["implicit_reward_l"].mean().item()))
        margin_mean = sum(per_pair_margins) / len(per_pair_margins)
        rc_mean = sum(per_pair_rcs) / len(per_pair_rcs)
        rr_mean = sum(per_pair_rrs) / len(per_pair_rrs)
        margins.append(margin_mean)
        rcs.append(rc_mean)
        rrs.append(rr_mean)
        score_gaps.append(score_gap)
        accs.append(1.0 if margin_mean > 0 else 0.0)
    out = {
        "n_pairs":   len(ds),
        "accuracy":  sum(accs) / len(accs),
        "margin_mean":  stat.mean(margins),
        "margin_std":   stat.stdev(margins) if len(margins) > 1 else 0.0,
        "R_chosen_mean":   stat.mean(rcs),
        "R_rejected_mean": stat.mean(rrs),
        "R_gap_mean":      stat.mean([c - r for c, r in zip(rcs, rrs)]),
        "score_gap_mean":  stat.mean(score_gaps),
    }
    # Spearman / pearson with score gap
    try:
        import numpy as _np
        m = _np.asarray(margins); s = _np.asarray(score_gaps)
        if m.var() > 0 and s.var() > 0:
            out["pearson_margin_vs_score_gap"] = float(_np.corrcoef(m, s)[0, 1])
            rm = _np.argsort(_np.argsort(m)).astype(float)
            rs = _np.argsort(_np.argsort(s)).astype(float)
            out["spearman_margin_vs_score_gap"] = float(_np.corrcoef(rm, rs)[0, 1])
    except Exception:
        pass
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pref-root", required=True)
    ap.add_argument("--runs", nargs="+", required=True,
                    help="NAME=PATH pairs; PATH=baseline means no LoRA.")
    ap.add_argument("--beta", type=float, default=10.0)
    ap.add_argument("--T", type=float, default=1000.0)
    ap.add_argument("--num-draws", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="runs/eval_dpo_metrics.json")
    args = ap.parse_args()

    setup_logging()
    set_seed(args.seed)
    pref_root = Path(args.pref_root)
    if not pref_root.exists():
        raise FileNotFoundError(pref_root)

    log.info("loading Wan2.1-T2V-1.3B...")
    wan = sampling.load_wan_t2v(t5_cpu=True)

    results: dict[str, dict] = {}
    from peft import PeftModel

    def _detach():
        if isinstance(wan.model, PeftModel):
            wan.model = wan.model.get_base_model()

    for spec in args.runs:
        name, path = spec.split("=", 1)
        log.info("=== run %s ===", name)
        _detach()
        if path.lower() != "baseline":
            wan.model.requires_grad_(False)
            wan.model = PeftModel.from_pretrained(wan.model, path)
            wan.model.eval()
        out = measure_one_run(wan, pref_root, args.num_draws, args.beta, args.T, args.seed)
        log.info("%s: %s", name, out)
        results[name] = out

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    log.info("written: %s", args.out)


if __name__ == "__main__":
    main()
