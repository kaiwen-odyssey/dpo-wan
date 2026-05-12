"""Diffusion-DPO training loop for Wan2.1 with PEFT-LoRA.

Key tradeoffs:
  * LoRA on the DiT only — leaves T5/VAE frozen and saves >25 GB of VRAM.
  * The reference model is the same DiT with LoRA *disabled* (PEFT
    `disable_adapter()` context manager). No second copy of weights.
  * Latents and text embeddings are pre-cached, so the GPU only ever holds
    DiT activations and the small LoRA optimizer state.

Logged metrics (W&B):
  loss/...                 raw and EMA-smoothed DPO loss
  accuracy/...             fraction of pairs with margin>0 (and EMA)
  margin/...               implicit-reward margin (mean / per-pair hist)
  implicit_reward/{w,l,gap}   DPO's own log-ratio reward, decomposed
  alignment/score_gap      reward-side preference gap from VideoReward (data)
  alignment/spearman       per-batch Spearman ρ(implicit-margin, score-gap)
  alignment/pearson        per-batch Pearson ρ                "
  timestep/{mean,hist}     diffusion-timestep distribution per step
  timestep_loss/...        loss bucketed by timestep band (early/mid/late)
  optim/{grad_norm,param_delta_norm,lr}   gradient health + drift from init
"""
from __future__ import annotations

import dataclasses
import json
import logging
import time
from pathlib import Path
from typing import Iterable

import math
import numpy as _np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from . import dpo, paths, sampling
from .data import LatentPreferenceDataset, collate_preferences

log = logging.getLogger(__name__)


def _safe_correlations(margin: _np.ndarray, score_gap: _np.ndarray) -> tuple[float, float]:
    """Return (spearman, pearson) ρ; NaN-safe; handles single-element windows."""
    valid = ~_np.isnan(margin) & ~_np.isnan(score_gap)
    if int(valid.sum()) < 3 or _np.var(margin[valid]) < 1e-12 or _np.var(score_gap[valid]) < 1e-12:
        return float("nan"), float("nan")
    m = margin[valid]
    s = score_gap[valid]
    pearson = float(_np.corrcoef(m, s)[0, 1])
    rm = _np.argsort(_np.argsort(m)).astype(float)
    rs = _np.argsort(_np.argsort(s)).astype(float)
    if _np.var(rm) < 1e-12 or _np.var(rs) < 1e-12:
        spearman = float("nan")
    else:
        spearman = float(_np.corrcoef(rm, rs)[0, 1])
    return spearman, pearson


def _bucket_by_timestep(margin: _np.ndarray, t: _np.ndarray) -> dict[str, float]:
    """Mean implicit margin in three diffusion-timestep bands.

    Diagnoses *where* in the noise schedule DPO is making progress.  Late-t
    progress (t close to 1, near-pure-noise) is the typical bottleneck for
    rectified-flow DPO; mid-t is where v-prediction error is highest signal.
    """
    out: dict[str, float] = {}
    for name, lo, hi in [("early", 0.0, 0.33), ("mid", 0.33, 0.66), ("late", 0.66, 1.01)]:
        mask = (t >= lo) & (t < hi)
        out[name] = float(margin[mask].mean()) if mask.any() else float("nan")
    return out


@dataclasses.dataclass
class TrainConfig:
    pref_root: str
    output_dir: str
    reward_dim: str
    lora_rank: int = 16
    lora_alpha: float = 16.0
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    max_steps: int = 200
    log_every: int = 5
    save_every: int = 100
    eval_every: int = 100
    seed: int = 0
    dpo_beta: float = 500.0
    dpo_T: float = 1000.0
    shared_noise: bool = True
    lambda_sft: float = 0.0           # SFT anchor on chosen (DPO+SFT/cDPO)
    bf16: bool = True
    grad_clip: float = 1.0
    dataloader_workers: int = 2

    # Stability extras
    warmup_steps: int = 5
    lr_min_ratio: float = 0.1          # cosine decay floor

    # W&B
    wandb_project: str = "dpo-wan"
    wandb_run_name: str | None = None
    wandb_mode: str = "offline"        # "online" / "offline" / "disabled"
    wandb_tags: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _model_forward(wan, latents_list, t_scalar, ctx_list, seq_len):
    """Wrap WanModel.forward to fit our (B, Z, F, H, W) batches.

    WanModel expects a *list* of (Z, F, H, W) tensors.  We unpack on entry and
    re-stack on exit.  The diffusion timestep is scaled to [0, num_train_timesteps]
    (Wan's internal convention) before being passed through.
    """
    B = len(latents_list)
    # Wan's flow-matching scheduler internally treats t in [0, num_train_timesteps).
    timestep = (t_scalar * wan.num_train_timesteps).clamp(min=1.0).to(latents_list[0].dtype)
    out = wan.model(latents_list, t=timestep, context=ctx_list, seq_len=seq_len)
    return torch.stack(out, dim=0)


def train_dpo(cfg: TrainConfig, wan: sampling.WanT2V) -> Path:
    """Run training for `cfg.max_steps` and save the LoRA adapter."""
    from peft import set_peft_model_state_dict, get_peft_model_state_dict

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))

    # Initialise W&B (offline by default — produces a run dir under wandb/ that
    # `wandb sync` can ship later).
    wb = _init_wandb(cfg, out_dir)

    # Build dataset / dataloader
    ds = LatentPreferenceDataset(cfg.pref_root)
    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.dataloader_workers,
        collate_fn=collate_preferences,
        drop_last=True,
        persistent_workers=cfg.dataloader_workers > 0,
    )
    log.info("Loaded %d preference pairs from %s", len(ds), cfg.pref_root)

    # Attach LoRA -- this swaps wan.model in place.
    peft_model = sampling.attach_lora_to_dit(
        wan, lora_rank=cfg.lora_rank, lora_alpha=cfg.lora_alpha
    )
    peft_model.train()
    trainable = [p for p in peft_model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    log.info("LoRA trainable params: %.2f M", n_trainable / 1e6)

    optim = AdamW(trainable, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    dpo_cfg = dpo.DPOConfig(
        beta=cfg.dpo_beta, timestep_T=cfg.dpo_T, shared_noise=cfg.shared_noise,
        lambda_sft=cfg.lambda_sft,
    )

    def _lr_at_step(s: int) -> float:
        # Linear warmup, then cosine decay to lr_min_ratio * base.
        if s < cfg.warmup_steps:
            return cfg.learning_rate * (s + 1) / max(cfg.warmup_steps, 1)
        decay_steps = max(cfg.max_steps - cfg.warmup_steps, 1)
        progress = (s - cfg.warmup_steps) / decay_steps
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return cfg.learning_rate * (cfg.lr_min_ratio + (1 - cfg.lr_min_ratio) * cosine)

    # Snapshot LoRA-A param init for drift diagnostic.
    lora_init_norm_sq = 0.0
    init_param_snapshot: dict[str, torch.Tensor] = {}
    for n, p in peft_model.named_parameters():
        if p.requires_grad:
            init_param_snapshot[n] = p.detach().clone()
            lora_init_norm_sq += float(p.detach().pow(2).sum().item())

    history: list[dict] = []
    # Per-optimizer-step UUID log: list of dicts {step: int, uuids: [str]}.
    # With grad_accum > 1, each step's uuids = concatenation of the uuids in
    # all micro-batches that contributed to that gradient.  Used by
    # `checkpoint_reward_eval.py` to score the policy on the prompts seen in
    # the just-trained 50-step window.
    uuid_log: list[dict] = []
    step_uuid_buffer: list[str] = []
    step = 0
    t0 = time.time()
    accum = 0
    optim.zero_grad(set_to_none=True)

    # EMA state for smoothed metrics.
    ema = {"loss": None, "accuracy": None, "margin": None, "impl_margin": None}
    ema_alpha = 0.1

    # Cumulative buffers for per-pair scatter (reset each log window).
    buf_margin: list[float] = []
    buf_score_gap: list[float] = []
    buf_t: list[float] = []
    buf_loss_per_pair: list[float] = []

    while step < cfg.max_steps:
        for batch in loader:
            if step >= cfg.max_steps:
                break

            chosen   = batch["chosen"].to(wan.device,   non_blocking=True)
            rejected = batch["rejected"].to(wan.device, non_blocking=True)
            text_list = [t.to(wan.device, non_blocking=True) for t in batch["text"]]

            x_w_t, x_l_t, v_w_tgt, v_l_tgt, t = dpo.dpo_step_inputs(
                chosen, rejected, dpo_cfg
            )

            seq_len = sampling.make_seq_len(wan, tuple(chosen.shape[1:]))

            # Predictions from the trainable (LoRA-on) model.
            with torch.amp.autocast("cuda", dtype=torch.bfloat16 if cfg.bf16 else torch.float32):
                v_pred_w = _model_forward(
                    wan, list(x_w_t), t, text_list, seq_len
                )
                v_pred_l = _model_forward(
                    wan, list(x_l_t), t, text_list, seq_len
                )

                # Reference model = same weights with LoRA disabled.
                with torch.no_grad(), peft_model.disable_adapter():
                    v_ref_w = _model_forward(
                        wan, list(x_w_t), t, text_list, seq_len
                    )
                    v_ref_l = _model_forward(
                        wan, list(x_l_t), t, text_list, seq_len
                    )

                metrics = dpo.diffusion_dpo_loss(
                    v_pred_w, v_pred_l, v_ref_w, v_ref_l, v_w_tgt, v_l_tgt, dpo_cfg
                )
                loss = metrics["loss"] / cfg.gradient_accumulation_steps

            loss.backward()

            # ---- track which UUIDs this opt step touched -----
            step_uuid_buffer.extend(batch.get("ids", []))

            # ---- accumulate per-pair signal across micro-batches ------
            mp = metrics["margin_per_pair"].detach().float().cpu().tolist()
            tp = t.detach().float().cpu().tolist()
            sg = []
            cs = batch.get("chosen_score") if isinstance(batch, dict) else None
            rs = batch.get("rejected_score") if isinstance(batch, dict) else None
            # `cs`/`rs` are produced by `LatentPreferenceDataset` per-sample;
            # `collate_preferences` already packs them into the batch keys
            # "chosen" / "rejected" but the *score* fields come through as
            # plain Python lists if they're attached.  Defensively rebuild
            # them from the dataset entries below if not present.
            chosen_scores  = batch.get("chosen_scores")  or []
            rejected_scores = batch.get("rejected_scores") or []
            for i in range(len(mp)):
                buf_margin.append(float(mp[i]))
                buf_t.append(float(tp[i]))
                gap = (float(chosen_scores[i]) - float(rejected_scores[i])
                       if chosen_scores and rejected_scores else float("nan"))
                buf_score_gap.append(gap)
                buf_loss_per_pair.append(float(metrics["loss"].detach().item()))

            accum += 1

            if accum >= cfg.gradient_accumulation_steps:
                # Pre-clip grad norm for stability tracking.
                grad_norm_pre = float(torch.nn.utils.clip_grad_norm_(
                    trainable, max_norm=float("inf")
                ).item())
                if cfg.grad_clip:
                    torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
                # warmup + cosine schedule
                lr_now = _lr_at_step(step)
                for pg in optim.param_groups:
                    pg["lr"] = lr_now
                optim.step()
                optim.zero_grad(set_to_none=True)
                accum = 0
                step += 1
                # Flush per-step uuid log
                uuid_log.append({"step": step, "uuids": list(step_uuid_buffer)})
                step_uuid_buffer = []

                # ----- EMA smoothing -----
                cur_loss = float(metrics["loss"].item())
                cur_acc  = float(metrics["accuracy"].item())
                cur_marg = float(metrics["margin"].item())
                cur_imrg = float(metrics["implicit_reward_margin"].mean().item())
                for k, v_cur in [("loss", cur_loss), ("accuracy", cur_acc),
                                 ("margin", cur_marg), ("impl_margin", cur_imrg)]:
                    ema[k] = v_cur if ema[k] is None else (
                        ema_alpha * v_cur + (1 - ema_alpha) * ema[k]
                    )

                if step % cfg.log_every == 0:
                    # Aggregate per-pair signal collected since last log.
                    mp_arr   = _np.array(buf_margin or [0.0])
                    sg_arr   = _np.array(buf_score_gap or [_np.nan])
                    t_arr    = _np.array(buf_t or [0.0])

                    # Spearman / Pearson between implicit margin and score gap
                    # — high positive ρ means DPO is genuinely tracking the
                    # VideoReward preference signal, not just memorising.
                    spearman_rho, pearson_rho = _safe_correlations(mp_arr, sg_arr)

                    # Bucket loss by timestep band (early=low t, late=high t).
                    buckets = _bucket_by_timestep(mp_arr, t_arr)

                    # Param drift from init (relative).
                    cur_norm_sq = 0.0
                    delta_norm_sq = 0.0
                    for n, p in peft_model.named_parameters():
                        if p.requires_grad and n in init_param_snapshot:
                            cur_norm_sq   += float(p.detach().pow(2).sum().item())
                            delta_norm_sq += float(
                                (p.detach() - init_param_snapshot[n]).pow(2).sum().item()
                            )
                    param_norm = cur_norm_sq ** 0.5
                    param_delta_norm = delta_norm_sq ** 0.5
                    relative_drift = (param_delta_norm /
                                      max(lora_init_norm_sq ** 0.5, 1e-9))

                    dt = time.time() - t0
                    rec = {
                        "step": step,
                        "elapsed_s": dt,

                        # Loss / accuracy (raw + EMA)
                        "loss/raw":        cur_loss,
                        "loss/ema":        ema["loss"],
                        "loss/dpo":        float(metrics["loss_dpo"].item()),
                        "loss/sft":        float(metrics["loss_sft"].item()),
                        "accuracy/raw":    cur_acc,
                        "accuracy/ema":    ema["accuracy"],

                        # Margin and per-pair distribution
                        "margin/mean":     cur_marg,
                        "margin/ema":      ema["margin"],
                        "margin/p10":      float(_np.percentile(mp_arr, 10)),
                        "margin/p90":      float(_np.percentile(mp_arr, 90)),
                        "margin/positive_frac": float((mp_arr > 0).mean()),

                        # Implicit DPO rewards (= -beta * delta)
                        "implicit_reward/chosen":   float(metrics["implicit_reward_w"].mean().item()),
                        "implicit_reward/rejected": float(metrics["implicit_reward_l"].mean().item()),
                        "implicit_reward/gap":      float(metrics["implicit_reward_margin"].mean().item()),
                        "implicit_reward/gap_ema":  ema["impl_margin"],

                        # Did the policy improve over ref on chosen?
                        "delta_w/mean": float(metrics["delta_w"].item()),
                        "delta_l/mean": float(metrics["delta_l"].item()),

                        # Reward-side data signal (from VideoReward at pair-build time).
                        "alignment/score_gap_mean": float(_np.nanmean(sg_arr)),
                        "alignment/score_gap_min":  float(_np.nanmin(sg_arr)),
                        "alignment/spearman":       spearman_rho,
                        "alignment/pearson":        pearson_rho,

                        # Per-timestep diagnostic loss
                        "timestep/mean":          float(t_arr.mean()),
                        "timestep_margin/early":  buckets["early"],
                        "timestep_margin/mid":    buckets["mid"],
                        "timestep_margin/late":   buckets["late"],

                        # Optimisation health
                        "optim/grad_norm":           grad_norm_pre,
                        "optim/lr":                  optim.param_groups[0]["lr"],
                        "optim/param_norm":          param_norm,
                        "optim/param_delta_norm":    param_delta_norm,
                        "optim/relative_drift":      relative_drift,
                    }
                    history.append(rec)
                    log.info(
                        "step %4d | loss %.4f (ema %.4f) | acc %.3f | "
                        "margin %+.4f | rho(impl,score)=%+.2f | gn %.2f | "
                        "drift %.3f%% | %.1fs",
                        rec["step"], rec["loss/raw"], rec["loss/ema"], rec["accuracy/raw"],
                        rec["margin/mean"], rec["alignment/spearman"],
                        rec["optim/grad_norm"], 100 * rec["optim/relative_drift"],
                        rec["elapsed_s"],
                    )
                    if wb is not None:
                        try:
                            log_payload = dict(rec)
                            # Histograms and a small scatter for the dashboard.
                            try:
                                import wandb as _wb
                                log_payload["margin/hist"] = _wb.Histogram(mp_arr)
                                log_payload["timestep/hist"] = _wb.Histogram(t_arr)
                                if not _np.isnan(sg_arr).all():
                                    log_payload["alignment/score_gap_hist"] = _wb.Histogram(
                                        sg_arr[~_np.isnan(sg_arr)]
                                    )
                            except Exception:                          # noqa: BLE001
                                pass
                            wb.log(log_payload, step=step)
                        except Exception:                              # noqa: BLE001
                            pass

                    # reset accumulator buffers for next log window
                    buf_margin.clear(); buf_score_gap.clear()
                    buf_t.clear(); buf_loss_per_pair.clear()

                if step % cfg.save_every == 0 or step == cfg.max_steps:
                    _save_checkpoint(peft_model, out_dir / f"step_{step:06d}", history)
                    # Flush uuid log alongside the checkpoint
                    (out_dir / "uuid_log.json").write_text(json.dumps(uuid_log, indent=0))

    # Final save
    final = _save_checkpoint(peft_model, out_dir / "final", history)
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    if wb is not None:
        try:
            wb.summary["final_loss"] = history[-1]["loss"] if history else None
            wb.summary["final_accuracy"] = history[-1]["accuracy"] if history else None
            wb.finish()
        except Exception:                                              # noqa: BLE001
            pass
    return final


def _init_wandb(cfg: TrainConfig, out_dir: Path):
    if cfg.wandb_mode == "disabled":
        return None
    try:
        import wandb
    except ImportError:
        log.warning("wandb not installed; skipping logging")
        return None
    run_name = cfg.wandb_run_name or out_dir.name
    try:
        run = wandb.init(
            project=cfg.wandb_project,
            name=run_name,
            mode=cfg.wandb_mode,
            dir=str(out_dir),
            tags=list(cfg.wandb_tags) + [cfg.reward_dim],
            config=cfg.to_dict(),
            reinit=True,
        )
        return run
    except Exception as exc:                                            # noqa: BLE001
        log.warning("wandb init failed (%s); continuing without logging", exc)
        return None


def _save_checkpoint(peft_model, ckpt_dir: Path, history: list[dict]) -> Path:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(str(ckpt_dir))
    (ckpt_dir / "history.json").write_text(json.dumps(history, indent=2))
    return ckpt_dir
