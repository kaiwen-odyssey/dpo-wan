"""Diffusion-DPO loss for Wan2.1 (rectified-flow / v-prediction parameterization).

Reference: Wallace et al. 2023, *Diffusion Model Alignment Using Direct
Preference Optimization*. We adapt the original epsilon-DPO objective to the
flow-matching parameterization used by Wan2.1 by replacing the noise-prediction
error with the velocity-prediction error along the rectified-flow probability
path.

For a preference pair (x_w, x_l, c) at timestep t we compute:

    eps         ~ N(0, I)
    x_w_t       = (1 - t) * x_w + t * eps
    x_l_t       = (1 - t) * x_l + t * eps        # shared noise (Wallace §B.1)
    v_target_w  = eps - x_w
    v_target_l  = eps - x_l
    Δ_w         = ||v_θ(x_w_t) - v_w_target||² - ||v_ref(x_w_t) - v_w_target||²
    Δ_l         = ||v_θ(x_l_t) - v_l_target||² - ||v_ref(x_l_t) - v_l_target||²
    L_DPO       = -log σ(-β · T · (Δ_w - Δ_l))

`β` plays the role of the DPO inverse-temperature; the multiplicative `T = 1000`
is a convention from the Wallace implementation that keeps β at the same
~1e3 scale across pixel- and latent-space models.
"""
from __future__ import annotations

import dataclasses
from typing import Sequence

import torch
import torch.nn.functional as F


@dataclasses.dataclass
class DPOConfig:
    beta: float = 500.0
    timestep_T: float = 1000.0       # scale factor in front of (Δ_w - Δ_l)
    timestep_min: float = 1e-3
    timestep_max: float = 1.0
    shared_noise: bool = True        # use the same eps for chosen and rejected
    lambda_sft: float = 0.0          # SFT-anchor weight on chosen (cDPO/DPO+SFT)


def sample_timesteps(batch: int, cfg: DPOConfig, device: torch.device) -> torch.Tensor:
    """Sample t ~ U(t_min, t_max) for a flow-matching DPO step."""
    return torch.empty(batch, device=device).uniform_(cfg.timestep_min, cfg.timestep_max)


def add_flow_noise(
    x0: torch.Tensor, eps: torch.Tensor, t: torch.Tensor
) -> torch.Tensor:
    """Rectified-flow probability path: x_t = (1 - t) x_0 + t eps."""
    while t.dim() < x0.dim():
        t = t.unsqueeze(-1)
    return (1.0 - t) * x0 + t * eps


def flow_velocity_target(x0: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    """Target velocity along the rectified flow: v = eps - x_0."""
    return eps - x0


def _per_sample_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error reduced over everything except the batch dim."""
    diff = (pred - target).flatten(1)
    return (diff * diff).mean(dim=1)


def diffusion_dpo_loss(
    v_pred_w: torch.Tensor,
    v_pred_l: torch.Tensor,
    v_ref_w: torch.Tensor,
    v_ref_l: torch.Tensor,
    v_target_w: torch.Tensor,
    v_target_l: torch.Tensor,
    cfg: DPOConfig,
) -> dict[str, torch.Tensor]:
    r"""Diffusion-DPO loss + per-pair diagnostic tensors.

    Returned tensors that are *vectors* of length B let the caller compute
    correlations against per-pair side information (e.g.\ the gap in
    VideoReward scores), which is the most direct test of whether DPO is
    tracking the intended preference signal rather than overfitting noise.
    """
    err_pred_w = _per_sample_mse(v_pred_w, v_target_w)
    err_ref_w  = _per_sample_mse(v_ref_w,  v_target_w)
    err_pred_l = _per_sample_mse(v_pred_l, v_target_l)
    err_ref_l  = _per_sample_mse(v_ref_l,  v_target_l)

    delta_w = err_pred_w - err_ref_w        # policy minus ref error on chosen
    delta_l = err_pred_l - err_ref_l        # policy minus ref error on rejected
    margin  = -(delta_w - delta_l)          # implicit-reward margin (per pair)
    inside  = cfg.beta * cfg.timestep_T * margin
    loss_dpo = -F.logsigmoid(inside).mean()

    # Optional SFT anchor on chosen: penalise the policy's velocity error on
    # chosen directly, which prevents the "DPO deflation" failure mode where
    # both R_chosen and R_rejected collapse with R_rejected collapsing more.
    loss_sft = err_pred_w.mean()
    loss    = loss_dpo + cfg.lambda_sft * loss_sft

    with torch.no_grad():
        accuracy = (margin > 0).float().mean()
        # Implicit DPO rewards (per pair, in nats / unitless).
        # r_implicit = -beta * (||v_pi - v*||^2 - ||v_ref - v*||^2)
        impl_r_w = (-cfg.beta * delta_w).detach()
        impl_r_l = (-cfg.beta * delta_l).detach()
        impl_r_margin = (impl_r_w - impl_r_l).detach()      # = beta * margin

    return {
        "loss":              loss,
        "loss_dpo":          loss_dpo.detach(),
        "loss_sft":          loss_sft.detach(),
        "delta_w":           delta_w.mean().detach(),
        "delta_l":           delta_l.mean().detach(),
        "margin":            margin.mean().detach(),
        "accuracy":          accuracy,
        # Per-pair tensors -- the trainer pairs them with score gaps from data.
        "delta_w_per_pair":      delta_w.detach(),
        "delta_l_per_pair":      delta_l.detach(),
        "margin_per_pair":       margin.detach(),
        "implicit_reward_w":     impl_r_w,
        "implicit_reward_l":     impl_r_l,
        "implicit_reward_margin": impl_r_margin,
    }


# Convenience wrapper used by the trainer ----------------------------------------------------

def dpo_step_inputs(
    chosen_latents: torch.Tensor,
    rejected_latents: torch.Tensor,
    cfg: DPOConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample shared (or independent) noise + timesteps and return:

        (x_w_t, x_l_t, v_target_w, v_target_l, t)
    """
    device = chosen_latents.device
    B = chosen_latents.shape[0]
    t = sample_timesteps(B, cfg, device)

    eps_w = torch.randn_like(chosen_latents)
    eps_l = eps_w if cfg.shared_noise else torch.randn_like(rejected_latents)

    x_w_t = add_flow_noise(chosen_latents,   eps_w, t)
    x_l_t = add_flow_noise(rejected_latents, eps_l, t)

    v_target_w = flow_velocity_target(chosen_latents,   eps_w)
    v_target_l = flow_velocity_target(rejected_latents, eps_l)

    return x_w_t, x_l_t, v_target_w, v_target_l, t
