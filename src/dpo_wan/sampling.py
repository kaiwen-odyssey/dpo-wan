"""Wan2.1-T2V-1.3B sampling wrapper for candidate generation.

Wraps `external/wan/text2video.py:WanT2V.generate` with:
  * Fixed deterministic seeds for K-candidate sampling per prompt.
  * LoRA hooks (loadable adapter checkpoint applied to `self.model`).
  * Latent-only sampling for the DPO training pass (no VAE decode).
"""
from __future__ import annotations

import dataclasses
import gc
import math
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.cuda.amp as amp
from tqdm import tqdm

from . import paths

paths.add_external_to_sys_path()

from wan import configs as wan_configs  # noqa: E402
from wan.text2video import WanT2V  # noqa: E402
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler  # noqa: E402


@dataclasses.dataclass
class SampleSpec:
    """All settings for a single rollout (we keep small/fast defaults for DPO)."""

    size: tuple[int, int] = (832, 480)            # (W, H) — supported by 1.3B
    frame_num: int = 33                           # 4n+1, ~2 sec at 16 fps after VAE
    fps: int = 16
    sampling_steps: int = 25
    guide_scale: float = 5.0
    shift: float = 5.0
    sample_solver: str = "unipc"


def load_wan_t2v(
    checkpoint_dir: Path | str | None = None,
    device_id: int = 0,
    t5_cpu: bool = True,
) -> WanT2V:
    """Load Wan2.1-T2V-1.3B (T5 on CPU keeps GPU memory flat)."""
    cfg = wan_configs.t2v_1_3B
    ckpt = str(checkpoint_dir or paths.WAN_CHECKPOINT_DIR)
    return WanT2V(config=cfg, checkpoint_dir=ckpt, device_id=device_id, t5_cpu=t5_cpu)


def attach_lora_to_dit(
    wan: WanT2V,
    lora_state_dict: dict | None = None,
    lora_rank: int = 16,
    lora_alpha: float = 16.0,
    target_modules: tuple[str, ...] = (
        # Attention QKV/output and MLP projections inside WanAttentionBlock.
        "q", "k", "v", "o",
        "ffn.0", "ffn.2",
    ),
    gradient_checkpointing: bool = True,
):
    """Wrap the diffusion transformer with PEFT-LoRA.

    Returns the PEFT-wrapped model; the wrapper still exposes `.forward(x, t, ...)` so
    sampling code paths continue to work.
    """
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict

    base = wan.model
    base.requires_grad_(False)

    config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=list(target_modules),
        lora_dropout=0.0,
        bias="none",
    )
    peft_model = get_peft_model(base, config)
    if lora_state_dict is not None:
        set_peft_model_state_dict(peft_model, lora_state_dict)

    if gradient_checkpointing:
        # Wrap each WanAttentionBlock in torch.utils.checkpoint so activations
        # don't pile up.  We do this on the unwrapped base model so PEFT's
        # forward-hooks still see the right call signature.
        import torch.utils.checkpoint as ckpt
        from functools import partial as _partial

        def _ckpt_forward(orig_forward, *a, **kw):
            return ckpt.checkpoint(orig_forward, *a, use_reentrant=False, **kw)

        for blk in base.blocks:
            if not getattr(blk, "_dpo_wan_ckpt", False):
                orig = blk.forward
                blk.forward = _partial(_ckpt_forward, orig)
                blk._dpo_wan_ckpt = True

    # Re-wire so existing sampling code (`self.model(...)`) hits the LoRA-wrapped model.
    wan.model = peft_model
    return peft_model


@torch.no_grad()
def generate_video(
    wan: WanT2V,
    prompt: str,
    seed: int,
    spec: SampleSpec | None = None,
) -> torch.Tensor:
    """Run a full T2V rollout. Returns (C, F, H, W) tensor in [-1, 1]."""
    spec = spec or SampleSpec()
    out = wan.generate(
        input_prompt=prompt,
        size=spec.size,
        frame_num=spec.frame_num,
        shift=spec.shift,
        sample_solver=spec.sample_solver,
        sampling_steps=spec.sampling_steps,
        guide_scale=spec.guide_scale,
        seed=seed,
        offload_model=False,
    )
    return out


@torch.no_grad()
def generate_K_candidates(
    wan: WanT2V,
    prompt: str,
    base_seed: int,
    K: int = 2,
    spec: SampleSpec | None = None,
) -> list[torch.Tensor]:
    """Sample K diverse candidates by varying the seed."""
    spec = spec or SampleSpec()
    return [generate_video(wan, prompt, seed=base_seed + i, spec=spec) for i in range(K)]


@torch.no_grad()
def encode_video_to_latent(wan: WanT2V, video: torch.Tensor) -> torch.Tensor:
    """VAE-encode a (C, F, H, W) video in [-1, 1] -> latent (Z, F', H/8, W/8)."""
    v = video.to(wan.device).unsqueeze(0)  # (1, C, F, H, W)
    latents = wan.vae.encode([v[0]])
    return latents[0].detach().cpu()


@torch.no_grad()
def decode_latent_to_video(wan: WanT2V, latent: torch.Tensor) -> torch.Tensor:
    """VAE-decode a latent -> (C, F, H, W) in [-1, 1]."""
    z = latent.to(wan.device)
    out = wan.vae.decode([z])
    return out[0].detach().cpu()


# ---------------------------------------------------------------------------
# Helpers used by the DPO trainer
# ---------------------------------------------------------------------------

def encode_text(wan: WanT2V, prompts: Sequence[str]) -> list[torch.Tensor]:
    """Encode a batch of prompts to T5 hidden states (list of [L, C] tensors)."""
    if wan.t5_cpu:
        ctx = wan.text_encoder(list(prompts), torch.device("cpu"))
        ctx = [t.to(wan.device) for t in ctx]
    else:
        wan.text_encoder.model.to(wan.device)
        ctx = wan.text_encoder(list(prompts), wan.device)
    return ctx


def encode_negative(wan: WanT2V, batch: int = 1) -> list[torch.Tensor]:
    """Encode the configured negative prompt batch times."""
    return encode_text(wan, [wan.sample_neg_prompt] * batch)


def make_seq_len(wan: WanT2V, latent_shape: tuple[int, int, int, int]) -> int:
    """Replicate the seq_len computation from `WanT2V.generate`."""
    z, f, h, w = latent_shape
    p1, p2 = wan.patch_size[1], wan.patch_size[2]
    return math.ceil((h * w) / (p1 * p2) * f / wan.sp_size) * wan.sp_size
