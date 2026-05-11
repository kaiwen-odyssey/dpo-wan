# dpo-wan — experimental log

Diffusion-DPO on Wan2.1-T2V-1.3B with VideoAlign (KwaiVGI/VideoReward) rewards.
This file is the authoritative record of what's been tried, what worked, what
didn't, and the numbers behind it. Future sessions should read this before
proposing experiments.

---

## Project structure

```
src/dpo_wan/      library: rewards, sampling, dpo loss, training, eval, prefs
external/wan/     vendored Wan2.1 (SDPA fallback patched into attention.py)
external/VideoAlign/  cloned; checkpoints/VideoReward/ has the reward weights
data/             prompts, videos, latents, T5 text embeds, preferences, scores
runs/             trainings (abl_*, main_*, main_MQ_sft, etc.) + evals
paper/            NeurIPS 2024 template at paper.tex; compile.sh
scripts/          numbered drivers (00..08) plus stage launchers
```

## Stable training recipe

After exhausting many failure modes, this is the validated stable DPO recipe:

* `lr = 1e-5`
* `beta = 10`
* `grad_clip = 0.1`
* `warmup_steps = 10`, cosine to 0.1·lr
* `batch_size = 1`, `grad_accum = 1`
* `lora_rank = 16` on q/k/v/o/ffn.0/ffn.2 of every WanAttentionBlock
* gradient checkpointing on each block
* SFT anchor: λ ∈ [0, 1] safe; λ ≥ 10 dominates DPO at scale and *hurts*
* Sampling for video gen: 832×480, 21 frames, 15 UniPC steps, cfg=5.0

DO NOT use β=500 (literature default) — it saturates log-σ and explodes
gradients with our latent scale. DO NOT use β=1 — gradient signal is too
small to actually move the model.

## Datasets

* VidProM prompts, seed=0 sample.
* Train: 1000 prompts, K=2 candidates each → 1000 preference pairs per
  reward dim after argmax/argmin ranking.
* Ablation: 20 prompts, K=2 → 20 MQ pairs (shared across sweep configs).
* Eval (holdout): 20 prompts, 1 seed per (run, prompt).

Score-gap distribution on training (1000 pairs):

| dim | mean | p25 | p50 | p75 | p90 |
|---|---|---|---|---|---|
| MQ | 0.67 | 0.21 | 0.50 | 0.93 | 1.52 |
| VQ | 0.56 | 0.18 | 0.42 | 0.76 | 1.25 |
| TA | 0.44 | 0.15 | 0.33 | 0.63 | 0.95 |

## Stability ablation (stage 3, 20-prompt set, 80 steps, β-lr sweep)

| η | β | loss std | loss EMA final | acc EMA | mean ρ | gn p90 | drift |
|---|---|---|---|---|---|---|---|
| 1e-5 | 1   | **0.035** | 0.695 | 0.51 | **+0.10** | **26** | 0.79 % |
| 1e-5 | 10  | 0.129 | 0.667 | 0.59 | -0.02 | 223 | 0.89 % |
| 1e-5 | 50  | 0.595 | 0.716 | 0.60 | +0.07 | 609 | 0.78 % |
| 1e-5 | 100 | 0.562 | 1.182 | 0.63 | -0.20 | 1438 | 0.78 % |
| 5e-6 | 10  | 0.174 | 0.699 | 0.43 | +0.02 | 432 | 0.39 % |
| 5e-6 | 50  | 0.574 | 0.588 | 0.79 | -0.14 | 756 | 0.37 % |
| 5e-6 | 100 | 2.017 | 0.911 | 0.75 | -0.15 | 3257 | 0.42 % |

W&B: https://wandb.ai/odyssey/dpo-wan-ablation

**Lesson:** β=1 is most stable but trains too slowly to move the model at
scale; β=10 is the sweet spot. β≥50 saturates log-σ.

## SFT-anchor ablation (80 steps on ablation split, β=10, lr=1e-5)

The 4-run sweep predicted λ=10 would win, but it did not generalise to scale.

| λ | loss_dpo | R_chosen | R_rej | R_gap | drift |
|---|---|---|---|---|---|
| 0.0 (from main_MQ_m0.2) | n/a | -42e-5 | -57e-5 | +15e-5 | 1.85 % |
| 0.1 | 0.765 | -19e-5 | -10e-5 | -8e-5 | 0.90 % |
| **1.0** | 0.543 | -2e-5 | -16e-5 | **+14e-5** | 0.89 % |
| 10.0 | 0.510 | -9e-5 | -16e-5 | +6e-5 | 0.90 % |

Holdout (20 prompts) for these short λ runs:
* λ=10.0: **MQ Δ=+0.019**, win 55 % (BEST short-scale)
* λ=1.0: MQ Δ=-0.014
* λ=0.1: MQ Δ=+0.005

W&B: https://wandb.ai/odyssey/dpo-wan-ablation (look for `abl_MQ_lam*` runs)

## Main MQ at scale (1000 pairs, 400 steps, β=10, lr=1e-5)

Three variants trained on the same prompt set:

| run | filter | λ | loss EMA | acc EMA | R_chosen late | R_gap late | drift |
|---|---|---|---|---|---|---|---|
| **main_MQ**       | none           | 0  | 0.659 | 0.49 | -7e-5 | +2e-5 | 1.63 % |
| main_MQ_m0.2     | score gap≥0.2  | 0  | 0.655 | 0.48 | **-45e-5 (deflation)** | +16e-5 | 1.85 % |
| main_MQ_sft      | none           | 10 | 2.323 | 0.62 | -7e-5 | **-6e-5 (negative gap)** | 1.63 % |

W&B (main):
* main_MQ:       https://wandb.ai/odyssey/dpo-wan/runs/dry4ttou  (was β=50; redone at β=10)
* main_MQ_m0.2:  https://wandb.ai/odyssey/dpo-wan-ablation/runs/srn2brxk (offline-only earlier, may need sync)
* main_MQ_sft:   https://wandb.ai/odyssey/dpo-wan/runs/m75n6jxz   (offline-only, may need sync)

### Holdout eval (20 prompts, 1 seed) for the three MQ variants

| variant | MQ mean | MQ Δ | MQ win | VQ Δ | VQ win | TA Δ | TA win |
|---|---|---|---|---|---|---|---|
| baseline (no LoRA)         | 0.845 | —      | —    | —      | —    | —      | —    |
| **main_MQ** (best)         | 0.900 | **+0.055** | 50 % | +0.029 | 50 % | -0.002 | 50 % |
| main_MQ_m0.2 (filtered)    | 0.835 | -0.010 | 55 % | +0.015 | 65 % | -0.032 | 30 % |
| main_MQ_sft (λ=10)         | 0.798 | -0.047 | 40 % | -0.027 | 45 % | -0.024 | 35 % |

Confidence: ±22 pp 95 % CI on win rates at N=20 / 1 seed; mean shifts are
more credible than win-rate ranks.

**Headline finding:** plain DPO at β=10 over 1000 unfiltered pairs is the
best single config we've tested. Filtering and SFT-anchor (at scales we
explored) both hurt the absolute reward despite winning some training-side
diagnostics.

### Why filtering hurt

Filtering at margin ≥ 0.2 (767 pairs kept) gave the strongest per-pair
gradient signal — R_gap reached +16×10⁻⁵ in the last training quartile.
But the model used that signal to push *both* chosen and rejected
log-probabilities *down* (chosen by -45e-5, rejected by -57e-5). Classic
DPO deflation (Pal et al. 2024). Holdout reward dropped because the
policy got worse at *every* generation, just slightly less worse at chosen.

### Why SFT anchor at λ=10 hurt

Total loss at λ=10 was ~70 % SFT, ~30 % DPO. The model effectively did
diffusion-SFT on chosen videos with a tiny DPO correction. But the "chosen"
videos are themselves not aspirational — they're just the better of two
mediocre samples per prompt. SFT-anchoring to them pulled both chosen
*and* rejected log-probs up; the optimizer found it could satisfy the
combined loss by lifting both, which yielded a *negative* implicit-margin
gap (R_gap = -6e-5).

## Failed approaches (record for future-me)

* β=500 / lr=1e-4 (literature defaults): loss oscillates 0 ↔ 100, grad
  norms 10⁴+. Caused by log-σ saturation. Switched to β=10 / lr=1e-5.
* β=1 with cosine LR: technically the most stable training, but loss/margin
  stay flat at the DPO equilibrium (ln 2). Gradient too small to actually
  move the policy at scale.
* Margin-filter (≥0.2): tighter signal → deflation. See above.
* SFT anchor λ=10 at scale: SFT dominates the loss, holdout reward drops.

## Open questions / unfinished work

* Try λ=1.0 at scale — 80-step ablation showed it had the cleanest training
  signal (R_gap=+14e-5 with R_chosen near zero). We never ran it at
  1000-pair / 400-step scale.
* Run VQ and TA at the chosen winning config (currently only MQ has scale
  results — paper headline table needs all 3 dims).
* Multi-seed evaluation: N=20 / 1 seed has ±22 pp CI on win rates.
* Generation budget: 21-frame clips at 832×480, 15 sampling steps. Wan2.1
  default is 81 frames / 50 steps — our trends may not transfer.

## Paper

`paper/paper.tex` (NeurIPS 2024 template). Compile via `paper/compile.sh`.
Currently has 4 tables + 6 figures rendered by `scripts/07_render_paper.py`.
The compiled PDF is named
`dpo-wan_bidirectional-DPO-on-Wan2.1_2026-05-10.pdf`
and is mirrored at the root of the user's Google Drive (rclone remote
`gdrive:`).

Last paper update reflects β=50 results — needs a re-render once the
β=10 main runs (and VQ/TA scale runs) are settled.

## Operational notes

* Single RTX PRO 6000 (96 GB) is shared with a co-located Wan_RF
  inference server holding 42 GB. Effective budget = ~54 GB.
* T5 encoder stays on CPU (`t5_cpu=True`) to fit; this is non-negotiable.
* `02_generate_and_score.py` is idempotent: skipping cached videos is
  safe across kills/restarts.
* Drive uploads use the locally-configured rclone remote `gdrive:`; the
  claude.ai MCP Drive connector needs OAuth in the web app and is not
  usable from Claude Code.
