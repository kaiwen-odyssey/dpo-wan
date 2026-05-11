# dpo-wan

Bidirectional Diffusion-DPO on Wan2.1-T2V-1.3B with three separate VideoAlign reward signals
(Motion Quality, Visual Quality, Text Alignment).

## Layout
- `src/dpo_wan/` — library code (rewards, sampling, dpo loss, training loop, eval)
- `scripts/` — pipeline drivers (prompts -> pairs -> train -> eval)
- `external/` — vendored Wan2.1 reference code and cloned VideoAlign
- `configs/` — YAML configs for runs and ablations
- `data/` — prompts, generated videos, latent caches, preference pairs
- `runs/` — training logs and LoRA checkpoints
- `paper/` — LaTeX source and compiled PDF

## Pipeline
1. **Prompts**: sample VidProM into train/eval/ablation splits
2. **Generate**: K candidates per prompt via Wan2.1-T2V-1.3B
3. **Score**: VideoAlign returns `{VQ, MQ, TA}` per video
4. **Pairs**: per reward dimension, take argmax/argmin -> (chosen, rejected)
5. **DPO train**: LoRA on Wan diffusion transformer, Diffusion-DPO loss
6. **Eval**: holdout generation + reward scoring + win rates
