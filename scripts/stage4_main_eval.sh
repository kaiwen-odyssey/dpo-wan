#!/usr/bin/env bash
# Stage 4: 3 main DPO runs at the chosen-stable config + holdout eval.
#
# Hyperparameters chosen from stage 3 stability sweep:
#   lr=1e-5, beta=50, grad_clip=0.1, warmup_steps=10
# Highest positive Spearman correlation between implicit margin and the
# VideoReward score gap (the most diagnostic metric for "DPO is tracking
# the right signal"), with bounded loss EMA.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${REPO}/.venv/bin/python"
export PYTHONPATH="${REPO}/src:${REPO}/external:${REPO}/external/VideoAlign:${PYTHONPATH:-}"
export PYTORCH_ALLOC_CONF=expandable_segments:True

cd "${REPO}"
LOG="${REPO}/runs/stage4.log"
mkdir -p "${REPO}/runs"
exec > >(tee -a "$LOG") 2>&1
echo "=== stage4 main+eval started $(date) ==="

for dim in MQ VQ TA; do
    ${PY} scripts/04_train_dpo.py \
        --reward-dim "${dim}" \
        --split train \
        --run-name "main_${dim}" \
        --lr 1e-5 --beta 50 \
        --batch-size 1 --grad-accum 1 \
        --max-steps 100 --lora-rank 16 \
        --wandb-mode offline
done

# Holdout eval (baseline + 3 LoRAs on 20 prompts).
${PY} scripts/05_eval.py \
    --runs \
        baseline=baseline \
        MQ="${REPO}/runs/main_MQ/final" \
        VQ="${REPO}/runs/main_VQ/final" \
        TA="${REPO}/runs/main_TA/final" \
    --out "${REPO}/runs/eval_holdout" \
    --frame-num 21 --steps 15

# Render paper artefacts and compile.
${PY} scripts/07_render_paper.py \
    --eval-dir "${REPO}/runs/eval_holdout" \
    --abl-eval-dir "${REPO}/runs/ablation_eval" \
    --scale "${REPO}/configs/scale.json" || true

bash "${REPO}/paper/compile.sh" || true

# Sync everything to W&B online.
${REPO}/.venv/bin/wandb sync \
    "${REPO}/runs/main_MQ/wandb/offline-run-"* \
    "${REPO}/runs/main_VQ/wandb/offline-run-"* \
    "${REPO}/runs/main_TA/wandb/offline-run-"* || true

echo "=== stage4 main+eval DONE $(date) ==="
