#!/usr/bin/env bash
# Stage 6: redo main DPO at beta=10 (the actual sweet spot from the sweep).
# Reuses generated train videos + scores + ablation preferences -- only the
# train preferences and main LoRAs are rebuilt.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${REPO}/.venv/bin/python"
export PYTHONPATH="${REPO}/src:${REPO}/external:${REPO}/external/VideoAlign:${PYTHONPATH:-}"
export PYTORCH_ALLOC_CONF=expandable_segments:True

cd "${REPO}"
LOG="${REPO}/runs/stage6.log"
mkdir -p "${REPO}/runs"
exec > >(tee -a "$LOG") 2>&1
echo "=== stage6 (beta=10, 1000 prompts, 400 steps) started $(date) ==="

# Rebuild train preferences (idempotent; uses cached scores + latents).
${PY} scripts/03_build_preferences.py --split train --reward-dims MQ VQ TA

# Three main DPO runs.
for dim in MQ VQ TA; do
    ${PY} scripts/04_train_dpo.py \
        --reward-dim "${dim}" \
        --split train \
        --run-name "main_${dim}" \
        --lr 1e-5 --beta 10 \
        --batch-size 1 --grad-accum 1 \
        --max-steps 400 --lora-rank 16 \
        --wandb-mode offline
done

# Holdout eval (baseline + 3 LoRAs, 20 prompts).
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

# Sync W&B and re-upload to Drive.
${REPO}/.venv/bin/wandb sync \
    "${REPO}/runs/main_MQ/wandb/offline-run-"* \
    "${REPO}/runs/main_VQ/wandb/offline-run-"* \
    "${REPO}/runs/main_TA/wandb/offline-run-"* || true

cp "${REPO}/paper/paper.pdf" \
   "${REPO}/paper/dpo-wan_bidirectional-DPO-on-Wan2.1_2026-05-10.pdf"
rclone copyto \
    "${REPO}/paper/dpo-wan_bidirectional-DPO-on-Wan2.1_2026-05-10.pdf" \
    "gdrive:/dpo-wan_bidirectional-DPO-on-Wan2.1_2026-05-10.pdf" || true

echo "=== stage6 DONE $(date) ==="
