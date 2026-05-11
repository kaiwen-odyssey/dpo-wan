#!/usr/bin/env bash
# Stage 5: scaled main experiment (1000 train prompts, 400 training steps).
# Uses the stable config from stage 3: lr=1e-5, beta=50, grad_clip=0.1, warmup=10.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${REPO}/.venv/bin/python"
export PYTHONPATH="${REPO}/src:${REPO}/external:${REPO}/external/VideoAlign:${PYTHONPATH:-}"
export PYTORCH_ALLOC_CONF=expandable_segments:True

cd "${REPO}"
LOG="${REPO}/runs/stage5.log"
mkdir -p "${REPO}/runs"
exec > >(tee -a "$LOG") 2>&1
echo "=== stage5 scaled started $(date) ==="

# 1. Train candidate generation (1000 prompts × K=2 = 2000 videos, ~13h).
${PY} scripts/02_generate_and_score.py --split train --K 2 --frame-num 21 --steps 15

# 2. Build preferences for all 3 reward dims.
${PY} scripts/03_build_preferences.py --split train --reward-dims MQ VQ TA

# 3. Three main DPO runs at the chosen-stable config, 400 steps each.
# Switched to beta=1 after a follow-up ablation: it's 17x more stable in
# loss-stdev, 23x lower gradient norm, and the highest mean Spearman rho
# (+0.10) of any config.  beta=50 was saturating log-sigmoid; beta=1 keeps
# beta*T*delta in the linear regime.
for dim in MQ VQ TA; do
    ${PY} scripts/04_train_dpo.py \
        --reward-dim "${dim}" \
        --split train \
        --run-name "main_${dim}" \
        --lr 1e-5 --beta 1.0 \
        --batch-size 1 --grad-accum 1 \
        --max-steps 400 --lora-rank 16 \
        --wandb-mode offline
done

# 4. Holdout eval (baseline + 3 LoRAs on the new 20-prompt set).
${PY} scripts/05_eval.py \
    --runs \
        baseline=baseline \
        MQ="${REPO}/runs/main_MQ/final" \
        VQ="${REPO}/runs/main_VQ/final" \
        TA="${REPO}/runs/main_TA/final" \
    --out "${REPO}/runs/eval_holdout" \
    --frame-num 21 --steps 15

# 5. Render paper artefacts and compile.
${PY} scripts/07_render_paper.py \
    --eval-dir "${REPO}/runs/eval_holdout" \
    --abl-eval-dir "${REPO}/runs/ablation_eval" \
    --scale "${REPO}/configs/scale.json" || true
bash "${REPO}/paper/compile.sh" || true

# 6. Sync W&B and upload to Drive.
${REPO}/.venv/bin/wandb sync \
    "${REPO}/runs/main_MQ/wandb/offline-run-"* \
    "${REPO}/runs/main_VQ/wandb/offline-run-"* \
    "${REPO}/runs/main_TA/wandb/offline-run-"* || true

cp "${REPO}/paper/paper.pdf" \
   "${REPO}/paper/dpo-wan_bidirectional-DPO-on-Wan2.1_2026-05-10.pdf"
rclone copyto \
    "${REPO}/paper/dpo-wan_bidirectional-DPO-on-Wan2.1_2026-05-10.pdf" \
    "gdrive:/dpo-wan_bidirectional-DPO-on-Wan2.1_2026-05-10.pdf" || true

echo "=== stage5 scaled DONE $(date) ==="
