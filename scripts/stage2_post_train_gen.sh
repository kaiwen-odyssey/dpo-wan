#!/usr/bin/env bash
# Continues the pipeline once train candidate generation + scoring has finished.
# Order:
#   1. Build preference sets for all 3 reward dims on train
#   2. Generate + score ablation candidates (sequential, same GPU)
#   3. Build preference sets on ablation
#   4. Run hyperparameter ablation
#   5. Run 3 main DPO trainings
#   6. Generate + score eval candidates for baseline + 3 LoRAs
#   7. Render paper
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${REPO}/.venv/bin/python"
export PYTHONPATH="${REPO}/src:${REPO}/external:${REPO}/external/VideoAlign:${PYTHONPATH:-}"
export PYTORCH_ALLOC_CONF=expandable_segments:True

cd "${REPO}"

LOG="${REPO}/runs/stage2.log"
mkdir -p "${REPO}/runs"
exec > >(tee -a "$LOG") 2>&1
echo "=== stage2 started $(date) ==="

# 1. preferences (train)
${PY} scripts/03_build_preferences.py --split train --reward-dims MQ VQ TA

# 2. ablation gen + score
${PY} scripts/02_generate_and_score.py --split ablation --K 2 --frame-num 21 --steps 15

# 3. preferences (ablation)
${PY} scripts/03_build_preferences.py --split ablation --reward-dims MQ

# 4. hyperparameter ablation
${PY} scripts/06_run_ablation.py \
    --reward-dim MQ \
    --ablation-split ablation \
    --eval-split eval \
    --max-steps 60 \
    --lrs 5e-5 1e-4 \
    --betas 100 500 \
    --batch-sizes 1

# 5. eval the ablation winners on the holdout (need to gen videos first)
${PY} scripts/08_eval_ablation.py --frame-num 21 --steps 15

# 6. main DPO training (3 reward dims)
for dim in MQ VQ TA; do
    ${PY} scripts/04_train_dpo.py \
        --reward-dim "${dim}" \
        --split train \
        --run-name "main_${dim}" \
        --lr 1e-4 --beta 500 \
        --batch-size 1 --grad-accum 4 \
        --max-steps 120 --lora-rank 16 \
        --wandb-mode offline
done

# 7. eval baseline + 3 main LoRAs on holdout
${PY} scripts/05_eval.py \
    --runs \
        baseline=baseline \
        MQ="${REPO}/runs/main_MQ/final" \
        VQ="${REPO}/runs/main_VQ/final" \
        TA="${REPO}/runs/main_TA/final" \
    --out "${REPO}/runs/eval_holdout" \
    --frame-num 21 --steps 15

# 8. render paper figures + tables, compile PDF
${PY} scripts/07_render_paper.py \
    --eval-dir "${REPO}/runs/eval_holdout" \
    --abl-eval-dir "${REPO}/runs/ablation_eval" \
    --scale "${REPO}/configs/scale.json" || true

bash "${REPO}/paper/compile.sh" || true

echo "=== stage2 DONE $(date) ==="
