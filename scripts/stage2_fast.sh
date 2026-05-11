#!/usr/bin/env bash
# Faster stage 2 with grad_accum=1 (4x speedup) and a 2-config ablation grid.
# All other steps identical to stage2_post_train_gen.sh.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${REPO}/.venv/bin/python"
export PYTHONPATH="${REPO}/src:${REPO}/external:${REPO}/external/VideoAlign:${PYTHONPATH:-}"
export PYTORCH_ALLOC_CONF=expandable_segments:True

cd "${REPO}"
LOG="${REPO}/runs/stage2.log"
mkdir -p "${REPO}/runs"
exec > >(tee -a "$LOG") 2>&1
echo "=== stage2_fast started $(date) ==="

# (already done): preferences/train, ablation gen + score, preferences/ablation
# Re-run preferences just in case the previous stage2 was interrupted mid-build.
${PY} scripts/03_build_preferences.py --split train     --reward-dims MQ VQ TA
${PY} scripts/03_build_preferences.py --split ablation  --reward-dims MQ

# 1. Hyperparameter ablation: 2x2 grid, 30 steps each, grad_accum=1
${PY} scripts/06_run_ablation.py \
    --reward-dim MQ \
    --ablation-split ablation \
    --eval-split eval \
    --max-steps 30 \
    --lrs 5e-5 1e-4 \
    --betas 100 500 \
    --batch-sizes 1

# 2. Main DPO training, 80 steps, grad_accum=1
for dim in MQ VQ TA; do
    ${PY} scripts/04_train_dpo.py \
        --reward-dim "${dim}" \
        --split train \
        --run-name "main_${dim}" \
        --lr 1e-4 --beta 500 \
        --batch-size 1 --grad-accum 1 \
        --max-steps 80 --lora-rank 16 \
        --wandb-mode offline
done

# 3. Eval baseline + 3 main LoRAs on holdout
${PY} scripts/05_eval.py \
    --runs \
        baseline=baseline \
        MQ="${REPO}/runs/main_MQ/final" \
        VQ="${REPO}/runs/main_VQ/final" \
        TA="${REPO}/runs/main_TA/final" \
    --out "${REPO}/runs/eval_holdout" \
    --frame-num 21 --steps 15

# 4. (best-effort) eval the ablation winners on the holdout
${PY} scripts/08_eval_ablation.py --frame-num 21 --steps 15 || true

# 5. render paper figures + tables, compile PDF
${PY} scripts/07_render_paper.py \
    --eval-dir "${REPO}/runs/eval_holdout" \
    --abl-eval-dir "${REPO}/runs/ablation_eval" \
    --scale "${REPO}/configs/scale.json" || true

bash "${REPO}/paper/compile.sh" || true

echo "=== stage2_fast DONE $(date) ==="
