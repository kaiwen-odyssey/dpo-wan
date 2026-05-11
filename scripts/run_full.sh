#!/usr/bin/env bash
# Full pipeline driver, scaled to a single-GPU session budget.
# Runs: candidates -> score -> preferences -> ablation -> main -> eval -> paper.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${REPO}/.venv/bin/python"
export PYTHONPATH="${REPO}/src:${REPO}/external:${REPO}/external/VideoAlign:${PYTHONPATH:-}"
export PYTORCH_ALLOC_CONF=expandable_segments:True

mkdir -p "${REPO}/runs"
LOG="${REPO}/runs/run_full.log"
exec > >(tee -a "$LOG") 2>&1
echo "=== run_full started $(date) ==="

cd "${REPO}"

#######################################
# 1. prompts
#######################################
${PY} scripts/01_prepare_prompts.py --n-train 50 --n-ablation 20 --n-eval 15 --seed 0

#######################################
# 2. candidate generation (train + ablation only; eval is per-LoRA)
#######################################
for split in train ablation; do
    ${PY} scripts/02_generate_and_score.py \
        --split "${split}" --K 2 --frame-num 21 --steps 15
done

#######################################
# 3. preference sets per reward dim
#######################################
for split in train ablation; do
    ${PY} scripts/03_build_preferences.py \
        --split "${split}" --reward-dims MQ VQ TA
done

#######################################
# 4. hyperparameter ablation (smaller sweep)
#######################################
${PY} scripts/06_run_ablation.py \
    --reward-dim MQ \
    --ablation-split ablation \
    --eval-split eval \
    --max-steps 60 \
    --lrs 5e-5 1e-4 \
    --betas 100 500 \
    --batch-sizes 1

#######################################
# 5. main runs (3 reward dims, identical hyperparameters)
#######################################
for dim in MQ VQ TA; do
    ${PY} scripts/04_train_dpo.py \
        --reward-dim "${dim}" \
        --split train \
        --run-name "main_${dim}" \
        --lr 1e-4 --beta 500 \
        --batch-size 1 --grad-accum 4 \
        --max-steps 120 --lora-rank 16
done

#######################################
# 6. eval: baseline + 3 LoRAs on holdout
#######################################
${PY} scripts/05_eval.py \
    --runs \
        baseline=baseline \
        MQ="${REPO}/runs/main_MQ/final" \
        VQ="${REPO}/runs/main_VQ/final" \
        TA="${REPO}/runs/main_TA/final" \
    --out "${REPO}/runs/eval_holdout" \
    --frame-num 21 --steps 15

#######################################
# 7. (optional) eval on ablation winners — keep fast, just baseline + best
#######################################
# left for separate driver

#######################################
# 8. render paper figures + tables
#######################################
${PY} scripts/07_render_paper.py \
    --eval-dir "${REPO}/runs/eval_holdout" \
    --abl-eval-dir "${REPO}/runs/ablation_eval" \
    --scale "${REPO}/configs/scale.json" || true

bash "${REPO}/paper/compile.sh" || true

echo "=== run_full DONE $(date) ==="
