#!/usr/bin/env bash
# End-to-end pipeline for the 5-second / 100-prompt MQ-DPO experiment.
#
#   prompts -> gen 200 vids (5 s, 81 f, 832x480, 15 steps)
#            -> score MQ (fps=2.0)
#            -> build unfiltered MQ prefs
#            -> DPO train 100 steps, save every 20
#            -> per-checkpoint eval (online policy) on 20 eval prompts
#               + on 20 most-recent train prompts (moving mean)
#               + reference-model baseline
#            -> plot curves
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
RUN_NAME=${RUN_NAME:-dpo_5s_v1_MQ_b10}
TRAIN_SPLIT=${TRAIN_SPLIT:-train_5s}
EVAL_SPLIT=${EVAL_SPLIT:-eval_5s}
FRAME_NUM=${FRAME_NUM:-81}    # 4n+1, ~5 s at 16 fps
STEPS=${STEPS:-15}
REWARD_DIM=${REWARD_DIM:-MQ}

echo "=== Stage A: generate 200 candidates + score (MQ, fps=2.0) ==="
GEN_EXTRA_ARGS=()
if [[ -n "${REWARD_NUM_FRAMES:-}" ]]; then
    GEN_EXTRA_ARGS+=(--reward-num-frames "$REWARD_NUM_FRAMES")
fi
$PY scripts/02_generate_and_score.py \
    --split "$TRAIN_SPLIT" --K 2 \
    --frame-num "$FRAME_NUM" --steps "$STEPS" \
    "${GEN_EXTRA_ARGS[@]}"

echo "=== Stage B: build unfiltered MQ preferences ==="
$PY scripts/03_build_preferences.py --split "$TRAIN_SPLIT" --reward-dims "$REWARD_DIM"

echo "=== Stage C: DPO train (100 steps, save every 20) ==="
$PY scripts/04_train_dpo.py \
    --reward-dim "$REWARD_DIM" \
    --split "$TRAIN_SPLIT" \
    --run-name "$RUN_NAME" \
    --lr 1e-5 --beta 10 \
    --batch-size 1 --grad-accum 1 \
    --max-steps 100 --save-every 20 \
    --lora-rank 16 \
    --wandb-mode offline

echo "=== Stage D: per-checkpoint eval + baseline ==="
$PY scripts/checkpoint_reward_eval.py \
    --run-name "$RUN_NAME" \
    --train-split "$TRAIN_SPLIT" \
    --eval-split "$EVAL_SPLIT" \
    --reward-dim "$REWARD_DIM" \
    --interval 20 --max-step 100 \
    --frame-num "$FRAME_NUM" --steps "$STEPS" \
    --eval-seeds 1234 \
    --include-baseline

echo "=== Stage E: plot ==="
$PY scripts/plot_dpo_5s.py --run-name "$RUN_NAME" --reward-dim "$REWARD_DIM"

echo "=== DONE: $RUN_NAME ==="
