#!/usr/bin/env bash
# Stage 3: stabilize DPO training before any large-scale run.
#
# The previous ablation showed loss oscillating between 0 and 100s with
# gradient norms in the thousands.  Root causes (in priority order):
#   1. β·T = 500·1000 multiplied tiny per-pair Δ's into huge logits ⇒
#      log-sigmoid saturates ⇒ gradients explode/vanish in alternation.
#   2. No LR warmup: step 0 hits the model with full LR on a still-cold ref.
#   3. grad_clip=1.0 was too loose given the spikes.
#
# This script:
#   * sweeps a lower-β grid (β ∈ {1, 10, 50, 100}, T=1000 unchanged)
#   * sweeps lower LRs (1e-5, 5e-6) so warmup has room
#   * pins grad_clip=0.1 and warmup_steps=10
#   * evaluates every config on a 20-prompt holdout and ranks by win-rate
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${REPO}/.venv/bin/python"
export PYTHONPATH="${REPO}/src:${REPO}/external:${REPO}/external/VideoAlign:${PYTHONPATH:-}"
export PYTORCH_ALLOC_CONF=expandable_segments:True

cd "${REPO}"
LOG="${REPO}/runs/stage3.log"
mkdir -p "${REPO}/runs"
exec > >(tee -a "$LOG") 2>&1
echo "=== stage3 stability started $(date) ==="

# Make sure eval video cache for the new 20-prompt holdout exists for the
# baseline before we run per-LoRA evaluation; otherwise eval.run_eval will
# regenerate them unconditionally per run, which is fine but wasteful.
# (Skipped here — eval will lazily generate on demand.)

${PY} scripts/06_run_ablation.py \
    --reward-dim MQ \
    --ablation-split ablation \
    --eval-split eval \
    --max-steps 80 \
    --lrs   1e-5 5e-6 \
    --betas 10  50  100 \
    --batch-sizes 1

echo "=== stage3 stability DONE $(date) ==="
