#!/usr/bin/env bash
#
# Top-level driver for the dpo-wan pipeline.  Steps are idempotent so the script
# can be re-run after a failure.
#
#   ./run_pipeline.sh smoke          # 3-prompt end-to-end sanity check
#   ./run_pipeline.sh prepare        # download / cache prompts + reward model
#   ./run_pipeline.sh generate SPLIT # generate K candidates and score them
#   ./run_pipeline.sh prefs    SPLIT # build chosen/rejected pairs per reward
#   ./run_pipeline.sh ablate         # hyperparameter sweep on ablation split
#   ./run_pipeline.sh main           # 3 main DPO runs (MQ / VQ / TA)
#   ./run_pipeline.sh eval           # evaluate baseline + 3 LoRAs
#
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${REPO}/.venv/bin/python"
export PYTHONPATH="${REPO}/src:${REPO}/external:${REPO}/external/VideoAlign:${PYTHONPATH:-}"

cmd="${1:-smoke}"
shift || true

case "$cmd" in
    smoke)
        ${PY} "${REPO}/scripts/99_smoke.py" "$@"
        ;;

    prepare)
        bash "${REPO}/scripts/00_setup_videoalign.sh"
        ${PY} "${REPO}/scripts/01_prepare_prompts.py" "$@"
        ;;

    generate)
        split="${1:-train}"; shift || true
        ${PY} "${REPO}/scripts/02_generate_and_score.py" --split "$split" "$@"
        ;;

    prefs)
        split="${1:-train}"; shift || true
        ${PY} "${REPO}/scripts/03_build_preferences.py" --split "$split" "$@"
        ;;

    ablate)
        ${PY} "${REPO}/scripts/06_run_ablation.py" "$@"
        ;;

    main)
        # Three main runs — each uses a different reward signal but the same
        # preference dataset definition.  The chosen/rejected pair changes per
        # reward dim because argmax/argmin re-rank the same K candidates.
        for dim in MQ VQ TA; do
            ${PY} "${REPO}/scripts/04_train_dpo.py" \
                --reward-dim "$dim" \
                --run-name   "main_${dim}" "$@"
        done
        ;;

    eval)
        ${PY} "${REPO}/scripts/05_eval.py" \
            --runs \
                "baseline=baseline" \
                "MQ=${REPO}/runs/main_MQ/final" \
                "VQ=${REPO}/runs/main_VQ/final" \
                "TA=${REPO}/runs/main_TA/final" \
            --out "${REPO}/runs/eval_holdout" \
            "$@"
        ;;

    *)
        echo "unknown command: $cmd" >&2
        exit 2
        ;;
esac
