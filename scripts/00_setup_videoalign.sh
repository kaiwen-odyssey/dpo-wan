#!/usr/bin/env bash
#
# Pull the VideoReward (VideoAlign) checkpoint from HuggingFace via huggingface_hub
# CLI.  Skips the download if the checkpoint directory already exists.
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT_DIR="${VIDEOREWARD_CHECKPOINT_DIR:-${REPO_ROOT}/external/VideoAlign/checkpoints/VideoReward}"

if [[ -f "${CKPT_DIR}/model_config.json" ]]; then
    echo "VideoReward checkpoint already present at ${CKPT_DIR}; skipping."
    exit 0
fi

mkdir -p "${CKPT_DIR%/*}"

if [[ -x "${REPO_ROOT}/.venv/bin/hf" ]]; then
    HF_BIN="${REPO_ROOT}/.venv/bin/hf"
elif [[ -x "${REPO_ROOT}/.venv/bin/huggingface-cli" ]]; then
    HF_BIN="${REPO_ROOT}/.venv/bin/huggingface-cli"
else
    HF_BIN="huggingface-cli"
fi

echo "Pulling KwaiVGI/VideoReward into ${CKPT_DIR}..."
"${HF_BIN}" download KwaiVGI/VideoReward --local-dir "${CKPT_DIR}"

echo "Done."
