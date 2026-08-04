#!/usr/bin/env bash

set -Eeuo pipefail

TRAINING_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REAL_ROOT="$(cd -- "${TRAINING_DIR}/.." && pwd)"
if [[ -z "${VLN_TRAIN_PYTHON:-}" ]]; then
    if [[ -x "${HOME}/.local/share/mamba/envs/vlnce_real/bin/python" ]]; then
        VLN_TRAIN_PYTHON="${HOME}/.local/share/mamba/envs/vlnce_real/bin/python"
    else
        VLN_TRAIN_PYTHON="${HOME}/.local/share/mamba/envs/vlnce/bin/python"
    fi
fi

VLN_TRAIN_DATA="${VLN_TRAIN_DATA:-training/data/real_episodes_0p4m_30deg}"
VLN_BASE_CHECKPOINT="${VLN_BASE_CHECKPOINT:-data/checkpoints/CMA_PM_DA_Aug_robot.pth}"
VLN_TRAIN_OUTPUT="${VLN_TRAIN_OUTPUT:-training/checkpoints/real_cma_0p4m_30deg}"
VLN_TRAIN_EPOCHS="${VLN_TRAIN_EPOCHS:-10}"
VLN_TRAIN_BATCH_SIZE="${VLN_TRAIN_BATCH_SIZE:-2}"
VLN_TRAIN_SEQUENCE_LENGTH="${VLN_TRAIN_SEQUENCE_LENGTH:-8}"
VLN_TRAIN_LEARNING_RATE="${VLN_TRAIN_LEARNING_RATE:-0.00001}"

if [[ ! -x "${VLN_TRAIN_PYTHON}" ]]; then
    echo "[finetune_real] Python not found: ${VLN_TRAIN_PYTHON}" >&2
    exit 1
fi

cd "${REAL_ROOT}"
exec "${VLN_TRAIN_PYTHON}" training/finetune_real_cma.py \
    --data-dir "${VLN_TRAIN_DATA}" \
    --checkpoint "${VLN_BASE_CHECKPOINT}" \
    --output-dir "${VLN_TRAIN_OUTPUT}" \
    --epochs "${VLN_TRAIN_EPOCHS}" \
    --batch-size "${VLN_TRAIN_BATCH_SIZE}" \
    --sequence-length "${VLN_TRAIN_SEQUENCE_LENGTH}" \
    --sequence-stride "${VLN_TRAIN_SEQUENCE_LENGTH}" \
    --learning-rate "${VLN_TRAIN_LEARNING_RATE}" \
    "$@"
