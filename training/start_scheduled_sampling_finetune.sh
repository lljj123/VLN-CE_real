#!/usr/bin/env bash

set -Eeuo pipefail

TRAINING_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REAL_ROOT="$(cd -- "${TRAINING_DIR}/.." && pwd)"
VLN_SS_PYTHON="${VLN_SS_PYTHON:-/usr/bin/python3}"
VLN_SS_CUDA_VISIBLE_DEVICES="${VLN_SS_CUDA_VISIBLE_DEVICES:-0}"

# This is a second-stage fine-tune. It starts from the already converged
# sequence-64 robot checkpoint, but deliberately creates a fresh optimizer and
# writes to a new directory so the original checkpoint remains untouched.
VLN_SS_DATA="${VLN_SS_DATA:-training/data/Scheduled Sampling_withoutfirst}"
VLN_SS_BASE_CHECKPOINT="${VLN_SS_BASE_CHECKPOINT:-training/checkpoints/real_cma_seq64_frozen_language/best_robot.pth}"
VLN_SS_OUTPUT="${VLN_SS_OUTPUT:-training/checkpoints/real_cma_seq64_scheduled_sampling}"
VLN_SS_EPOCHS="${VLN_SS_EPOCHS:-100}"
VLN_SS_BATCH_SIZE="${VLN_SS_BATCH_SIZE:-50}"
VLN_SS_SEQUENCE_LENGTH="${VLN_SS_SEQUENCE_LENGTH:-64}"
VLN_SS_LEARNING_RATE="${VLN_SS_LEARNING_RATE:-0.000005}"
VLN_SS_MAX_PROB="${VLN_SS_MAX_PROB:-0.20}"
VLN_SS_WARMUP_EPOCHS="${VLN_SS_WARMUP_EPOCHS:-5}"
VLN_SS_RAMP_EPOCHS="${VLN_SS_RAMP_EPOCHS:-45}"

if [[ ! -x "${VLN_SS_PYTHON}" ]]; then
    echo "[scheduled_sampling] Python not found: ${VLN_SS_PYTHON}" >&2
    exit 1
fi

if [[ ! -f "${REAL_ROOT}/${VLN_SS_BASE_CHECKPOINT}" && ! -f "${VLN_SS_BASE_CHECKPOINT}" ]]; then
    echo "[scheduled_sampling] Base checkpoint not found: ${VLN_SS_BASE_CHECKPOINT}" >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${VLN_SS_CUDA_VISIBLE_DEVICES}"

echo "[scheduled_sampling] Initial checkpoint: ${VLN_SS_BASE_CHECKPOINT}"
echo "[scheduled_sampling] Output directory: ${VLN_SS_OUTPUT}"
echo "[scheduled_sampling] Epochs: ${VLN_SS_EPOCHS}"
echo "[scheduled_sampling] Probability: 0 -> ${VLN_SS_MAX_PROB}"
echo "[scheduled_sampling] Warmup/ramp: ${VLN_SS_WARMUP_EPOCHS}/${VLN_SS_RAMP_EPOCHS} epochs"

cd "${REAL_ROOT}"
exec "${VLN_SS_PYTHON}" training/finetune_real_cma.py \
    --data-dir "${VLN_SS_DATA}" \
    --checkpoint "${VLN_SS_BASE_CHECKPOINT}" \
    --output-dir "${VLN_SS_OUTPUT}" \
    --epochs "${VLN_SS_EPOCHS}" \
    --batch-size "${VLN_SS_BATCH_SIZE}" \
    --sequence-length "${VLN_SS_SEQUENCE_LENGTH}" \
    --sequence-stride "${VLN_SS_SEQUENCE_LENGTH}" \
    --learning-rate "${VLN_SS_LEARNING_RATE}" \
    --scheduled-sampling-max-prob "${VLN_SS_MAX_PROB}" \
    --scheduled-sampling-warmup-epochs "${VLN_SS_WARMUP_EPOCHS}" \
    --scheduled-sampling-ramp-epochs "${VLN_SS_RAMP_EPOCHS}" \
    "$@"
