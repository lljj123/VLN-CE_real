#!/usr/bin/env bash

set -Eeuo pipefail

TRAINING_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REAL_ROOT="$(cd -- "${TRAINING_DIR}/.." && pwd)"
VLN_TRAIN_PYTHON="${VLN_TRAIN_PYTHON:-/usr/bin/python3}"
VLN_TRAIN_CUDA_VISIBLE_DEVICES="${VLN_TRAIN_CUDA_VISIBLE_DEVICES:-0}"

VLN_TRAIN_DATA="${VLN_TRAIN_DATA:-training/data/real_episodes_0p4m_15deg}"
VLN_BASE_CHECKPOINT="${VLN_BASE_CHECKPOINT:-data/checkpoints/CMA_PM_DA_Aug_robot.pth}"
VLN_TRAIN_OUTPUT="${VLN_TRAIN_OUTPUT:-training/checkpoints/real_cma_0p4m_15deg}"
VLN_TRAIN_EPOCHS="${VLN_TRAIN_EPOCHS:-400}"
VLN_TRAIN_BATCH_SIZE="${VLN_TRAIN_BATCH_SIZE:-50}"
VLN_TRAIN_SEQUENCE_LENGTH="${VLN_TRAIN_SEQUENCE_LENGTH:-64}"
VLN_TRAIN_LEARNING_RATE="${VLN_TRAIN_LEARNING_RATE:-0.00001}"

if [[ ! -x "${VLN_TRAIN_PYTHON}" ]]; then
    echo "[finetune_real] Python not found: ${VLN_TRAIN_PYTHON}" >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${VLN_TRAIN_CUDA_VISIBLE_DEVICES}"
CUDA_SUMMARY="$("${VLN_TRAIN_PYTHON}" - <<'PY'
import sys

try:
    import torch
except Exception as error:
    print("[finetune_real] Failed to import PyTorch: {}".format(error), file=sys.stderr)
    raise SystemExit(1)

if not torch.cuda.is_available():
    print(
        "[finetune_real] CUDA is unavailable in {} (torch={}, built_cuda={}).".format(
            sys.executable,
            torch.__version__,
            torch.version.cuda,
        ),
        file=sys.stderr,
    )
    print(
        "[finetune_real] Select a CUDA-enabled interpreter with VLN_TRAIN_PYTHON.",
        file=sys.stderr,
    )
    raise SystemExit(1)

print(
    "{}; torch={}; cuda={}".format(
        torch.cuda.get_device_name(0),
        torch.__version__,
        torch.version.cuda,
    )
)
PY
)"

echo "[finetune_real] GPU: ${CUDA_SUMMARY}"
echo "[finetune_real] Python: ${VLN_TRAIN_PYTHON}"
echo "[finetune_real] Batch size: ${VLN_TRAIN_BATCH_SIZE}"
echo "[finetune_real] Sequence length: ${VLN_TRAIN_SEQUENCE_LENGTH}"

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
