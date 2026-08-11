#!/usr/bin/env bash

set -Eeuo pipefail

TRAINING_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REAL_ROOT="$(cd -- "${TRAINING_DIR}/.." && pwd)"
VLN_DAGGER_CONFIG="${VLN_DAGGER_CONFIG:-config/dagger_collection.json}"

if [[ "${VLN_DAGGER_CONFIG}" = /* ]]; then
    VLN_DAGGER_CONFIG_PATH="${VLN_DAGGER_CONFIG}"
else
    VLN_DAGGER_CONFIG_PATH="${REAL_ROOT}/${VLN_DAGGER_CONFIG}"
fi

if [[ ! -f "${VLN_DAGGER_CONFIG_PATH}" ]]; then
    echo "[dagger_collection] Config not found: ${VLN_DAGGER_CONFIG_PATH}" >&2
    exit 1
fi

if [[ -z "${VLN_DAGGER_CHECKPOINT:-}" ]]; then
    VLN_CONFIG_PYTHON="${VLN_PYTHON:-$(command -v python3 || true)}"
    if [[ ! -x "${VLN_CONFIG_PYTHON}" ]]; then
        echo "[dagger_collection] Python is required to read the config." >&2
        exit 1
    fi
    VLN_DAGGER_CHECKPOINT="$(${VLN_CONFIG_PYTHON} - "${VLN_DAGGER_CONFIG_PATH}" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as input_file:
    config = json.load(input_file)
checkpoint = config.get("checkpoint")
if not isinstance(checkpoint, str) or not checkpoint.strip():
    raise ValueError("checkpoint must be a non-empty string")
print(checkpoint.strip())
PY
)"
fi

if [[ "${VLN_DAGGER_CHECKPOINT}" = /* ]]; then
    VLN_DAGGER_CHECKPOINT_PATH="${VLN_DAGGER_CHECKPOINT}"
else
    VLN_DAGGER_CHECKPOINT_PATH="${REAL_ROOT}/${VLN_DAGGER_CHECKPOINT}"
fi
if [[ ! -f "${VLN_DAGGER_CHECKPOINT_PATH}" ]]; then
    echo "[dagger_collection] Checkpoint not found: ${VLN_DAGGER_CHECKPOINT_PATH}" >&2
    exit 1
fi

VLN_DAGGER_CPU="${VLN_DAGGER_CPU:-0}"
VLN_DAGGER_SAMPLE="${VLN_DAGGER_SAMPLE:-0}"
VLN_DAGGER_ALLOW_MODEL_MISTAKE="${VLN_DAGGER_ALLOW_MODEL_MISTAKE:-0}"
for value_name in \
    VLN_DAGGER_CPU \
    VLN_DAGGER_SAMPLE \
    VLN_DAGGER_ALLOW_MODEL_MISTAKE; do
    if [[ "${!value_name}" != "0" && "${!value_name}" != "1" ]]; then
        echo "[dagger_collection] ${value_name} must be 0 or 1." >&2
        exit 1
    fi
done

DAGGER_ARGS=(--checkpoint-path "${VLN_DAGGER_CHECKPOINT}")
if [[ "${VLN_DAGGER_CPU}" = "1" ]]; then
    DAGGER_ARGS+=(--cpu)
fi
if [[ "${VLN_DAGGER_SAMPLE}" = "1" ]]; then
    DAGGER_ARGS+=(--sample)
fi
if [[ "${VLN_DAGGER_ALLOW_MODEL_MISTAKE}" = "1" ]]; then
    DAGGER_ARGS+=(--allow-model-mistake-execution)
fi

export VLN_COLLECTION_CONFIG="${VLN_DAGGER_CONFIG_PATH}"
export VLN_COLLECTOR_SCRIPT="training/ros_dagger_collector.py"

echo "[dagger_collection] Policy: ${VLN_DAGGER_CHECKPOINT}"
echo "[dagger_collection] Human approval is required before every action."
echo "[dagger_collection] Ctrl-C is the emergency stop during an action."

exec "${TRAINING_DIR}/start_expert_collection.sh" \
    "${DAGGER_ARGS[@]}" \
    "$@"
