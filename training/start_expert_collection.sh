#!/usr/bin/env bash

set -Eeuo pipefail

TRAINING_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REAL_ROOT="$(cd -- "${TRAINING_DIR}/.." && pwd)"
COLLECTOR_CLI_ARGS=("$@")
if [[ -z "${VLN_PYTHON:-}" ]]; then
    VLN_PYTHON_CANDIDATES=(
        "${HOME}/.local/share/mamba/envs/vlnce_real/bin/python"
        "${HOME}/.local/share/mamba/envs/vlnce/bin/python"
    )
    for VLN_PYTHON_CANDIDATE in "${VLN_PYTHON_CANDIDATES[@]}"; do
        if [[ -x "${VLN_PYTHON_CANDIDATE}" ]]; then
            VLN_PYTHON="${VLN_PYTHON_CANDIDATE}"
            break
        fi
    done
    if [[ -z "${VLN_PYTHON:-}" ]]; then
        VLN_PYTHON="$(command -v python3 || true)"
    fi
fi

VLN_ROS_SETUP="${VLN_ROS_SETUP:-/opt/ros/noetic/setup.bash}"
VLN_DEPTH_LOG_EVERY="${VLN_DEPTH_LOG_EVERY:-0}"
VLN_DEPTH_PID=""

cleanup() {
    if [[ -n "${VLN_DEPTH_PID}" ]] \
        && kill -0 "${VLN_DEPTH_PID}" 2>/dev/null; then
        kill -INT "${VLN_DEPTH_PID}" 2>/dev/null || true
        wait "${VLN_DEPTH_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

if [[ ! -f "${VLN_ROS_SETUP}" ]]; then
    echo "[expert_collection] ROS setup not found: ${VLN_ROS_SETUP}" >&2
    exit 1
fi
if [[ ! -x "${VLN_PYTHON}" ]]; then
    echo "[expert_collection] No executable Python 3 was found." >&2
    echo "Set it explicitly, for example:" >&2
    echo "  VLN_PYTHON=/usr/bin/python3 ./training/start_expert_collection.sh" >&2
    exit 1
fi

VLN_COLLECTION_CONFIG="${VLN_COLLECTION_CONFIG:-config/expert_collection.json}"
if [[ "${VLN_COLLECTION_CONFIG}" = /* ]]; then
    VLN_COLLECTION_CONFIG_PATH="${VLN_COLLECTION_CONFIG}"
else
    VLN_COLLECTION_CONFIG_PATH="${REAL_ROOT}/${VLN_COLLECTION_CONFIG}"
fi
if [[ ! -f "${VLN_COLLECTION_CONFIG_PATH}" ]]; then
    echo "[expert_collection] Config not found: ${VLN_COLLECTION_CONFIG_PATH}" >&2
    exit 1
fi

CONFIG_TEXT="$("${VLN_PYTHON}" - "${VLN_COLLECTION_CONFIG_PATH}" <<'PY'
import json
import math
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as input_file:
    config = json.load(input_file)
if not isinstance(config, dict):
    raise ValueError("configuration root must be an object")


def required_text(mapping, key):
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError("{} must be a non-empty string".format(key))
    value = value.strip()
    if "\n" in value or "\r" in value:
        raise ValueError("{} must be a single line".format(key))
    return value


def positive_number(mapping, key):
    value = mapping.get(key)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value <= 0.0
    ):
        raise ValueError("{} must be a positive number".format(key))
    return str(float(value))


topics = config.get("topics")
sync = config.get("synchronization")
if not isinstance(topics, dict):
    raise ValueError("topics must be an object")
if not isinstance(sync, dict):
    raise ValueError("synchronization must be an object")
split = required_text(config, "split")
if split not in ("train", "val"):
    raise ValueError("split must be train or val")

values = [
    required_text(config, "instruction"),
    required_text(config, "episode_prefix"),
    split,
    required_text(config, "output_dir"),
    required_text(topics, "rgb"),
    required_text(topics, "depth_raw"),
    required_text(topics, "depth_filled"),
    required_text(topics, "rgb_camera_info"),
    required_text(topics, "depth_camera_info"),
    required_text(topics, "expert_action"),
    required_text(topics, "cmd_vel"),
    required_text(config, "motion_config"),
    positive_number(sync, "slop_seconds"),
    positive_number(sync, "max_pair_age_seconds"),
    positive_number(config, "settle_time_seconds"),
]
print("\n".join(values))
PY
)"
mapfile -t CONFIG_VALUES <<< "${CONFIG_TEXT}"
if [[ "${#CONFIG_VALUES[@]}" -ne 15 ]]; then
    echo "[expert_collection] Config parser returned incomplete data." >&2
    exit 1
fi

VLN_INSTRUCTION="${VLN_INSTRUCTION:-${CONFIG_VALUES[0]}}"
VLN_EPISODE_PREFIX="${VLN_EPISODE_PREFIX:-${CONFIG_VALUES[1]}}"
VLN_DATA_SPLIT="${VLN_DATA_SPLIT:-${CONFIG_VALUES[2]}}"
VLN_DATA_OUTPUT="${VLN_DATA_OUTPUT:-${CONFIG_VALUES[3]}}"
VLN_RGB_TOPIC="${VLN_RGB_TOPIC:-${CONFIG_VALUES[4]}}"
VLN_DEPTH_RAW_TOPIC="${VLN_DEPTH_RAW_TOPIC:-${CONFIG_VALUES[5]}}"
VLN_DEPTH_FILLED_TOPIC="${VLN_DEPTH_FILLED_TOPIC:-${CONFIG_VALUES[6]}}"
VLN_RGB_CAMERA_INFO_TOPIC="${VLN_RGB_CAMERA_INFO_TOPIC:-${CONFIG_VALUES[7]}}"
VLN_DEPTH_CAMERA_INFO_TOPIC="${VLN_DEPTH_CAMERA_INFO_TOPIC:-${CONFIG_VALUES[8]}}"
VLN_EXPERT_ACTION_TOPIC="${VLN_EXPERT_ACTION_TOPIC:-${CONFIG_VALUES[9]}}"
VLN_CMD_VEL_TOPIC="${VLN_CMD_VEL_TOPIC:-${CONFIG_VALUES[10]}}"
VLN_MOTION_CONFIG="${VLN_MOTION_CONFIG:-${CONFIG_VALUES[11]}}"
VLN_SYNC_SLOP="${VLN_SYNC_SLOP:-${CONFIG_VALUES[12]}}"
VLN_MAX_PAIR_AGE="${VLN_MAX_PAIR_AGE:-${CONFIG_VALUES[13]}}"
VLN_SETTLE_TIME="${VLN_SETTLE_TIME:-${CONFIG_VALUES[14]}}"
VLN_EPISODE_ID="${VLN_EPISODE_ID:-${VLN_EPISODE_PREFIX}_$(date +%Y%m%d_%H%M%S_%N)}"

# shellcheck disable=SC1091
set --
source "${VLN_ROS_SETUP}"
set -- "${COLLECTOR_CLI_ARGS[@]}"
if ! "${VLN_PYTHON}" - >/dev/null 2>&1 <<'PY'
import glob
import os
import sys

paths = ["/usr/lib/python3/dist-packages"]
ros_distro = os.environ.get("ROS_DISTRO")
if ros_distro:
    paths.append(
        "/opt/ros/{}/lib/python3/dist-packages".format(ros_distro)
    )
else:
    paths.extend(sorted(glob.glob("/opt/ros/*/lib/python3/dist-packages")))
for path in paths:
    if os.path.isdir(path) and path not in sys.path:
        sys.path.append(path)

import cv2
import cv_bridge
import message_filters
import numpy
import rospy
PY
then
    echo "[expert_collection] Python dependencies are unavailable in:" >&2
    echo "  ${VLN_PYTHON}" >&2
    echo "Required modules: cv2, numpy, rospy, message_filters, cv_bridge" >&2
    echo "Use VLN_PYTHON to select a compatible environment." >&2
    exit 1
fi
if ! rostopic list >/dev/null 2>&1; then
    echo "[expert_collection] ROS master is unreachable." >&2
    exit 1
fi

cd "${REAL_ROOT}"
if rosnode list 2>/dev/null | grep -Fxq /ros_depth_hole_filler; then
    echo "[expert_collection] Reusing /ros_depth_hole_filler."
else
    "${VLN_PYTHON}" scripts/ros_depth_hole_filler.py \
        --input-topic "${VLN_DEPTH_RAW_TOPIC}" \
        --output-topic "${VLN_DEPTH_FILLED_TOPIC}" \
        --log-every "${VLN_DEPTH_LOG_EVERY}" &
    VLN_DEPTH_PID=$!
fi

echo "[expert_collection] RGB: ${VLN_RGB_TOPIC}"
echo "[expert_collection] Depth: ${VLN_DEPTH_FILLED_TOPIC}"
echo "[expert_collection] Chassis: ${VLN_CMD_VEL_TOPIC}"
echo "[expert_collection] Motion config: ${VLN_MOTION_CONFIG}"
echo "[expert_collection] Collection config: ${VLN_COLLECTION_CONFIG_PATH}"
echo "[expert_collection] Python: ${VLN_PYTHON}"
echo "[expert_collection] Episode: ${VLN_EPISODE_ID} (${VLN_DATA_SPLIT})"
echo "[expert_collection] Instruction: ${VLN_INSTRUCTION}"
echo "[expert_collection] Stop VLN inference/action-converter nodes before collecting."

"${VLN_PYTHON}" training/ros_expert_drive_collector.py \
    --instruction "${VLN_INSTRUCTION}" \
    --episode-id "${VLN_EPISODE_ID}" \
    --split "${VLN_DATA_SPLIT}" \
    --output-dir "${VLN_DATA_OUTPUT}" \
    --rgb-topic "${VLN_RGB_TOPIC}" \
    --depth-topic "${VLN_DEPTH_FILLED_TOPIC}" \
    --rgb-camera-info-topic "${VLN_RGB_CAMERA_INFO_TOPIC}" \
    --depth-camera-info-topic "${VLN_DEPTH_CAMERA_INFO_TOPIC}" \
    --expert-action-topic "${VLN_EXPERT_ACTION_TOPIC}" \
    --cmd-vel-topic "${VLN_CMD_VEL_TOPIC}" \
    --motion-config "${VLN_MOTION_CONFIG}" \
    --sync-slop "${VLN_SYNC_SLOP}" \
    --max-pair-age "${VLN_MAX_PAIR_AGE}" \
    --settle-time "${VLN_SETTLE_TIME}" \
    "${COLLECTOR_CLI_ARGS[@]}"
