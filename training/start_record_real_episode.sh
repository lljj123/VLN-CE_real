#!/usr/bin/env bash

set -Eeuo pipefail

TRAINING_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REAL_ROOT="$(cd -- "${TRAINING_DIR}/.." && pwd)"
if [[ -z "${VLN_PYTHON:-}" ]]; then
    if [[ -x "${HOME}/.local/share/mamba/envs/vlnce_real/bin/python" ]]; then
        VLN_PYTHON="${HOME}/.local/share/mamba/envs/vlnce_real/bin/python"
    else
        VLN_PYTHON="${HOME}/.local/share/mamba/envs/vlnce/bin/python"
    fi
fi
VLN_ROS_SETUP="${VLN_ROS_SETUP:-/opt/ros/noetic/setup.bash}"
VLN_INSTRUCTION="${VLN_INSTRUCTION:-Go straight down the hallway from the starting room. Turn left at the end, pass the first doorway on your left, then turn right. Continue forward to the next doorway on your left. Turn left into the office and stop in front of the chair.}"
VLN_EPISODE_ID="${VLN_EPISODE_ID:-episode_$(date +%Y%m%d_%H%M%S)}"
VLN_DATA_SPLIT="${VLN_DATA_SPLIT:-train}"
VLN_EXPERT_ACTION_TOPIC="${VLN_EXPERT_ACTION_TOPIC:-/vln/expert_action}"
VLN_SYNC_SLOP="${VLN_SYNC_SLOP:-0.10}"
VLN_MAX_PAIR_AGE="${VLN_MAX_PAIR_AGE:-0.50}"

VLN_DEPTH_RAW_TOPIC="/camera/depth_registered/image_raw"
VLN_DEPTH_FILLED_TOPIC="/camera/depth_registered/image_filled"
VLN_RGB_TOPIC="/camera/rgb/image_color"
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
    echo "[record_real] ROS setup not found: ${VLN_ROS_SETUP}" >&2
    exit 1
fi
if [[ ! -x "${VLN_PYTHON}" ]]; then
    echo "[record_real] Python not found: ${VLN_PYTHON}" >&2
    exit 1
fi

# shellcheck disable=SC1091
source "${VLN_ROS_SETUP}"
if ! rostopic list >/dev/null 2>&1; then
    echo "[record_real] ROS master is unreachable." >&2
    exit 1
fi

cd "${REAL_ROOT}"
if rosnode ping -c 1 /ros_depth_hole_filler >/dev/null 2>&1; then
    echo "[record_real] Reusing /ros_depth_hole_filler."
else
    "${VLN_PYTHON}" scripts/ros_depth_hole_filler.py \
        --input-topic "${VLN_DEPTH_RAW_TOPIC}" \
        --output-topic "${VLN_DEPTH_FILLED_TOPIC}" &
    VLN_DEPTH_PID=$!
fi

echo "[record_real] episode=${VLN_EPISODE_ID} split=${VLN_DATA_SPLIT}"
echo "[record_real] instruction=${VLN_INSTRUCTION}"
echo "[record_real] Publish HUMAN/TRUSTED labels to ${VLN_EXPERT_ACTION_TOPIC}"
echo "[record_real] Allowed: STOP MOVE_FORWARD TURN_LEFT TURN_RIGHT"

"${VLN_PYTHON}" training/ros_record_real_episode.py \
    --instruction "${VLN_INSTRUCTION}" \
    --episode-id "${VLN_EPISODE_ID}" \
    --split "${VLN_DATA_SPLIT}" \
    --rgb-topic "${VLN_RGB_TOPIC}" \
    --depth-topic "${VLN_DEPTH_FILLED_TOPIC}" \
    --expert-action-topic "${VLN_EXPERT_ACTION_TOPIC}" \
    --sync-slop "${VLN_SYNC_SLOP}" \
    --max-pair-age "${VLN_MAX_PAIR_AGE}"
