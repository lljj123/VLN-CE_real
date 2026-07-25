#!/usr/bin/env bash

set -Eeuo pipefail

VLN_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VLN_REPO_ROOT="$(cd -- "${VLN_SCRIPT_DIR}/.." && pwd)"
if [[ -z "${VLN_PYTHON:-}" ]]; then
    if [[ -x "${HOME}/.local/share/mamba/envs/vlnce_real/bin/python" ]]; then
        VLN_PYTHON="${HOME}/.local/share/mamba/envs/vlnce_real/bin/python"
    else
        VLN_PYTHON="${HOME}/.local/share/mamba/envs/vlnce/bin/python"
    fi
fi
VLN_ROS_SETUP="${VLN_ROS_SETUP:-/opt/ros/noetic/setup.bash}"

VLN_INSTRUCTION="${VLN_INSTRUCTION:-Go straight down the hallway from the starting room. Turn left at the end, pass the first doorway on your left, then turn right. Continue forward to the next doorway on your left. Turn left into the office and stop in front of the chair.}"
VLN_ACTION_TOPIC="${VLN_ACTION_TOPIC:-/vln/action}"
VLN_MAX_ACTIONS="${VLN_MAX_ACTIONS:-0}"
VLN_MIN_ACTION_INTERVAL="${VLN_MIN_ACTION_INTERVAL:-5.0}"
VLN_SYNC_SLOP="${VLN_SYNC_SLOP:-0.10}"
VLN_SYNC_QUEUE_SIZE="${VLN_SYNC_QUEUE_SIZE:-20}"
VLN_INPUT_TIMEOUT="${VLN_INPUT_TIMEOUT:-30.0}"
VLN_PUBLISHER_WAIT="${VLN_PUBLISHER_WAIT:-5.0}"
VLN_STARTUP_TIMEOUT="${VLN_STARTUP_TIMEOUT:-20}"
VLN_FORCE_CPU="${VLN_FORCE_CPU:-0}"
VLN_CHECKPOINT="${VLN_CHECKPOINT:-data/checkpoints/CMA_PM_DA_Aug_robot.pth}"

VLN_DEPTH_RAW_TOPIC="/camera/depth_registered/image_raw"
VLN_DEPTH_FILLED_TOPIC="/camera/depth_registered/image_filled"
VLN_RGB_TOPIC="/camera/rgb/image_color"
VLN_DEPTH_PID=""

cleanup() {
    if [[ -n "${VLN_DEPTH_PID}" ]] \
        && kill -0 "${VLN_DEPTH_PID}" 2>/dev/null; then
        echo "[start_vln_real] Stopping depth preprocessing node..."
        kill -INT "${VLN_DEPTH_PID}" 2>/dev/null || true
        wait "${VLN_DEPTH_PID}" 2>/dev/null || true
    fi
}

wait_for_image() {
    local topic_name="$1"
    local timeout_seconds="$2"

    # Keep one TCPROS subscription alive for the whole timeout. Reconnecting
    # every second can repeatedly abort before a remote camera's first frame.
    if timeout "${timeout_seconds}s" rostopic echo -n 1 \
        "${topic_name}/header" >/dev/null 2>&1; then
        return 0
    fi

    if [[ -n "${VLN_DEPTH_PID}" ]] \
        && ! kill -0 "${VLN_DEPTH_PID}" 2>/dev/null; then
        echo "[start_vln_real] Depth preprocessing node exited early." >&2
        return 1
    fi
    echo "[start_vln_real] No image received from ${topic_name} within " \
        "${timeout_seconds}s." >&2
    return 1
}

trap cleanup EXIT INT TERM

if [[ ! -f "${VLN_ROS_SETUP}" ]]; then
    echo "[start_vln_real] ROS setup not found: ${VLN_ROS_SETUP}" >&2
    exit 1
fi
if [[ ! -x "${VLN_PYTHON}" ]]; then
    echo "[start_vln_real] VLN Python not found: ${VLN_PYTHON}" >&2
    exit 1
fi

# shellcheck disable=SC1091
source "${VLN_ROS_SETUP}"

if ! "${VLN_PYTHON}" -c \
    'import cv2, message_filters, numpy, rospy, torch; import cv_bridge' \
    >/dev/null 2>&1; then
    echo "[start_vln_real] Missing a required runtime dependency. Need " \
        "NumPy, PyTorch, OpenCV, rospy, cv_bridge and message_filters." >&2
    exit 1
fi

if ! rostopic list >/dev/null 2>&1; then
    echo "[start_vln_real] ROS master is unreachable. Start roscore and " \
        "check ROS_MASTER_URI/ROS_IP." >&2
    exit 1
fi

cd "${VLN_REPO_ROOT}"

echo "[start_vln_real] Starting depth preprocessing:"
echo "  ${VLN_DEPTH_RAW_TOPIC} -> ${VLN_DEPTH_FILLED_TOPIC}"
if rosnode ping -c 1 /ros_depth_hole_filler >/dev/null 2>&1; then
    echo "[start_vln_real] Reusing existing /ros_depth_hole_filler node."
else
    "${VLN_PYTHON}" "${VLN_SCRIPT_DIR}/ros_depth_hole_filler.py" \
        --input-topic "${VLN_DEPTH_RAW_TOPIC}" \
        --output-topic "${VLN_DEPTH_FILLED_TOPIC}" &
    VLN_DEPTH_PID=$!
fi

echo "[start_vln_real] Waiting for processed depth..."
wait_for_image "${VLN_DEPTH_FILLED_TOPIC}" "${VLN_STARTUP_TIMEOUT}"

echo "[start_vln_real] Waiting for RGB..."
wait_for_image "${VLN_RGB_TOPIC}" "${VLN_STARTUP_TIMEOUT}"

VLN_INFERENCE_ARGS=(
    "${VLN_SCRIPT_DIR}/ros_vln_inference.py"
    --checkpoint-path "${VLN_CHECKPOINT}"
    --instruction "${VLN_INSTRUCTION}"
    --rgb-topic "${VLN_RGB_TOPIC}"
    --depth-topic "${VLN_DEPTH_FILLED_TOPIC}"
    --action-topic "${VLN_ACTION_TOPIC}"
    --max-actions "${VLN_MAX_ACTIONS}"
    --min-action-interval "${VLN_MIN_ACTION_INTERVAL}"
    --sync-slop "${VLN_SYNC_SLOP}"
    --sync-queue-size "${VLN_SYNC_QUEUE_SIZE}"
    --input-timeout "${VLN_INPUT_TIMEOUT}"
    --publisher-wait-timeout "${VLN_PUBLISHER_WAIT}"
    --keep-running-after-stop
)
if [[ "${VLN_FORCE_CPU}" == "1" ]]; then
    VLN_INFERENCE_ARGS+=(--cpu)
fi

echo "[start_vln_real] Starting standalone PyTorch VLN inference."
echo "  checkpoint: ${VLN_CHECKPOINT}"
echo "  instruction: ${VLN_INSTRUCTION}"
echo "  action topic: ${VLN_ACTION_TOPIC}"
echo "  max actions: ${VLN_MAX_ACTIONS}"
echo "  minimum action interval: ${VLN_MIN_ACTION_INTERVAL}s"
echo "  RGB-D sync slop: ${VLN_SYNC_SLOP}s"
echo "  action message type: std_msgs/String"

VLN_EXIT_STATUS=0
"${VLN_PYTHON}" "${VLN_INFERENCE_ARGS[@]}" || VLN_EXIT_STATUS=$?
exit "${VLN_EXIT_STATUS}"
