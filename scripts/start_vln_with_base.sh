#!/usr/bin/env bash

# Start the action-to-Twist safety layer first, then the existing RGB-D VLN
# pipeline.  The converter is stopped automatically when inference exits.

set -Eeuo pipefail

VLN_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VLN_REPO_ROOT="$(cd -- "${VLN_SCRIPT_DIR}/.." && pwd)"
VLN_ROS_SETUP="${VLN_ROS_SETUP:-/opt/ros/noetic/setup.bash}"
VLN_CONTROL_PYTHON="${VLN_PYTHON:-/usr/bin/python3}"

VLN_ACTION_TOPIC="${VLN_ACTION_TOPIC:-/vln/action}"
VLN_ACTION_CONFIG="${VLN_ACTION_CONFIG:-${VLN_REPO_ROOT}/config/action_to_cmd_vel.json}"
VLN_CONVERTER_PID=""
VLN_INFERENCE_PID=""

cleanup() {
    if [[ -n "${VLN_INFERENCE_PID}" ]] \
        && kill -0 "${VLN_INFERENCE_PID}" 2>/dev/null; then
        kill -INT "${VLN_INFERENCE_PID}" 2>/dev/null || true
        wait "${VLN_INFERENCE_PID}" 2>/dev/null || true
    fi
    if [[ -n "${VLN_CONVERTER_PID}" ]] \
        && kill -0 "${VLN_CONVERTER_PID}" 2>/dev/null; then
        echo "[start_vln_with_base] Stopping action converter and chassis..."
        kill -INT "${VLN_CONVERTER_PID}" 2>/dev/null || true
        wait "${VLN_CONVERTER_PID}" 2>/dev/null || true
    fi
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ ! -f "${VLN_ROS_SETUP}" ]]; then
    echo "[start_vln_with_base] ROS setup not found: ${VLN_ROS_SETUP}" >&2
    exit 1
fi
if [[ ! -x "${VLN_CONTROL_PYTHON}" ]]; then
    echo "[start_vln_with_base] Python not found: ${VLN_CONTROL_PYTHON}" >&2
    exit 1
fi

set +u
# shellcheck disable=SC1090
source "${VLN_ROS_SETUP}"
set -u

if ! rostopic list >/dev/null 2>&1; then
    echo "[start_vln_with_base] ROS master is unreachable." >&2
    exit 1
fi
if rosnode info /ros_vln_action_to_cmd_vel 2>/dev/null \
    | grep -Eq '^Pid: [0-9]+$'; then
    echo "[start_vln_with_base] /ros_vln_action_to_cmd_vel is already running." >&2
    echo "Stop it first so two nodes cannot command the chassis." >&2
    exit 1
fi

echo "[start_vln_with_base] Starting action converter:"
echo "  config: ${VLN_ACTION_CONFIG}"
echo "  action topic: ${VLN_ACTION_TOPIC}"
VLN_CONVERTER_ARGS=(
    "${VLN_SCRIPT_DIR}/ros_action_to_cmd_vel.py"
    --config "${VLN_ACTION_CONFIG}"
    --action-topic "${VLN_ACTION_TOPIC}"
)
if [[ -n "${VLN_CMD_VEL_TOPIC:-}" ]]; then
    VLN_CONVERTER_ARGS+=(--cmd-vel-topic "${VLN_CMD_VEL_TOPIC}")
fi
if [[ -n "${VLN_LINEAR_SPEED:-}" ]]; then
    VLN_CONVERTER_ARGS+=(--linear-speed "${VLN_LINEAR_SPEED}")
fi
if [[ -n "${VLN_ANGULAR_SPEED:-}" ]]; then
    VLN_CONVERTER_ARGS+=(--angular-speed "${VLN_ANGULAR_SPEED}")
fi
if [[ -n "${VLN_FORWARD_DISTANCE:-}" ]]; then
    VLN_CONVERTER_ARGS+=(--forward-distance "${VLN_FORWARD_DISTANCE}")
fi
if [[ -n "${VLN_TURN_ANGLE_DEG:-}" ]]; then
    VLN_CONVERTER_ARGS+=(--turn-angle-deg "${VLN_TURN_ANGLE_DEG}")
fi
if [[ -n "${VLN_CMD_RATE:-}" ]]; then
    VLN_CONVERTER_ARGS+=(--publish-rate "${VLN_CMD_RATE}")
fi
if [[ -n "${VLN_ACTION_WATCHDOG:-}" ]]; then
    VLN_CONVERTER_ARGS+=(--watchdog-timeout "${VLN_ACTION_WATCHDOG}")
fi
"${VLN_CONTROL_PYTHON}" "${VLN_CONVERTER_ARGS[@]}" &
VLN_CONVERTER_PID=$!

sleep 0.5
if ! kill -0 "${VLN_CONVERTER_PID}" 2>/dev/null; then
    echo "[start_vln_with_base] Action converter exited during startup." >&2
    wait "${VLN_CONVERTER_PID}" || true
    exit 1
fi

export VLN_ACTION_TOPIC VLN_ROS_SETUP
"${VLN_SCRIPT_DIR}/start_vln_real.sh" &
VLN_INFERENCE_PID=$!

# Keep the two processes coupled.  If the velocity safety layer dies, stop
# inference instead of continuing to publish actions with no executor.
while kill -0 "${VLN_CONVERTER_PID}" 2>/dev/null \
    && kill -0 "${VLN_INFERENCE_PID}" 2>/dev/null; do
    sleep 0.2
done

if ! kill -0 "${VLN_CONVERTER_PID}" 2>/dev/null; then
    echo "[start_vln_with_base] Action converter exited unexpectedly; " \
        "stopping VLN inference." >&2
    wait "${VLN_CONVERTER_PID}" 2>/dev/null || true
    VLN_CONVERTER_PID=""
    if kill -0 "${VLN_INFERENCE_PID}" 2>/dev/null; then
        kill -INT "${VLN_INFERENCE_PID}" 2>/dev/null || true
    fi
    wait "${VLN_INFERENCE_PID}" 2>/dev/null || true
    VLN_INFERENCE_PID=""
    exit 1
fi

VLN_EXIT_STATUS=0
wait "${VLN_INFERENCE_PID}" || VLN_EXIT_STATUS=$?
VLN_INFERENCE_PID=""
exit "${VLN_EXIT_STATUS}"
