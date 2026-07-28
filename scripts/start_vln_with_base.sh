#!/usr/bin/env bash

# Start the action-to-Twist safety layer first, then the existing RGB-D VLN
# pipeline.  The converter is stopped automatically when inference exits.

set -Eeuo pipefail

VLN_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VLN_ROS_SETUP="${VLN_ROS_SETUP:-/opt/ros/noetic/setup.bash}"
VLN_CONTROL_PYTHON="${VLN_PYTHON:-/usr/bin/python3}"

VLN_ACTION_TOPIC="${VLN_ACTION_TOPIC:-/vln/action}"
VLN_CMD_VEL_TOPIC="${VLN_CMD_VEL_TOPIC:-/cmd_vel}"
VLN_LINEAR_SPEED="${VLN_LINEAR_SPEED:-0.10}"
VLN_ANGULAR_SPEED="${VLN_ANGULAR_SPEED:-0.30}"
VLN_FORWARD_DISTANCE="${VLN_FORWARD_DISTANCE:-0.25}"
VLN_TURN_ANGLE_DEG="${VLN_TURN_ANGLE_DEG:-15.0}"
VLN_CMD_RATE="${VLN_CMD_RATE:-20.0}"
VLN_ACTION_WATCHDOG="${VLN_ACTION_WATCHDOG:-10.0}"
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
echo "  ${VLN_ACTION_TOPIC} -> ${VLN_CMD_VEL_TOPIC}"
echo "  forward: ${VLN_FORWARD_DISTANCE} m at ${VLN_LINEAR_SPEED} m/s"
echo "  turn: ${VLN_TURN_ANGLE_DEG} deg at ${VLN_ANGULAR_SPEED} rad/s"
"${VLN_CONTROL_PYTHON}" "${VLN_SCRIPT_DIR}/ros_action_to_cmd_vel.py" \
    --action-topic "${VLN_ACTION_TOPIC}" \
    --cmd-vel-topic "${VLN_CMD_VEL_TOPIC}" \
    --linear-speed "${VLN_LINEAR_SPEED}" \
    --angular-speed "${VLN_ANGULAR_SPEED}" \
    --forward-distance "${VLN_FORWARD_DISTANCE}" \
    --turn-angle-deg "${VLN_TURN_ANGLE_DEG}" \
    --publish-rate "${VLN_CMD_RATE}" \
    --watchdog-timeout "${VLN_ACTION_WATCHDOG}" &
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
