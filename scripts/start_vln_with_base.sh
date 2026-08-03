#!/usr/bin/env bash

# Start the action-to-Twist safety layer first, then the existing RGB-D VLN
# pipeline.  The converter is stopped automatically when inference exits.

set -Eeuo pipefail

VLN_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VLN_REPO_ROOT="$(cd -- "${VLN_SCRIPT_DIR}/.." && pwd)"
VLN_ROS_SETUP="${VLN_ROS_SETUP:-/opt/ros/noetic/setup.bash}"
VLN_CONTROL_PYTHON="${VLN_PYTHON:-/usr/bin/python3}"

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

VLN_INFERENCE_CONFIG="${VLN_INFERENCE_CONFIG:-config/vln_inference.json}"
if [[ "${VLN_INFERENCE_CONFIG}" = /* ]]; then
    VLN_INFERENCE_CONFIG_PATH="${VLN_INFERENCE_CONFIG}"
else
    VLN_INFERENCE_CONFIG_PATH="${VLN_REPO_ROOT}/${VLN_INFERENCE_CONFIG}"
fi
if [[ ! -f "${VLN_INFERENCE_CONFIG_PATH}" ]]; then
    echo "[start_vln_with_base] Inference config not found: " \
        "${VLN_INFERENCE_CONFIG_PATH}" >&2
    exit 1
fi

TOPIC_TEXT="$("${VLN_CONTROL_PYTHON}" - "${VLN_INFERENCE_CONFIG_PATH}" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as input_file:
    config = json.load(input_file)
topics = config.get("topics")
if not isinstance(topics, dict):
    raise ValueError("topics must be an object")
values = []
for key in ("action", "cmd_vel"):
    value = topics.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError("topics.{} must be a non-empty string".format(key))
    value = value.strip()
    if "\n" in value or "\r" in value:
        raise ValueError("topics.{} must be a single line".format(key))
    values.append(value)
print("\n".join(values))
PY
)"
mapfile -t CONFIG_TOPICS <<< "${TOPIC_TEXT}"
if [[ "${#CONFIG_TOPICS[@]}" -ne 2 ]]; then
    echo "[start_vln_with_base] Config parser returned incomplete topics." >&2
    exit 1
fi
VLN_ACTION_TOPIC="${VLN_ACTION_TOPIC:-${CONFIG_TOPICS[0]}}"
VLN_CMD_VEL_TOPIC="${VLN_CMD_VEL_TOPIC:-${CONFIG_TOPICS[1]}}"

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
echo "  cmd_vel topic: ${VLN_CMD_VEL_TOPIC}"
VLN_CONVERTER_ARGS=(
    "${VLN_SCRIPT_DIR}/ros_action_to_cmd_vel.py"
    --config "${VLN_ACTION_CONFIG}"
    --action-topic "${VLN_ACTION_TOPIC}"
    --cmd-vel-topic "${VLN_CMD_VEL_TOPIC}"
)
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

export VLN_ACTION_TOPIC VLN_CMD_VEL_TOPIC VLN_ROS_SETUP VLN_INFERENCE_CONFIG
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
