#!/usr/bin/env bash

set -Eeuo pipefail

VLN_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VLN_REPO_ROOT="$(cd -- "${VLN_SCRIPT_DIR}/.." && pwd)"
VLN_CLI_ARGS=("$@")
if [[ -z "${VLN_PYTHON:-}" ]]; then
    VLN_PYTHON="/usr/bin/python3"
fi
VLN_ROS_SETUP="${VLN_ROS_SETUP:-/opt/ros/noetic/setup.bash}"
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

VLN_INFERENCE_CONFIG="${VLN_INFERENCE_CONFIG:-config/vln_inference.json}"
if [[ "${VLN_INFERENCE_CONFIG}" = /* ]]; then
    VLN_INFERENCE_CONFIG_PATH="${VLN_INFERENCE_CONFIG}"
else
    VLN_INFERENCE_CONFIG_PATH="${VLN_REPO_ROOT}/${VLN_INFERENCE_CONFIG}"
fi
if [[ ! -f "${VLN_INFERENCE_CONFIG_PATH}" ]]; then
    echo "[start_vln_real] Inference config not found: " \
        "${VLN_INFERENCE_CONFIG_PATH}" >&2
    exit 1
fi

CONFIG_TEXT="$("${VLN_PYTHON}" - "${VLN_INFERENCE_CONFIG_PATH}" <<'PY'
import json
import math
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as input_file:
    config = json.load(input_file)
if not isinstance(config, dict):
    raise ValueError("configuration root must be an object")


def section(mapping, key):
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise ValueError("{} must be an object".format(key))
    return value


def text(mapping, key):
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError("{} must be a non-empty string".format(key))
    value = value.strip()
    if "\n" in value or "\r" in value:
        raise ValueError("{} must be a single line".format(key))
    return value


def integer(mapping, key, minimum):
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError("{} must be an integer >= {}".format(key, minimum))
    return str(value)


def number(mapping, key, minimum, strict=False):
    value = mapping.get(key)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or (float(value) <= minimum if strict else float(value) < minimum)
    ):
        operator = ">" if strict else ">="
        raise ValueError("{} must be finite and {} {}".format(key, operator, minimum))
    return str(float(value))


def boolean(mapping, key):
    value = mapping.get(key)
    if not isinstance(value, bool):
        raise ValueError("{} must be true or false".format(key))
    return "1" if value else "0"


topics = section(config, "topics")
inference = section(config, "inference")
sync = section(config, "synchronization")
startup = section(config, "startup")
depth = section(config, "depth")
min_depth = number(depth, "min_meters", 0.0)
max_depth = number(depth, "max_meters", 0.0, strict=True)
if float(max_depth) <= float(min_depth):
    raise ValueError("depth.max_meters must be greater than depth.min_meters")

values = [
    text(config, "instruction"),
    text(config, "checkpoint"),
    text(topics, "rgb"),
    text(topics, "depth_raw"),
    text(topics, "depth_filled"),
    text(topics, "action"),
    integer(inference, "max_actions", 0),
    number(inference, "min_action_interval_seconds", 0.0),
    integer(inference, "instruction_length", 1),
    boolean(inference, "force_cpu"),
    boolean(inference, "sample_actions"),
    boolean(inference, "publish_actions"),
    boolean(inference, "keep_running_after_stop"),
    number(sync, "slop_seconds", 0.0),
    integer(sync, "queue_size", 1),
    number(sync, "input_timeout_seconds", 0.0),
    boolean(sync, "exit_on_input_timeout"),
    number(startup, "image_timeout_seconds", 0.0, strict=True),
    number(startup, "action_subscriber_wait_seconds", 0.0),
    min_depth,
    max_depth,
    integer(depth, "log_every_frames", 0),
]
print("\n".join(values))
PY
)"
mapfile -t CONFIG_VALUES <<< "${CONFIG_TEXT}"
if [[ "${#CONFIG_VALUES[@]}" -ne 22 ]]; then
    echo "[start_vln_real] Config parser returned incomplete data." >&2
    exit 1
fi

VLN_INSTRUCTION="${VLN_INSTRUCTION:-${CONFIG_VALUES[0]}}"
VLN_CHECKPOINT="${VLN_CHECKPOINT:-${CONFIG_VALUES[1]}}"
VLN_RGB_TOPIC="${VLN_RGB_TOPIC:-${CONFIG_VALUES[2]}}"
VLN_DEPTH_RAW_TOPIC="${VLN_DEPTH_RAW_TOPIC:-${CONFIG_VALUES[3]}}"
VLN_DEPTH_FILLED_TOPIC="${VLN_DEPTH_FILLED_TOPIC:-${CONFIG_VALUES[4]}}"
VLN_ACTION_TOPIC="${VLN_ACTION_TOPIC:-${CONFIG_VALUES[5]}}"
VLN_MAX_ACTIONS="${VLN_MAX_ACTIONS:-${CONFIG_VALUES[6]}}"
VLN_MIN_ACTION_INTERVAL="${VLN_MIN_ACTION_INTERVAL:-${CONFIG_VALUES[7]}}"
VLN_INSTRUCTION_LENGTH="${VLN_INSTRUCTION_LENGTH:-${CONFIG_VALUES[8]}}"
VLN_FORCE_CPU="${VLN_FORCE_CPU:-${CONFIG_VALUES[9]}}"
VLN_SAMPLE_ACTIONS="${VLN_SAMPLE_ACTIONS:-${CONFIG_VALUES[10]}}"
VLN_PUBLISH_ACTIONS="${VLN_PUBLISH_ACTIONS:-${CONFIG_VALUES[11]}}"
VLN_KEEP_RUNNING_AFTER_STOP="${VLN_KEEP_RUNNING_AFTER_STOP:-${CONFIG_VALUES[12]}}"
VLN_SYNC_SLOP="${VLN_SYNC_SLOP:-${CONFIG_VALUES[13]}}"
VLN_SYNC_QUEUE_SIZE="${VLN_SYNC_QUEUE_SIZE:-${CONFIG_VALUES[14]}}"
VLN_INPUT_TIMEOUT="${VLN_INPUT_TIMEOUT:-${CONFIG_VALUES[15]}}"
VLN_EXIT_ON_INPUT_TIMEOUT="${VLN_EXIT_ON_INPUT_TIMEOUT:-${CONFIG_VALUES[16]}}"
VLN_STARTUP_TIMEOUT="${VLN_STARTUP_TIMEOUT:-${CONFIG_VALUES[17]}}"
VLN_PUBLISHER_WAIT="${VLN_PUBLISHER_WAIT:-${CONFIG_VALUES[18]}}"
VLN_MIN_DEPTH="${VLN_MIN_DEPTH:-${CONFIG_VALUES[19]}}"
VLN_MAX_DEPTH="${VLN_MAX_DEPTH:-${CONFIG_VALUES[20]}}"
VLN_DEPTH_LOG_EVERY="${VLN_DEPTH_LOG_EVERY:-${CONFIG_VALUES[21]}}"

# ROS Noetic's setup scripts are not safe under Bash nounset when a clean
# terminal has not inherited ROS_DISTRO yet. Temporarily disable nounset only
# while loading ROS, then restore the launcher's strict mode.
set +u
# shellcheck disable=SC1091
set --
source "${VLN_ROS_SETUP}"
set -- "${VLN_CLI_ARGS[@]}"
set -u

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
# rosnode commands return status 0 even for an unknown node on Noetic.
# A live rospy node's rosnode-info output contains its process ID.
if rosnode info /ros_depth_hole_filler 2>/dev/null \
    | grep -Eq '^Pid: [0-9]+$'; then
    echo "[start_vln_real] Reusing existing /ros_depth_hole_filler node."
else
    "${VLN_PYTHON}" "${VLN_SCRIPT_DIR}/ros_depth_hole_filler.py" \
        --input-topic "${VLN_DEPTH_RAW_TOPIC}" \
        --output-topic "${VLN_DEPTH_FILLED_TOPIC}" \
        --log-every "${VLN_DEPTH_LOG_EVERY}" &
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
    --instruction-length "${VLN_INSTRUCTION_LENGTH}"
    --min-depth "${VLN_MIN_DEPTH}"
    --max-depth "${VLN_MAX_DEPTH}"
)
if [[ "${VLN_FORCE_CPU}" == "1" ]]; then
    VLN_INFERENCE_ARGS+=(--cpu)
fi
if [[ "${VLN_SAMPLE_ACTIONS}" == "1" ]]; then
    VLN_INFERENCE_ARGS+=(--sample)
fi
if [[ "${VLN_PUBLISH_ACTIONS}" != "1" ]]; then
    VLN_INFERENCE_ARGS+=(--no-publish)
fi
if [[ "${VLN_KEEP_RUNNING_AFTER_STOP}" == "1" ]]; then
    VLN_INFERENCE_ARGS+=(--keep-running-after-stop)
fi
if [[ "${VLN_EXIT_ON_INPUT_TIMEOUT}" == "1" ]]; then
    VLN_INFERENCE_ARGS+=(--exit-on-input-timeout)
fi
VLN_INFERENCE_ARGS+=("${VLN_CLI_ARGS[@]}")

echo "[start_vln_real] Starting standalone PyTorch VLN inference."
echo "  config: ${VLN_INFERENCE_CONFIG_PATH}"
echo "  checkpoint: ${VLN_CHECKPOINT}"
echo "  instruction: ${VLN_INSTRUCTION}"
echo "  action topic: ${VLN_ACTION_TOPIC}"
echo "  configured max actions: ${VLN_MAX_ACTIONS}"
echo "  configured minimum action interval: ${VLN_MIN_ACTION_INTERVAL}s"
echo "  configured RGB-D sync slop: ${VLN_SYNC_SLOP}s"
if [[ "${#VLN_CLI_ARGS[@]}" -gt 0 ]]; then
    printf '  command-line overrides:'
    printf ' %q' "${VLN_CLI_ARGS[@]}"
    printf '\n'
fi
echo "  action message type: std_msgs/String"

VLN_EXIT_STATUS=0
"${VLN_PYTHON}" "${VLN_INFERENCE_ARGS[@]}" || VLN_EXIT_STATUS=$?
exit "${VLN_EXIT_STATUS}"
