#!/usr/bin/env bash

# Start only the passive VLN timing monitor. This process is intentionally
# independent from start_vln_with_base.sh and never publishes chassis commands.

set -Eeuo pipefail

VLN_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VLN_REPO_ROOT="$(cd -- "${VLN_SCRIPT_DIR}/.." && pwd)"
VLN_ROS_SETUP="${VLN_ROS_SETUP:-/opt/ros/noetic/setup.bash}"
VLN_CONTROL_PYTHON="${VLN_PYTHON:-/usr/bin/python3}"
VLN_INFERENCE_CONFIG="${VLN_INFERENCE_CONFIG:-config/vln_inference.json}"
VLN_MONITOR_CLI_ARGS=("$@")

if [[ ! -f "${VLN_ROS_SETUP}" ]]; then
    echo "[start_vln_action_monitor] ROS setup not found: ${VLN_ROS_SETUP}" >&2
    exit 1
fi
if [[ ! -x "${VLN_CONTROL_PYTHON}" ]]; then
    echo "[start_vln_action_monitor] Python not found: ${VLN_CONTROL_PYTHON}" >&2
    exit 1
fi
if [[ "${VLN_INFERENCE_CONFIG}" = /* ]]; then
    VLN_INFERENCE_CONFIG_PATH="${VLN_INFERENCE_CONFIG}"
else
    VLN_INFERENCE_CONFIG_PATH="${VLN_REPO_ROOT}/${VLN_INFERENCE_CONFIG}"
fi
if [[ ! -f "${VLN_INFERENCE_CONFIG_PATH}" ]]; then
    echo "[start_vln_action_monitor] Inference config not found: " \
        "${VLN_INFERENCE_CONFIG_PATH}" >&2
    exit 1
fi

MONITOR_CONFIG_TEXT="$("${VLN_CONTROL_PYTHON}" - \
    "${VLN_INFERENCE_CONFIG_PATH}" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as input_file:
    config = json.load(input_file)

topics = config.get("topics")
if not isinstance(topics, dict):
    raise ValueError("topics must be an object")
values = []
for key in ("action", "action_result", "inference_metrics"):
    value = topics.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError("topics.{} must be a non-empty string".format(key))
    value = value.strip()
    if "\n" in value or "\r" in value:
        raise ValueError("topics.{} must be a single line".format(key))
    values.append(value)

monitor = config.get("monitor")
if not isinstance(monitor, dict):
    raise ValueError("monitor must be an object")
gui = monitor.get("gui")
if not isinstance(gui, bool):
    raise ValueError("monitor.gui must be true or false")
values.append("1" if gui else "0")

output_directory = monitor.get("output_directory")
if not isinstance(output_directory, str) or not output_directory.strip():
    raise ValueError("monitor.output_directory must be a non-empty string")
output_directory = output_directory.strip()
if "\n" in output_directory or "\r" in output_directory:
    raise ValueError("monitor.output_directory must be a single line")
values.append(output_directory)

history_size = monitor.get("history_size")
if (
    isinstance(history_size, bool)
    or not isinstance(history_size, int)
    or history_size < 0
):
    raise ValueError("monitor.history_size must be an integer >= 0")
values.append(str(history_size))
print("\n".join(values))
PY
)"
mapfile -t MONITOR_CONFIG <<< "${MONITOR_CONFIG_TEXT}"
if [[ "${#MONITOR_CONFIG[@]}" -ne 6 ]]; then
    echo "[start_vln_action_monitor] Config parser returned incomplete data." >&2
    exit 1
fi

VLN_ACTION_TOPIC="${VLN_ACTION_TOPIC:-${MONITOR_CONFIG[0]}}"
VLN_ACTION_RESULT_TOPIC="${VLN_ACTION_RESULT_TOPIC:-${MONITOR_CONFIG[1]}}"
VLN_INFERENCE_METRICS_TOPIC="${VLN_INFERENCE_METRICS_TOPIC:-${MONITOR_CONFIG[2]}}"
VLN_MONITOR_GUI="${VLN_MONITOR_GUI:-${MONITOR_CONFIG[3]}}"
VLN_MONITOR_OUTPUT_DIRECTORY="${VLN_MONITOR_OUTPUT_DIRECTORY:-${MONITOR_CONFIG[4]}}"
VLN_MONITOR_HISTORY_SIZE="${VLN_MONITOR_HISTORY_SIZE:-${MONITOR_CONFIG[5]}}"

if [[ "${VLN_MONITOR_GUI}" != "0" && "${VLN_MONITOR_GUI}" != "1" ]]; then
    echo "[start_vln_action_monitor] VLN_MONITOR_GUI must be 0 or 1." >&2
    exit 1
fi
if [[ -z "${VLN_MONITOR_OUTPUT_DIRECTORY}" ]]; then
    echo "[start_vln_action_monitor] Output directory cannot be empty." >&2
    exit 1
fi
if [[ ! "${VLN_MONITOR_HISTORY_SIZE}" =~ ^[0-9]+$ ]]; then
    echo "[start_vln_action_monitor] History size must be >= 0." >&2
    exit 1
fi

set +u
# shellcheck disable=SC1090
source "${VLN_ROS_SETUP}"
set -u

if ! rostopic list >/dev/null 2>&1; then
    echo "[start_vln_action_monitor] ROS master is unreachable." >&2
    exit 1
fi
if rosnode info /ros_vln_action_monitor 2>/dev/null \
    | grep -Eq '^Pid: [0-9]+$'; then
    echo "[start_vln_action_monitor] /ros_vln_action_monitor is already running." >&2
    exit 1
fi

VLN_MONITOR_ARGS=(
    "${VLN_SCRIPT_DIR}/ros_vln_action_monitor.py"
    --action-topic "${VLN_ACTION_TOPIC}"
    --action-result-topic "${VLN_ACTION_RESULT_TOPIC}"
    --inference-metrics-topic "${VLN_INFERENCE_METRICS_TOPIC}"
    --output-directory "${VLN_MONITOR_OUTPUT_DIRECTORY}"
    --history-size "${VLN_MONITOR_HISTORY_SIZE}"
)
if [[ "${VLN_MONITOR_GUI}" != "1" ]]; then
    VLN_MONITOR_ARGS+=(--no-gui)
fi

echo "[start_vln_action_monitor] Starting independent timing monitor:"
echo "  action topic: ${VLN_ACTION_TOPIC}"
echo "  action result topic: ${VLN_ACTION_RESULT_TOPIC}"
echo "  inference metrics topic: ${VLN_INFERENCE_METRICS_TOPIC}"
echo "  GUI enabled: ${VLN_MONITOR_GUI}"
echo "  output directory: ${VLN_MONITOR_OUTPUT_DIRECTORY}"

exec "${VLN_CONTROL_PYTHON}" "${VLN_MONITOR_ARGS[@]}" \
    "${VLN_MONITOR_CLI_ARGS[@]}"
