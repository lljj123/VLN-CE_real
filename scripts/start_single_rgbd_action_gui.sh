#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REAL_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
VLN_GUI_PYTHON="${VLN_GUI_PYTHON:-/usr/bin/python3}"

if [[ ! -x "${VLN_GUI_PYTHON}" ]]; then
    echo "[single_rgbd_action_gui] Python not found: ${VLN_GUI_PYTHON}" >&2
    exit 1
fi

cd "${REAL_ROOT}"
exec "${VLN_GUI_PYTHON}" scripts/single_rgbd_action_gui.py "$@"
