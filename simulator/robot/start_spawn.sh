#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/setup_env.sh"
exec roslaunch robot_sim_bundle spawn_robot.launch "$@"
