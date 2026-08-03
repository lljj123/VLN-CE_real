#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/setup_env.sh"

required=(roslaunch rospack gz)
for command_name in "${required[@]}"; do
  command -v "$command_name" >/dev/null || {
    echo "Missing command: $command_name" >&2
    exit 1
  }
done
for package_name in gazebo_ros robot_state_publisher rospy geometry_msgs; do
  rospack find "$package_name" >/dev/null || {
    echo "Missing ROS package: $package_name" >&2
    exit 1
  }
done
rospack find robot_sim_bundle >/dev/null
python3 - "$SCRIPT_DIR" <<'PY'
import pathlib
import sys
import xml.etree.ElementTree as ET
root = pathlib.Path(sys.argv[1])
for relative in ("package.xml", "launch/spawn_robot.launch", "launch/robot_world.launch", "urdf/turtlebot_kinect_sim.urdf"):
    ET.parse(root / relative)
print("XML validation: OK")
PY
for plugin in "$SCRIPT_DIR"/lib/*.so; do
  if ldd "$plugin" | grep -q 'not found'; then
    echo "Unresolved library dependency: $plugin" >&2
    ldd "$plugin" | grep 'not found' >&2
    exit 1
  fi
done
roslaunch --nodes robot_sim_bundle robot_world.launch gui:=false >/dev/null
echo "robot_sim_bundle check: OK"
