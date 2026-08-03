#!/usr/bin/env bash
# Source this file from the other scripts; it does not change shell rc files.
ROBOT_BUNDLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROS_SETUP="${ROS_SETUP:-/opt/ros/noetic/setup.bash}"
GAZEBO_SETUP="${GAZEBO_SETUP:-/usr/share/gazebo/setup.sh}"
if [[ ! -f "$ROS_SETUP" ]]; then
  echo "ROS Noetic setup file not found: $ROS_SETUP" >&2
  return 1 2>/dev/null || exit 1
fi
if [[ ! -f "$GAZEBO_SETUP" ]]; then
  echo "Gazebo setup file not found: $GAZEBO_SETUP" >&2
  return 1 2>/dev/null || exit 1
fi
set +u
source "$ROS_SETUP"
source "$GAZEBO_SETUP"
set -u
export ROS_PACKAGE_PATH="$(dirname "$ROBOT_BUNDLE_DIR")${ROS_PACKAGE_PATH:+:$ROS_PACKAGE_PATH}"
export GAZEBO_PLUGIN_PATH="$ROBOT_BUNDLE_DIR/lib${GAZEBO_PLUGIN_PATH:+:$GAZEBO_PLUGIN_PATH}"
export LD_LIBRARY_PATH="$ROBOT_BUNDLE_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
