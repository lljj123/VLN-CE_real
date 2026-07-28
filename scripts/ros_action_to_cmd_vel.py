#!/usr/bin/env python3

"""Convert VLN English actions into time-bounded ROS1 Twist commands.

The node subscribes to ``std_msgs/String`` actions and continuously publishes
``geometry_msgs/Twist`` while an action is active.  Motion is deliberately
time bounded: the default 0.25 m forward step and 15 degree turn match the
discrete action scale used by VLN-CE, but they must be calibrated for the
actual chassis.
"""

import argparse
import glob
import json
import math
import os
import sys
import threading
import time
from pathlib import Path
from typing import NamedTuple


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "action_to_cmd_vel.json"


def add_ros_python_paths() -> None:
    """Expose ROS Noetic modules to a non-system Python environment."""

    candidates = ["/usr/lib/python3/dist-packages"]
    ros_distro = os.environ.get("ROS_DISTRO")
    if ros_distro:
        candidates.append(
            "/opt/ros/{}/lib/python3/dist-packages".format(ros_distro)
        )
    else:
        candidates.extend(
            sorted(glob.glob("/opt/ros/*/lib/python3/dist-packages"))
        )

    for candidate in candidates:
        if os.path.isdir(candidate) and candidate not in sys.path:
            sys.path.append(candidate)


add_ros_python_paths()

import rospy  # noqa: E402
from geometry_msgs.msg import Twist  # noqa: E402
from std_msgs.msg import String  # noqa: E402


VALID_ACTIONS = {
    "STOP",
    "MOVE_FORWARD",
    "TURN_LEFT",
    "TURN_RIGHT",
}


class Motion(NamedTuple):
    linear_x: float
    angular_z: float
    duration: float


class MotionSettings(NamedTuple):
    forward_linear_speed: float
    forward_distance: float
    left_angular_speed: float
    left_angle_deg: float
    right_angular_speed: float
    right_angle_deg: float


def normalize_action(action: str) -> str:
    """Normalize harmless case, whitespace, and hyphen differences."""

    return "_".join(action.strip().upper().replace("-", " ").split())


def action_to_motion(
    action: str,
    settings: MotionSettings,
) -> Motion:
    """Return the velocity and open-loop duration for one VLN action."""

    action = normalize_action(action)
    if action not in VALID_ACTIONS:
        raise ValueError("Unsupported VLN action: {!r}".format(action))
    if action == "STOP":
        return Motion(0.0, 0.0, 0.0)
    if action == "MOVE_FORWARD":
        return Motion(
            settings.forward_linear_speed,
            0.0,
            settings.forward_distance / settings.forward_linear_speed,
        )

    if action == "TURN_LEFT":
        angular_speed = settings.left_angular_speed
        angle_deg = settings.left_angle_deg
        angular_z = angular_speed
    else:
        angular_speed = settings.right_angular_speed
        angle_deg = settings.right_angle_deg
        angular_z = -angular_speed
    return Motion(
        0.0,
        angular_z,
        math.radians(angle_deg) / angular_speed,
    )


def resolve_config_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _section(mapping, name: str):
    value = mapping.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(
            "Configuration section {!r} must be an object.".format(name)
        )
    return value


def _first_not_none(*values):
    for value in values:
        if value is not None:
            return value
    raise RuntimeError("No configuration value or fallback was supplied.")


def load_configuration(args) -> None:
    """Load JSON defaults, then apply any explicit command-line overrides."""

    config_path = resolve_config_path(args.config)
    try:
        with config_path.open("r", encoding="utf-8") as config_file:
            config = json.load(config_file)
    except OSError as error:
        raise ValueError(
            "Cannot read configuration {}: {}".format(config_path, error)
        )
    except ValueError as error:
        raise ValueError(
            "Invalid JSON in configuration {}: {}".format(config_path, error)
        )
    if not isinstance(config, dict):
        raise ValueError("The configuration root must be a JSON object.")

    topics = _section(config, "topics")
    control = _section(config, "control")
    actions = _section(config, "actions")
    forward = _section(actions, "MOVE_FORWARD")
    turn_left = _section(actions, "TURN_LEFT")
    turn_right = _section(actions, "TURN_RIGHT")

    args.config_path = str(config_path)
    args.action_topic = _first_not_none(
        args.action_topic,
        topics.get("action"),
        "/vln/action",
    )
    args.cmd_vel_topic = _first_not_none(
        args.cmd_vel_topic,
        topics.get("cmd_vel"),
        "/cmd_vel",
    )
    args.publish_rate = _first_not_none(
        args.publish_rate,
        control.get("publish_rate_hz"),
        20.0,
    )
    args.watchdog_timeout = _first_not_none(
        args.watchdog_timeout,
        control.get("watchdog_timeout_s"),
        10.0,
    )
    args.stop_publish_count = _first_not_none(
        args.stop_publish_count,
        control.get("stop_publish_count"),
        3,
    )
    args.linear_speed = _first_not_none(
        args.linear_speed,
        forward.get("linear_speed_mps"),
        0.10,
    )
    args.forward_distance = _first_not_none(
        args.forward_distance,
        forward.get("distance_m"),
        0.25,
    )
    args.left_angular_speed = _first_not_none(
        args.left_angular_speed,
        args.shared_angular_speed,
        turn_left.get("angular_speed_radps"),
        0.30,
    )
    args.right_angular_speed = _first_not_none(
        args.right_angular_speed,
        args.shared_angular_speed,
        turn_right.get("angular_speed_radps"),
        0.30,
    )
    args.left_turn_angle_deg = _first_not_none(
        args.left_turn_angle_deg,
        args.shared_turn_angle_deg,
        turn_left.get("angle_deg"),
        15.0,
    )
    args.right_turn_angle_deg = _first_not_none(
        args.right_turn_angle_deg,
        args.shared_turn_angle_deg,
        turn_right.get("angle_deg"),
        15.0,
    )


def make_twist(linear_x: float = 0.0, angular_z: float = 0.0) -> Twist:
    message = Twist()
    message.linear.x = linear_x
    message.angular.z = angular_z
    return message


class ActionToCmdVelNode:
    """Preemptible action executor with automatic stop and watchdog."""

    def __init__(self, args) -> None:
        self.args = args
        self.motion_settings = MotionSettings(
            forward_linear_speed=args.linear_speed,
            forward_distance=args.forward_distance,
            left_angular_speed=args.left_angular_speed,
            left_angle_deg=args.left_turn_angle_deg,
            right_angular_speed=args.right_angular_speed,
            right_angle_deg=args.right_turn_angle_deg,
        )
        self.lock = threading.Lock()
        self.pending_action = "STOP"
        self.pending_sequence = 0
        self.handled_sequence = 0
        self.active_action = None
        self.active_motion = None
        self.active_until = 0.0
        self.last_action_time = None
        self.watchdog_reported = False
        self.stop_repeats_remaining = args.stop_publish_count

        # Construct the publisher before the subscriber so an early STOP can
        # always be forwarded to the chassis.
        self.velocity_publisher = rospy.Publisher(
            args.cmd_vel_topic,
            Twist,
            queue_size=1,
        )
        self.action_subscriber = rospy.Subscriber(
            args.action_topic,
            String,
            self._action_callback,
            queue_size=10,
        )
        rospy.on_shutdown(self._on_shutdown)

    def _action_callback(self, message: String) -> None:
        received = message.data
        action = normalize_action(received)
        is_valid = action in VALID_ACTIONS
        if not is_valid:
            rospy.logerr(
                "Unsupported action %r; stopping chassis. Expected one of %s.",
                received,
                ", ".join(sorted(VALID_ACTIONS)),
            )
            action = "STOP"

        now = time.monotonic()
        with self.lock:
            self.pending_sequence += 1
            self.pending_action = action
            self.last_action_time = now
            self.watchdog_reported = False

            # STOP and malformed messages preempt motion in the callback.  A
            # zero command is sent before releasing the lock, preventing the
            # control loop from publishing one more stale motion command.
            if action == "STOP":
                self.active_action = None
                self.active_motion = None
                self.active_until = 0.0
                self.stop_repeats_remaining = self.args.stop_publish_count
                self.velocity_publisher.publish(make_twist())

    def _start_pending_action(self, now: float) -> None:
        action = self.pending_action
        self.handled_sequence = self.pending_sequence

        # Give every newly accepted action a clean zero-velocity boundary.
        self.velocity_publisher.publish(make_twist())
        if action == "STOP":
            self.active_action = None
            self.active_motion = None
            self.active_until = 0.0
            self.stop_repeats_remaining = self.args.stop_publish_count
            rospy.loginfo("action=STOP cmd_vel=(0.000 m/s, 0.000 rad/s)")
            return

        motion = action_to_motion(
            action=action,
            settings=self.motion_settings,
        )
        self.active_action = action
        self.active_motion = motion
        self.active_until = now + motion.duration
        self.stop_repeats_remaining = 0
        rospy.loginfo(
            "action=%s cmd_vel=(%.3f m/s, %.3f rad/s) duration=%.3f s",
            action,
            motion.linear_x,
            motion.angular_z,
            motion.duration,
        )

    def _stop_active_action(self, reason: str) -> None:
        action = self.active_action
        self.active_action = None
        self.active_motion = None
        self.active_until = 0.0
        self.stop_repeats_remaining = self.args.stop_publish_count
        self.velocity_publisher.publish(make_twist())
        if action is not None:
            rospy.loginfo("action=%s finished (%s); chassis stopped", action, reason)

    def run(self) -> None:
        rospy.loginfo(
            "Action converter ready: %s (std_msgs/String) -> %s "
            "(geometry_msgs/Twist); config=%s; "
            "forward=%.3f m at %.3f m/s; "
            "left=%.2f deg at %.3f rad/s; "
            "right=%.2f deg at %.3f rad/s; rate=%.1f Hz",
            self.args.action_topic,
            self.args.cmd_vel_topic,
            self.args.config_path,
            self.args.forward_distance,
            self.args.linear_speed,
            self.args.left_turn_angle_deg,
            self.args.left_angular_speed,
            self.args.right_turn_angle_deg,
            self.args.right_angular_speed,
            self.args.publish_rate,
        )
        rate = rospy.Rate(self.args.publish_rate)

        while not rospy.is_shutdown():
            now = time.monotonic()
            with self.lock:
                if self.pending_sequence != self.handled_sequence:
                    self._start_pending_action(now)

                if (
                    self.active_motion is not None
                    and self.args.watchdog_timeout > 0.0
                    and self.last_action_time is not None
                    and now - self.last_action_time
                    >= self.args.watchdog_timeout
                ):
                    self._stop_active_action("action watchdog timeout")
                    if not self.watchdog_reported:
                        rospy.logwarn(
                            "No action received for %.1f s; chassis stopped.",
                            self.args.watchdog_timeout,
                        )
                        self.watchdog_reported = True

                if self.active_motion is not None:
                    if now >= self.active_until:
                        self._stop_active_action("target duration reached")
                    else:
                        self.velocity_publisher.publish(
                            make_twist(
                                self.active_motion.linear_x,
                                self.active_motion.angular_z,
                            )
                        )
                elif self.stop_repeats_remaining > 0:
                    self.velocity_publisher.publish(make_twist())
                    self.stop_repeats_remaining -= 1

            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                break

    def _on_shutdown(self) -> None:
        # Publish several zeros because a single final TCPROS packet may be
        # lost while ROS connections are shutting down.
        with self.lock:
            self.active_action = None
            self.active_motion = None
            for _ in range(self.args.stop_publish_count):
                self.velocity_publisher.publish(make_twist())
                time.sleep(0.01)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert VLN English actions to time-bounded ROS1 Twist "
            "commands."
        )
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help=(
            "JSON configuration file; relative paths resolve from the "
            "repository."
        ),
    )
    parser.add_argument("--action-topic")
    parser.add_argument("--cmd-vel-topic")
    parser.add_argument("--linear-speed", type=float)
    parser.add_argument("--forward-distance", type=float)
    parser.add_argument(
        "--angular-speed",
        dest="shared_angular_speed",
        type=float,
        help="Override both left and right angular speeds.",
    )
    parser.add_argument(
        "--turn-angle-deg",
        dest="shared_turn_angle_deg",
        type=float,
        help="Override both left and right turn angles.",
    )
    parser.add_argument("--left-angular-speed", type=float)
    parser.add_argument("--right-angular-speed", type=float)
    parser.add_argument("--left-turn-angle-deg", type=float)
    parser.add_argument("--right-turn-angle-deg", type=float)
    parser.add_argument("--publish-rate", type=float)
    parser.add_argument(
        "--watchdog-timeout",
        type=float,
        default=None,
        help="Stop an active command after this action-input silence; 0 disables.",
    )
    parser.add_argument(
        "--stop-publish-count",
        type=int,
        default=None,
        help="Number of redundant zero Twist messages sent when stopping.",
    )
    parser.add_argument("--node-name", default="ros_vln_action_to_cmd_vel")
    return parser


def validate_arguments(args) -> None:
    if not isinstance(args.action_topic, str) or not args.action_topic.strip():
        raise ValueError("action topic must be a non-empty string.")
    if not isinstance(args.cmd_vel_topic, str) or not args.cmd_vel_topic.strip():
        raise ValueError("cmd_vel topic must be a non-empty string.")

    positive_values = (
        ("linear speed", args.linear_speed),
        ("forward distance", args.forward_distance),
        ("left angular speed", args.left_angular_speed),
        ("right angular speed", args.right_angular_speed),
        ("left turn angle", args.left_turn_angle_deg),
        ("right turn angle", args.right_turn_angle_deg),
        ("publish rate", args.publish_rate),
    )
    for name, value in positive_values:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value <= 0.0
        ):
            raise ValueError("{} must be a positive finite number.".format(name))

    if (
        isinstance(args.watchdog_timeout, bool)
        or not isinstance(args.watchdog_timeout, (int, float))
        or not math.isfinite(float(args.watchdog_timeout))
        or args.watchdog_timeout < 0.0
    ):
        raise ValueError("watchdog timeout must be a finite number >= 0.")
    if (
        isinstance(args.stop_publish_count, bool)
        or not isinstance(args.stop_publish_count, int)
        or args.stop_publish_count <= 0
    ):
        raise ValueError("stop publish count must be a positive integer.")


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args(rospy.myargv(argv=sys.argv)[1:])
    try:
        load_configuration(args)
        validate_arguments(args)
    except ValueError as error:
        parser.error(str(error))

    rospy.init_node(args.node_name, anonymous=False)
    node = ActionToCmdVelNode(args)
    node.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
