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
import math
import os
import sys
import threading
import time
from typing import NamedTuple


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


def normalize_action(action: str) -> str:
    """Normalize harmless case, whitespace, and hyphen differences."""

    return "_".join(action.strip().upper().replace("-", " ").split())


def action_to_motion(
    action: str,
    linear_speed: float,
    angular_speed: float,
    forward_distance: float,
    turn_angle_deg: float,
) -> Motion:
    """Return the velocity and open-loop duration for one VLN action."""

    action = normalize_action(action)
    if action not in VALID_ACTIONS:
        raise ValueError("Unsupported VLN action: {!r}".format(action))
    if action == "STOP":
        return Motion(0.0, 0.0, 0.0)
    if action == "MOVE_FORWARD":
        return Motion(
            linear_speed,
            0.0,
            forward_distance / linear_speed,
        )

    angular_z = angular_speed if action == "TURN_LEFT" else -angular_speed
    return Motion(
        0.0,
        angular_z,
        math.radians(turn_angle_deg) / angular_speed,
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
            linear_speed=self.args.linear_speed,
            angular_speed=self.args.angular_speed,
            forward_distance=self.args.forward_distance,
            turn_angle_deg=self.args.turn_angle_deg,
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
            "(geometry_msgs/Twist); forward=%.3f m at %.3f m/s; "
            "turn=%.2f deg at %.3f rad/s; rate=%.1f Hz",
            self.args.action_topic,
            self.args.cmd_vel_topic,
            self.args.forward_distance,
            self.args.linear_speed,
            self.args.turn_angle_deg,
            self.args.angular_speed,
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
        description="Convert VLN English actions to time-bounded ROS1 Twist commands."
    )
    parser.add_argument("--action-topic", default="/vln/action")
    parser.add_argument("--cmd-vel-topic", default="/cmd_vel")
    parser.add_argument("--linear-speed", type=float, default=0.10)
    parser.add_argument("--angular-speed", type=float, default=0.30)
    parser.add_argument("--forward-distance", type=float, default=0.25)
    parser.add_argument("--turn-angle-deg", type=float, default=15.0)
    parser.add_argument("--publish-rate", type=float, default=20.0)
    parser.add_argument(
        "--watchdog-timeout",
        type=float,
        default=10.0,
        help="Stop an active command after this action-input silence; 0 disables.",
    )
    parser.add_argument(
        "--stop-publish-count",
        type=int,
        default=3,
        help="Number of redundant zero Twist messages sent when stopping.",
    )
    parser.add_argument("--node-name", default="ros_vln_action_to_cmd_vel")
    return parser


def validate_arguments(args) -> None:
    if args.linear_speed <= 0.0:
        raise ValueError("--linear-speed must be positive.")
    if args.angular_speed <= 0.0:
        raise ValueError("--angular-speed must be positive.")
    if args.forward_distance <= 0.0:
        raise ValueError("--forward-distance must be positive.")
    if args.turn_angle_deg <= 0.0:
        raise ValueError("--turn-angle-deg must be positive.")
    if args.publish_rate <= 0.0:
        raise ValueError("--publish-rate must be positive.")
    if args.watchdog_timeout < 0.0:
        raise ValueError("--watchdog-timeout must be >= 0.")
    if args.stop_publish_count <= 0:
        raise ValueError("--stop-publish-count must be positive.")


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args(rospy.myargv(argv=sys.argv)[1:])
    try:
        validate_arguments(args)
    except ValueError as error:
        parser.error(str(error))

    rospy.init_node(args.node_name, anonymous=False)
    node = ActionToCmdVelNode(args)
    node.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
