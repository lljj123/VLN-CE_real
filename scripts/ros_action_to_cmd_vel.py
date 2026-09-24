#!/usr/bin/env python3

"""Convert VLN English actions into odometry-bounded ROS1 Twist commands.

The node subscribes to ``std_msgs/String`` actions and continuously publishes
``geometry_msgs/Twist`` while an action is active.  With the default
configuration, ``nav_msgs/Odometry`` feedback closes each distance/angle
action and a time limit remains as a fail-safe.
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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


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
from nav_msgs.msg import Odometry  # noqa: E402
from std_msgs.msg import String  # noqa: E402

from vlnce_real.action_protocol import (  # noqa: E402
    decode_action_command,
    encode_action_result,
)
from vlnce_real.control_timing import WakeableRate  # noqa: E402
from vlnce_real.odom_control import (  # noqa: E402
    ClosedLoopMotion,
    OdomControlSettings,
    PlanarPose,
    quaternion_to_yaw,
)


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
    """Return maximum velocity and nominal duration for one VLN action."""

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


def _configuration_boolean(value, name):
    if not isinstance(value, bool):
        raise ValueError("{} must be true or false.".format(name))
    return value


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
    args.action_result_topic = _first_not_none(
        args.action_result_topic,
        topics.get("action_result"),
        "/vln/action_result",
    )
    args.cmd_vel_topic = _first_not_none(
        args.cmd_vel_topic,
        topics.get("cmd_vel"),
        "/cmd_vel",
    )
    args.odom_topic = _first_not_none(
        args.odom_topic,
        topics.get("odom"),
        "/odom",
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
    configured_use_odom = control.get("use_odom", True)
    args.use_odom = _first_not_none(
        args.use_odom,
        _configuration_boolean(configured_use_odom, "control.use_odom"),
    )
    args.odom_stale_timeout = _first_not_none(
        args.odom_stale_timeout,
        control.get("odom_stale_timeout_s"),
        0.5,
    )
    args.action_timeout_scale = _first_not_none(
        args.action_timeout_scale,
        control.get("action_timeout_scale"),
        3.0,
    )
    args.minimum_action_timeout = _first_not_none(
        args.minimum_action_timeout,
        control.get("minimum_action_timeout_s"),
        3.0,
    )
    args.distance_tolerance = _first_not_none(
        args.distance_tolerance,
        control.get("distance_tolerance_m"),
        0.02,
    )
    args.angle_tolerance_deg = _first_not_none(
        args.angle_tolerance_deg,
        control.get("angle_tolerance_deg"),
        2.0,
    )
    args.linear_slowdown_distance = _first_not_none(
        args.linear_slowdown_distance,
        control.get("linear_slowdown_distance_m"),
        0.10,
    )
    args.angular_slowdown_angle_deg = _first_not_none(
        args.angular_slowdown_angle_deg,
        control.get("angular_slowdown_angle_deg"),
        8.0,
    )
    args.minimum_linear_speed = _first_not_none(
        args.minimum_linear_speed,
        control.get("minimum_linear_speed_mps"),
        0.05,
    )
    args.minimum_angular_speed = _first_not_none(
        args.minimum_angular_speed,
        control.get("minimum_angular_speed_radps"),
        0.10,
    )
    args.forward_heading_kp = _first_not_none(
        args.forward_heading_kp,
        control.get("forward_heading_kp"),
        1.5,
    )
    args.maximum_heading_correction = _first_not_none(
        args.maximum_heading_correction,
        control.get("maximum_heading_correction_radps"),
        0.20,
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
        self.odom_settings = OdomControlSettings(
            distance_tolerance_m=args.distance_tolerance,
            angle_tolerance_rad=math.radians(args.angle_tolerance_deg),
            linear_slowdown_distance_m=args.linear_slowdown_distance,
            angular_slowdown_angle_rad=math.radians(
                args.angular_slowdown_angle_deg
            ),
            minimum_linear_speed_mps=args.minimum_linear_speed,
            minimum_angular_speed_radps=args.minimum_angular_speed,
            forward_heading_kp=args.forward_heading_kp,
            maximum_heading_correction_radps=(
                args.maximum_heading_correction
            ),
            action_timeout_scale=args.action_timeout_scale,
            minimum_action_timeout_s=args.minimum_action_timeout,
        )
        self.lock = threading.Lock()
        self.latest_odom = None
        self.pending_action = "STOP"
        self.pending_command_sequence = None
        self.pending_sequence = 0
        self.handled_sequence = 0
        self.active_action = None
        self.active_command_sequence = None
        self.active_started_at = None
        self.active_control_mode = None
        self.active_target_value = None
        self.active_progress_value = None
        self.active_target_unit = None
        self.active_previous_action_end_to_start_seconds = None
        self.active_motion = None
        self.active_controller = None
        self.active_until = 0.0
        self.last_action_finished_at = None
        self.last_action_time = None
        self.watchdog_reported = False
        self.stop_repeats_remaining = args.stop_publish_count
        self.control_rate = WakeableRate(args.publish_rate)

        # Construct the publisher before the subscriber so an early STOP can
        # always be forwarded to the chassis.
        self.velocity_publisher = rospy.Publisher(
            args.cmd_vel_topic,
            Twist,
            queue_size=1,
        )
        self.result_publisher = rospy.Publisher(
            args.action_result_topic,
            String,
            queue_size=10,
        )
        self.odom_subscriber = None
        if args.use_odom:
            self.odom_subscriber = rospy.Subscriber(
                args.odom_topic,
                Odometry,
                self._odom_callback,
                queue_size=1,
            )
        self.action_subscriber = rospy.Subscriber(
            args.action_topic,
            String,
            self._action_callback,
            queue_size=10,
        )
        rospy.on_shutdown(self._on_shutdown)

    def _odom_callback(self, message: Odometry) -> None:
        position = message.pose.pose.position
        orientation = message.pose.pose.orientation
        pose = PlanarPose(
            float(position.x),
            float(position.y),
            quaternion_to_yaw(
                orientation.x,
                orientation.y,
                orientation.z,
                orientation.w,
            ),
        )
        with self.lock:
            self.latest_odom = (
                pose,
                time.monotonic(),
                message.header.stamp.to_sec(),
            )

    def _fresh_odom(self, now):
        if self.latest_odom is None:
            return None
        pose, arrival_time, ros_stamp = self.latest_odom
        if now - arrival_time > self.args.odom_stale_timeout:
            return None
        return pose, arrival_time, ros_stamp

    def _publish_active_velocity(self, linear_x, angular_z) -> None:
        """Publish motion and capture the prior-end to current-start gap."""
        if (
            self.active_previous_action_end_to_start_seconds is None
            and self.last_action_finished_at is not None
            and (abs(linear_x) > 0.0 or abs(angular_z) > 0.0)
        ):
            motion_started_at = time.monotonic()
            self.active_previous_action_end_to_start_seconds = max(
                0.0,
                motion_started_at - self.last_action_finished_at,
            )
            rospy.loginfo(
                "action=%s sequence=%s previous_end_to_start_ms=%.2f",
                self.active_action,
                self.active_command_sequence,
                1000.0
                * self.active_previous_action_end_to_start_seconds,
            )
        self.velocity_publisher.publish(make_twist(linear_x, angular_z))

    def _publish_result(
        self,
        sequence,
        action,
        status,
        reason,
        execution_seconds=None,
        control_mode=None,
        target_value=None,
        progress_value=None,
        target_unit=None,
        previous_action_end_to_start_seconds=None,
    ) -> None:
        payload = encode_action_result(
            sequence=sequence,
            action=action,
            status=status,
            reason=reason,
            execution_seconds=execution_seconds,
            control_mode=control_mode,
            target_value=target_value,
            progress_value=progress_value,
            target_unit=target_unit,
            previous_action_end_to_start_seconds=(
                previous_action_end_to_start_seconds
            ),
        )
        self.result_publisher.publish(String(data=payload))

    def _preempt_pending_action(self, reason) -> None:
        if self.pending_sequence == self.handled_sequence:
            return
        self._publish_result(
            self.pending_command_sequence,
            self.pending_action,
            "preempted",
            reason,
            execution_seconds=0.0,
        )
        self.handled_sequence = self.pending_sequence

    def _action_callback(self, message: String) -> None:
        received = message.data
        try:
            command = decode_action_command(received)
        except ValueError as error:
            rospy.logerr(
                "Malformed action command %r (%s); stopping chassis.",
                received,
                error,
            )
            with self.lock:
                self._preempt_pending_action("superseded by malformed command")
                if self.active_action is not None:
                    self._stop_active_action(
                        "preempted by malformed command", "preempted"
                    )
                else:
                    self.velocity_publisher.publish(make_twist())
                    self.stop_repeats_remaining = self.args.stop_publish_count
                self._publish_result(
                    None,
                    "UNKNOWN",
                    "failed",
                    "malformed action command",
                    execution_seconds=0.0,
                )
            return

        action = normalize_action(command.action)
        if action not in VALID_ACTIONS:
            rospy.logerr(
                "Unsupported action %r; stopping chassis. Expected one of %s.",
                command.action,
                ", ".join(sorted(VALID_ACTIONS)),
            )
            with self.lock:
                self._preempt_pending_action("superseded by invalid command")
                if self.active_action is not None:
                    self._stop_active_action(
                        "preempted by invalid command", "preempted"
                    )
                else:
                    self.velocity_publisher.publish(make_twist())
                    self.stop_repeats_remaining = self.args.stop_publish_count
                self._publish_result(
                    command.sequence,
                    action,
                    "failed",
                    "unsupported action",
                    execution_seconds=0.0,
                )
            return

        now = time.monotonic()
        with self.lock:
            self._preempt_pending_action("superseded by newer command")
            self.pending_sequence += 1
            self.pending_action = action
            self.pending_command_sequence = command.sequence
            self.last_action_time = now
            self.watchdog_reported = False

            # STOP preempts motion in the callback.  A zero command is sent
            # before releasing the lock, preventing the control loop from
            # publishing one more stale motion command.
            if action == "STOP":
                self.handled_sequence = self.pending_sequence
                if self.active_action is not None:
                    self._stop_active_action(
                        "preempted by STOP command", "preempted"
                    )
                else:
                    self.stop_repeats_remaining = self.args.stop_publish_count
                    self.velocity_publisher.publish(make_twist())
                self._publish_result(
                    command.sequence,
                    action,
                    "stopped",
                    "stop command",
                    execution_seconds=0.0,
                    control_mode="stop",
                )
                rospy.loginfo(
                    "action=STOP sequence=%s cmd_vel=(0.000 m/s, 0.000 rad/s)",
                    command.sequence,
                )
        # Do not make a newly accepted action wait for the next periodic tick.
        # WakeableRate's generation counter preserves a wake-up that occurs
        # immediately before the control loop begins waiting.
        self.control_rate.wake()

    def _start_pending_action(self, now: float) -> None:
        action = self.pending_action
        command_sequence = self.pending_command_sequence
        self.handled_sequence = self.pending_sequence

        if self.active_action is not None:
            self._stop_active_action(
                "preempted by newer command", "preempted"
            )

        # Give every newly accepted action a clean zero-velocity boundary.
        self.velocity_publisher.publish(make_twist())
        if action == "STOP":
            self.active_action = None
            self.active_motion = None
            self.active_controller = None
            self.active_until = 0.0
            self.stop_repeats_remaining = self.args.stop_publish_count
            rospy.loginfo("action=STOP cmd_vel=(0.000 m/s, 0.000 rad/s)")
            self._publish_result(
                command_sequence,
                action,
                "stopped",
                "stop command",
                execution_seconds=0.0,
                control_mode="stop",
            )
            return

        motion = action_to_motion(
            action=action,
            settings=self.motion_settings,
        )
        controller = None
        if self.args.use_odom:
            odom = self._fresh_odom(now)
            if odom is None:
                self.active_action = None
                self.active_motion = None
                self.active_controller = None
                self.active_until = 0.0
                self.stop_repeats_remaining = self.args.stop_publish_count
                self._publish_result(
                    command_sequence,
                    action,
                    "failed",
                    "odometry unavailable or stale",
                    execution_seconds=0.0,
                    control_mode="odom_closed_loop",
                )
                rospy.logerr(
                    "action=%s rejected: no fresh odometry on %s within "
                    "%.3f s; chassis stopped",
                    action,
                    self.args.odom_topic,
                    self.args.odom_stale_timeout,
                )
                return
            controller = ClosedLoopMotion(
                action=action,
                start_pose=odom[0],
                nominal_linear_x=motion.linear_x,
                nominal_angular_z=motion.angular_z,
                nominal_duration=motion.duration,
                settings=self.odom_settings,
                start_time=now,
            )
        self.active_action = action
        self.active_command_sequence = command_sequence
        self.active_started_at = now
        self.active_previous_action_end_to_start_seconds = None
        self.active_motion = motion
        self.active_controller = controller
        self.active_until = now + (
            controller.timeout_s if controller is not None else motion.duration
        )
        self.stop_repeats_remaining = 0
        if controller is not None:
            self.active_control_mode = "odom_closed_loop"
            self.active_progress_value = 0.0
            if action == "MOVE_FORWARD":
                self.active_target_value = controller.target
                self.active_target_unit = "m"
            else:
                self.active_target_value = math.degrees(controller.target)
                self.active_target_unit = "deg"
            target = (
                "{:.3f} m".format(controller.target)
                if action == "MOVE_FORWARD"
                else "{:.2f} deg".format(math.degrees(controller.target))
            )
            rospy.loginfo(
                "action=%s odom_target=%s cmd_vel_max=(%.3f m/s, "
                "%.3f rad/s) timeout=%.3f s",
                action,
                target,
                motion.linear_x,
                motion.angular_z,
                controller.timeout_s,
            )
        else:
            self.active_control_mode = "open_loop"
            self.active_target_value = motion.duration
            self.active_progress_value = 0.0
            self.active_target_unit = "s"
            rospy.loginfo(
                "action=%s open_loop cmd_vel=(%.3f m/s, %.3f rad/s) "
                "duration=%.3f s",
                action,
                motion.linear_x,
                motion.angular_z,
                motion.duration,
            )

    def _stop_active_action(self, reason: str, status: str) -> None:
        action = self.active_action
        command_sequence = self.active_command_sequence
        execution_seconds = (
            None
            if self.active_started_at is None
            else max(0.0, time.monotonic() - self.active_started_at)
        )
        control_mode = self.active_control_mode
        target_value = self.active_target_value
        progress_value = self.active_progress_value
        target_unit = self.active_target_unit
        previous_action_end_to_start_seconds = (
            self.active_previous_action_end_to_start_seconds
        )
        self.active_action = None
        self.active_command_sequence = None
        self.active_started_at = None
        self.active_control_mode = None
        self.active_target_value = None
        self.active_progress_value = None
        self.active_target_unit = None
        self.active_previous_action_end_to_start_seconds = None
        self.active_motion = None
        self.active_controller = None
        self.active_until = 0.0
        self.stop_repeats_remaining = self.args.stop_publish_count
        self.velocity_publisher.publish(make_twist())
        self.last_action_finished_at = time.monotonic()
        if action is not None:
            self._publish_result(
                command_sequence,
                action,
                status,
                reason,
                execution_seconds=execution_seconds,
                control_mode=control_mode,
                target_value=target_value,
                progress_value=progress_value,
                target_unit=target_unit,
                previous_action_end_to_start_seconds=(
                    previous_action_end_to_start_seconds
                ),
            )
            rospy.loginfo("action=%s finished (%s); chassis stopped", action, reason)

    def run(self) -> None:
        rospy.loginfo(
            "Action converter ready: %s (std_msgs/String) -> %s "
            "(geometry_msgs/Twist); config=%s; "
            "forward=%.3f m at %.3f m/s; "
            "left=%.2f deg at %.3f rad/s; "
            "right=%.2f deg at %.3f rad/s; rate=%.1f Hz; odom=%s",
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
            (
                "{} (closed-loop)".format(self.args.odom_topic)
                if self.args.use_odom
                else "disabled (open-loop)"
            ),
        )
        rospy.loginfo(
            "Action execution results: %s (std_msgs/String)",
            self.args.action_result_topic,
        )
        wake_generation = self.control_rate.snapshot()

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
                    self._stop_active_action(
                        "action watchdog timeout", "failed"
                    )
                    if not self.watchdog_reported:
                        rospy.logwarn(
                            "No action received for %.1f s; chassis stopped.",
                            self.args.watchdog_timeout,
                        )
                        self.watchdog_reported = True

                if self.active_motion is not None:
                    if self.active_controller is not None:
                        odom = self._fresh_odom(now)
                        if odom is None:
                            self._stop_active_action(
                                "odometry missing or stale", "failed"
                            )
                            rospy.logerr(
                                "Odometry on %s is older than %.3f s; "
                                "active chassis action was stopped.",
                                self.args.odom_topic,
                                self.args.odom_stale_timeout,
                            )
                        else:
                            step = self.active_controller.step(odom[0], now)
                            if self.active_action == "MOVE_FORWARD":
                                self.active_progress_value = step.progress
                            else:
                                self.active_progress_value = math.degrees(
                                    step.progress
                                )
                            if step.status == "target_reached":
                                action = self.active_action
                                progress = step.progress
                                self._stop_active_action(
                                    "odometry target reached", "succeeded"
                                )
                                if action == "MOVE_FORWARD":
                                    rospy.loginfo(
                                        "action=%s measured_distance=%.3f m",
                                        action,
                                        progress,
                                    )
                                else:
                                    rospy.loginfo(
                                        "action=%s measured_angle=%.2f deg",
                                        action,
                                        math.degrees(progress),
                                    )
                            elif step.status == "timeout":
                                action = self.active_action
                                self._stop_active_action(
                                    "odometry target timeout", "failed"
                                )
                                rospy.logerr(
                                    "action=%s timed out with %.4f target "
                                    "remaining; chassis stopped",
                                    action,
                                    step.remaining,
                                )
                            else:
                                self._publish_active_velocity(
                                    step.linear_x,
                                    step.angular_z,
                                )
                    elif now >= self.active_until:
                        self.active_progress_value = self.active_target_value
                        self._stop_active_action(
                            "target duration reached", "succeeded"
                        )
                    else:
                        self.active_progress_value = min(
                            self.active_target_value,
                            max(0.0, now - self.active_started_at),
                        )
                        self._publish_active_velocity(
                            self.active_motion.linear_x,
                            self.active_motion.angular_z,
                        )
                elif self.stop_repeats_remaining > 0:
                    self.velocity_publisher.publish(make_twist())
                    self.stop_repeats_remaining -= 1

            wake_generation = self.control_rate.wait(wake_generation)

    def _on_shutdown(self) -> None:
        # Publish several zeros because a single final TCPROS packet may be
        # lost while ROS connections are shutting down.
        self.control_rate.wake()
        with self.lock:
            if self.active_action is not None:
                self._stop_active_action(
                    "action executor shutdown", "failed"
                )
            for _ in range(self.args.stop_publish_count):
                self.velocity_publisher.publish(make_twist())
                time.sleep(0.01)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert VLN English actions to odometry-bounded ROS1 Twist "
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
    parser.add_argument("--action-result-topic")
    parser.add_argument("--cmd-vel-topic")
    parser.add_argument("--odom-topic")
    odom_group = parser.add_mutually_exclusive_group()
    odom_group.add_argument(
        "--use-odom", dest="use_odom", action="store_true"
    )
    odom_group.add_argument(
        "--no-odom", dest="use_odom", action="store_false"
    )
    parser.set_defaults(use_odom=None)
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
    parser.add_argument("--odom-stale-timeout", type=float)
    parser.add_argument("--action-timeout-scale", type=float)
    parser.add_argument("--minimum-action-timeout", type=float)
    parser.add_argument("--distance-tolerance", type=float)
    parser.add_argument("--angle-tolerance-deg", type=float)
    parser.add_argument("--linear-slowdown-distance", type=float)
    parser.add_argument("--angular-slowdown-angle-deg", type=float)
    parser.add_argument("--minimum-linear-speed", type=float)
    parser.add_argument("--minimum-angular-speed", type=float)
    parser.add_argument("--forward-heading-kp", type=float)
    parser.add_argument("--maximum-heading-correction", type=float)
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
    if (
        not isinstance(args.action_result_topic, str)
        or not args.action_result_topic.strip()
    ):
        raise ValueError("action result topic must be a non-empty string.")
    if not isinstance(args.cmd_vel_topic, str) or not args.cmd_vel_topic.strip():
        raise ValueError("cmd_vel topic must be a non-empty string.")
    if (
        args.use_odom
        and (
            not isinstance(args.odom_topic, str)
            or not args.odom_topic.strip()
        )
    ):
        raise ValueError("odom topic must be a non-empty string.")

    positive_values = (
        ("linear speed", args.linear_speed),
        ("forward distance", args.forward_distance),
        ("left angular speed", args.left_angular_speed),
        ("right angular speed", args.right_angular_speed),
        ("left turn angle", args.left_turn_angle_deg),
        ("right turn angle", args.right_turn_angle_deg),
        ("publish rate", args.publish_rate),
        ("odom stale timeout", args.odom_stale_timeout),
        ("action timeout scale", args.action_timeout_scale),
        ("minimum action timeout", args.minimum_action_timeout),
        ("distance tolerance", args.distance_tolerance),
        ("angle tolerance", args.angle_tolerance_deg),
        ("linear slowdown distance", args.linear_slowdown_distance),
        ("angular slowdown angle", args.angular_slowdown_angle_deg),
        ("minimum linear speed", args.minimum_linear_speed),
        ("minimum angular speed", args.minimum_angular_speed),
        ("forward heading kp", args.forward_heading_kp),
        (
            "maximum heading correction",
            args.maximum_heading_correction,
        ),
    )
    for name, value in positive_values:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value <= 0.0
        ):
            raise ValueError("{} must be a positive finite number.".format(name))

    if args.minimum_linear_speed > args.linear_speed:
        raise ValueError(
            "minimum linear speed must not exceed MOVE_FORWARD speed."
        )
    if args.minimum_angular_speed > min(
        args.left_angular_speed, args.right_angular_speed
    ):
        raise ValueError(
            "minimum angular speed must not exceed either turn speed."
        )

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
