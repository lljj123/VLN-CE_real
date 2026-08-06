"""ROS-independent planar odometry control helpers for discrete actions."""

import math
from typing import NamedTuple


class PlanarPose(NamedTuple):
    x: float
    y: float
    yaw: float


class OdomControlSettings(NamedTuple):
    distance_tolerance_m: float
    angle_tolerance_rad: float
    linear_slowdown_distance_m: float
    angular_slowdown_angle_rad: float
    minimum_linear_speed_mps: float
    minimum_angular_speed_radps: float
    forward_heading_kp: float
    maximum_heading_correction_radps: float
    action_timeout_scale: float
    minimum_action_timeout_s: float


class ControlStep(NamedTuple):
    status: str
    linear_x: float
    angular_z: float
    progress: float
    remaining: float


def normalize_angle(angle):
    """Normalize an angle to [-pi, pi)."""

    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def quaternion_to_yaw(x, y, z, w):
    """Return REP-103 yaw from a geometry_msgs-style quaternion."""

    sin_yaw = 2.0 * (float(w) * float(z) + float(x) * float(y))
    cos_yaw = 1.0 - 2.0 * (
        float(y) * float(y) + float(z) * float(z)
    )
    return math.atan2(sin_yaw, cos_yaw)


def scaled_speed(nominal_speed, minimum_speed, remaining, slowdown_range):
    """Linearly slow near a target while retaining enough speed to move."""

    if remaining <= 0.0:
        return 0.0
    if slowdown_range <= 0.0 or remaining >= slowdown_range:
        return nominal_speed
    requested = nominal_speed * remaining / slowdown_range
    return min(nominal_speed, max(minimum_speed, requested))


class ClosedLoopMotion:
    """Track one forward or turn action relative to its starting odometry."""

    def __init__(
        self,
        action,
        start_pose,
        nominal_linear_x,
        nominal_angular_z,
        nominal_duration,
        settings,
        start_time,
    ):
        if action not in ("MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"):
            raise ValueError("Unsupported closed-loop action: {}".format(action))
        if nominal_duration <= 0.0:
            raise ValueError("nominal_duration must be positive")
        self.action = action
        self.start_pose = start_pose
        self.nominal_linear_x = float(nominal_linear_x)
        self.nominal_angular_z = float(nominal_angular_z)
        self.nominal_duration = float(nominal_duration)
        self.settings = settings
        self.start_time = float(start_time)
        self.timeout_s = max(
            settings.minimum_action_timeout_s,
            self.nominal_duration * settings.action_timeout_scale,
        )
        if action == "MOVE_FORWARD":
            self.target = abs(self.nominal_linear_x) * self.nominal_duration
            self.tolerance = settings.distance_tolerance_m
        else:
            self.target = abs(self.nominal_angular_z) * self.nominal_duration
            self.tolerance = settings.angle_tolerance_rad

    def progress(self, pose):
        if self.action == "MOVE_FORWARD":
            delta_x = pose.x - self.start_pose.x
            delta_y = pose.y - self.start_pose.y
            # Measure displacement along the starting heading so lateral odom
            # drift cannot falsely satisfy a forward-distance target.
            projected = (
                delta_x * math.cos(self.start_pose.yaw)
                + delta_y * math.sin(self.start_pose.yaw)
            )
            return max(0.0, projected)

        signed_yaw = normalize_angle(pose.yaw - self.start_pose.yaw)
        direction = 1.0 if self.action == "TURN_LEFT" else -1.0
        return max(0.0, direction * signed_yaw)

    def step(self, pose, now):
        progress = self.progress(pose)
        remaining = max(0.0, self.target - progress)
        if remaining <= self.tolerance:
            return ControlStep(
                "target_reached", 0.0, 0.0, progress, remaining
            )
        if float(now) - self.start_time >= self.timeout_s:
            return ControlStep("timeout", 0.0, 0.0, progress, remaining)

        if self.action == "MOVE_FORWARD":
            speed = scaled_speed(
                abs(self.nominal_linear_x),
                self.settings.minimum_linear_speed_mps,
                remaining,
                self.settings.linear_slowdown_distance_m,
            )
            heading_error = normalize_angle(
                self.start_pose.yaw - pose.yaw
            )
            angular_correction = max(
                -self.settings.maximum_heading_correction_radps,
                min(
                    self.settings.maximum_heading_correction_radps,
                    self.settings.forward_heading_kp * heading_error,
                ),
            )
            return ControlStep(
                "running",
                speed,
                angular_correction,
                progress,
                remaining,
            )

        speed = scaled_speed(
            abs(self.nominal_angular_z),
            self.settings.minimum_angular_speed_radps,
            remaining,
            self.settings.angular_slowdown_angle_rad,
        )
        direction = 1.0 if self.action == "TURN_LEFT" else -1.0
        return ControlStep(
            "running", 0.0, direction * speed, progress, remaining
        )
