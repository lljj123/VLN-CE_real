#!/usr/bin/env python3

"""Interactively collect one real-robot expert RGB-D/action episode.

For every expert command this node performs the following transaction:

1. Freeze the latest synchronized RGB-D pair captured before the action.
2. Save the source-resolution RGB and metric depth plus the expert label.
3. Execute the configured discrete action on ``cmd_vel``.
4. Stop the chassis before accepting the next expert command.

The resulting ``episode.json`` is directly compatible with
``training/real_dataset.py``.  Motion is open-loop and must be calibrated for
the real chassis before data collection.
"""

import argparse
import glob
import json
import math
import os
import re
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np


REAL_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REAL_ROOT / "training" / "data" / "real_episodes"
DEFAULT_MOTION_CONFIG = REAL_ROOT / "config" / "action_to_cmd_vel.json"
if str(REAL_ROOT) not in sys.path:
    sys.path.insert(0, str(REAL_ROOT))


def add_ros_python_paths():
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

import cv2  # noqa: E402
import message_filters  # noqa: E402
import rosgraph  # noqa: E402
import rospy  # noqa: E402
from cv_bridge import CvBridge  # noqa: E402
from geometry_msgs.msg import Twist  # noqa: E402
from sensor_msgs.msg import CameraInfo, Image  # noqa: E402
from std_msgs.msg import String  # noqa: E402

from scripts.ros_action_to_cmd_vel import (  # noqa: E402
    MotionSettings,
    action_to_motion,
    make_twist,
)


ACTION_LABELS = [
    "STOP",
    "MOVE_FORWARD",
    "TURN_LEFT",
    "TURN_RIGHT",
]
ACTION_TO_INDEX = {
    action: index for index, action in enumerate(ACTION_LABELS)
}
COMMANDS = {
    "w": "MOVE_FORWARD",
    "forward": "MOVE_FORWARD",
    "move_forward": "MOVE_FORWARD",
    "a": "TURN_LEFT",
    "left": "TURN_LEFT",
    "turn_left": "TURN_LEFT",
    "d": "TURN_RIGHT",
    "right": "TURN_RIGHT",
    "turn_right": "TURN_RIGHT",
    "s": "STOP",
    "stop": "STOP",
}
SAFE_EPISODE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


def atomic_write_json(path, payload):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
        output_file.flush()
        os.fsync(output_file.fileno())
    os.replace(str(temporary), str(path))


def resolve_real_path(path_text):
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = REAL_ROOT / path
    return path.resolve()


def _mapping_section(mapping, name):
    section = mapping.get(name)
    if not isinstance(section, dict):
        raise ValueError(
            "Motion configuration section {!r} must be an object.".format(
                name
            )
        )
    return section


def _positive_float(mapping, key, description):
    value = mapping.get(key)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value <= 0.0
    ):
        raise ValueError(
            "{} must be a positive finite number.".format(description)
        )
    return float(value)


def load_motion_configuration(path_text):
    config_path = resolve_real_path(path_text)
    try:
        with config_path.open("r", encoding="utf-8") as input_file:
            config = json.load(input_file)
    except OSError as error:
        raise ValueError(
            "Cannot read motion configuration {}: {}".format(
                config_path, error
            )
        )
    except ValueError as error:
        raise ValueError(
            "Invalid JSON in motion configuration {}: {}".format(
                config_path, error
            )
        )
    if not isinstance(config, dict):
        raise ValueError("Motion configuration root must be an object.")

    control = _mapping_section(config, "control")
    actions = _mapping_section(config, "actions")
    forward = _mapping_section(actions, "MOVE_FORWARD")
    left = _mapping_section(actions, "TURN_LEFT")
    right = _mapping_section(actions, "TURN_RIGHT")

    publish_rate = _positive_float(
        control, "publish_rate_hz", "publish_rate_hz"
    )
    stop_publish_count = control.get("stop_publish_count", 3)
    if (
        isinstance(stop_publish_count, bool)
        or not isinstance(stop_publish_count, int)
        or stop_publish_count <= 0
    ):
        raise ValueError("stop_publish_count must be a positive integer.")

    settings = MotionSettings(
        forward_linear_speed=_positive_float(
            forward, "linear_speed_mps", "MOVE_FORWARD linear speed"
        ),
        forward_distance=_positive_float(
            forward, "distance_m", "MOVE_FORWARD distance"
        ),
        left_angular_speed=_positive_float(
            left, "angular_speed_radps", "TURN_LEFT angular speed"
        ),
        left_angle_deg=_positive_float(
            left, "angle_deg", "TURN_LEFT angle"
        ),
        right_angular_speed=_positive_float(
            right, "angular_speed_radps", "TURN_RIGHT angular speed"
        ),
        right_angle_deg=_positive_float(
            right, "angle_deg", "TURN_RIGHT angle"
        ),
    )
    return config_path, settings, publish_rate, stop_publish_count


def camera_info_to_dict(message):
    return {
        "width": int(message.width),
        "height": int(message.height),
        "distortion_model": message.distortion_model,
        "D": list(message.D),
        "K": list(message.K),
        "R": list(message.R),
        "P": list(message.P),
        "binning_x": int(message.binning_x),
        "binning_y": int(message.binning_y),
        "roi": {
            "x_offset": int(message.roi.x_offset),
            "y_offset": int(message.roi.y_offset),
            "height": int(message.roi.height),
            "width": int(message.roi.width),
            "do_rectify": bool(message.roi.do_rectify),
        },
    }


def current_topic_publishers(topic):
    resolved_topic = rospy.resolve_name(topic)
    master = rosgraph.Master(rospy.get_name())
    publishers, _, _ = master.getSystemState()
    for published_topic, nodes in publishers:
        if published_topic == resolved_topic:
            return sorted(nodes)
    return []


class ExpertDriveCollector:
    def __init__(self, args):
        self.args = args
        self.bridge = CvBridge()
        self.pair_lock = threading.Lock()
        self.manifest_lock = threading.RLock()
        self.latest_pair = None
        self.pair_serial = 0
        self.last_recorded_pair_serial = -1
        self.samples = []
        self.finalized = False
        self.shutdown_stop_sent = False

        (
            self.motion_config_path,
            self.motion_settings,
            self.publish_rate,
            self.stop_publish_count,
        ) = load_motion_configuration(args.motion_config)

        other_publishers = current_topic_publishers(args.cmd_vel_topic)
        if other_publishers and not args.allow_other_cmd_vel_publishers:
            raise RuntimeError(
                "Refusing to collect while {} already has publishers: {}. "
                "Stop inference/action-control nodes, use a dedicated mux "
                "input topic, or explicitly pass "
                "--allow-other-cmd-vel-publishers.".format(
                    rospy.resolve_name(args.cmd_vel_topic),
                    ", ".join(other_publishers),
                )
            )

        output_root = resolve_real_path(args.output_dir)
        self.episode_dir = output_root / args.split / args.episode_id
        self.episode_dir.mkdir(parents=True, exist_ok=False)
        self.rgb_dir = self.episode_dir / "rgb"
        self.depth_dir = self.episode_dir / "depth"
        self.rgb_dir.mkdir()
        self.depth_dir.mkdir()
        self.manifest_path = self.episode_dir / "episode.json"
        self.instruction_path = self.episode_dir / "instruction.txt"
        with self.instruction_path.open("x", encoding="utf-8") as output:
            output.write(args.instruction.strip() + "\n")

        self.manifest = {
            "format_version": 1,
            "source": "real_robot_ros1",
            "status": "recording",
            "episode_id": args.episode_id,
            "split": args.split,
            "instruction": args.instruction.strip(),
            "instruction_file": "instruction.txt",
            "action_labels": ACTION_LABELS,
            "rgb_storage_encoding": "bgr8_png",
            "depth_storage_encoding": "32FC1_meters_npy",
            "created_at": datetime.now().astimezone().isoformat(),
            "topics": {
                "rgb": args.rgb_topic,
                "depth": args.depth_topic,
                "rgb_camera_info": args.rgb_camera_info_topic,
                "depth_camera_info": args.depth_camera_info_topic,
                "expert_action": args.expert_action_topic,
                "cmd_vel": args.cmd_vel_topic,
            },
            "synchronization": {
                "method": "ApproximateTimeSynchronizer",
                "slop_seconds": args.sync_slop,
                "max_pair_age_seconds": args.max_pair_age,
            },
            "camera_info": {
                "rgb": None,
                "depth": None,
            },
            "action_execution": {
                "mode": "dry_run" if args.dry_run else "open_loop_cmd_vel",
                "motion_config": str(self.motion_config_path),
                "publish_rate_hz": self.publish_rate,
                "settle_time_seconds": args.settle_time,
                "MOVE_FORWARD": {
                    "linear_speed_mps": (
                        self.motion_settings.forward_linear_speed
                    ),
                    "distance_m": self.motion_settings.forward_distance,
                },
                "TURN_LEFT": {
                    "angular_speed_radps": (
                        self.motion_settings.left_angular_speed
                    ),
                    "angle_deg": self.motion_settings.left_angle_deg,
                },
                "TURN_RIGHT": {
                    "angular_speed_radps": (
                        self.motion_settings.right_angular_speed
                    ),
                    "angle_deg": self.motion_settings.right_angle_deg,
                },
            },
            "samples": self.samples,
        }
        self._write_manifest()

        self.velocity_publisher = rospy.Publisher(
            args.cmd_vel_topic, Twist, queue_size=1
        )
        self.expert_action_publisher = rospy.Publisher(
            args.expert_action_topic, String, queue_size=10
        )

        self.rgb_subscriber = message_filters.Subscriber(
            args.rgb_topic,
            Image,
            queue_size=1,
            buff_size=args.subscriber_buffer_bytes,
        )
        self.depth_subscriber = message_filters.Subscriber(
            args.depth_topic,
            Image,
            queue_size=1,
            buff_size=args.subscriber_buffer_bytes,
        )
        self.synchronizer = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_subscriber, self.depth_subscriber],
            queue_size=args.sync_queue_size,
            slop=args.sync_slop,
        )
        self.synchronizer.registerCallback(self._synchronized_callback)

        self.rgb_info_subscriber = None
        self.depth_info_subscriber = None
        if args.rgb_camera_info_topic:
            self.rgb_info_subscriber = rospy.Subscriber(
                args.rgb_camera_info_topic,
                CameraInfo,
                self._rgb_camera_info_callback,
                queue_size=1,
            )
        if args.depth_camera_info_topic:
            self.depth_info_subscriber = rospy.Subscriber(
                args.depth_camera_info_topic,
                CameraInfo,
                self._depth_camera_info_callback,
                queue_size=1,
            )
        rospy.on_shutdown(self._emergency_stop)

    def _write_manifest(self):
        with self.manifest_lock:
            atomic_write_json(self.manifest_path, self.manifest)

    def _synchronized_callback(self, rgb_message, depth_message):
        with self.pair_lock:
            self.pair_serial += 1
            self.latest_pair = (
                self.pair_serial,
                rgb_message,
                depth_message,
                time.monotonic(),
            )

    def _set_camera_info_once(self, sensor, message):
        with self.manifest_lock:
            if self.manifest["camera_info"][sensor] is not None:
                return
            self.manifest["camera_info"][sensor] = camera_info_to_dict(
                message
            )
            self._write_manifest()

    def _rgb_camera_info_callback(self, message):
        self._set_camera_info_once("rgb", message)

    def _depth_camera_info_callback(self, message):
        self._set_camera_info_once("depth", message)

    def wait_until_ready(self):
        deadline = time.monotonic() + self.args.startup_timeout
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            with self.pair_lock:
                pair_ready = self.latest_pair is not None
            drive_ready = (
                self.args.dry_run
                or self.velocity_publisher.get_num_connections() > 0
            )
            if pair_ready and drive_ready:
                return
            time.sleep(0.1)

        missing = []
        with self.pair_lock:
            if self.latest_pair is None:
                missing.append("a synchronized RGB-D pair")
        if (
            not self.args.dry_run
            and self.velocity_publisher.get_num_connections() == 0
        ):
            missing.append(
                "a chassis subscriber on {}".format(
                    rospy.resolve_name(self.args.cmd_vel_topic)
                )
            )
        raise RuntimeError(
            "Timed out waiting for {}.".format(" and ".join(missing))
        )

    def _newest_pair_for_action(self, action):
        with self.pair_lock:
            pair = self.latest_pair
            if pair is None:
                raise RuntimeError("No synchronized RGB-D pair is available.")
            pair_serial, rgb_message, depth_message, arrival_time = pair
            pair_age = time.monotonic() - arrival_time
            if pair_age > self.args.max_pair_age:
                raise RuntimeError(
                    "Newest RGB-D pair is {:.0f} ms old; limit is {:.0f} "
                    "ms. Action {} was not recorded or executed.".format(
                        1000.0 * pair_age,
                        1000.0 * self.args.max_pair_age,
                        action,
                    )
                )
            if pair_serial == self.last_recorded_pair_serial:
                raise RuntimeError(
                    "Newest synchronized RGB-D frame was already recorded; "
                    "wait for a fresh camera frame."
                )
            return pair_serial, rgb_message, depth_message

    def record_before_action(self, action):
        pair_serial, rgb_message, depth_message = (
            self._newest_pair_for_action(action)
        )
        if depth_message.encoding.upper() != "32FC1":
            raise ValueError(
                "Expected 32FC1 depth in meters from the filled-depth topic, "
                "got {!r}.".format(depth_message.encoding)
            )

        bgr = self.bridge.imgmsg_to_cv2(
            rgb_message, desired_encoding="bgr8"
        )
        depth_m = self.bridge.imgmsg_to_cv2(
            depth_message, desired_encoding="passthrough"
        )
        bgr = np.ascontiguousarray(bgr, dtype=np.uint8)
        depth_m = np.asarray(depth_m, dtype=np.float32)
        if depth_m.ndim == 3 and depth_m.shape[2] == 1:
            depth_m = depth_m[:, :, 0]
        if depth_m.ndim != 2:
            raise ValueError(
                "Expected depth shape [H, W], got {}.".format(depth_m.shape)
            )
        if bgr.shape[:2] != depth_m.shape:
            raise ValueError(
                "Registered RGB/depth dimensions differ: {} vs {}.".format(
                    bgr.shape[:2], depth_m.shape
                )
            )

        valid = np.isfinite(depth_m) & (depth_m > 0.0)
        depth_m = depth_m.copy()
        depth_m[~valid] = 0.0
        invalid_fraction = float(1.0 - valid.mean())

        sample_index = len(self.samples)
        rgb_relative = "rgb/{:06d}.png".format(sample_index)
        depth_relative = "depth/{:06d}.npy".format(sample_index)
        rgb_path = self.episode_dir / rgb_relative
        depth_path = self.episode_dir / depth_relative

        rgb_temporary = rgb_path.with_name(
            rgb_path.stem + ".tmp" + rgb_path.suffix
        )
        if not cv2.imwrite(
            str(rgb_temporary),
            bgr,
            [cv2.IMWRITE_PNG_COMPRESSION, self.args.png_compression],
        ):
            raise IOError("Failed to save {}.".format(rgb_temporary))
        os.replace(str(rgb_temporary), str(rgb_path))

        depth_temporary = depth_path.with_name(depth_path.name + ".tmp")
        with depth_temporary.open("wb") as depth_file:
            np.save(depth_file, depth_m, allow_pickle=False)
            depth_file.flush()
            os.fsync(depth_file.fileno())
        os.replace(str(depth_temporary), str(depth_path))

        motion = action_to_motion(action, self.motion_settings)
        rgb_stamp = rgb_message.header.stamp.to_sec()
        depth_stamp = depth_message.header.stamp.to_sec()
        sample = {
            "index": sample_index,
            "rgb": rgb_relative,
            "depth": depth_relative,
            "action": action,
            "action_index": ACTION_TO_INDEX[action],
            "rgb_stamp": rgb_stamp,
            "depth_stamp": depth_stamp,
            "action_stamp": rospy.Time.now().to_sec(),
            "sync_delta_ms": 1000.0 * abs(rgb_stamp - depth_stamp),
            "invalid_depth_fraction": invalid_fraction,
            "source_rgb_shape": list(bgr.shape),
            "source_depth_shape": list(depth_m.shape),
            "source_rgb_encoding": rgb_message.encoding,
            "source_depth_encoding": depth_message.encoding,
            "planned_motion": {
                "linear_x_mps": motion.linear_x,
                "angular_z_radps": motion.angular_z,
                "duration_seconds": motion.duration,
            },
            "execution": {
                "status": "recorded_not_started",
            },
        }
        with self.manifest_lock:
            self.samples.append(sample)
            self._write_manifest()
        with self.pair_lock:
            self.last_recorded_pair_serial = pair_serial

        rospy.loginfo(
            "Recorded sample=%d action=%s rgb=%dx%d depth=%dx%d "
            "sync_delta_ms=%.2f invalid_depth=%.2f%%",
            sample_index,
            action,
            bgr.shape[1],
            bgr.shape[0],
            depth_m.shape[1],
            depth_m.shape[0],
            sample["sync_delta_ms"],
            100.0 * invalid_fraction,
        )
        return sample, motion

    def _update_execution(self, sample, status, **fields):
        with self.manifest_lock:
            sample["execution"]["status"] = status
            sample["execution"].update(fields)
            self._write_manifest()

    def _publish_stop(self):
        for _ in range(self.stop_publish_count):
            self.velocity_publisher.publish(make_twist())
            time.sleep(0.01)

    def execute_action(self, sample, action, motion):
        self.expert_action_publisher.publish(String(data=action))
        if self.args.dry_run:
            self._update_execution(
                sample,
                "dry_run",
                started_at=datetime.now().astimezone().isoformat(),
                ended_at=datetime.now().astimezone().isoformat(),
            )
            return

        if self.velocity_publisher.get_num_connections() == 0:
            self._update_execution(sample, "not_executed_no_subscriber")
            raise RuntimeError(
                "No chassis subscriber is connected to {}; data was saved "
                "but the action was not executed.".format(
                    rospy.resolve_name(self.args.cmd_vel_topic)
                )
            )

        started_at = datetime.now().astimezone().isoformat()
        self._update_execution(sample, "executing", started_at=started_at)
        if action == "STOP":
            self._publish_stop()
            self._update_execution(
                sample,
                "finished",
                ended_at=datetime.now().astimezone().isoformat(),
            )
            return

        self._publish_stop()
        command = make_twist(motion.linear_x, motion.angular_z)
        deadline = time.monotonic() + motion.duration
        rate = rospy.Rate(self.publish_rate)
        interrupted = False
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            self.velocity_publisher.publish(command)
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                interrupted = True
                break

        self._publish_stop()
        settle_deadline = time.monotonic() + self.args.settle_time
        while (
            not rospy.is_shutdown()
            and time.monotonic() < settle_deadline
        ):
            self.velocity_publisher.publish(make_twist())
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                interrupted = True
                break

        status = "interrupted" if interrupted else "finished"
        self._update_execution(
            sample,
            status,
            ended_at=datetime.now().astimezone().isoformat(),
        )
        if interrupted:
            raise RuntimeError("ROS shutdown interrupted chassis motion.")
        rospy.loginfo("Executed action=%s; chassis stopped", action)

    def handle_action(self, action):
        if (
            action != "STOP"
            and not self.args.dry_run
            and self.velocity_publisher.get_num_connections() == 0
        ):
            raise RuntimeError(
                "No chassis subscriber is connected to {}; action was not "
                "recorded or executed.".format(
                    rospy.resolve_name(self.args.cmd_vel_topic)
                )
            )
        sample, motion = self.record_before_action(action)
        self.execute_action(sample, action, motion)

    def finalize(self, status):
        if self.finalized:
            return
        self._publish_stop()
        with self.manifest_lock:
            self.manifest["status"] = status
            self.manifest["sample_count"] = len(self.samples)
            self.manifest["ended_at"] = (
                datetime.now().astimezone().isoformat()
            )
            self._write_manifest()
        self.finalized = True
        rospy.loginfo(
            "Episode %s closed status=%s samples=%d path=%s",
            self.args.episode_id,
            status,
            len(self.samples),
            self.episode_dir,
        )

    def _emergency_stop(self):
        if self.shutdown_stop_sent:
            return
        self.shutdown_stop_sent = True
        try:
            self._publish_stop()
        except Exception:
            pass

    def run(self):
        rospy.loginfo(
            "Expert collector waiting for RGB=%s Depth=%s cmd_vel=%s",
            self.args.rgb_topic,
            self.args.depth_topic,
            self.args.cmd_vel_topic,
        )
        try:
            self.wait_until_ready()
        except Exception:
            self.finalize("aborted")
            raise

        print("\nEnglish instruction:\n{}\n".format(self.args.instruction))
        print("Expert commands (press Enter after each command):")
        print("  w = MOVE_FORWARD")
        print("  a = TURN_LEFT")
        print("  d = TURN_RIGHT")
        print("  s = STOP, record final sample and complete episode")
        print("  q = emergency stop and abort episode (excluded from training)")
        if self.args.dry_run:
            print("  DRY RUN: actions will be recorded but chassis will not move")

        try:
            while not rospy.is_shutdown():
                command = input("expert> ").strip().lower().replace("-", "_")
                if not command:
                    continue
                if command in ("q", "quit", "abort"):
                    self.finalize("aborted")
                    return 0
                action = COMMANDS.get(command)
                if action is None:
                    print("Unknown command. Use w, a, d, s or q.")
                    continue
                sample_count_before = len(self.samples)
                try:
                    self.handle_action(action)
                except Exception as error:
                    rospy.logerr("%s", error)
                    print("Action failed safely; chassis STOP was sent.")
                    self._publish_stop()
                    if len(self.samples) > sample_count_before:
                        # The observation/label was already committed but its
                        # physical action did not complete. Continuing would
                        # make every later transition in this episode false.
                        self.finalize("error")
                        return 1
                    continue
                if action == "STOP":
                    self.finalize("complete")
                    return 0
        except (EOFError, KeyboardInterrupt):
            self.finalize("aborted")
            return 130

        self.finalize("aborted")
        return 1


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Interactively record synchronized real RGB-D expert samples "
            "and execute each expert action on the chassis."
        )
    )
    parser.add_argument("--instruction", required=True)
    parser.add_argument(
        "--episode-id",
        default=datetime.now().strftime("episode_%Y%m%d_%H%M%S"),
    )
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--rgb-topic", default="/camera/rgb/image_color")
    parser.add_argument(
        "--depth-topic",
        default="/camera/depth_registered/image_filled",
    )
    parser.add_argument(
        "--rgb-camera-info-topic", default="/camera/rgb/camera_info"
    )
    parser.add_argument(
        "--depth-camera-info-topic",
        default="/camera/depth_registered/camera_info",
    )
    parser.add_argument(
        "--expert-action-topic", default="/vln/expert_action"
    )
    parser.add_argument("--cmd-vel-topic", default="/cmd_vel")
    parser.add_argument(
        "--motion-config", default=str(DEFAULT_MOTION_CONFIG)
    )
    parser.add_argument("--sync-queue-size", type=int, default=20)
    parser.add_argument("--sync-slop", type=float, default=0.10)
    parser.add_argument("--max-pair-age", type=float, default=0.50)
    parser.add_argument("--startup-timeout", type=float, default=30.0)
    parser.add_argument("--settle-time", type=float, default=0.20)
    parser.add_argument("--png-compression", type=int, default=3)
    parser.add_argument(
        "--subscriber-buffer-bytes", type=int, default=2 ** 24
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Record actions without publishing non-zero Twist commands.",
    )
    parser.add_argument(
        "--allow-other-cmd-vel-publishers",
        action="store_true",
        help=(
            "Allow another node to publish the same cmd_vel topic. Unsafe "
            "without a correctly configured command multiplexer."
        ),
    )
    parser.add_argument(
        "--node-name", default="ros_expert_drive_collector"
    )
    return parser


def validate_args(args):
    if not args.instruction.strip():
        raise ValueError("--instruction must not be empty.")
    if not SAFE_EPISODE_ID.match(args.episode_id):
        raise ValueError(
            "--episode-id may contain only letters, digits, '.', '_' and '-'."
        )
    if args.sync_queue_size <= 0:
        raise ValueError("--sync-queue-size must be positive.")
    if args.sync_slop < 0.0:
        raise ValueError("--sync-slop must be >= 0.")
    if args.max_pair_age <= 0.0:
        raise ValueError("--max-pair-age must be positive.")
    if args.startup_timeout <= 0.0:
        raise ValueError("--startup-timeout must be positive.")
    if args.settle_time < 0.0:
        raise ValueError("--settle-time must be >= 0.")
    if not 0 <= args.png_compression <= 9:
        raise ValueError("--png-compression must be in [0, 9].")
    if args.subscriber_buffer_bytes <= 0:
        raise ValueError("--subscriber-buffer-bytes must be positive.")


def main():
    parser = build_parser()
    args = parser.parse_args(rospy.myargv(argv=sys.argv)[1:])
    try:
        validate_args(args)
    except ValueError as error:
        parser.error(str(error))

    rospy.init_node(args.node_name, anonymous=False)
    try:
        collector = ExpertDriveCollector(args)
    except FileExistsError:
        rospy.logfatal(
            "Episode directory already exists; choose another --episode-id."
        )
        return 1
    except Exception as error:
        rospy.logfatal("Cannot initialize expert collector: %s", error)
        return 1

    try:
        return collector.run()
    except Exception as error:
        rospy.logfatal("Expert collection failed: %s", error)
        collector.finalize("error")
        return 1


if __name__ == "__main__":
    sys.exit(main())
