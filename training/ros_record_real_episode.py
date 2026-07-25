#!/usr/bin/env python3

"""Record one real-robot RGB-D/action episode for CMA fine-tuning.

Every expert action message captures the newest synchronized RGB-D pair that
arrived before that action. The expert action topic must come from a human or
trusted controller; never use the model prediction topic as supervision.
"""

import argparse
import glob
import json
import os
import queue
import re
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np


REAL_ROOT = Path(__file__).resolve().parents[1]
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
import rospy  # noqa: E402
from cv_bridge import CvBridge  # noqa: E402
from sensor_msgs.msg import Image  # noqa: E402
from std_msgs.msg import String  # noqa: E402


ACTION_LABELS = [
    "STOP",
    "MOVE_FORWARD",
    "TURN_LEFT",
    "TURN_RIGHT",
]
ACTION_TO_INDEX = {
    label: index for index, label in enumerate(ACTION_LABELS)
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


def normalize_action(action_text):
    action = action_text.strip().upper().replace(" ", "_")
    aliases = {
        "FORWARD": "MOVE_FORWARD",
        "LEFT": "TURN_LEFT",
        "RIGHT": "TURN_RIGHT",
    }
    action = aliases.get(action, action)
    if action not in ACTION_TO_INDEX:
        raise ValueError(
            "Unsupported expert action '{}'; expected one of {}.".format(
                action_text, ACTION_LABELS
            )
        )
    return action


class RealEpisodeRecorder:
    def __init__(self, args):
        self.args = args
        self.bridge = CvBridge()
        self.latest_pair_lock = threading.Lock()
        self.latest_pair = None
        self.latest_pair_serial = 0
        self.last_recorded_pair_serial = -1
        self.write_queue = queue.Queue(maxsize=args.writer_queue_size)
        self.writer_error = None
        self.samples = []
        self.closed = False

        output_root = Path(args.output_dir).expanduser()
        if not output_root.is_absolute():
            output_root = REAL_ROOT / output_root
        self.episode_dir = (
            output_root.resolve() / args.split / args.episode_id
        )
        self.episode_dir.mkdir(parents=True, exist_ok=False)
        self.rgb_dir = self.episode_dir / "rgb"
        self.depth_dir = self.episode_dir / "depth"
        self.rgb_dir.mkdir()
        self.depth_dir.mkdir()
        self.manifest_path = self.episode_dir / "episode.json"

        self.manifest = {
            "format_version": 1,
            "source": "real_robot_ros1",
            "status": "recording",
            "episode_id": args.episode_id,
            "split": args.split,
            "instruction": args.instruction.strip(),
            "action_labels": ACTION_LABELS,
            "rgb_storage_encoding": "bgr8_jpeg",
            "depth_storage_encoding": "32FC1_meters_npy",
            "created_at": datetime.now().astimezone().isoformat(),
            "topics": {
                "rgb": args.rgb_topic,
                "depth": args.depth_topic,
                "expert_action": args.expert_action_topic,
            },
            "sync_slop_seconds": args.sync_slop,
            "samples": self.samples,
        }
        atomic_write_json(self.manifest_path, self.manifest)

        self.writer = threading.Thread(
            target=self._writer_loop,
            name="real_episode_writer",
            daemon=True,
        )
        self.writer.start()

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
        self.action_subscriber = rospy.Subscriber(
            args.expert_action_topic,
            String,
            self._expert_action_callback,
            queue_size=10,
        )

    def _synchronized_callback(self, rgb_message, depth_message):
        with self.latest_pair_lock:
            self.latest_pair_serial += 1
            self.latest_pair = (
                self.latest_pair_serial,
                rgb_message,
                depth_message,
                time.monotonic(),
            )

    def _expert_action_callback(self, action_message):
        try:
            action = normalize_action(action_message.data)
        except ValueError as error:
            rospy.logerr("%s", error)
            return

        with self.latest_pair_lock:
            pair = self.latest_pair
            if pair is None:
                rospy.logwarn(
                    "Ignoring expert action %s: no synchronized RGB-D pair "
                    "has arrived yet.",
                    action,
                )
                return
            pair_serial, rgb_message, depth_message, arrival_time = pair
            pair_age = time.monotonic() - arrival_time
            if pair_age > self.args.max_pair_age:
                rospy.logwarn(
                    "Ignoring expert action %s: newest RGB-D pair is %.0f ms "
                    "old (limit %.0f ms).",
                    action,
                    1000.0 * pair_age,
                    1000.0 * self.args.max_pair_age,
                )
                return
            if (
                not self.args.allow_reuse_frame
                and pair_serial == self.last_recorded_pair_serial
            ):
                rospy.logwarn(
                    "Ignoring expert action %s: synchronized frame %d was "
                    "already recorded.",
                    action,
                    pair_serial,
                )
                return
            self.last_recorded_pair_serial = pair_serial

        record = {
            "pair_serial": pair_serial,
            "rgb_message": rgb_message,
            "depth_message": depth_message,
            "action": action,
            "action_stamp": rospy.Time.now().to_sec(),
        }
        try:
            self.write_queue.put_nowait(record)
        except queue.Full:
            rospy.logerr(
                "Recorder writer queue is full; expert action %s was not "
                "saved. Stop the robot and reduce action frequency.",
                action,
            )
            return

        if action == "STOP" and self.args.stop_ends_episode:
            rospy.signal_shutdown("expert STOP completed the episode")

    def _writer_loop(self):
        while True:
            record = self.write_queue.get()
            if record is None:
                return
            try:
                self._write_record(record)
            except Exception as error:
                self.writer_error = error
                rospy.logerr("Failed to save real episode sample: %s", error)
                rospy.signal_shutdown("episode writer failure")
                return

    def _write_record(self, record):
        rgb_message = record["rgb_message"]
        depth_message = record["depth_message"]
        if depth_message.encoding.upper() != "32FC1":
            raise ValueError(
                "Expected depth topic encoding 32FC1 meters, got '{}'. "
                "Record /camera/depth_registered/image_filled."
                .format(depth_message.encoding)
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
                "Registered RGB/depth dimensions differ: {} vs {}."
                .format(bgr.shape[:2], depth_m.shape)
            )

        valid = np.isfinite(depth_m) & (depth_m > 0.0)
        depth_m = depth_m.copy()
        depth_m[~valid] = 0.0
        invalid_fraction = float(1.0 - valid.mean())

        sample_index = len(self.samples)
        rgb_relative = "rgb/{:06d}.jpg".format(sample_index)
        depth_relative = "depth/{:06d}.npy".format(sample_index)
        rgb_path = self.episode_dir / rgb_relative
        depth_path = self.episode_dir / depth_relative

        rgb_temporary = rgb_path.with_name(
            rgb_path.stem + ".tmp" + rgb_path.suffix
        )
        if not cv2.imwrite(
            str(rgb_temporary),
            bgr,
            [cv2.IMWRITE_JPEG_QUALITY, self.args.jpeg_quality],
        ):
            raise IOError("cv2.imwrite failed for {}".format(rgb_temporary))
        os.replace(str(rgb_temporary), str(rgb_path))

        depth_temporary = depth_path.with_name(depth_path.name + ".tmp")
        with depth_temporary.open("wb") as depth_file:
            np.save(depth_file, depth_m, allow_pickle=False)
            depth_file.flush()
            os.fsync(depth_file.fileno())
        os.replace(str(depth_temporary), str(depth_path))

        rgb_stamp = rgb_message.header.stamp.to_sec()
        depth_stamp = depth_message.header.stamp.to_sec()
        sample = {
            "index": sample_index,
            "rgb": rgb_relative,
            "depth": depth_relative,
            "action": record["action"],
            "action_index": ACTION_TO_INDEX[record["action"]],
            "rgb_stamp": rgb_stamp,
            "depth_stamp": depth_stamp,
            "action_stamp": record["action_stamp"],
            "sync_delta_ms": 1000.0 * abs(rgb_stamp - depth_stamp),
            "invalid_depth_fraction": invalid_fraction,
        }
        self.samples.append(sample)
        atomic_write_json(self.manifest_path, self.manifest)
        rospy.loginfo(
            "Recorded sample=%d action=%s sync_delta_ms=%.2f "
            "invalid_depth=%.2f%%",
            sample_index,
            record["action"],
            sample["sync_delta_ms"],
            100.0 * invalid_fraction,
        )

    def run(self):
        rospy.loginfo(
            "Recording real episode %s (%s) to %s; instruction=%s; "
            "expert_action_topic=%s",
            self.args.episode_id,
            self.args.split,
            self.episode_dir,
            self.args.instruction,
            self.args.expert_action_topic,
        )
        rospy.spin()
        self.close()
        return 1 if self.writer_error is not None else 0

    def close(self):
        if self.closed:
            return
        self.closed = True
        while self.writer.is_alive():
            try:
                self.write_queue.put(None, timeout=0.2)
                break
            except queue.Full:
                continue
        self.writer.join()
        self.manifest["status"] = (
            "error" if self.writer_error is not None else "complete"
        )
        self.manifest["ended_at"] = (
            datetime.now().astimezone().isoformat()
        )
        self.manifest["sample_count"] = len(self.samples)
        if self.writer_error is not None:
            self.manifest["error"] = str(self.writer_error)
        atomic_write_json(self.manifest_path, self.manifest)
        rospy.loginfo(
            "Episode closed with %d samples: %s",
            len(self.samples),
            self.episode_dir,
        )


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Record synchronized real-robot RGB-D observations whenever a "
            "human/trusted controller publishes an expert action."
        )
    )
    parser.add_argument("--instruction", required=True)
    parser.add_argument(
        "--episode-id",
        default=datetime.now().strftime("episode_%Y%m%d_%H%M%S"),
    )
    parser.add_argument(
        "--split", choices=["train", "val"], default="train"
    )
    parser.add_argument(
        "--output-dir", default="training/data/real_episodes"
    )
    parser.add_argument(
        "--rgb-topic", default="/camera/rgb/image_color"
    )
    parser.add_argument(
        "--depth-topic",
        default="/camera/depth_registered/image_filled",
    )
    parser.add_argument(
        "--expert-action-topic", default="/vln/expert_action"
    )
    parser.add_argument("--sync-queue-size", type=int, default=20)
    parser.add_argument("--sync-slop", type=float, default=0.10)
    parser.add_argument("--max-pair-age", type=float, default=0.50)
    parser.add_argument("--writer-queue-size", type=int, default=8)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument(
        "--allow-reuse-frame", action="store_true"
    )
    parser.add_argument(
        "--no-stop-ends-episode",
        dest="stop_ends_episode",
        action="store_false",
    )
    parser.set_defaults(stop_ends_episode=True)
    parser.add_argument(
        "--subscriber-buffer-bytes", type=int, default=2 ** 24
    )
    parser.add_argument(
        "--node-name", default="ros_record_real_vln_episode"
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
    if args.writer_queue_size <= 0:
        raise ValueError("--writer-queue-size must be positive.")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be in [1, 100].")
    if args.expert_action_topic == "/vln/action":
        raise ValueError(
            "Refusing to use /vln/action as expert supervision. Publish "
            "human/trusted-controller labels on a separate topic such as "
            "/vln/expert_action."
        )


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        validate_args(args)
    except ValueError as error:
        parser.error(str(error))

    rospy.init_node(args.node_name, anonymous=False)
    try:
        recorder = RealEpisodeRecorder(args)
    except FileExistsError:
        rospy.logfatal(
            "Episode directory already exists; choose another --episode-id."
        )
        return 1
    except Exception as error:
        rospy.logfatal("Failed to initialize real episode recorder: %s", error)
        return 1
    return recorder.run()


if __name__ == "__main__":
    sys.exit(main())
