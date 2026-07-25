#!/usr/bin/env python3

"""Run standalone CMA inference from ROS1 RGB and metric-depth topics.

This node expects the depth stream published by
``ros_depth_hole_filler.py``: ``32FC1`` values measured in meters with the
original camera timestamp preserved.  ROS supplies the observations to a
minimal PyTorch CMA policy.  Habitat, Habitat-Sim, Habitat-Baselines, Gym, and
TorchVision are not imported or required.

The default invocation predicts one action and exits.  Set ``--max-actions 0``
for continuous inference, but only do that when every published action is
actually executed before the next policy step.
"""

import argparse
import glob
import os
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Dict

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def add_ros_python_paths() -> None:
    """Make ROS Noetic's Python modules visible from the Python 3.6 VLN env."""

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
            # Append instead of prepending so the VLN environment keeps its
            # own NumPy, PyTorch, and other binary packages.
            sys.path.append(candidate)


add_ros_python_paths()

import message_filters  # noqa: E402
import rospy  # noqa: E402
import torch  # noqa: E402
from cv_bridge import CvBridge  # noqa: E402
from sensor_msgs.msg import Image  # noqa: E402
from std_msgs.msg import String  # noqa: E402

from vlnce_real.model import (  # noqa: E402
    CMAPolicy,
    DEPTH_SIZE,
    RGB_SIZE,
    STATE_HIDDEN_SIZE,
    batch_observation,
    encode_instruction,
)
from vlnce_real.preprocessing import preprocess_rgbd  # noqa: E402


ACTION_LABELS = {
    0: "STOP",
    1: "MOVE_FORWARD",
    2: "TURN_LEFT",
    3: "TURN_RIGHT",
}


def resolve_repo_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


class CMARunner:
    """CMA checkpoint wrapper that keeps recurrent state across policy steps."""

    def __init__(
        self,
        checkpoint_path: str,
        instruction: str,
        instruction_length: int,
        force_cpu: bool,
        sample_actions: bool,
    ) -> None:
        self.device = torch.device(
            "cpu"
            if force_cpu or not torch.cuda.is_available()
            else "cuda:0"
        )
        self.sample_actions = sample_actions

        checkpoint = torch.load(
            str(resolve_repo_path(checkpoint_path)), map_location="cpu"
        )
        if checkpoint.get("format_version") != 1:
            raise ValueError(
                "Expected the Habitat-free robot checkpoint format version 1."
            )
        expected_actions = [
            ACTION_LABELS[index] for index in range(len(ACTION_LABELS))
        ]
        if checkpoint.get("action_labels") != expected_actions:
            raise ValueError(
                "Checkpoint action order does not match {}.".format(
                    expected_actions
                )
            )

        self.rgb_size = tuple(checkpoint.get("rgb_size", RGB_SIZE))
        self.depth_size = tuple(checkpoint.get("depth_size", DEPTH_SIZE))
        if self.rgb_size != RGB_SIZE or self.depth_size != DEPTH_SIZE:
            raise ValueError(
                "Checkpoint sensor sizes {} / {} do not match the fixed CMA "
                "architecture {} / {}.".format(
                    self.rgb_size,
                    self.depth_size,
                    RGB_SIZE,
                    DEPTH_SIZE,
                )
            )

        word_list = checkpoint.get("word_list")
        if not isinstance(word_list, list) or not word_list:
            raise ValueError(
                "Checkpoint does not contain a valid R2R word list."
            )
        self.instruction_tokens, self.instruction_stats = encode_instruction(
            word_list, instruction, instruction_length
        )
        self.policy = CMAPolicy(
            vocab_size=len(word_list), num_actions=len(ACTION_LABELS)
        )
        self.policy.load_state_dict(checkpoint["state_dict"], strict=True)
        self.policy.to(self.device)
        self.policy.eval()
        self.reset()

    def reset(self) -> None:
        self.rnn_states = torch.zeros(
            1,
            self.policy.net.num_recurrent_layers,
            STATE_HIDDEN_SIZE,
            device=self.device,
        )
        self.prev_actions = torch.zeros(
            1, 1, device=self.device, dtype=torch.long
        )
        self.not_done_masks = torch.zeros(
            1, 1, device=self.device, dtype=torch.bool
        )

    def predict(self, observations: Dict[str, np.ndarray]) -> int:
        observations = dict(observations)
        observations["instruction"] = self.instruction_tokens
        batch = batch_observation(observations, self.device)

        with torch.no_grad():
            actions, self.rnn_states = self.policy.act(
                batch,
                self.rnn_states,
                self.prev_actions,
                self.not_done_masks,
                deterministic=not self.sample_actions,
            )
            self.prev_actions.copy_(actions)
            self.not_done_masks.fill_(1)

        return int(actions[0].item())


class RosVlnInferenceNode:
    def __init__(self, args, runner: CMARunner) -> None:
        self.args = args
        self.runner = runner
        self.bridge = CvBridge()
        self.work_queue = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.action_count = 0
        self.exit_code = 0
        self.last_inference_start_time = 0.0
        self.last_pair_time = time.monotonic()
        self.last_rgb_message_time = self.last_pair_time
        self.last_depth_message_time = self.last_pair_time

        # Publishers are set up before subscribers so callbacks never race
        # against partially initialized outputs.
        self.action_publisher = None
        if not args.no_publish:
            self.action_publisher = rospy.Publisher(
                args.action_topic, String, queue_size=10
            )
            wait_started = time.monotonic()
            while (
                self.action_publisher.get_num_connections() == 0
                and not rospy.is_shutdown()
                and time.monotonic() - wait_started
                < args.publisher_wait_timeout
            ):
                rospy.sleep(0.05)
            if self.action_publisher.get_num_connections() == 0:
                rospy.logwarn(
                    "No subscriber connected to %s after %.1f s; the "
                    "initial action may not be delivered. Use --no-publish "
                    "if only console output is needed.",
                    args.action_topic,
                    args.publisher_wait_timeout,
                )

        self.worker = threading.Thread(
            target=self._worker_loop,
            name="vln_rgbd_inference_worker",
            daemon=True,
        )
        self.worker.start()

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
        self.rgb_subscriber.registerCallback(self._record_rgb_arrival)
        self.depth_subscriber.registerCallback(self._record_depth_arrival)
        self.synchronizer = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_subscriber, self.depth_subscriber],
            queue_size=args.sync_queue_size,
            slop=args.sync_slop,
        )
        self.synchronizer.registerCallback(self._synchronized_callback)
        rospy.on_shutdown(self.stop_event.set)

    def _record_rgb_arrival(self, _message: Image) -> None:
        self.last_rgb_message_time = time.monotonic()

    def _record_depth_arrival(self, _message: Image) -> None:
        self.last_depth_message_time = time.monotonic()

    def _synchronized_callback(
        self, rgb_message: Image, depth_message: Image
    ) -> None:
        self.last_pair_time = time.monotonic()
        pair = (rgb_message, depth_message)
        try:
            self.work_queue.put_nowait(pair)
            return
        except queue.Full:
            pass

        # Keep only the newest synchronized pair; stale image queues make a
        # physical robot act on an old view.
        try:
            self.work_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self.work_queue.put_nowait(pair)
        except queue.Full:
            pass

    def _worker_loop(self) -> None:
        while not self.stop_event.is_set() and not rospy.is_shutdown():
            try:
                rgb_message, depth_message = self.work_queue.get(
                    timeout=0.2
                )
            except queue.Empty:
                now = time.monotonic()
                if (
                    self.args.input_timeout > 0.0
                    and now - self.last_pair_time
                    >= self.args.input_timeout
                ):
                    log_function = (
                        rospy.logerr
                        if self.args.exit_on_input_timeout
                        else rospy.logwarn
                    )
                    log_function(
                        "No synchronized RGB-D pair received for %.1f s. "
                        "Last RGB %.1f s ago; last processed depth %.1f s "
                        "ago; sync slop %.0f ms.",
                        self.args.input_timeout,
                        now - self.last_rgb_message_time,
                        now - self.last_depth_message_time,
                        1000.0 * self.args.sync_slop,
                    )
                    if self.args.exit_on_input_timeout:
                        self.exit_code = 2
                        rospy.signal_shutdown("RGB-D input timeout")
                        return
                    self.last_pair_time = now
                continue

            now = time.monotonic()
            if (
                self.args.min_action_interval > 0.0
                and now - self.last_inference_start_time
                < self.args.min_action_interval
            ):
                continue
            self.last_inference_start_time = now

            try:
                rgb = self.bridge.imgmsg_to_cv2(
                    rgb_message, desired_encoding="rgb8"
                )
                depth = self.bridge.imgmsg_to_cv2(
                    depth_message, desired_encoding="passthrough"
                )
                observations, invalid_fraction = preprocess_rgbd(
                    rgb=rgb,
                    depth_m=depth,
                    depth_encoding=depth_message.encoding,
                    rgb_size=self.runner.rgb_size,
                    depth_size=self.runner.depth_size,
                    min_depth=self.args.min_depth,
                    max_depth=self.args.max_depth,
                )
                action = self.runner.predict(observations)
            except Exception as error:
                rospy.logerr("RGB-D inference failed: %s", error)
                self.exit_code = 1
                rospy.signal_shutdown("RGB-D inference failure")
                return

            self.action_count += 1
            action_name = ACTION_LABELS.get(action, "UNKNOWN")
            if self.action_publisher is not None:
                self.action_publisher.publish(String(data=action_name))

            timestamp_delta_ms = abs(
                rgb_message.header.stamp.to_sec()
                - depth_message.header.stamp.to_sec()
            ) * 1000.0
            rospy.loginfo(
                "action=%s count=%d stamp_delta_ms=%.2f "
                "processed_invalid_depth=%.2f%%",
                action_name,
                self.action_count,
                timestamp_delta_ms,
                100.0 * invalid_fraction,
            )

            if (
                self.args.max_actions > 0
                and self.action_count >= self.args.max_actions
            ):
                rospy.signal_shutdown("requested action count reached")
                return
            if action == 0 and not self.args.keep_running_after_stop:
                rospy.signal_shutdown("policy predicted STOP")
                return

    def run(self) -> int:
        rospy.spin()
        self.stop_event.set()
        if self.worker.is_alive():
            self.worker.join(timeout=2.0)
        return self.exit_code


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Synchronize ROS1 RGB with preprocessed 32FC1 metric depth and "
            "run the standalone PyTorch VLN-CE CMA policy."
        )
    )
    parser.add_argument(
        "--instruction",
        required=True,
        help="English R2R navigation instruction used for this episode.",
    )
    parser.add_argument(
        "--checkpoint-path",
        default="data/checkpoints/CMA_PM_DA_Aug_robot.pth",
        help=(
            "Habitat-free CMA robot checkpoint, relative to the repository "
            "root."
        ),
    )
    parser.add_argument(
        "--rgb-topic",
        default="/camera/rgb/image_color",
        help="ROS sensor_msgs/Image RGB topic.",
    )
    parser.add_argument(
        "--depth-topic",
        default="/camera/depth_registered/image_filled",
        help=(
            "Processed 32FC1 metric-depth topic published by "
            "ros_depth_hole_filler.py."
        ),
    )
    parser.add_argument(
        "--action-topic",
        default="/vln/action",
        help="std_msgs/String topic for predicted English action names.",
    )
    parser.add_argument(
        "--no-publish",
        action="store_true",
        help="Print/log actions without publishing them to ROS.",
    )
    parser.add_argument(
        "--publisher-wait-timeout",
        type=float,
        default=2.0,
        help=(
            "Seconds to wait for an action-topic subscriber before starting "
            "inference. Use 0 to skip the wait."
        ),
    )
    parser.add_argument(
        "--max-actions",
        type=int,
        default=1,
        help="Number of actions before exit. Use 0 for continuous inference.",
    )
    parser.add_argument(
        "--min-action-interval",
        type=float,
        default=0.5,
        help="Minimum seconds between policy steps in continuous mode.",
    )
    parser.add_argument(
        "--keep-running-after-stop",
        action="store_true",
        help="Do not exit when the policy predicts STOP.",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Sample from the action distribution instead of using argmax.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force inference on CPU even if CUDA is available.",
    )
    parser.add_argument(
        "--instruction-length",
        type=int,
        default=200,
        help="Padded instruction token length expected by the R2R model.",
    )
    parser.add_argument(
        "--sync-queue-size",
        type=int,
        default=10,
        help="ApproximateTimeSynchronizer queue size.",
    )
    parser.add_argument(
        "--sync-slop",
        type=float,
        default=0.05,
        help="Maximum RGB/depth timestamp difference in seconds.",
    )
    parser.add_argument(
        "--input-timeout",
        type=float,
        default=30.0,
        help=(
            "Warn if no synchronized RGB-D pair arrives for this many "
            "seconds. Use 0 to disable timeout warnings."
        ),
    )
    parser.add_argument(
        "--exit-on-input-timeout",
        action="store_true",
        help="Exit instead of continuing after an RGB-D input timeout.",
    )
    parser.add_argument(
        "--subscriber-buffer-bytes",
        type=int,
        default=2 ** 24,
        help="TCPROS receive buffer for each image subscriber.",
    )
    parser.add_argument(
        "--min-depth",
        type=float,
        default=0.0,
        help="Minimum valid depth in meters.",
    )
    parser.add_argument(
        "--max-depth",
        type=float,
        default=10.0,
        help="Maximum valid depth in meters; also defines normalization.",
    )
    parser.add_argument(
        "--node-name",
        default="ros_vln_inference",
        help="ROS node name.",
    )
    return parser


def validate_arguments(args) -> None:
    if args.max_actions < 0:
        raise ValueError("--max-actions must be >= 0.")
    if args.publisher_wait_timeout < 0.0:
        raise ValueError("--publisher-wait-timeout must be >= 0.")
    if args.min_action_interval < 0.0:
        raise ValueError("--min-action-interval must be >= 0.")
    if args.instruction_length <= 0:
        raise ValueError("--instruction-length must be positive.")
    if args.sync_queue_size <= 0:
        raise ValueError("--sync-queue-size must be positive.")
    if args.sync_slop < 0.0:
        raise ValueError("--sync-slop must be >= 0.")
    if args.input_timeout < 0.0:
        raise ValueError("--input-timeout must be >= 0.")
    if not args.max_depth > args.min_depth:
        raise ValueError("--max-depth must be greater than --min-depth.")


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()
    try:
        validate_arguments(args)
    except ValueError as error:
        parser.error(str(error))

    os.chdir(str(REPO_ROOT))
    rospy.init_node(args.node_name, anonymous=False)

    try:
        runner = CMARunner(
            checkpoint_path=args.checkpoint_path,
            instruction=args.instruction,
            instruction_length=args.instruction_length,
            force_cpu=args.cpu,
            sample_actions=args.sample,
        )
    except Exception as error:
        rospy.logfatal("Failed to initialize CMA policy: %s", error)
        return 1

    instruction_stats = runner.instruction_stats
    rospy.loginfo(
        "CMA ready on %s; instruction_tokens=%d unknown=%d/%d; "
        "rgb_topic=%s depth_topic=%s",
        runner.device,
        instruction_stats["length"],
        instruction_stats["unknown_count"],
        instruction_stats["length"],
        args.rgb_topic,
        args.depth_topic,
    )
    if (
        instruction_stats["length"] > 0
        and instruction_stats["unknown_count"]
        / float(instruction_stats["length"])
        > 0.5
    ):
        rospy.logwarn(
            "More than half of the instruction tokens are unknown. "
            "The current R2R CMA checkpoint expects English instructions."
        )

    node = RosVlnInferenceNode(args, runner)
    return node.run()


if __name__ == "__main__":
    sys.exit(main())
