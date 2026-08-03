#!/usr/bin/env python3

"""Show and publish the exact RGB-D arrays passed to the CMA policy.

The node synchronizes the same ROS RGB and processed metric-depth topics used
by ``ros_vln_inference.py`` and calls the shared ``preprocess_rgbd`` function.
It does not load a checkpoint, run CMA, or publish navigation actions.

Published images:

* ``/vln/preprocessed/rgb``: exact ``rgb8`` CMA input (224x224).
* ``/vln/preprocessed/depth``: exact normalized ``32FC1`` CMA input
  (256x256, range 0..1).
* ``/vln/preprocessed/depth_color``: a display-only ``bgr8`` colour map.
"""

import argparse
import glob
import os
import queue
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def add_ros_python_paths() -> None:
    """Make ROS Noetic Python modules visible from the VLN environment."""

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

from vlnce_real.model import DEPTH_SIZE, RGB_SIZE  # noqa: E402
from vlnce_real.preprocessing import preprocess_rgbd  # noqa: E402


def colorize_normalized_depth(depth_normalized: np.ndarray) -> np.ndarray:
    """Return a display image; near is red, far is blue, zero is black."""

    depth = np.asarray(depth_normalized, dtype=np.float32)
    if depth.ndim == 3 and depth.shape[2] == 1:
        depth = depth[:, :, 0]
    if depth.ndim != 2:
        raise ValueError(
            "Expected normalized depth [H, W] or [H, W, 1], got {}."
            .format(depth.shape)
        )

    finite = np.isfinite(depth)
    clipped = np.clip(depth, 0.0, 1.0)
    # JET maps high values to red. Invert normalized metric depth so close
    # pixels are warm and distant pixels are cool.
    color_index = np.rint((1.0 - clipped) * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(color_index, cv2.COLORMAP_JET)
    colored[~finite | (depth <= 0.0)] = (0, 0, 0)
    return colored


def window_is_open(window_name: str) -> bool:
    """Return whether an OpenCV HighGUI window still exists."""

    try:
        visible = cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE)
        if visible >= 0.0:
            return visible >= 1.0
        return (
            cv2.getWindowProperty(window_name, cv2.WND_PROP_AUTOSIZE)
            >= 0.0
        )
    except cv2.error:
        return False


class PreprocessedRgbdVisualizerNode:
    """Synchronize RGB-D, reuse CMA preprocessing, then publish its output."""

    def __init__(self, args) -> None:
        self.args = args
        self.bridge = CvBridge()
        self.message_queue = queue.Queue(maxsize=1)
        self.received_frames = 0
        self.window_has_image = False
        self.exit_code = 0

        # Create publishers before subscribers so the first synchronized pair
        # can always be published.
        self.rgb_publisher = rospy.Publisher(
            args.rgb_output_topic, Image, queue_size=1
        )
        self.depth_publisher = rospy.Publisher(
            args.depth_output_topic, Image, queue_size=1
        )
        self.depth_color_publisher = rospy.Publisher(
            args.depth_color_topic, Image, queue_size=1
        )

        if not args.no_window:
            cv2.namedWindow(args.rgb_window_name, cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow(args.depth_window_name, cv2.WINDOW_AUTOSIZE)

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

    def _synchronized_callback(
        self, rgb_message: Image, depth_message: Image
    ) -> None:
        pair = (rgb_message, depth_message)
        try:
            self.message_queue.put_nowait(pair)
            return
        except queue.Full:
            pass

        # Visualization should always use the newest frame rather than build a
        # delayed image queue behind the live camera.
        try:
            self.message_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self.message_queue.put_nowait(pair)
        except queue.Full:
            pass

    def _consume_and_publish(self) -> bool:
        try:
            rgb_message, depth_message = self.message_queue.get(timeout=0.03)
        except queue.Empty:
            return False

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
                rgb_size=RGB_SIZE,
                depth_size=DEPTH_SIZE,
                min_depth=self.args.min_depth,
                max_depth=self.args.max_depth,
            )
        except Exception as error:
            rospy.logerr_throttle(
                2.0, "Cannot preprocess synchronized RGB-D: %s", error
            )
            self.exit_code = 1
            return False

        model_rgb = observations["rgb"]
        model_depth = observations["depth"][:, :, 0]
        depth_color = colorize_normalized_depth(model_depth)

        rgb_output = self.bridge.cv2_to_imgmsg(model_rgb, encoding="rgb8")
        rgb_output.header = rgb_message.header
        self.rgb_publisher.publish(rgb_output)

        depth_output = self.bridge.cv2_to_imgmsg(
            model_depth, encoding="32FC1"
        )
        depth_output.header = depth_message.header
        self.depth_publisher.publish(depth_output)

        depth_color_output = self.bridge.cv2_to_imgmsg(
            depth_color, encoding="bgr8"
        )
        depth_color_output.header = depth_message.header
        self.depth_color_publisher.publish(depth_color_output)

        self.received_frames += 1
        if self.received_frames == 1 or (
            self.args.log_every > 0
            and self.received_frames % self.args.log_every == 0
        ):
            stamp_delta_ms = abs(
                rgb_message.header.stamp.to_sec()
                - depth_message.header.stamp.to_sec()
            ) * 1000.0
            rospy.loginfo(
                "preprocessed frame=%d source_rgb=%dx%d source_depth=%dx%d "
                "model_rgb=%dx%d model_depth=%dx%d stamp_delta_ms=%.2f "
                "invalid_depth=%.2f%%",
                self.received_frames,
                rgb.shape[1],
                rgb.shape[0],
                depth.shape[1],
                depth.shape[0],
                model_rgb.shape[1],
                model_rgb.shape[0],
                model_depth.shape[1],
                model_depth.shape[0],
                stamp_delta_ms,
                100.0 * invalid_fraction,
            )

        if not self.args.no_window:
            rgb_bgr = cv2.cvtColor(model_rgb, cv2.COLOR_RGB2BGR)
            if self.args.display_scale != 1.0:
                rgb_bgr = cv2.resize(
                    rgb_bgr,
                    None,
                    fx=self.args.display_scale,
                    fy=self.args.display_scale,
                    interpolation=cv2.INTER_NEAREST,
                )
                depth_color = cv2.resize(
                    depth_color,
                    None,
                    fx=self.args.display_scale,
                    fy=self.args.display_scale,
                    interpolation=cv2.INTER_NEAREST,
                )
            cv2.imshow(self.args.rgb_window_name, rgb_bgr)
            cv2.imshow(self.args.depth_window_name, depth_color)
            self.window_has_image = True
        return True

    def run(self) -> int:
        try:
            while not rospy.is_shutdown():
                self._consume_and_publish()
                if not self.args.no_window:
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        rospy.signal_shutdown("visualizer window closed")
                        break
                    if self.window_has_image and (
                        not window_is_open(self.args.rgb_window_name)
                        or not window_is_open(self.args.depth_window_name)
                    ):
                        rospy.signal_shutdown("visualizer window closed")
                        break
        finally:
            if not self.args.no_window:
                cv2.destroyAllWindows()
        return self.exit_code


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Display and publish the exact RGB and normalized-depth images "
            "produced by the CMA preprocessing function."
        )
    )
    parser.add_argument(
        "--rgb-topic",
        default="/camera/rgb/image_raw",
        help="Input sensor_msgs/Image RGB topic.",
    )
    parser.add_argument(
        "--depth-topic",
        default="/camera/depth_registered/image_filled",
        help="Input processed 32FC1 metric-depth topic.",
    )
    parser.add_argument(
        "--rgb-output-topic",
        default="/vln/preprocessed/rgb",
        help="Exact rgb8 image sent to CMA.",
    )
    parser.add_argument(
        "--depth-output-topic",
        default="/vln/preprocessed/depth",
        help="Exact normalized 32FC1 depth image sent to CMA.",
    )
    parser.add_argument(
        "--depth-color-topic",
        default="/vln/preprocessed/depth_color",
        help="Display-only bgr8 colour map of normalized CMA depth.",
    )
    parser.add_argument(
        "--min-depth",
        type=float,
        default=0.0,
        help="Minimum valid metric depth, matching ros_vln_inference.py.",
    )
    parser.add_argument(
        "--max-depth",
        type=float,
        default=10.0,
        help="Maximum valid metric depth and normalization range.",
    )
    parser.add_argument(
        "--sync-slop",
        type=float,
        default=0.10,
        help="Maximum RGB/depth timestamp difference in seconds.",
    )
    parser.add_argument(
        "--sync-queue-size",
        type=int,
        default=20,
        help="ApproximateTimeSynchronizer queue size.",
    )
    parser.add_argument(
        "--subscriber-buffer-bytes",
        type=int,
        default=2 ** 24,
        help="TCPROS receive buffer for each image topic.",
    )
    parser.add_argument(
        "--display-scale",
        type=float,
        default=2.0,
        help="Window-only scale; published resolutions remain unchanged.",
    )
    parser.add_argument(
        "--rgb-window-name",
        default="CMA RGB 224x224",
        help="OpenCV RGB window title.",
    )
    parser.add_argument(
        "--depth-window-name",
        default="CMA Depth 256x256 (normalized)",
        help="OpenCV depth window title.",
    )
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="Do not open GUI windows; only publish diagnostic topics.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=100,
        help="Log dimensions/statistics every N frames; 0 disables repeats.",
    )
    parser.add_argument(
        "--node-name",
        default="ros_preprocessed_rgbd_visualizer",
        help="ROS node name.",
    )
    return parser


def validate_arguments(args) -> None:
    input_topics = {args.rgb_topic, args.depth_topic}
    output_topics = {
        args.rgb_output_topic,
        args.depth_output_topic,
        args.depth_color_topic,
    }
    if len(output_topics) != 3:
        raise ValueError("The three output topics must be different.")
    if input_topics & output_topics:
        raise ValueError("Input topics must not also be output topics.")
    if not args.max_depth > args.min_depth >= 0.0:
        raise ValueError(
            "Require --max-depth > --min-depth >= 0, in metres."
        )
    if args.sync_slop < 0.0:
        raise ValueError("--sync-slop must be >= 0.")
    if args.sync_queue_size <= 0:
        raise ValueError("--sync-queue-size must be positive.")
    if args.subscriber_buffer_bytes <= 0:
        raise ValueError("--subscriber-buffer-bytes must be positive.")
    if args.display_scale <= 0.0:
        raise ValueError("--display-scale must be positive.")
    if args.log_every < 0:
        raise ValueError("--log-every must be >= 0.")


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args(rospy.myargv()[1:])
    try:
        validate_arguments(args)
    except ValueError as error:
        parser.error(str(error))

    rospy.init_node(args.node_name, anonymous=False)
    rospy.loginfo(
        "CMA preprocessing visualizer ready: RGB %s + depth %s -> "
        "RGB %s (%dx%d), depth %s (%dx%d), colour %s",
        args.rgb_topic,
        args.depth_topic,
        args.rgb_output_topic,
        RGB_SIZE[1],
        RGB_SIZE[0],
        args.depth_output_topic,
        DEPTH_SIZE[1],
        DEPTH_SIZE[0],
        args.depth_color_topic,
    )
    node = PreprocessedRgbdVisualizerNode(args)
    return node.run()


if __name__ == "__main__":
    sys.exit(main())
