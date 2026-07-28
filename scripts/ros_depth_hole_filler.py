#!/usr/bin/env python3

"""Convert registered ROS1 depth images to meters and fill small holes.

Input is normally ``16UC1`` depth measured in millimeters.  Output is always
``32FC1`` depth measured in meters.  The output keeps the input header exactly,
including its timestamp and frame ID, so another node can synchronize it with
the original RGB image.

Only small invalid connected components that do not touch the image border are
filled.  Large missing regions and registered-depth border gaps remain zero.
"""

import argparse
import glob
import os
import queue
import sys
import threading

import numpy as np


def add_ros_python_paths() -> None:
    """Make ROS Noetic modules visible from the VLN Python environment."""

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
import rospy  # noqa: E402
from cv_bridge import CvBridge  # noqa: E402
from sensor_msgs.msg import Image  # noqa: E402


def depth_to_meters(
    depth: np.ndarray,
    encoding: str,
    depth_scale=None,
) -> np.ndarray:
    """Return a single-channel float32 depth image measured in meters."""

    depth = np.asarray(depth)
    if depth.ndim == 3 and depth.shape[2] == 1:
        depth = depth[:, :, 0]
    if depth.ndim != 2:
        raise ValueError(
            "Expected a single-channel depth image, got shape {}".format(
                depth.shape
            )
        )

    if depth_scale is None:
        encoding_upper = encoding.upper()
        if encoding_upper == "16UC1":
            depth_scale = 0.001
        elif encoding_upper == "32FC1":
            depth_scale = 1.0
        else:
            raise ValueError(
                "Unsupported depth encoding '{}'. Use --depth-scale for an "
                "explicit raw-value-to-meter multiplier.".format(encoding)
            )

    return depth.astype(np.float32) * float(depth_scale)


def fill_small_depth_holes(
    depth_m: np.ndarray,
    radius: int,
    max_area: int,
):
    """Fill bounded small holes and return ``(filled_depth, pixel_count)``."""

    depth_m = np.asarray(depth_m, dtype=np.float32)
    if radius <= 0 or max_area <= 0:
        return depth_m.copy(), 0

    valid = np.isfinite(depth_m) & (depth_m > 0.0)
    invalid_u8 = (~valid).astype(np.uint8)
    if not np.any(invalid_u8):
        return depth_m.copy(), 0

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        invalid_u8, connectivity=8
    )
    allowed_labels = np.zeros(num_labels, dtype=np.bool_)
    image_height, image_width = depth_m.shape

    for label in range(1, num_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        touches_border = (
            x == 0
            or y == 0
            or x + width >= image_width
            or y + height >= image_height
        )
        if not touches_border and area <= max_area:
            allowed_labels[label] = True

    eligible = allowed_labels[labels]
    if not np.any(eligible):
        return depth_m.copy(), 0

    filled = depth_m.copy()
    filled[~valid] = 0.0
    working_valid = valid.copy()
    initially_valid_count = int(np.count_nonzero(working_valid))

    for _ in range(radius):
        neighbor_sum = cv2.boxFilter(
            filled,
            ddepth=-1,
            ksize=(3, 3),
            normalize=False,
            borderType=cv2.BORDER_REPLICATE,
        )
        neighbor_count = cv2.boxFilter(
            working_valid.astype(np.float32),
            ddepth=-1,
            ksize=(3, 3),
            normalize=False,
            borderType=cv2.BORDER_REPLICATE,
        )
        candidates = eligible & (~working_valid) & (neighbor_count > 0.0)
        if not np.any(candidates):
            break
        filled[candidates] = (
            neighbor_sum[candidates] / neighbor_count[candidates]
        )
        working_valid[candidates] = True

    filled_count = (
        int(np.count_nonzero(working_valid)) - initially_valid_count
    )
    return filled, filled_count


def process_depth(
    raw_depth: np.ndarray,
    encoding: str,
    min_depth: float,
    max_depth: float,
    depth_scale,
    hole_fill_radius: int,
    hole_fill_max_area: int,
):
    """Convert, sanitize, and conservatively fill one depth frame."""

    depth_m = depth_to_meters(raw_depth, encoding, depth_scale)
    valid = (
        np.isfinite(depth_m)
        & (depth_m > min_depth)
        & (depth_m <= max_depth)
    )
    invalid_fraction_before = float(1.0 - valid.mean())
    depth_m = depth_m.copy()
    depth_m[~valid] = 0.0

    filled_depth, filled_count = fill_small_depth_holes(
        depth_m,
        radius=hole_fill_radius,
        max_area=hole_fill_max_area,
    )
    invalid_fraction_after = float(
        1.0 - np.count_nonzero(filled_depth > 0.0) / filled_depth.size
    )
    return np.ascontiguousarray(filled_depth, dtype=np.float32), {
        "invalid_fraction_before": invalid_fraction_before,
        "invalid_fraction_after": invalid_fraction_after,
        "filled_pixels": filled_count,
    }


class DepthHoleFillerNode:
    def __init__(self, args) -> None:
        self.args = args
        self.bridge = CvBridge()
        self.work_queue = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.frame_count = 0
        self.exit_code = 0

        # Publisher first, subscriber last: callbacks cannot race with output
        # initialization.  Sensor queues stay at one to avoid stale frames.
        self.publisher = rospy.Publisher(
            args.output_topic, Image, queue_size=1
        )
        self.worker = threading.Thread(
            target=self._worker_loop,
            name="depth_hole_filler_worker",
            daemon=True,
        )
        self.worker.start()
        self.subscriber = rospy.Subscriber(
            args.input_topic,
            Image,
            self._depth_callback,
            queue_size=1,
            buff_size=args.subscriber_buffer_bytes,
        )
        rospy.on_shutdown(self.stop_event.set)

    def _depth_callback(self, message: Image) -> None:
        try:
            self.work_queue.put_nowait(message)
            return
        except queue.Full:
            pass

        try:
            self.work_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self.work_queue.put_nowait(message)
        except queue.Full:
            pass

    def _worker_loop(self) -> None:
        while not self.stop_event.is_set() and not rospy.is_shutdown():
            try:
                message = self.work_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                raw_depth = self.bridge.imgmsg_to_cv2(
                    message, desired_encoding="passthrough"
                )
                filled_depth, stats = process_depth(
                    raw_depth=raw_depth,
                    encoding=message.encoding,
                    min_depth=self.args.min_depth,
                    max_depth=self.args.max_depth,
                    depth_scale=self.args.depth_scale,
                    hole_fill_radius=self.args.hole_fill_radius,
                    hole_fill_max_area=self.args.hole_fill_max_area,
                )
                output_message = self.bridge.cv2_to_imgmsg(
                    filled_depth, encoding="32FC1"
                )
                output_message.header = message.header
                self.publisher.publish(output_message)
            except Exception as error:
                rospy.logerr("Depth preprocessing failed: %s", error)
                self.exit_code = 1
                rospy.signal_shutdown("depth preprocessing failure")
                return

            self.frame_count += 1
            if self.frame_count == 1 or (
                self.args.log_every > 0
                and self.frame_count % self.args.log_every == 0
            ):
                rospy.loginfo(
                    "depth_frame=%d encoding=%s->32FC1(m) "
                    "invalid=%.2f%%->%.2f%% filled_pixels=%d",
                    self.frame_count,
                    message.encoding,
                    100.0 * stats["invalid_fraction_before"],
                    100.0 * stats["invalid_fraction_after"],
                    stats["filled_pixels"],
                )

            if (
                self.args.max_frames > 0
                and self.frame_count >= self.args.max_frames
            ):
                rospy.signal_shutdown("requested frame count reached")
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
            "Convert registered ROS depth to 32FC1 meters and fill only "
            "small internal holes."
        )
    )
    parser.add_argument(
        "--input-topic",
        default="/camera/depth_registered/image_raw",
        help="Input registered depth sensor_msgs/Image topic.",
    )
    parser.add_argument(
        "--output-topic",
        default="/camera/depth_registered/image_filled",
        help="Output 32FC1 metric-depth sensor_msgs/Image topic.",
    )
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=None,
        help=(
            "Optional raw-depth-to-meter multiplier. Defaults to 0.001 for "
            "16UC1 and 1.0 for 32FC1."
        ),
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
        help="Maximum valid depth in meters.",
    )
    parser.add_argument(
        "--hole-fill-radius",
        type=int,
        default=5,
        help="Maximum inward fill radius for eligible holes.",
    )
    parser.add_argument(
        "--hole-fill-max-area",
        type=int,
        default=500,
        help=(
            "Maximum invalid connected-component area to fill. Border "
            "components are always retained as invalid."
        ),
    )
    parser.add_argument(
        "--subscriber-buffer-bytes",
        type=int,
        default=2 ** 24,
        help="TCPROS receive buffer for the input depth image.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=30,
        help="Log every N frames; use 0 to disable periodic statistics.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after N output frames. Use 0 for continuous processing.",
    )
    parser.add_argument(
        "--node-name",
        default="ros_depth_hole_filler",
        help="ROS node name.",
    )
    return parser


def validate_arguments(args) -> None:
    if args.depth_scale is not None and args.depth_scale <= 0.0:
        raise ValueError("--depth-scale must be positive.")
    if not args.max_depth > args.min_depth:
        raise ValueError("--max-depth must be greater than --min-depth.")
    if args.hole_fill_radius < 0:
        raise ValueError("--hole-fill-radius must be >= 0.")
    if args.hole_fill_max_area < 0:
        raise ValueError("--hole-fill-max-area must be >= 0.")
    if args.subscriber_buffer_bytes <= 0:
        raise ValueError("--subscriber-buffer-bytes must be positive.")
    if args.log_every < 0:
        raise ValueError("--log-every must be >= 0.")
    if args.max_frames < 0:
        raise ValueError("--max-frames must be >= 0.")


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()
    try:
        validate_arguments(args)
    except ValueError as error:
        parser.error(str(error))

    rospy.init_node(args.node_name, anonymous=False)
    rospy.loginfo(
        "Depth hole filler ready: %s -> %s (32FC1 meters)",
        args.input_topic,
        args.output_topic,
    )
    node = DepthHoleFillerNode(args)
    return node.run()


if __name__ == "__main__":
    sys.exit(main())
