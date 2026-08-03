#!/usr/bin/env python3

"""Visualize a ROS1 depth image with metric distance annotations.

The node accepts the common ``16UC1`` (millimetres) and ``32FC1`` (metres)
depth encodings.  It publishes a colourized ``bgr8`` image and, unless
``--no-window`` is used, opens an interactive OpenCV window.  Grid labels give
an overview of the scene while the mouse reports the distance of any exact
pixel.  Left-click locks a pixel and right-click returns to hover mode.
"""

import argparse
import glob
import os
import queue
import sys
from typing import Optional, Tuple

import numpy as np


def add_ros_python_paths() -> None:
    """Make ROS Noetic modules visible from a non-system Python environment."""

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


Point = Tuple[int, int]


def depth_to_meters(
    raw_depth: np.ndarray,
    encoding: str,
    depth_scale: Optional[float] = None,
) -> np.ndarray:
    """Convert a single-channel ROS depth image to float32 metres."""

    raw_depth = np.asarray(raw_depth)
    if raw_depth.ndim == 3 and raw_depth.shape[2] == 1:
        raw_depth = raw_depth[:, :, 0]
    if raw_depth.ndim != 2:
        raise ValueError(
            "Expected a single-channel depth image, got shape {}.".format(
                raw_depth.shape
            )
        )

    if depth_scale is None:
        encoding_upper = encoding.upper()
        if encoding_upper in ("16UC1", "MONO16"):
            depth_scale = 0.001
        elif encoding_upper in ("32FC1", "64FC1"):
            depth_scale = 1.0
        else:
            raise ValueError(
                "Unsupported depth encoding '{}'. Supply --depth-scale "
                "(raw value to metres).".format(encoding)
            )

    depth_m = raw_depth.astype(np.float32) * float(depth_scale)
    depth_m[~np.isfinite(depth_m) | (depth_m <= 0.0)] = np.nan
    return np.ascontiguousarray(depth_m)


def sample_distance(
    depth_m: np.ndarray,
    point: Point,
    radius: int = 0,
) -> Optional[float]:
    """Return the pixel value, or the local valid median for radius > 0."""

    x, y = point
    height, width = depth_m.shape
    if x < 0 or x >= width or y < 0 or y >= height:
        return None

    x0 = max(0, x - radius)
    x1 = min(width, x + radius + 1)
    y0 = max(0, y - radius)
    y1 = min(height, y + radius + 1)
    values = depth_m[y0:y1, x0:x1]
    valid_values = values[np.isfinite(values) & (values > 0.0)]
    if valid_values.size == 0:
        return None
    return float(np.median(valid_values))


def format_distance(distance_m: Optional[float]) -> str:
    if distance_m is None:
        return "N/A"
    return "{:.2f} m".format(distance_m)


def draw_outlined_text(
    image: np.ndarray,
    text: str,
    origin: Point,
    scale: float = 0.45,
    color: Tuple[int, int, int] = (255, 255, 255),
    thickness: int = 1,
) -> None:
    """Draw legible text over both bright and dark colour-map regions."""

    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def window_is_open(window_name: str) -> bool:
    """Return whether a HighGUI window exists across OpenCV GUI backends."""

    try:
        visible = cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE)
        if visible >= 0.0:
            return visible >= 1.0

        # OpenCV 4.2's GTK backend returns -1 for WND_PROP_VISIBLE even for
        # an open window.  WND_PROP_AUTOSIZE is supported there and becomes
        # negative (or raises cv2.error) after the window is destroyed.
        return (
            cv2.getWindowProperty(window_name, cv2.WND_PROP_AUTOSIZE)
            >= 0.0
        )
    except cv2.error:
        return False


def colorize_depth(
    depth_m: np.ndarray,
    min_depth: float,
    max_depth: float,
) -> np.ndarray:
    """Map near points to red, far points to blue, and invalid points black."""

    valid = np.isfinite(depth_m) & (depth_m > 0.0)
    clipped = np.clip(depth_m, min_depth, max_depth)
    normalized = (
        (max_depth - clipped) * (255.0 / (max_depth - min_depth))
    )
    normalized[~valid] = 0.0
    normalized_u8 = normalized.astype(np.uint8)
    colored = cv2.applyColorMap(normalized_u8, cv2.COLORMAP_JET)
    colored[~valid] = (0, 0, 0)
    return colored


def _label_origin(
    image_shape: Tuple[int, ...],
    point: Point,
    text: str,
) -> Point:
    """Place a sample label near its marker while keeping it in the image."""

    height, width = image_shape[:2]
    (text_width, text_height), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
    )
    x, y = point
    label_x = min(max(2, x + 5), max(2, width - text_width - 2))
    label_y = y - 5
    if label_y - text_height < 30:
        label_y = y + text_height + 7
    label_y = min(max(text_height + 2, label_y), height - 3)
    return label_x, label_y


def render_depth_frame(
    depth_m: np.ndarray,
    min_depth: float,
    max_depth: float,
    grid_rows: int,
    grid_columns: int,
    sample_radius: int,
    selected_point: Optional[Point] = None,
    selection_locked: bool = False,
) -> np.ndarray:
    """Create a colourized depth frame with grid and cursor annotations."""

    visualization = colorize_depth(depth_m, min_depth, max_depth)
    height, width = depth_m.shape

    # Cell borders make it clear which part of the image each label samples.
    for column in range(1, grid_columns):
        x = int(round(column * width / float(grid_columns)))
        cv2.line(visualization, (x, 0), (x, height - 1), (70, 70, 70), 1)
    for row in range(1, grid_rows):
        y = int(round(row * height / float(grid_rows)))
        cv2.line(visualization, (0, y), (width - 1, y), (70, 70, 70), 1)

    for row in range(grid_rows):
        y = min(
            height - 1,
            int(round((row + 0.5) * height / float(grid_rows))),
        )
        for column in range(grid_columns):
            x = min(
                width - 1,
                int(
                    round(
                        (column + 0.5) * width / float(grid_columns)
                    )
                ),
            )
            point = (x, y)
            distance_text = format_distance(
                sample_distance(depth_m, point, sample_radius)
            )
            cv2.drawMarker(
                visualization,
                point,
                (255, 255, 255),
                markerType=cv2.MARKER_CROSS,
                markerSize=7,
                thickness=1,
                line_type=cv2.LINE_AA,
            )
            draw_outlined_text(
                visualization,
                distance_text,
                _label_origin(visualization.shape, point, distance_text),
            )

    # A compact top banner reports the exact pixel under the mouse.  It also
    # masks grid text underneath, making the live value easy to read.
    banner = visualization.copy()
    cv2.rectangle(banner, (0, 0), (width - 1, 29), (0, 0, 0), -1)
    visualization = cv2.addWeighted(banner, 0.70, visualization, 0.30, 0.0)

    if selected_point is None:
        status = "Move mouse: inspect | Left click: lock | Right click: unlock"
    else:
        selected_x, selected_y = selected_point
        selected_distance = sample_distance(
            depth_m, selected_point, sample_radius
        )
        mode = "LOCKED" if selection_locked else "HOVER"
        status = "{} pixel=({}, {}) distance={}".format(
            mode,
            selected_x,
            selected_y,
            format_distance(selected_distance),
        )
        cv2.drawMarker(
            visualization,
            selected_point,
            (0, 255, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=19,
            thickness=2,
            line_type=cv2.LINE_AA,
        )
        cv2.circle(visualization, selected_point, 7, (0, 255, 255), 1)

    draw_outlined_text(
        visualization,
        status,
        (8, 20),
        scale=0.5,
        color=(0, 255, 255),
    )
    range_text = "red=near  blue=far  color range {:.2f}-{:.2f} m".format(
        min_depth, max_depth
    )
    draw_outlined_text(
        visualization,
        range_text,
        (8, height - 8),
        scale=0.43,
    )
    return visualization


class DepthVisualizerNode:
    def __init__(self, args) -> None:
        self.args = args
        self.bridge = CvBridge()
        self.message_queue = queue.Queue(maxsize=1)
        self.publisher = rospy.Publisher(
            args.output_topic, Image, queue_size=1
        )

        self.depth_m = None
        self.header = None
        self.encoding = ""
        self.selected_point = None
        self.selection_locked = False
        self.render_dirty = False
        self.window_has_image = False
        self.received_frames = 0
        self.exit_code = 0

        if not args.no_window:
            cv2.namedWindow(args.window_name, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(args.window_name, self._mouse_callback)

        # The large TCPROS buffer avoids truncating common 640x480/1280x720
        # depth messages; queue_size=1 prevents stale sensor frames piling up.
        self.subscriber = rospy.Subscriber(
            args.input_topic,
            Image,
            self._depth_callback,
            queue_size=1,
            buff_size=args.subscriber_buffer_bytes,
        )

    def _depth_callback(self, message: Image) -> None:
        try:
            self.message_queue.put_nowait(message)
            return
        except queue.Full:
            pass

        try:
            self.message_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self.message_queue.put_nowait(message)
        except queue.Full:
            pass

    def _display_to_source_point(self, x: int, y: int) -> Optional[Point]:
        if self.depth_m is None:
            return None
        source_x = int(x / self.args.display_scale)
        source_y = int(y / self.args.display_scale)
        height, width = self.depth_m.shape
        if source_x < 0 or source_x >= width:
            return None
        if source_y < 0 or source_y >= height:
            return None
        return source_x, source_y

    def _mouse_callback(self, event, x, y, _flags, _userdata) -> None:
        point = self._display_to_source_point(x, y)
        if event == cv2.EVENT_RBUTTONDOWN:
            self.selection_locked = False
            self.selected_point = point
            self.render_dirty = True
            return
        if event == cv2.EVENT_LBUTTONDOWN and point is not None:
            self.selection_locked = True
            self.selected_point = point
            self.render_dirty = True
            return
        if event == cv2.EVENT_MOUSEMOVE and not self.selection_locked:
            self.selected_point = point
            self.render_dirty = True

    def _consume_latest_message(self) -> bool:
        try:
            message = self.message_queue.get(timeout=0.03)
        except queue.Empty:
            return False

        try:
            raw_depth = self.bridge.imgmsg_to_cv2(
                message, desired_encoding="passthrough"
            )
            self.depth_m = depth_to_meters(
                raw_depth,
                message.encoding,
                self.args.depth_scale,
            )
        except Exception as error:
            rospy.logerr_throttle(2.0, "Cannot decode depth image: %s", error)
            self.exit_code = 1
            return False

        self.header = message.header
        self.encoding = message.encoding
        self.received_frames += 1
        self.render_dirty = True
        if self.received_frames == 1:
            rospy.loginfo(
                "Received %s depth image: %dx%d. Distances are shown in "
                "metres.",
                message.encoding,
                self.depth_m.shape[1],
                self.depth_m.shape[0],
            )
        return True

    def _render_and_publish(self) -> None:
        if self.depth_m is None or not self.render_dirty:
            return

        visualization = render_depth_frame(
            self.depth_m,
            self.args.min_depth,
            self.args.max_depth,
            self.args.grid_rows,
            self.args.grid_columns,
            self.args.sample_radius,
            self.selected_point,
            self.selection_locked,
        )
        output_message = self.bridge.cv2_to_imgmsg(
            visualization, encoding="bgr8"
        )
        output_message.header = self.header
        self.publisher.publish(output_message)

        if not self.args.no_window:
            if self.args.display_scale == 1.0:
                displayed = visualization
            else:
                displayed = cv2.resize(
                    visualization,
                    None,
                    fx=self.args.display_scale,
                    fy=self.args.display_scale,
                    interpolation=cv2.INTER_NEAREST,
                )
            cv2.imshow(self.args.window_name, displayed)
            self.window_has_image = True
        self.render_dirty = False

    def run(self) -> int:
        try:
            while not rospy.is_shutdown():
                self._consume_latest_message()
                self._render_and_publish()

                if not self.args.no_window:
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        rospy.signal_shutdown("visualizer window closed")
                        break
                    # Some GUI backends do not implement WND_PROP_VISIBLE;
                    # window_is_open() uses a supported fallback for them.
                    if self.window_has_image and not window_is_open(
                        self.args.window_name
                    ):
                        rospy.signal_shutdown("visualizer window closed")
                        break
        finally:
            if not self.args.no_window:
                cv2.destroyWindow(self.args.window_name)
        return self.exit_code


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Colourize a ROS1 depth topic and annotate metric distances."
        )
    )
    parser.add_argument(
        "--input-topic",
        default="/camera/depth_registered/image_raw",
        help="Input sensor_msgs/Image depth topic.",
    )
    parser.add_argument(
        "--output-topic",
        default="/camera/depth_registered/image_visualized",
        help="Output sensor_msgs/Image topic containing bgr8 annotations.",
    )
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=None,
        help=(
            "Raw-value-to-metre multiplier. Auto: 0.001 for 16UC1/mono16, "
            "1.0 for 32FC1/64FC1."
        ),
    )
    parser.add_argument(
        "--min-depth",
        type=float,
        default=0.10,
        help="Near end of the colour range in metres.",
    )
    parser.add_argument(
        "--max-depth",
        type=float,
        default=10.0,
        help="Far end of the colour range in metres.",
    )
    parser.add_argument(
        "--grid-rows",
        type=int,
        default=6,
        help="Number of annotated sampling rows.",
    )
    parser.add_argument(
        "--grid-columns",
        type=int,
        default=8,
        help="Number of annotated sampling columns.",
    )
    parser.add_argument(
        "--sample-radius",
        type=int,
        default=0,
        help=(
            "Median sampling radius in pixels. 0 shows the exact pixel; "
            "2 gives a more stable 5x5 median."
        ),
    )
    parser.add_argument(
        "--display-scale",
        type=float,
        default=1.0,
        help="OpenCV window scale; does not change output-topic resolution.",
    )
    parser.add_argument(
        "--window-name",
        default="ROS Depth Distance Viewer",
        help="OpenCV window title.",
    )
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="Do not open a GUI; only publish the annotated output topic.",
    )
    parser.add_argument(
        "--subscriber-buffer-bytes",
        type=int,
        default=2 ** 24,
        help="TCPROS receive buffer for the depth image.",
    )
    parser.add_argument(
        "--node-name",
        default="ros_depth_visualizer",
        help="ROS node name.",
    )
    return parser


def validate_arguments(args) -> None:
    if args.input_topic == args.output_topic:
        raise ValueError("Input and output topics must be different.")
    if args.depth_scale is not None and args.depth_scale <= 0.0:
        raise ValueError("--depth-scale must be positive.")
    if not args.max_depth > args.min_depth >= 0.0:
        raise ValueError(
            "Require --max-depth > --min-depth >= 0, in metres."
        )
    if args.grid_rows <= 0 or args.grid_columns <= 0:
        raise ValueError("--grid-rows and --grid-columns must be positive.")
    if args.sample_radius < 0:
        raise ValueError("--sample-radius must be >= 0.")
    if args.display_scale <= 0.0:
        raise ValueError("--display-scale must be positive.")
    if args.subscriber_buffer_bytes <= 0:
        raise ValueError("--subscriber-buffer-bytes must be positive.")


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args(rospy.myargv()[1:])
    try:
        validate_arguments(args)
    except ValueError as error:
        parser.error(str(error))

    rospy.init_node(args.node_name, anonymous=False)
    rospy.loginfo(
        "Depth visualizer ready: %s -> %s (grid=%dx%d)",
        args.input_topic,
        args.output_topic,
        args.grid_columns,
        args.grid_rows,
    )
    node = DepthVisualizerNode(args)
    return node.run()


if __name__ == "__main__":
    sys.exit(main())
