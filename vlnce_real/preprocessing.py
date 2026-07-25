"""RGB-D preprocessing shared by robot inference and real-data training."""

from typing import Dict, Tuple

import cv2
import numpy as np


def preprocess_rgbd(
    rgb: np.ndarray,
    depth_m: np.ndarray,
    depth_encoding: str,
    rgb_size: Tuple[int, int],
    depth_size: Tuple[int, int],
    min_depth: float,
    max_depth: float,
) -> Tuple[Dict[str, np.ndarray], float]:
    """Convert aligned RGB and metric depth to CMA observation arrays."""

    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(
            "Expected RGB shape [H, W, 3], got {}".format(rgb.shape)
        )
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    if depth_encoding.upper() != "32FC1":
        raise ValueError(
            "Expected processed depth encoding 32FC1 in meters, got '{}'."
            .format(depth_encoding)
        )
    depth_m = np.asarray(depth_m, dtype=np.float32)
    if depth_m.ndim == 3 and depth_m.shape[2] == 1:
        depth_m = depth_m[:, :, 0]
    if depth_m.ndim != 2:
        raise ValueError(
            "Expected a single-channel depth image, got shape {}".format(
                depth_m.shape
            )
        )
    if rgb.shape[:2] != depth_m.shape[:2]:
        raise ValueError(
            "Registered RGB and depth sizes differ: {} vs {}".format(
                rgb.shape[:2], depth_m.shape[:2]
            )
        )
    if not max_depth > min_depth:
        raise ValueError("max_depth must be greater than min_depth.")

    valid = (
        np.isfinite(depth_m)
        & (depth_m > min_depth)
        & (depth_m <= max_depth)
    )
    invalid_fraction = float(1.0 - valid.mean())
    depth_m = depth_m.copy()
    depth_m[~valid] = 0.0

    rgb_height, rgb_width = rgb_size
    depth_height, depth_width = depth_size
    rgb_resized = cv2.resize(
        rgb,
        (rgb_width, rgb_height),
        interpolation=cv2.INTER_AREA,
    )
    depth_resized_m = cv2.resize(
        depth_m,
        (depth_width, depth_height),
        interpolation=cv2.INTER_NEAREST,
    )

    depth_normalized = np.zeros_like(depth_resized_m, dtype=np.float32)
    resized_valid = depth_resized_m > 0.0
    depth_normalized[resized_valid] = np.clip(
        (
            depth_resized_m[resized_valid] - float(min_depth)
        )
        / float(max_depth - min_depth),
        0.0,
        1.0,
    )

    observations = {
        "rgb": np.ascontiguousarray(rgb_resized, dtype=np.uint8),
        "depth": np.ascontiguousarray(
            depth_normalized[:, :, None], dtype=np.float32
        ),
    }
    return observations, invalid_fraction
