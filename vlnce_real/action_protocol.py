"""ROS-message-independent protocol for discrete action execution.

The ROS nodes carry these payloads in ``std_msgs/String`` so the standalone
project does not need a generated catkin message package.  Plain action names
remain valid commands for backwards-compatible manual testing.
"""

import json
import math
from typing import NamedTuple, Optional


PROTOCOL_VERSION = 1


class ActionCommand(NamedTuple):
    sequence: Optional[int]
    action: str


class ActionResult(NamedTuple):
    sequence: Optional[int]
    action: str
    status: str
    reason: str
    execution_seconds: Optional[float]
    control_mode: Optional[str]
    target_value: Optional[float]
    progress_value: Optional[float]
    target_unit: Optional[str]


class InferenceMetrics(NamedTuple):
    sequence: Optional[int]
    action: str
    action_count: int
    device: str
    image_conversion_seconds: float
    preprocess_seconds: float
    model_seconds: float
    total_seconds: float
    rgb_depth_delta_seconds: float
    invalid_depth_fraction: float
    first_inference: bool
    result_to_inference_start_seconds: Optional[float]
    fresh_rgbd_wait_seconds: Optional[float]
    rgbd_queue_seconds: Optional[float]


def _sequence(value):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("sequence must be a positive integer")
    return value


def _text(mapping, key):
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError("{} must be a non-empty string".format(key))
    return value.strip()


def _optional_text(mapping, key):
    value = mapping.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("{} must be null or a non-empty string".format(key))
    return value.strip()


def _nonnegative_float(mapping, key, optional=False):
    value = mapping.get(key)
    if value is None and optional:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        qualifier = "null or " if optional else ""
        raise ValueError(
            "{} must be {}a finite number >= 0".format(key, qualifier)
        )
    return float(value)


def _positive_integer(mapping, key):
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("{} must be a positive integer".format(key))
    return value


def encode_action_command(sequence, action):
    payload = {
        "version": PROTOCOL_VERSION,
        "sequence": _sequence(sequence),
        "action": action,
    }
    _text(payload, "action")
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def decode_action_command(payload):
    if not isinstance(payload, str) or not payload.strip():
        raise ValueError("action command must be a non-empty string")
    text = payload.strip()
    if not text.startswith("{"):
        return ActionCommand(None, text)

    try:
        mapping = json.loads(text)
    except ValueError as error:
        raise ValueError("invalid action command JSON: {}".format(error))
    if not isinstance(mapping, dict):
        raise ValueError("action command JSON must be an object")
    if mapping.get("version") != PROTOCOL_VERSION:
        raise ValueError("unsupported action command protocol version")
    return ActionCommand(
        _sequence(mapping.get("sequence")),
        _text(mapping, "action"),
    )


def encode_action_result(
    sequence,
    action,
    status,
    reason,
    execution_seconds=None,
    control_mode=None,
    target_value=None,
    progress_value=None,
    target_unit=None,
):
    payload = {
        "version": PROTOCOL_VERSION,
        "sequence": _sequence(sequence),
        "action": action,
        "status": status,
        "reason": reason,
        "execution_seconds": execution_seconds,
        "control_mode": control_mode,
        "target_value": target_value,
        "progress_value": progress_value,
        "target_unit": target_unit,
    }
    for key in ("action", "status", "reason"):
        _text(payload, key)
    _nonnegative_float(payload, "execution_seconds", optional=True)
    _optional_text(payload, "control_mode")
    _nonnegative_float(payload, "target_value", optional=True)
    _nonnegative_float(payload, "progress_value", optional=True)
    _optional_text(payload, "target_unit")
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def decode_action_result(payload):
    if not isinstance(payload, str) or not payload.strip():
        raise ValueError("action result must be a non-empty string")
    try:
        mapping = json.loads(payload)
    except ValueError as error:
        raise ValueError("invalid action result JSON: {}".format(error))
    if not isinstance(mapping, dict):
        raise ValueError("action result JSON must be an object")
    if mapping.get("version") != PROTOCOL_VERSION:
        raise ValueError("unsupported action result protocol version")
    return ActionResult(
        _sequence(mapping.get("sequence")),
        _text(mapping, "action"),
        _text(mapping, "status"),
        _text(mapping, "reason"),
        _nonnegative_float(mapping, "execution_seconds", optional=True),
        _optional_text(mapping, "control_mode"),
        _nonnegative_float(mapping, "target_value", optional=True),
        _nonnegative_float(mapping, "progress_value", optional=True),
        _optional_text(mapping, "target_unit"),
    )


def encode_inference_metrics(
    sequence,
    action,
    action_count,
    device,
    image_conversion_seconds,
    preprocess_seconds,
    model_seconds,
    total_seconds,
    rgb_depth_delta_seconds,
    invalid_depth_fraction,
    first_inference,
    result_to_inference_start_seconds=None,
    fresh_rgbd_wait_seconds=None,
    rgbd_queue_seconds=None,
):
    payload = {
        "version": PROTOCOL_VERSION,
        "sequence": _sequence(sequence),
        "action": action,
        "action_count": action_count,
        "device": device,
        "image_conversion_seconds": image_conversion_seconds,
        "preprocess_seconds": preprocess_seconds,
        "model_seconds": model_seconds,
        "total_seconds": total_seconds,
        "rgb_depth_delta_seconds": rgb_depth_delta_seconds,
        "invalid_depth_fraction": invalid_depth_fraction,
        "first_inference": first_inference,
        "result_to_inference_start_seconds": (
            result_to_inference_start_seconds
        ),
        "fresh_rgbd_wait_seconds": fresh_rgbd_wait_seconds,
        "rgbd_queue_seconds": rgbd_queue_seconds,
    }
    for key in ("action", "device"):
        _text(payload, key)
    _positive_integer(payload, "action_count")
    for key in (
        "image_conversion_seconds",
        "preprocess_seconds",
        "model_seconds",
        "total_seconds",
        "rgb_depth_delta_seconds",
        "invalid_depth_fraction",
    ):
        _nonnegative_float(payload, key)
    if not isinstance(first_inference, bool):
        raise ValueError("first_inference must be true or false")
    _nonnegative_float(
        payload,
        "result_to_inference_start_seconds",
        optional=True,
    )
    _nonnegative_float(
        payload,
        "fresh_rgbd_wait_seconds",
        optional=True,
    )
    _nonnegative_float(
        payload,
        "rgbd_queue_seconds",
        optional=True,
    )
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def decode_inference_metrics(payload):
    if not isinstance(payload, str) or not payload.strip():
        raise ValueError("inference metrics must be a non-empty string")
    try:
        mapping = json.loads(payload)
    except ValueError as error:
        raise ValueError("invalid inference metrics JSON: {}".format(error))
    if not isinstance(mapping, dict):
        raise ValueError("inference metrics JSON must be an object")
    if mapping.get("version") != PROTOCOL_VERSION:
        raise ValueError("unsupported inference metrics protocol version")
    first_inference = mapping.get("first_inference")
    if not isinstance(first_inference, bool):
        raise ValueError("first_inference must be true or false")
    return InferenceMetrics(
        _sequence(mapping.get("sequence")),
        _text(mapping, "action"),
        _positive_integer(mapping, "action_count"),
        _text(mapping, "device"),
        _nonnegative_float(mapping, "image_conversion_seconds"),
        _nonnegative_float(mapping, "preprocess_seconds"),
        _nonnegative_float(mapping, "model_seconds"),
        _nonnegative_float(mapping, "total_seconds"),
        _nonnegative_float(mapping, "rgb_depth_delta_seconds"),
        _nonnegative_float(mapping, "invalid_depth_fraction"),
        first_inference,
        _nonnegative_float(
            mapping,
            "result_to_inference_start_seconds",
            optional=True,
        ),
        _nonnegative_float(
            mapping,
            "fresh_rgbd_wait_seconds",
            optional=True,
        ),
        _nonnegative_float(
            mapping,
            "rgbd_queue_seconds",
            optional=True,
        ),
    )
