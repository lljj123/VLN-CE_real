"""ROS-message-independent protocol for discrete action execution.

The ROS nodes carry these payloads in ``std_msgs/String`` so the standalone
project does not need a generated catkin message package.  Plain action names
remain valid commands for backwards-compatible manual testing.
"""

import json
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


def encode_action_result(sequence, action, status, reason):
    payload = {
        "version": PROTOCOL_VERSION,
        "sequence": _sequence(sequence),
        "action": action,
        "status": status,
        "reason": reason,
    }
    for key in ("action", "status", "reason"):
        _text(payload, key)
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
    )
