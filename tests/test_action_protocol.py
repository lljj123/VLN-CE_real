import unittest

from vlnce_real.action_protocol import (
    ActionCommand,
    ActionResult,
    decode_action_command,
    decode_action_result,
    encode_action_command,
    encode_action_result,
)


class ActionProtocolTest(unittest.TestCase):
    def test_command_round_trip(self):
        payload = encode_action_command(7, "TURN_LEFT")
        self.assertEqual(
            decode_action_command(payload),
            ActionCommand(7, "TURN_LEFT"),
        )

    def test_plain_action_is_accepted_for_manual_control(self):
        self.assertEqual(
            decode_action_command("  MOVE_FORWARD  "),
            ActionCommand(None, "MOVE_FORWARD"),
        )

    def test_result_round_trip(self):
        payload = encode_action_result(
            9,
            "MOVE_FORWARD",
            "succeeded",
            "odometry target reached",
        )
        self.assertEqual(
            decode_action_result(payload),
            ActionResult(
                9,
                "MOVE_FORWARD",
                "succeeded",
                "odometry target reached",
            ),
        )

    def test_invalid_sequence_is_rejected(self):
        with self.assertRaises(ValueError):
            decode_action_command(
                '{"version":1,"sequence":0,"action":"STOP"}'
            )

    def test_result_must_be_json(self):
        with self.assertRaises(ValueError):
            decode_action_result("MOVE_FORWARD")


if __name__ == "__main__":
    unittest.main()
