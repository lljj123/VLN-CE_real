import unittest

from vlnce_real.action_protocol import (
    ActionCommand,
    ActionResult,
    InferenceMetrics,
    decode_action_command,
    decode_action_result,
    decode_inference_metrics,
    encode_action_command,
    encode_action_result,
    encode_inference_metrics,
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
                None,
                None,
                None,
                None,
                None,
                None,
            ),
        )

    def test_detailed_result_round_trip(self):
        payload = encode_action_result(
            4,
            "TURN_LEFT",
            "succeeded",
            "odometry target reached",
            execution_seconds=0.61,
            control_mode="odom_closed_loop",
            target_value=15.0,
            progress_value=14.7,
            target_unit="deg",
            previous_action_end_to_start_seconds=0.123,
        )
        self.assertEqual(
            decode_action_result(payload),
            ActionResult(
                4,
                "TURN_LEFT",
                "succeeded",
                "odometry target reached",
                0.61,
                "odom_closed_loop",
                15.0,
                14.7,
                "deg",
                0.123,
            ),
        )

    def test_old_result_without_execution_gap_remains_compatible(self):
        payload = (
            '{"version":1,"sequence":2,"action":"TURN_RIGHT",'
            '"status":"succeeded","reason":"done",'
            '"execution_seconds":0.5,"control_mode":"open_loop",'
            '"target_value":0.5,"progress_value":0.5,'
            '"target_unit":"s"}'
        )
        self.assertIsNone(
            decode_action_result(
                payload
            ).previous_action_end_to_start_seconds
        )

    def test_inference_metrics_round_trip(self):
        payload = encode_inference_metrics(
            sequence=3,
            action="MOVE_FORWARD",
            action_count=3,
            device="cuda:0",
            image_conversion_seconds=0.001,
            preprocess_seconds=0.002,
            model_seconds=0.011,
            total_seconds=0.014,
            rgb_depth_delta_seconds=0.004,
            invalid_depth_fraction=0.03,
            first_inference=False,
            result_to_inference_start_seconds=0.087,
            fresh_rgbd_wait_seconds=0.080,
            rgbd_queue_seconds=0.007,
        )
        self.assertEqual(
            decode_inference_metrics(payload),
            InferenceMetrics(
                3,
                "MOVE_FORWARD",
                3,
                "cuda:0",
                0.001,
                0.002,
                0.011,
                0.014,
                0.004,
                0.03,
                False,
                0.087,
                0.080,
                0.007,
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
