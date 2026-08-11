#!/usr/bin/env python3

"""Collect safety-gated recurrent DAgger episodes on a ROS1 robot.

At each stopped decision state this node freezes one synchronized RGB-D pair,
advances the CMA recurrent state using the previously *executed* action, and
shows the model proposal.  The operator then either approves the proposal or
enters an expert override before the discrete chassis action is executed.

The legacy ``action`` field remains the expert training target.  Additional
``model_action`` and ``executed_action`` fields make recurrent action history
unambiguous.  Existing expert episodes remain compatible because their
executed action implicitly equals their expert action.
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np


REAL_ROOT = Path(__file__).resolve().parents[1]
if str(REAL_ROOT) not in sys.path:
    sys.path.insert(0, str(REAL_ROOT))

from training.ros_expert_drive_collector import (  # noqa: E402
    ACTION_LABELS,
    ACTION_TO_INDEX,
    COMMANDS,
    ExpertDriveCollector,
    action_to_motion,
    build_parser as build_expert_parser,
    rospy,
    String,
    validate_args as validate_expert_args,
)
from scripts.ros_vln_inference import CMARunner  # noqa: E402
from vlnce_real.preprocessing import preprocess_rgbd  # noqa: E402


DEFAULT_OUTPUT_DIR = (
    REAL_ROOT / "training" / "data" / "real_episodes_0p4m_15deg"
)
DEFAULT_CHECKPOINT = (
    REAL_ROOT
    / "training"
    / "checkpoints"
    / "real_cma_seq64_scheduled_sampling"
    / "best_robot.pth"
)


def action_probability_dict(probabilities):
    if len(probabilities) != len(ACTION_LABELS):
        raise ValueError("Policy returned an invalid action distribution.")
    return {
        action: float(probabilities[index])
        for index, action in enumerate(ACTION_LABELS)
    }


class DaggerCollector(ExpertDriveCollector):
    """Human-gated DAgger collection with executed-action recurrent memory."""

    def __init__(self, args, runner):
        self.runner = runner
        self.previous_executed_action = None
        super().__init__(args)

        self.model_action_publisher = rospy.Publisher(
            args.model_action_topic, String, queue_size=10
        )
        self.executed_action_publisher = rospy.Publisher(
            args.executed_action_topic, String, queue_size=10
        )
        with self.manifest_lock:
            self.manifest["collection_mode"] = "recurrent_dagger_human_gated"
            self.manifest["topics"]["model_action"] = args.model_action_topic
            self.manifest["topics"]["executed_action"] = (
                args.executed_action_topic
            )
            self.manifest["dagger"] = {
                "policy_checkpoint": str(
                    Path(args.checkpoint_path).expanduser().resolve()
                ),
                "policy_action_selection": (
                    "sample" if args.sample else "argmax"
                ),
                "operator_gate_required": True,
                "model_mistake_execution_enabled": (
                    args.allow_model_mistake_execution
                ),
                "memory_previous_action_source": "executed_action",
                "expert_target_field": "expert_action_index",
                "executed_history_field": "executed_action_index",
            }
            self._write_manifest()

    def _infer_frozen_pair(self, frozen_pair):
        _, rgb_message, depth_message = frozen_pair
        if depth_message.encoding.upper() != "32FC1":
            raise ValueError(
                "Expected 32FC1 depth in meters, got {!r}.".format(
                    depth_message.encoding
                )
            )
        rgb = self.bridge.imgmsg_to_cv2(
            rgb_message, desired_encoding="rgb8"
        )
        depth = self.bridge.imgmsg_to_cv2(
            depth_message, desired_encoding="passthrough"
        )
        observations, invalid_fraction = preprocess_rgbd(
            rgb=np.asarray(rgb),
            depth_m=np.asarray(depth),
            depth_encoding=depth_message.encoding,
            rgb_size=self.runner.rgb_size,
            depth_size=self.runner.depth_size,
            min_depth=self.args.min_depth,
            max_depth=self.args.max_depth,
        )
        details = self.runner.predict_with_details(observations)
        details["processed_invalid_depth_fraction"] = float(invalid_fraction)
        return details

    def _print_proposal(self, details):
        proposal = ACTION_LABELS[int(details["action"])]
        previous = self.previous_executed_action or "START"
        probabilities = details["probabilities"]
        print("\nstep={} previous_executed={}".format(len(self.samples), previous))
        print(
            "model={} confidence={:.2%} | STOP={:.2%} FWD={:.2%} "
            "LEFT={:.2%} RIGHT={:.2%}".format(
                proposal,
                details["confidence"],
                probabilities[0],
                probabilities[1],
                probabilities[2],
                probabilities[3],
            )
        )

    def _read_operator_decision(self, model_action):
        while not rospy.is_shutdown():
            command = input(
                "expert [Enter=approve, w/a/d/s=override, e=end, q=abort]> "
            ).strip().lower().replace("-", "_")
            if not command:
                return model_action, model_action, "model_approved"
            if command in ("q", "quit", "abort"):
                return None, None, "abort"
            if command in ("e", "end", "finish", "complete"):
                return None, None, "complete"

            if command.startswith("m "):
                if not self.args.allow_model_mistake_execution:
                    print(
                        "Model-mistake execution is disabled. Use an ordinary "
                        "expert override, or restart with "
                        "--allow-model-mistake-execution."
                    )
                    continue
                expert_command = command[2:].strip()
                expert_action = COMMANDS.get(expert_command)
                if expert_action is None:
                    print("Use 'm w', 'm a' or 'm d' to label the state.")
                    continue
                if expert_action == "STOP" and model_action != "STOP":
                    print(
                        "Refusing to move when the expert label is STOP. "
                        "Execute the expert STOP instead."
                    )
                    continue
                return expert_action, model_action, "model_exploration"

            expert_action = COMMANDS.get(command)
            if expert_action is None:
                print("Unknown command. Use Enter, w, a, d, s, e or q.")
                continue
            source = (
                "model_approved"
                if expert_action == model_action
                else "expert_override"
            )
            return expert_action, expert_action, source
        return None, None, "abort"

    def _ensure_execution_ready(self, action):
        if action == "STOP" or self.args.dry_run:
            return
        if self.velocity_publisher.get_num_connections() == 0:
            raise RuntimeError(
                "No chassis subscriber is connected to {}; action was not "
                "recorded or executed.".format(
                    rospy.resolve_name(self.args.cmd_vel_topic)
                )
            )
        if self.use_odom and self._fresh_odom() is None:
            raise RuntimeError(
                "No fresh nav_msgs/Odometry is available on {}; action was "
                "not recorded or executed.".format(
                    rospy.resolve_name(self.odom_topic)
                )
            )

    def collect_decision(
        self,
        frozen_pair,
        details,
        expert_action,
        executed_action,
        behavior_source,
    ):
        self._ensure_execution_ready(executed_action)
        sample, _ = self.record_before_action(
            expert_action, frozen_pair=frozen_pair
        )
        executed_motion = action_to_motion(
            executed_action, self.motion_settings
        )
        sample["expert_action"] = expert_action
        sample["expert_action_index"] = ACTION_TO_INDEX[expert_action]
        sample["model_action"] = ACTION_LABELS[int(details["action"])]
        sample["model_action_index"] = int(details["action"])
        sample["executed_action"] = executed_action
        sample["executed_action_index"] = ACTION_TO_INDEX[executed_action]
        sample["previous_executed_action"] = self.previous_executed_action
        sample["previous_executed_action_index"] = (
            None
            if self.previous_executed_action is None
            else ACTION_TO_INDEX[self.previous_executed_action]
        )
        sample["dagger"] = {
            "behavior_source": behavior_source,
            "expert_intervened": expert_action != sample["model_action"],
            "expert_and_executed_differ": expert_action != executed_action,
            "model_confidence": float(details["confidence"]),
            "model_probabilities": action_probability_dict(
                details["probabilities"]
            ),
            "rnn_state_norm_before": float(
                details["rnn_state_norm_before"]
            ),
            "rnn_state_norm_after": float(details["rnn_state_norm_after"]),
            "processed_invalid_depth_fraction": float(
                details["processed_invalid_depth_fraction"]
            ),
        }
        sample["planned_motion"] = {
            "action": executed_action,
            "linear_x_mps": executed_motion.linear_x,
            "angular_z_radps": executed_motion.angular_z,
            "duration_seconds": executed_motion.duration,
        }
        with self.manifest_lock:
            self._write_manifest()

        self.model_action_publisher.publish(
            String(data=sample["model_action"])
        )
        self.expert_action_publisher.publish(String(data=expert_action))
        self.executed_action_publisher.publish(
            String(data=executed_action)
        )
        self.execute_action(
            sample,
            executed_action,
            executed_motion,
            publish_expert_action=False,
        )
        self.runner.set_previous_action(ACTION_TO_INDEX[executed_action])
        self.previous_executed_action = executed_action
        return sample

    def finalize(self, status):
        if not self.finalized:
            with self.manifest_lock:
                dagger_samples = [
                    sample
                    for sample in self.samples
                    if isinstance(sample.get("dagger"), dict)
                ]
                self.manifest["dagger"].update(
                    {
                        "model_approved_count": sum(
                            sample["dagger"]["behavior_source"]
                            == "model_approved"
                            for sample in dagger_samples
                        ),
                        "expert_override_count": sum(
                            sample["dagger"]["behavior_source"]
                            == "expert_override"
                            for sample in dagger_samples
                        ),
                        "model_exploration_count": sum(
                            sample["dagger"]["behavior_source"]
                            == "model_exploration"
                            for sample in dagger_samples
                        ),
                    }
                )
                self._write_manifest()
        super().finalize(status)

    def run(self):
        rospy.loginfo(
            "DAgger collector waiting for RGB=%s Depth=%s cmd_vel=%s odom=%s",
            self.args.rgb_topic,
            self.args.depth_topic,
            self.args.cmd_vel_topic,
            (
                self.odom_topic
                if self.use_odom and not self.args.dry_run
                else "not required"
            ),
        )
        try:
            self.wait_until_ready()
        except Exception:
            self.finalize("aborted")
            raise

        print("\nEnglish instruction:\n{}\n".format(self.args.instruction))
        print("Safety-gated recurrent DAgger:")
        print("  Enter = approve and execute the model proposal")
        print("  w/a/d/s = expert override and execute that expert action")
        print("  e = complete without adding a STOP sample")
        print("  q or Ctrl-C = emergency stop and abort")
        if self.args.allow_model_mistake_execution:
            print(
                "  m w / m a / m d = save that expert label but deliberately "
                "execute the model proposal"
            )
        if self.args.dry_run:
            print("  DRY RUN: samples are saved but the chassis will not move")

        try:
            while not rospy.is_shutdown():
                frozen_pair = self._newest_pair_for_action("MODEL_PROPOSAL")
                details = self._infer_frozen_pair(frozen_pair)
                model_action = ACTION_LABELS[int(details["action"])]
                self._print_proposal(details)
                expert_action, executed_action, source = (
                    self._read_operator_decision(model_action)
                )
                if source == "abort":
                    self.finalize("aborted")
                    return 0
                if source == "complete":
                    if not self.samples:
                        print(
                            "Cannot complete an empty episode; approve or "
                            "override at least one action first."
                        )
                        self.finalize("aborted")
                        return 1
                    self.finalize("complete")
                    return 0

                try:
                    self.collect_decision(
                        frozen_pair,
                        details,
                        expert_action,
                        executed_action,
                        source,
                    )
                except Exception as error:
                    rospy.logerr("%s", error)
                    print("DAgger transaction failed; chassis STOP was sent.")
                    self._publish_stop()
                    self.finalize("error")
                    return 1

                if expert_action == "STOP" and executed_action == "STOP":
                    self.finalize("complete")
                    return 0
        except (EOFError, KeyboardInterrupt):
            self.finalize("aborted")
            return 130

        self.finalize("aborted")
        return 1


def build_parser():
    parser = build_expert_parser()
    parser.description = (
        "Collect a human-gated recurrent DAgger episode with model, expert, "
        "and executed actions stored separately."
    )
    parser.set_defaults(
        output_dir=str(DEFAULT_OUTPUT_DIR),
        episode_id=datetime.now().strftime("dagger_%Y%m%d_%H%M%S"),
        node_name="ros_dagger_collector",
    )
    parser.add_argument(
        "--checkpoint-path", default=str(DEFAULT_CHECKPOINT)
    )
    parser.add_argument("--instruction-length", type=int, default=200)
    parser.add_argument("--min-depth", type=float, default=0.0)
    parser.add_argument("--max-depth", type=float, default=10.0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Sample model proposals instead of using deterministic argmax.",
    )
    parser.add_argument(
        "--model-action-topic", default="/vln/dagger/model_action"
    )
    parser.add_argument(
        "--executed-action-topic", default="/vln/dagger/executed_action"
    )
    parser.add_argument(
        "--allow-model-mistake-execution",
        action="store_true",
        help=(
            "Enable the explicit 'm <expert key>' command that records an "
            "expert label but executes the differing model proposal. Unsafe; "
            "disabled by default."
        ),
    )
    return parser


def validate_args(args):
    validate_expert_args(args)
    if args.instruction_length <= 0:
        raise ValueError("--instruction-length must be positive.")
    if not args.max_depth > args.min_depth:
        raise ValueError("--max-depth must be greater than --min-depth.")
    for name in ("model_action_topic", "executed_action_topic"):
        value = getattr(args, name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError("--{} must not be empty.".format(name.replace("_", "-")))


def main():
    parser = build_parser()
    args = parser.parse_args(rospy.myargv(argv=sys.argv)[1:])
    try:
        validate_args(args)
    except ValueError as error:
        parser.error(str(error))

    rospy.init_node(args.node_name, anonymous=False)
    try:
        runner = CMARunner(
            checkpoint_path=args.checkpoint_path,
            instruction=args.instruction,
            instruction_length=args.instruction_length,
            force_cpu=args.cpu,
            sample_actions=args.sample,
        )
        collector = DaggerCollector(args, runner)
    except FileExistsError:
        rospy.logfatal(
            "Episode directory already exists; choose another --episode-id."
        )
        return 1
    except Exception as error:
        rospy.logfatal("Cannot initialize DAgger collector: %s", error)
        return 1

    try:
        return collector.run()
    except Exception as error:
        rospy.logfatal("DAgger collection failed: %s", error)
        collector.finalize("error")
        return 1


if __name__ == "__main__":
    sys.exit(main())
