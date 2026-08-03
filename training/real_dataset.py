"""Dataset utilities for Habitat-free real-robot CMA fine-tuning."""

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from vlnce_real.model import encode_instruction
from vlnce_real.preprocessing import preprocess_rgbd


ACTION_LABELS = [
    "STOP",
    "MOVE_FORWARD",
    "TURN_LEFT",
    "TURN_RIGHT",
]


def _contained_path(root, relative_path):
    root = root.resolve()
    result = (root / relative_path).resolve()
    try:
        result.relative_to(root)
    except ValueError:
        raise ValueError(
            "Episode sample path escapes its directory: {}".format(
                relative_path
            )
        )
    return result


def discover_episodes(data_dir, split):
    data_root = Path(data_dir).expanduser().resolve()
    manifests = sorted(data_root.rglob("episode.json"))
    episodes = []
    errors = []
    for manifest_path in manifests:
        try:
            with manifest_path.open("r", encoding="utf-8") as input_file:
                manifest = json.load(input_file)
            if manifest.get("format_version") != 1:
                raise ValueError("unsupported format_version")
            if manifest.get("source") != "real_robot_ros1":
                raise ValueError("source is not real_robot_ros1")
            if manifest.get("split") != split:
                continue
            if manifest.get("status") != "complete":
                # Interrupted, aborted and currently-recording episodes are
                # intentionally retained for diagnosis/recovery but must not
                # block otherwise valid training data.
                continue
            if manifest.get("action_labels") != ACTION_LABELS:
                raise ValueError("action_labels/order mismatch")
            instruction = manifest.get("instruction", "").strip()
            if not instruction:
                raise ValueError("instruction is empty")
            samples = manifest.get("samples")
            if not isinstance(samples, list) or not samples:
                raise ValueError("samples are empty")

            episode_root = manifest_path.parent
            normalized_samples = []
            for expected_index, sample in enumerate(samples):
                if sample.get("index") != expected_index:
                    raise ValueError(
                        "sample indices are not contiguous at {}".format(
                            expected_index
                        )
                    )
                action_index = int(sample["action_index"])
                action = sample["action"]
                if (
                    action_index < 0
                    or action_index >= len(ACTION_LABELS)
                    or ACTION_LABELS[action_index] != action
                ):
                    raise ValueError(
                        "invalid action at sample {}".format(expected_index)
                    )
                rgb_path = _contained_path(episode_root, sample["rgb"])
                depth_path = _contained_path(
                    episode_root, sample["depth"]
                )
                if not rgb_path.is_file() or not depth_path.is_file():
                    raise ValueError(
                        "missing RGB/depth at sample {}".format(
                            expected_index
                        )
                    )
                normalized_samples.append(
                    {
                        "rgb_path": rgb_path,
                        "depth_path": depth_path,
                        "action_index": action_index,
                    }
                )

            episodes.append(
                {
                    "episode_id": manifest["episode_id"],
                    "manifest_path": manifest_path,
                    "instruction": instruction,
                    "samples": normalized_samples,
                }
            )
        except Exception as error:
            errors.append("{}: {}".format(manifest_path, error))

    if errors:
        raise ValueError(
            "Invalid real-robot episode manifests:\n{}".format(
                "\n".join(errors)
            )
        )
    return episodes


class RealCMASequenceDataset(Dataset):
    def __init__(
        self,
        episodes,
        word_list,
        rgb_size,
        depth_size,
        instruction_length=200,
        sequence_length=8,
        sequence_stride=8,
        min_depth=0.0,
        max_depth=10.0,
    ):
        if (
            sequence_length <= 0
            or sequence_stride <= 0
            or sequence_stride > sequence_length
        ):
            raise ValueError(
                "sequence_length must be positive and sequence_stride must "
                "be in [1, sequence_length]."
            )
        self.episodes = list(episodes)
        self.word_list = list(word_list)
        self.rgb_size = tuple(rgb_size)
        self.depth_size = tuple(depth_size)
        self.instruction_length = instruction_length
        self.sequence_length = sequence_length
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.windows = []
        self.encoded_instructions = []
        self.instruction_stats = []

        for episode_index, episode in enumerate(self.episodes):
            tokens, stats = encode_instruction(
                self.word_list,
                episode["instruction"],
                self.instruction_length,
            )
            self.encoded_instructions.append(tokens)
            self.instruction_stats.append(stats)
            sample_count = len(episode["samples"])
            for start in range(0, sample_count, sequence_stride):
                end = min(start + sequence_length, sample_count)
                self.windows.append((episode_index, start, end))
                if end == sample_count:
                    break

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, item):
        episode_index, start, end = self.windows[item]
        episode = self.episodes[episode_index]
        rgb_sequence = []
        depth_sequence = []
        actions = []

        for sample in episode["samples"][start:end]:
            bgr = cv2.imread(
                str(sample["rgb_path"]), cv2.IMREAD_COLOR
            )
            if bgr is None:
                raise IOError(
                    "Failed to read {}".format(sample["rgb_path"])
                )
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            depth_m = np.load(
                str(sample["depth_path"]), allow_pickle=False
            )
            observations, _ = preprocess_rgbd(
                rgb=rgb,
                depth_m=depth_m,
                depth_encoding="32FC1",
                rgb_size=self.rgb_size,
                depth_size=self.depth_size,
                min_depth=self.min_depth,
                max_depth=self.max_depth,
            )
            rgb_sequence.append(observations["rgb"])
            depth_sequence.append(observations["depth"])
            actions.append(sample["action_index"])

        return {
            "episode_id": episode["episode_id"],
            "instruction": self.encoded_instructions[episode_index],
            "rgb": torch.from_numpy(np.stack(rgb_sequence)),
            "depth": torch.from_numpy(np.stack(depth_sequence)),
            "actions": torch.tensor(actions, dtype=torch.long),
        }

    def action_counts(self):
        counts = [0] * len(ACTION_LABELS)
        for episode in self.episodes:
            for sample in episode["samples"]:
                counts[sample["action_index"]] += 1
        return counts


def collate_real_sequences(batch):
    batch_size = len(batch)
    max_steps = max(item["actions"].numel() for item in batch)
    instruction_length = batch[0]["instruction"].numel()
    rgb_shape = tuple(batch[0]["rgb"].shape[1:])
    depth_shape = tuple(batch[0]["depth"].shape[1:])

    instructions = torch.zeros(
        batch_size, instruction_length, dtype=torch.long
    )
    rgb = torch.zeros(
        (batch_size, max_steps) + rgb_shape, dtype=torch.uint8
    )
    depth = torch.zeros(
        (batch_size, max_steps) + depth_shape, dtype=torch.float32
    )
    actions = torch.zeros(
        batch_size, max_steps, dtype=torch.long
    )
    valid = torch.zeros(
        batch_size, max_steps, dtype=torch.bool
    )
    episode_ids = []

    for batch_index, item in enumerate(batch):
        steps = item["actions"].numel()
        instructions[batch_index].copy_(item["instruction"])
        rgb[batch_index, :steps].copy_(item["rgb"])
        depth[batch_index, :steps].copy_(item["depth"])
        actions[batch_index, :steps].copy_(item["actions"])
        valid[batch_index, :steps] = True
        episode_ids.append(item["episode_id"])

    return {
        "episode_ids": episode_ids,
        "instruction": instructions,
        "rgb": rgb,
        "depth": depth,
        "actions": actions,
        "valid": valid,
    }
