#!/usr/bin/env python3

"""Inspect a VLN-CE Real CMA checkpoint without starting ROS or inference."""

import argparse
import json
import sys
import warnings
from collections import OrderedDict
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = (
    "training/checkpoints/real_cma_0p4m_30deg/best_robot.pth"
)


def resolve_checkpoint(path_text):
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def load_checkpoint(path):
    if not path.is_file():
        raise FileNotFoundError("Checkpoint not found: {}".format(path))
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="You are using `torch.load` with `weights_only=False`",
            category=FutureWarning,
        )
        checkpoint = torch.load(str(path), map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint root must be a dictionary.")
    return checkpoint


def checkpoint_metadata(checkpoint):
    nested = checkpoint.get("robot_metadata")
    if isinstance(nested, dict):
        return nested
    return checkpoint


def state_dict_from_checkpoint(checkpoint):
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, (dict, OrderedDict)):
        raise ValueError("Checkpoint does not contain a state_dict.")
    invalid = [
        name for name, value in state_dict.items()
        if not isinstance(name, str) or not torch.is_tensor(value)
    ]
    if invalid:
        raise ValueError("state_dict contains invalid entries.")
    return state_dict


def human_size(byte_count):
    value = float(byte_count)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024.0 or unit == "GiB":
            return "{:.2f} {}".format(value, unit)
        value /= 1024.0
    return "{} B".format(byte_count)


def print_json_value(label, value):
    print(label)
    print(json.dumps(value, indent=2, ensure_ascii=False, default=str))


def module_name(tensor_name):
    parts = tensor_name.split(".")
    if len(parts) >= 2 and parts[0] == "net":
        return ".".join(parts[:2])
    return parts[0]


def summarize_modules(state_dict):
    modules = OrderedDict()
    for name, tensor in state_dict.items():
        group = module_name(name)
        summary = modules.setdefault(
            group, {"tensors": 0, "parameters": 0}
        )
        summary["tensors"] += 1
        summary["parameters"] += tensor.numel()
    return modules


def print_summary(path, checkpoint, metadata, state_dict):
    word_list = metadata.get("word_list")
    tensor_count = len(state_dict)
    state_value_count = sum(tensor.numel() for tensor in state_dict.values())
    checkpoint_type = (
        "training/resume"
        if checkpoint.get("training_format_version") is not None
        else "robot/inference"
    )

    print("=== Checkpoint file ===")
    print("path: {}".format(path))
    print("size: {}".format(human_size(path.stat().st_size)))
    print("type: {}".format(checkpoint_type))
    print("top-level keys: {}".format(", ".join(checkpoint.keys())))
    print()

    print("=== Model metadata ===")
    print("format_version: {}".format(metadata.get("format_version")))
    print("model_name: {}".format(metadata.get("model_name")))
    print(
        "source_checkpoint_sha256: {}".format(
            metadata.get("source_checkpoint_sha256")
        )
    )
    print("rgb_size: {}".format(metadata.get("rgb_size")))
    print("depth_size: {}".format(metadata.get("depth_size")))
    print("action_labels: {}".format(metadata.get("action_labels")))
    print(
        "vocabulary_size: {}".format(
            len(word_list) if isinstance(word_list, list) else "missing"
        )
    )
    print("tensor_count: {}".format(tensor_count))
    print("state_dict_value_count: {:,}".format(state_value_count))
    print(
        "state_dict_memory_float32: {}".format(
            human_size(state_value_count * 4)
        )
    )
    print()

    fine_tuning = checkpoint.get("fine_tuning")
    if fine_tuning is not None:
        print_json_value("=== Fine-tuning metadata ===", fine_tuning)
        print()
    if checkpoint.get("epoch") is not None:
        print("resume_epoch: {}".format(checkpoint.get("epoch")))
        print("best_loss: {}".format(checkpoint.get("best_val_loss")))
        print()

    print("=== Parameter groups ===")
    print("{:<36} {:>10} {:>16}".format("module", "tensors", "values"))
    for name, summary in summarize_modules(state_dict).items():
        print(
            "{:<36} {:>10} {:>16,}".format(
                name, summary["tensors"], summary["parameters"]
            )
        )


def print_vocabulary(word_list):
    if not isinstance(word_list, list):
        raise ValueError("Checkpoint does not contain a word_list.")
    print("\n=== Full R2R vocabulary ===")
    for index, word in enumerate(word_list):
        print("{:04d}  {}".format(index, word))


def find_words(word_list, queries):
    if not isinstance(word_list, list):
        raise ValueError("Checkpoint does not contain a word_list.")
    index_by_word = {word: index for index, word in enumerate(word_list)}
    print("\n=== Vocabulary lookup ===")
    for query in queries:
        index = index_by_word.get(query)
        if index is None:
            index = index_by_word.get(query.lower())
        if index is None:
            print("{}: not found -> <unk> index {}".format(
                query, index_by_word.get("<unk>")
            ))
        else:
            print("{}: index {}".format(query, index))


def print_tensor_list(state_dict):
    print("\n=== state_dict tensors ===")
    print("{:<78} {:<24} {:>14}".format("name", "shape", "values"))
    for name, tensor in state_dict.items():
        print(
            "{:<78} {:<24} {:>14,}".format(
                name, str(tuple(tensor.shape)), tensor.numel()
            )
        )


def print_tensor_details(state_dict, tensor_name, preview_count):
    if tensor_name not in state_dict:
        matches = [name for name in state_dict if tensor_name in name]
        message = "Tensor not found: {}".format(tensor_name)
        if matches:
            message += "\nPossible matches:\n  " + "\n  ".join(matches[:20])
        raise KeyError(message)

    tensor = state_dict[tensor_name].detach().cpu()
    flattened = tensor.reshape(-1)
    preview = flattened[:preview_count].tolist()
    print("\n=== Tensor details ===")
    print("name: {}".format(tensor_name))
    print("shape: {}".format(tuple(tensor.shape)))
    print("dtype: {}".format(tensor.dtype))
    print("values: {:,}".format(tensor.numel()))
    if tensor.numel() and (tensor.is_floating_point() or tensor.is_complex()):
        values = tensor.float()
        print("min: {:.8g}".format(values.min().item()))
        print("max: {:.8g}".format(values.max().item()))
        print("mean: {:.8g}".format(values.mean().item()))
        print("std: {:.8g}".format(values.std(unbiased=False).item()))
    print("first_{}_values: {}".format(len(preview), preview))


def build_parser():
    parser = argparse.ArgumentParser(
        description="Inspect metadata, vocabulary and tensors in a CMA pth file."
    )
    parser.add_argument(
        "checkpoint",
        nargs="?",
        default=DEFAULT_CHECKPOINT,
        help="Checkpoint path, relative paths are resolved from repository root.",
    )
    parser.add_argument(
        "--show-vocab",
        action="store_true",
        help="Print the complete indexed R2R vocabulary.",
    )
    parser.add_argument(
        "--find-word",
        action="append",
        default=[],
        metavar="WORD",
        help="Find a word's token index; may be specified more than once.",
    )
    parser.add_argument(
        "--list-tensors",
        action="store_true",
        help="Print every state_dict tensor name, shape and parameter count.",
    )
    parser.add_argument(
        "--tensor",
        metavar="NAME",
        help="Show statistics and a short value preview for one tensor.",
    )
    parser.add_argument(
        "--preview-count",
        type=int,
        default=16,
        help="Number of values shown by --tensor (default: 16).",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.preview_count < 0:
        parser.error("--preview-count must be >= 0")

    try:
        path = resolve_checkpoint(args.checkpoint)
        checkpoint = load_checkpoint(path)
        metadata = checkpoint_metadata(checkpoint)
        state_dict = state_dict_from_checkpoint(checkpoint)
        word_list = metadata.get("word_list")
        print_summary(path, checkpoint, metadata, state_dict)
        if args.find_word:
            find_words(word_list, args.find_word)
        if args.show_vocab:
            print_vocabulary(word_list)
        if args.list_tensors:
            print_tensor_list(state_dict)
        if args.tensor:
            print_tensor_details(
                state_dict, args.tensor, args.preview_count
            )
    except (FileNotFoundError, ValueError, KeyError) as error:
        print("ERROR: {}".format(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
