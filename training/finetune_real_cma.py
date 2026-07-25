#!/usr/bin/env python3

"""Fine-tune the Habitat-free CMA policy on real-robot episodes."""

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


REAL_ROOT = Path(__file__).resolve().parents[1]
if str(REAL_ROOT) not in sys.path:
    sys.path.insert(0, str(REAL_ROOT))

from training.real_dataset import (  # noqa: E402
    ACTION_LABELS,
    RealCMASequenceDataset,
    collate_real_sequences,
    discover_episodes,
)
from vlnce_real.model import (  # noqa: E402
    CMAPolicy,
    STATE_HIDDEN_SIZE,
)


def resolve_real_path(path_text):
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = REAL_ROOT / path
    return path.resolve()


def atomic_torch_save(payload, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    torch.save(payload, str(temporary))
    os.replace(str(temporary), str(output_path))


def checkpoint_metadata(checkpoint):
    return {
        key: value
        for key, value in checkpoint.items()
        if key != "state_dict"
    }


def cpu_state_dict(module):
    return {
        key: value.detach().cpu()
        for key, value in module.state_dict().items()
    }


def set_trainable_modules(policy, train_visual_encoders):
    for parameter in policy.parameters():
        parameter.requires_grad_(True)
    policy.net.instruction_encoder.embedding_layer.weight.requires_grad_(
        False
    )
    for parameter in policy.net.progress_monitor.parameters():
        parameter.requires_grad_(False)
    if not train_visual_encoders:
        for encoder in [
            policy.net.rgb_encoder,
            policy.net.depth_encoder,
        ]:
            for parameter in encoder.parameters():
                parameter.requires_grad_(False)


def set_policy_mode(policy, training, train_visual_encoders):
    policy.train(training)
    if not train_visual_encoders:
        policy.net.rgb_encoder.eval()
        policy.net.depth_encoder.eval()


def build_class_weights(counts, device):
    total = float(sum(counts))
    weights = []
    for count in counts:
        weights.append(
            total / (len(counts) * count) if count > 0 else 0.0
        )
    return torch.tensor(weights, dtype=torch.float32, device=device)


def run_loader(
    policy,
    loader,
    device,
    optimizer,
    class_weights,
    gradient_clip,
    train_visual_encoders,
):
    training = optimizer is not None
    set_policy_mode(policy, training, train_visual_encoders)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            instructions = batch["instruction"].to(device)
            rgb = batch["rgb"].to(device)
            depth = batch["depth"].to(device)
            actions = batch["actions"].to(device)
            valid = batch["valid"].to(device)
            batch_size, sequence_steps = actions.shape

            rnn_states = torch.zeros(
                batch_size,
                policy.net.num_recurrent_layers,
                STATE_HIDDEN_SIZE,
                device=device,
            )
            previous_actions = torch.zeros(
                batch_size, 1, dtype=torch.long, device=device
            )
            loss_sum = torch.zeros((), device=device)
            valid_count = 0

            if training:
                optimizer.zero_grad()

            for step in range(sequence_steps):
                step_valid = valid[:, step]
                masks = (
                    torch.zeros(
                        batch_size, 1, dtype=torch.bool, device=device
                    )
                    if step == 0
                    else (
                        valid[:, step - 1] & step_valid
                    ).view(batch_size, 1)
                )
                observations = {
                    "instruction": instructions,
                    "rgb": rgb[:, step],
                    "depth": depth[:, step],
                }
                features, rnn_states = policy.net(
                    observations,
                    rnn_states,
                    previous_actions,
                    masks,
                    detach_state=not training,
                )
                logits = policy.action_distribution.linear(features)
                step_losses = F.cross_entropy(
                    logits,
                    actions[:, step],
                    weight=class_weights,
                    reduction="none",
                )
                loss_sum = loss_sum + step_losses[step_valid].sum()
                valid_count += int(step_valid.sum().item())

                predicted = logits.argmax(dim=1)
                total_correct += int(
                    (predicted[step_valid] == actions[:, step][step_valid])
                    .sum()
                    .item()
                )
                previous_actions = actions[:, step].view(
                    batch_size, 1
                )

            if valid_count == 0:
                continue
            loss = loss_sum / float(valid_count)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [
                        parameter
                        for parameter in policy.parameters()
                        if parameter.requires_grad
                    ],
                    gradient_clip,
                )
                optimizer.step()

            total_loss += float(loss_sum.detach().item())
            total_samples += valid_count

    if total_samples == 0:
        raise ValueError("The data loader produced no valid action samples.")
    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / float(total_samples),
        "samples": total_samples,
    }


def make_dataset(args, checkpoint, split):
    episodes = discover_episodes(args.data_dir, split)
    if not episodes:
        return None
    return RealCMASequenceDataset(
        episodes=episodes,
        word_list=checkpoint["word_list"],
        rgb_size=checkpoint["rgb_size"],
        depth_size=checkpoint["depth_size"],
        instruction_length=args.instruction_length,
        sequence_length=args.sequence_length,
        sequence_stride=args.sequence_stride,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
    )


def make_loader(dataset, args, shuffle):
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        collate_fn=collate_real_sequences,
        pin_memory=torch.cuda.is_available(),
    )


def save_training_outputs(
    policy,
    optimizer,
    base_checkpoint,
    output_dir,
    epoch,
    best_val_loss,
    args,
    is_best,
):
    state_dict = cpu_state_dict(policy)
    fine_tuning = {
        "type": "real_robot_behavior_cloning",
        "epoch": epoch,
        "saved_at": datetime.now().astimezone().isoformat(),
        "data_dir": str(Path(args.data_dir).expanduser().resolve()),
        "sequence_length": args.sequence_length,
        "sequence_stride": args.sequence_stride,
        "min_depth": args.min_depth,
        "max_depth": args.max_depth,
        "train_visual_encoders": args.train_visual_encoders,
    }
    robot_checkpoint = checkpoint_metadata(base_checkpoint)
    robot_checkpoint["state_dict"] = state_dict
    robot_checkpoint["fine_tuning"] = fine_tuning
    atomic_torch_save(
        robot_checkpoint, output_dir / "latest_robot.pth"
    )
    if is_best:
        atomic_torch_save(
            robot_checkpoint, output_dir / "best_robot.pth"
        )

    training_checkpoint = {
        "training_format_version": 1,
        "state_dict": state_dict,
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "robot_metadata": checkpoint_metadata(base_checkpoint),
        "fine_tuning": fine_tuning,
    }
    atomic_torch_save(
        training_checkpoint, output_dir / "latest_training.pth"
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Offline behavior-cloning fine-tuning of the standalone CMA "
            "policy using real ROS RGB-D/action episodes."
        )
    )
    parser.add_argument(
        "--data-dir", default="training/data/real_episodes"
    )
    parser.add_argument(
        "--checkpoint",
        default="data/checkpoints/CMA_PM_DA_Aug_robot.pth",
    )
    parser.add_argument(
        "--output-dir", default="training/checkpoints/real_cma"
    )
    parser.add_argument("--resume")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--sequence-stride", type=int, default=8)
    parser.add_argument("--instruction-length", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--min-depth", type=float, default=0.0)
    parser.add_argument("--max-depth", type=float, default=10.0)
    parser.add_argument("--class-balance", action="store_true")
    parser.add_argument(
        "--train-visual-encoders", action="store_true"
    )
    parser.add_argument("--cpu", action="store_true")
    return parser


def validate_args(args):
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if (
        args.sequence_length <= 0
        or args.sequence_stride <= 0
        or args.sequence_stride > args.sequence_length
    ):
        raise ValueError(
            "--sequence-length must be positive and --sequence-stride must "
            "be in [1, sequence-length]."
        )
    if args.instruction_length <= 0:
        raise ValueError("--instruction-length must be positive.")
    if args.learning_rate <= 0.0:
        raise ValueError("--learning-rate must be positive.")
    if args.gradient_clip <= 0.0:
        raise ValueError("--gradient-clip must be positive.")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0.")
    if not args.max_depth > args.min_depth:
        raise ValueError("--max-depth must exceed --min-depth.")


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        validate_args(args)
    except ValueError as error:
        parser.error(str(error))

    args.data_dir = str(resolve_real_path(args.data_dir))
    checkpoint_path = resolve_real_path(args.checkpoint)
    output_dir = resolve_real_path(args.output_dir)
    device = torch.device(
        "cpu"
        if args.cpu or not torch.cuda.is_available()
        else "cuda:0"
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    if checkpoint.get("format_version") != 1:
        raise ValueError("Expected robot checkpoint format version 1.")
    if checkpoint.get("action_labels") != ACTION_LABELS:
        raise ValueError("Checkpoint action labels/order do not match.")
    if not isinstance(checkpoint.get("word_list"), list):
        raise ValueError("Checkpoint word_list is missing.")

    policy = CMAPolicy(
        vocab_size=len(checkpoint["word_list"]),
        num_actions=len(ACTION_LABELS),
    )
    policy.load_state_dict(checkpoint["state_dict"], strict=True)
    set_trainable_modules(policy, args.train_visual_encoders)
    policy.to(device)
    optimizer = torch.optim.Adam(
        [
            parameter
            for parameter in policy.parameters()
            if parameter.requires_grad
        ],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    start_epoch = 1
    best_val_loss = float("inf")
    if args.resume:
        resume_path = resolve_real_path(args.resume)
        resume = torch.load(str(resume_path), map_location="cpu")
        if resume.get("training_format_version") != 1:
            raise ValueError("Unsupported training resume checkpoint.")
        resume_metadata = resume.get("robot_metadata", {})
        if (
            resume_metadata.get("action_labels")
            != checkpoint["action_labels"]
            or resume_metadata.get("word_list")
            != checkpoint["word_list"]
        ):
            raise ValueError(
                "Resume checkpoint vocabulary/action metadata does not "
                "match --checkpoint."
            )
        previous_visual_setting = resume.get("fine_tuning", {}).get(
            "train_visual_encoders"
        )
        if previous_visual_setting != args.train_visual_encoders:
            raise ValueError(
                "--train-visual-encoders must match the resumed run."
            )
        policy.load_state_dict(resume["state_dict"], strict=True)
        optimizer.load_state_dict(resume["optimizer_state"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
        start_epoch = int(resume["epoch"]) + 1
        best_val_loss = float(resume["best_val_loss"])
        if start_epoch > args.epochs:
            raise ValueError(
                "--epochs is the final epoch number; set it to at least {} "
                "when resuming.".format(start_epoch)
            )

    train_dataset = make_dataset(args, checkpoint, "train")
    if train_dataset is None:
        raise ValueError(
            "No complete train episodes found under {}.".format(
                args.data_dir
            )
        )
    val_dataset = make_dataset(args, checkpoint, "val")
    train_loader = make_loader(train_dataset, args, shuffle=True)
    val_loader = (
        make_loader(val_dataset, args, shuffle=False)
        if val_dataset is not None
        else None
    )
    counts = train_dataset.action_counts()
    class_weights = (
        build_class_weights(counts, device)
        if args.class_balance
        else None
    )

    run_summary = {
        "device": str(device),
        "train_episodes": len(train_dataset.episodes),
        "train_windows": len(train_dataset),
        "val_episodes": (
            len(val_dataset.episodes) if val_dataset is not None else 0
        ),
        "val_windows": len(val_dataset) if val_dataset is not None else 0,
        "action_counts": dict(zip(ACTION_LABELS, counts)),
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in policy.parameters()
            if parameter.requires_grad
        ),
        "all_parameters": sum(
            parameter.numel() for parameter in policy.parameters()
        ),
    }
    print(json.dumps(run_summary, indent=2, sort_keys=True))
    if val_loader is None:
        print(
            "WARNING: no validation episodes found; latest_robot.pth will "
            "also be used as best_robot.pth."
        )

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = run_loader(
            policy=policy,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            class_weights=class_weights,
            gradient_clip=args.gradient_clip,
            train_visual_encoders=args.train_visual_encoders,
        )
        val_metrics = None
        if val_loader is not None:
            val_metrics = run_loader(
                policy=policy,
                loader=val_loader,
                device=device,
                optimizer=None,
                class_weights=class_weights,
                gradient_clip=args.gradient_clip,
                train_visual_encoders=args.train_visual_encoders,
            )
            score = val_metrics["loss"]
        else:
            score = train_metrics["loss"]

        is_best = score < best_val_loss
        if is_best:
            best_val_loss = score
        save_training_outputs(
            policy=policy,
            optimizer=optimizer,
            base_checkpoint=checkpoint,
            output_dir=output_dir,
            epoch=epoch,
            best_val_loss=best_val_loss,
            args=args,
            is_best=is_best,
        )
        print(
            "epoch={} train_loss={:.6f} train_accuracy={:.2%}{}{}".format(
                epoch,
                train_metrics["loss"],
                train_metrics["accuracy"],
                (
                    " val_loss={:.6f}".format(val_metrics["loss"])
                    if val_metrics is not None
                    else ""
                ),
                (
                    " val_accuracy={:.2%}".format(
                        val_metrics["accuracy"]
                    )
                    if val_metrics is not None
                    else ""
                ),
            )
        )

    print("Robot checkpoint: {}".format(output_dir / "best_robot.pth"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
