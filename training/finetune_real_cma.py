#!/usr/bin/env python3

"""Fine-tune the Habitat-free CMA policy on real-robot episodes."""

import argparse
import csv
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
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


def atomic_json_save(payload, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2)
        output_file.write("\n")
        output_file.flush()
        os.fsync(output_file.fileno())
    os.replace(str(temporary), str(output_path))


def atomic_csv_save(history, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    fieldnames = [
        "epoch",
        "train_loss",
        "train_accuracy",
        "scheduled_sampling_probability",
        "model_previous_fraction",
        "wrong_model_previous_fraction",
        "free_running_loss",
        "free_running_accuracy",
        "val_loss",
        "val_accuracy",
    ]
    with temporary.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)
        output_file.flush()
        os.fsync(output_file.fileno())
    os.replace(str(temporary), str(output_path))


def save_curve(history, output_path, metric, ylabel, percentage=False):
    epochs = [record["epoch"] for record in history]
    train_values = [record["train_{}".format(metric)] for record in history]
    val_points = [
        (record["epoch"], record["val_{}".format(metric)])
        for record in history
        if record["val_{}".format(metric)] is not None
    ]
    scale = 100.0 if percentage else 1.0

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(
        epochs,
        [value * scale for value in train_values],
        marker="o",
        label="Train",
    )
    if val_points:
        axis.plot(
            [point[0] for point in val_points],
            [point[1] * scale for point in val_points],
            marker="o",
            label="Validation",
        )
    axis.set_xlabel("Epoch")
    axis.set_ylabel(ylabel)
    axis.set_title("{} Curve".format(ylabel))
    axis.grid(True, alpha=0.3)
    axis.legend()
    axis.set_xticks(epochs)
    if percentage:
        axis.set_ylim(0.0, 100.0)
    figure.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    figure.savefig(str(temporary), format="png", dpi=150)
    plt.close(figure)
    os.replace(str(temporary), str(output_path))


def save_metric_artifacts(history, output_dir):
    atomic_json_save(history, output_dir / "metrics_history.json")
    atomic_csv_save(history, output_dir / "metrics_history.csv")
    save_curve(
        history,
        output_dir / "loss_curve.png",
        metric="loss",
        ylabel="Loss",
    )
    save_curve(
        history,
        output_dir / "accuracy_curve.png",
        metric="accuracy",
        ylabel="Accuracy (%)",
        percentage=True,
    )


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


def set_trainable_modules(
    policy, train_visual_encoders, train_instruction_lstm
):
    for parameter in policy.parameters():
        parameter.requires_grad_(True)
    policy.net.instruction_encoder.embedding_layer.weight.requires_grad_(
        False
    )
    if not train_instruction_lstm:
        for parameter in (
            policy.net.instruction_encoder.encoder_rnn.parameters()
        ):
            parameter.requires_grad_(False)
    for parameter in policy.net.progress_monitor.parameters():
        parameter.requires_grad_(False)
    if not train_visual_encoders:
        for encoder in [
            policy.net.rgb_encoder,
            policy.net.depth_encoder,
        ]:
            for parameter in encoder.parameters():
                parameter.requires_grad_(False)


def set_policy_mode(
    policy, training, train_visual_encoders, train_instruction_lstm
):
    policy.train(training)
    if not train_instruction_lstm:
        policy.net.instruction_encoder.encoder_rnn.eval()
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


def scheduled_sampling_probability(
    epoch, max_probability, warmup_epochs, ramp_epochs
):
    """Return the second-stage sampling probability for one epoch."""

    if max_probability <= 0.0 or epoch <= warmup_epochs:
        return 0.0
    progress = min(
        1.0,
        float(epoch - warmup_epochs) / float(ramp_epochs),
    )
    return max_probability * progress


def select_previous_actions(
    expert_actions,
    predicted_actions,
    eligible,
    mode,
    sampling_probability,
):
    """Choose the action embedding input for the following sequence step."""

    if mode not in {"teacher", "scheduled", "free_running"}:
        raise ValueError("Unsupported previous-action mode: {}".format(mode))

    if mode == "teacher":
        use_model = torch.zeros_like(eligible)
    elif mode == "free_running":
        use_model = eligible
    elif sampling_probability <= 0.0:
        use_model = torch.zeros_like(eligible)
    elif sampling_probability >= 1.0:
        use_model = eligible
    else:
        use_model = (
            torch.rand(eligible.shape, device=eligible.device)
            < sampling_probability
        ) & eligible

    selected = torch.where(
        use_model,
        predicted_actions.detach(),
        expert_actions,
    )
    return selected, use_model


def run_loader(
    policy,
    loader,
    device,
    optimizer,
    class_weights,
    gradient_clip,
    train_visual_encoders,
    train_instruction_lstm,
    previous_action_mode="teacher",
    sampling_probability=0.0,
):
    training = optimizer is not None
    set_policy_mode(
        policy,
        training,
        train_visual_encoders,
        train_instruction_lstm,
    )
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    eligible_previous_count = 0
    model_previous_count = 0
    wrong_model_previous_count = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            instructions = batch["instruction"].to(device)
            rgb = batch["rgb"].to(device)
            depth = batch["depth"].to(device)
            actions = batch["actions"].to(device)
            executed_actions = batch.get("executed_actions", batch["actions"]).to(
                device
            )
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
                # DAgger supervision and recurrent history are intentionally
                # different: loss uses the expert label above, while the next
                # action embedding uses what the robot physically executed.
                executed_previous = executed_actions[:, step].view(
                    batch_size, 1
                )
                predicted_previous = predicted.view(batch_size, 1)
                if step + 1 < sequence_steps:
                    eligible = (
                        step_valid & valid[:, step + 1]
                    ).view(batch_size, 1)
                    previous_actions, use_model = select_previous_actions(
                        expert_actions=executed_previous,
                        predicted_actions=predicted_previous,
                        eligible=eligible,
                        mode=previous_action_mode,
                        sampling_probability=sampling_probability,
                    )
                    eligible_previous_count += int(eligible.sum().item())
                    model_previous_count += int(use_model.sum().item())
                    wrong_model_previous_count += int(
                        (
                            use_model
                            & (predicted_previous != executed_previous)
                        )
                        .sum()
                        .item()
                    )
                else:
                    previous_actions = executed_previous

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
        "eligible_previous_count": eligible_previous_count,
        "model_previous_count": model_previous_count,
        "wrong_model_previous_count": wrong_model_previous_count,
        "model_previous_fraction": (
            model_previous_count / float(eligible_previous_count)
            if eligible_previous_count
            else 0.0
        ),
        "wrong_model_previous_fraction": (
            wrong_model_previous_count / float(model_previous_count)
            if model_previous_count
            else 0.0
        ),
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
    history,
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
        "train_instruction_lstm": args.train_instruction_lstm,
        "scheduled_sampling_max_prob": (
            args.scheduled_sampling_max_prob
        ),
        "scheduled_sampling_warmup_epochs": (
            args.scheduled_sampling_warmup_epochs
        ),
        "scheduled_sampling_ramp_epochs": (
            args.scheduled_sampling_ramp_epochs
        ),
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
        "history": history,
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
        "--data-dir",
        default="training/data/real_episodes_0p4m_15deg",
    )
    parser.add_argument(
        "--checkpoint",
        default="data/checkpoints/CMA_PM_DA_Aug_robot.pth",
    )
    parser.add_argument(
        "--output-dir",
        default="training/checkpoints/real_cma_0p4m_15deg",
    )
    parser.add_argument("--resume")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--sequence-stride", type=int, default=16)
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
        "--scheduled-sampling-max-prob",
        type=float,
        default=0.0,
        help=(
            "Maximum probability of feeding the model's previous predicted "
            "action during training. Zero preserves teacher forcing."
        ),
    )
    parser.add_argument(
        "--scheduled-sampling-warmup-epochs",
        type=int,
        default=0,
        help="Number of second-stage epochs kept at pure teacher forcing.",
    )
    parser.add_argument(
        "--scheduled-sampling-ramp-epochs",
        type=int,
        default=1,
        help="Epochs used to linearly reach the maximum probability.",
    )
    parser.add_argument(
        "--train-visual-encoders", action="store_true"
    )
    parser.add_argument(
        "--train-instruction-lstm",
        action="store_true",
        help=(
            "Fine-tune the pretrained instruction BiLSTM. Its word "
            "embedding remains frozen. By default both are frozen."
        ),
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
    if not 0.0 <= args.scheduled_sampling_max_prob <= 1.0:
        raise ValueError(
            "--scheduled-sampling-max-prob must be in [0, 1]."
        )
    if args.scheduled_sampling_warmup_epochs < 0:
        raise ValueError(
            "--scheduled-sampling-warmup-epochs must be >= 0."
        )
    if args.scheduled_sampling_ramp_epochs <= 0:
        raise ValueError(
            "--scheduled-sampling-ramp-epochs must be positive."
        )
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
    set_trainable_modules(
        policy,
        args.train_visual_encoders,
        args.train_instruction_lstm,
    )
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
    history = []
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
        previous_instruction_setting = resume.get(
            "fine_tuning", {}
        ).get("train_instruction_lstm")
        if previous_instruction_setting is None:
            raise ValueError(
                "The resume checkpoint predates instruction-LSTM freeze "
                "metadata. Start a new run from --checkpoint instead."
            )
        if previous_instruction_setting != args.train_instruction_lstm:
            raise ValueError(
                "--train-instruction-lstm must match the resumed run."
            )
        previous_fine_tuning = resume.get("fine_tuning", {})
        previous_max_probability = previous_fine_tuning.get(
            "scheduled_sampling_max_prob", 0.0
        )
        if (
            previous_max_probability > 0.0
            or args.scheduled_sampling_max_prob > 0.0
        ):
            schedule_settings = {
                "scheduled_sampling_max_prob": (
                    args.scheduled_sampling_max_prob
                ),
                "scheduled_sampling_warmup_epochs": (
                    args.scheduled_sampling_warmup_epochs
                ),
                "scheduled_sampling_ramp_epochs": (
                    args.scheduled_sampling_ramp_epochs
                ),
            }
            for name, current_value in schedule_settings.items():
                previous_value = previous_fine_tuning.get(name)
                if previous_value != current_value:
                    raise ValueError(
                        "--{} must match the resumed run. Start a new "
                        "second-stage run from --checkpoint to change "
                        "scheduled sampling."
                        .format(name.replace("_", "-"))
                    )
        policy.load_state_dict(resume["state_dict"], strict=True)
        optimizer.load_state_dict(resume["optimizer_state"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
        start_epoch = int(resume["epoch"]) + 1
        best_val_loss = float(resume["best_val_loss"])
        history = list(resume.get("history", []))
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
    scheduled_sampling_enabled = (
        args.scheduled_sampling_max_prob > 0.0
    )
    free_running_loader = None
    free_running_split = None
    if scheduled_sampling_enabled:
        free_running_loader = val_loader
        free_running_split = "val"
        if free_running_loader is None:
            free_running_loader = make_loader(
                train_dataset, args, shuffle=False
            )
            free_running_split = "train"
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
        "train_visual_encoders": args.train_visual_encoders,
        "train_instruction_lstm": args.train_instruction_lstm,
        "initial_checkpoint": str(checkpoint_path),
        "scheduled_sampling_max_prob": (
            args.scheduled_sampling_max_prob
        ),
        "scheduled_sampling_warmup_epochs": (
            args.scheduled_sampling_warmup_epochs
        ),
        "scheduled_sampling_ramp_epochs": (
            args.scheduled_sampling_ramp_epochs
        ),
        "free_running_selection_split": free_running_split,
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
        if scheduled_sampling_enabled:
            print(
                "WARNING: no validation episodes found; best_robot.pth will "
                "be selected by train-set open-loop free-running loss."
            )
        else:
            print(
                "WARNING: no validation episodes found; best_robot.pth will "
                "be selected by train loss."
            )

    for epoch in range(start_epoch, args.epochs + 1):
        sampling_probability = scheduled_sampling_probability(
            epoch=epoch,
            max_probability=args.scheduled_sampling_max_prob,
            warmup_epochs=args.scheduled_sampling_warmup_epochs,
            ramp_epochs=args.scheduled_sampling_ramp_epochs,
        )
        train_metrics = run_loader(
            policy=policy,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            class_weights=class_weights,
            gradient_clip=args.gradient_clip,
            train_visual_encoders=args.train_visual_encoders,
            train_instruction_lstm=args.train_instruction_lstm,
            previous_action_mode="scheduled",
            sampling_probability=sampling_probability,
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
                train_instruction_lstm=args.train_instruction_lstm,
            )
        free_running_metrics = None
        if scheduled_sampling_enabled:
            free_running_metrics = run_loader(
                policy=policy,
                loader=free_running_loader,
                device=device,
                optimizer=None,
                class_weights=class_weights,
                gradient_clip=args.gradient_clip,
                train_visual_encoders=args.train_visual_encoders,
                train_instruction_lstm=args.train_instruction_lstm,
                previous_action_mode="free_running",
                sampling_probability=1.0,
            )
            score = free_running_metrics["loss"]
        elif val_metrics is not None:
            score = val_metrics["loss"]
        else:
            score = train_metrics["loss"]

        is_best = score < best_val_loss
        if is_best:
            best_val_loss = score
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "scheduled_sampling_probability": sampling_probability,
                "model_previous_fraction": train_metrics[
                    "model_previous_fraction"
                ],
                "wrong_model_previous_fraction": train_metrics[
                    "wrong_model_previous_fraction"
                ],
                "free_running_loss": (
                    free_running_metrics["loss"]
                    if free_running_metrics is not None
                    else None
                ),
                "free_running_accuracy": (
                    free_running_metrics["accuracy"]
                    if free_running_metrics is not None
                    else None
                ),
                "val_loss": (
                    val_metrics["loss"]
                    if val_metrics is not None
                    else None
                ),
                "val_accuracy": (
                    val_metrics["accuracy"]
                    if val_metrics is not None
                    else None
                ),
            }
        )
        save_training_outputs(
            policy=policy,
            optimizer=optimizer,
            base_checkpoint=checkpoint,
            output_dir=output_dir,
            epoch=epoch,
            best_val_loss=best_val_loss,
            history=history,
            args=args,
            is_best=is_best,
        )
        save_metric_artifacts(history, output_dir)
        print(
            "epoch={} train_loss={:.6f} train_accuracy={:.2%} "
            "ss_prob={:.2%} model_prev={:.2%} wrong_model_prev={:.2%}{}{}{}"
            .format(
                epoch,
                train_metrics["loss"],
                train_metrics["accuracy"],
                sampling_probability,
                train_metrics["model_previous_fraction"],
                train_metrics["wrong_model_previous_fraction"],
                (
                    " free_running_loss={:.6f} free_running_accuracy={:.2%}"
                    .format(
                        free_running_metrics["loss"],
                        free_running_metrics["accuracy"],
                    )
                    if free_running_metrics is not None
                    else ""
                ),
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
    print("Loss curve: {}".format(output_dir / "loss_curve.png"))
    print(
        "Accuracy curve: {}".format(
            output_dir / "accuracy_curve.png"
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
