#!/usr/bin/env python3

"""Record and visualize per-action VLN inference and execution timings."""

import argparse
import csv
import glob
import json
import os
import queue
import sys
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def add_ros_python_paths():
    candidates = ["/usr/lib/python3/dist-packages"]
    ros_distro = os.environ.get("ROS_DISTRO")
    if ros_distro:
        candidates.append(
            "/opt/ros/{}/lib/python3/dist-packages".format(ros_distro)
        )
    else:
        candidates.extend(
            sorted(glob.glob("/opt/ros/*/lib/python3/dist-packages"))
        )
    for candidate in candidates:
        if os.path.isdir(candidate) and candidate not in sys.path:
            sys.path.append(candidate)


add_ros_python_paths()

import rospy  # noqa: E402
from std_msgs.msg import String  # noqa: E402

from vlnce_real.action_protocol import (  # noqa: E402
    decode_action_command,
    decode_action_result,
    decode_inference_metrics,
)


CSV_FIELDS = [
    "sequence",
    "action",
    "action_count",
    "status",
    "reason",
    "device",
    "first_inference",
    "previous_action_end_to_start_ms",
    "result_to_inference_start_ms",
    "fresh_rgbd_wait_ms",
    "rgbd_queue_ms",
    "image_conversion_ms",
    "preprocess_ms",
    "model_ms",
    "inference_total_ms",
    "execution_seconds",
    "command_to_result_seconds",
    "control_mode",
    "target_value",
    "progress_value",
    "target_unit",
    "rgb_depth_delta_ms",
    "invalid_depth_percent",
    "command_received_at",
    "result_received_at",
]


def _timestamp_text():
    return datetime.now().astimezone().isoformat()


def _record_key(sequence, fallback):
    if sequence is not None:
        return "sequence:{}".format(sequence)
    return fallback


class ActionMetricsRecorder:
    """Merge three ROS event streams and persist completed action records."""

    def __init__(self, output_directory, history_size):
        self.history_size = history_size
        self.events = queue.Queue()
        self.records = OrderedDict()
        self.written_keys = set()
        self.legacy_counter = 0
        self.latest_legacy_key = None
        self.changed = True
        self.closed = False

        output_path = Path(output_directory).expanduser()
        if not output_path.is_absolute():
            output_path = REPO_ROOT / output_path
        output_path.mkdir(parents=True, exist_ok=True)
        session_name = datetime.now().strftime("action_metrics_%Y%m%d_%H%M%S")
        self.csv_path = output_path / (session_name + ".csv")
        self.jsonl_path = output_path / (session_name + ".jsonl")
        self.csv_file = self.csv_path.open("w", newline="", encoding="utf-8")
        self.jsonl_file = self.jsonl_path.open("w", encoding="utf-8")
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=CSV_FIELDS)
        self.csv_writer.writeheader()
        self.csv_file.flush()

    def _new_record(self, sequence=None, action=""):
        return {
            "sequence": sequence,
            "action": action,
            "action_count": None,
            "status": "pending",
            "reason": "",
            "device": "",
            "first_inference": False,
            "previous_action_end_to_start_ms": None,
            "result_to_inference_start_ms": None,
            "fresh_rgbd_wait_ms": None,
            "rgbd_queue_ms": None,
            "image_conversion_ms": None,
            "preprocess_ms": None,
            "model_ms": None,
            "inference_total_ms": None,
            "execution_seconds": None,
            "command_to_result_seconds": None,
            "control_mode": "",
            "target_value": None,
            "progress_value": None,
            "target_unit": "",
            "rgb_depth_delta_ms": None,
            "invalid_depth_percent": None,
            "command_received_at": "",
            "result_received_at": "",
            "_command_monotonic": None,
            "_result_monotonic": None,
        }

    def command_callback(self, message):
        try:
            command = decode_action_command(message.data)
        except ValueError as error:
            rospy.logwarn("Action monitor ignored malformed command: %s", error)
            return
        self.events.put(("command", command, time.monotonic(), _timestamp_text()))

    def metrics_callback(self, message):
        try:
            metrics = decode_inference_metrics(message.data)
        except ValueError as error:
            rospy.logwarn("Action monitor ignored malformed metrics: %s", error)
            return
        self.events.put(("metrics", metrics, time.monotonic(), _timestamp_text()))

    def result_callback(self, message):
        try:
            result = decode_action_result(message.data)
        except ValueError as error:
            rospy.logwarn("Action monitor ignored malformed result: %s", error)
            return
        self.events.put(("result", result, time.monotonic(), _timestamp_text()))

    def _get_record(self, key, sequence=None, action=""):
        record = self.records.get(key)
        if record is None:
            record = self._new_record(sequence=sequence, action=action)
            self.records[key] = record
        elif action and not record["action"]:
            record["action"] = action
        return record

    def _apply_command(self, command, received_monotonic, received_at):
        if command.sequence is None:
            self.legacy_counter += 1
            key = "legacy:{}".format(self.legacy_counter)
            self.latest_legacy_key = key
        else:
            key = _record_key(command.sequence, "")
        record = self._get_record(key, command.sequence, command.action)
        record["command_received_at"] = received_at
        record["_command_monotonic"] = received_monotonic
        if record["_result_monotonic"] is not None:
            record["command_to_result_seconds"] = max(
                0.0,
                record["_result_monotonic"] - received_monotonic,
            )

    def _apply_metrics(self, metrics):
        key = _record_key(
            metrics.sequence,
            "legacy:{}".format(metrics.action_count),
        )
        record = self._get_record(key, metrics.sequence, metrics.action)
        record.update(
            {
                "action_count": metrics.action_count,
                "device": metrics.device,
                "first_inference": metrics.first_inference,
                "result_to_inference_start_ms": (
                    None
                    if metrics.result_to_inference_start_seconds is None
                    else 1000.0
                    * metrics.result_to_inference_start_seconds
                ),
                "fresh_rgbd_wait_ms": (
                    None
                    if metrics.fresh_rgbd_wait_seconds is None
                    else 1000.0 * metrics.fresh_rgbd_wait_seconds
                ),
                "rgbd_queue_ms": (
                    None
                    if metrics.rgbd_queue_seconds is None
                    else 1000.0 * metrics.rgbd_queue_seconds
                ),
                "image_conversion_ms": (
                    1000.0 * metrics.image_conversion_seconds
                ),
                "preprocess_ms": 1000.0 * metrics.preprocess_seconds,
                "model_ms": 1000.0 * metrics.model_seconds,
                "inference_total_ms": 1000.0 * metrics.total_seconds,
                "rgb_depth_delta_ms": (
                    1000.0 * metrics.rgb_depth_delta_seconds
                ),
                "invalid_depth_percent": (
                    100.0 * metrics.invalid_depth_fraction
                ),
            }
        )

    def _apply_result(self, result, received_monotonic, received_at):
        if result.sequence is None and self.latest_legacy_key is not None:
            key = self.latest_legacy_key
        else:
            key = _record_key(result.sequence, "result:legacy")
        record = self._get_record(key, result.sequence, result.action)
        record.update(
            {
                "status": result.status,
                "reason": result.reason,
                "execution_seconds": result.execution_seconds,
                "previous_action_end_to_start_ms": (
                    None
                    if result.previous_action_end_to_start_seconds is None
                    else 1000.0
                    * result.previous_action_end_to_start_seconds
                ),
                "control_mode": result.control_mode or "",
                "target_value": result.target_value,
                "progress_value": result.progress_value,
                "target_unit": result.target_unit or "",
                "result_received_at": received_at,
                "_result_monotonic": received_monotonic,
            }
        )
        if record["_command_monotonic"] is not None:
            record["command_to_result_seconds"] = max(
                0.0,
                received_monotonic - record["_command_monotonic"],
            )

    def process_pending(self):
        if self.closed:
            return False
        processed = False
        while True:
            try:
                event_type, payload, monotonic_time, wall_time = (
                    self.events.get_nowait()
                )
            except queue.Empty:
                break
            processed = True
            if event_type == "command":
                self._apply_command(payload, monotonic_time, wall_time)
            elif event_type == "metrics":
                self._apply_metrics(payload)
            else:
                self._apply_result(payload, monotonic_time, wall_time)
        if processed:
            self.changed = True
            self._trim_history()
        self._write_completed_records()
        return processed

    def _write_completed_records(self):
        for key, record in self.records.items():
            if key in self.written_keys or not record["result_received_at"]:
                continue
            # In the normal pipeline metrics are published before the command.
            # Keep waiting if cross-topic delivery reorders those messages.
            missing_command = (
                record["sequence"] is not None
                and not record["command_received_at"]
            )
            if record["inference_total_ms"] is None or missing_command:
                result_age = time.monotonic() - record["_result_monotonic"]
                if result_age < 1.0:
                    continue
            public_record = {
                field: record.get(field) for field in CSV_FIELDS
            }
            self.csv_writer.writerow(public_record)
            self.csv_file.flush()
            self.jsonl_file.write(
                json.dumps(public_record, ensure_ascii=False, sort_keys=True)
                + "\n"
            )
            self.jsonl_file.flush()
            self.written_keys.add(key)

    def _trim_history(self):
        if self.history_size == 0:
            return
        while len(self.records) > self.history_size:
            first_key = next(iter(self.records))
            if first_key not in self.written_keys:
                break
            self.records.pop(first_key)

    def public_records(self):
        return [
            {field: record.get(field) for field in CSV_FIELDS}
            for record in self.records.values()
        ]

    def close(self):
        if self.closed:
            return
        self.process_pending()
        self._write_completed_records()
        self.csv_file.close()
        self.jsonl_file.close()
        self.closed = True


def _display_value(value, precision=2, suffix=""):
    if value is None:
        return "-"
    return ("{:.%df}{}" % precision).format(float(value), suffix)


class ActionMonitorGui:
    def __init__(self, recorder):
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.recorder = recorder
        self.selected_action_id = None
        self.episode_hit_regions = []
        self.root = tk.Tk()
        self.root.title("VLN Action Timing Monitor")
        self.root.geometry("1400x850")
        self.root.minsize(1100, 700)
        self.root.protocol("WM_DELETE_WINDOW", self._close)

        self.summary_variables = {
            name: tk.StringVar(value="-")
            for name in (
                "status",
                "actions",
                "last_action",
                "latest_execution_gap",
                "avg_inference",
                "avg_execution",
                "avg_gap",
                "device",
            )
        }
        self._build_layout()
        self.root.after(100, self._refresh)

    def _build_layout(self):
        ttk = self.ttk
        root = self.root
        summary = ttk.Frame(root, padding=10)
        summary.pack(fill="x")
        cards = [
            ("Pipeline", "status"),
            ("Actions", "actions"),
            ("Last action", "last_action"),
            ("END→NEXT START", "latest_execution_gap"),
            ("Avg inference", "avg_inference"),
            ("Avg execution", "avg_execution"),
            ("Avg result→infer", "avg_gap"),
            ("Device", "device"),
        ]
        for index, (title, key) in enumerate(cards):
            card = ttk.LabelFrame(summary, text=title, padding=(12, 8))
            card.grid(row=0, column=index, padx=4, sticky="nsew")
            label_options = {
                "textvariable": self.summary_variables[key],
                "font": (
                    "TkDefaultFont",
                    14 if key == "latest_execution_gap" else 11,
                    "bold",
                ),
            }
            if key == "latest_execution_gap":
                label_options["foreground"] = "#000000"
            ttk.Label(card, **label_options).pack()
            summary.columnconfigure(index, weight=1)

        table_frame = ttk.LabelFrame(root, text="Action records", padding=8)
        table_frame.pack(fill="both", expand=True, padx=10, pady=(0, 8))
        columns = (
            "seq",
            "action",
            "mode",
            "execution_gap",
            "gap",
            "convert",
            "preprocess",
            "model",
            "inference",
            "execution",
            "cycle",
            "progress",
            "status",
            "reason",
        )
        self.table = ttk.Treeview(
            table_frame,
            columns=columns,
            show="headings",
            height=12,
        )
        headings = {
            "seq": "Seq",
            "action": "Action",
            "mode": "Control mode",
            "execution_gap": "END→START ms",
            "gap": "Result→infer ms",
            "convert": "Convert ms",
            "preprocess": "Preprocess ms",
            "model": "Model ms",
            "inference": "Inference ms",
            "execution": "Execution s",
            "cycle": "Command→result s",
            "progress": "Progress / target",
            "status": "Status",
            "reason": "Result reason",
        }
        widths = {
            "seq": 55,
            "action": 125,
            "mode": 125,
            "execution_gap": 125,
            "gap": 115,
            "convert": 90,
            "preprocess": 100,
            "model": 85,
            "inference": 95,
            "execution": 90,
            "cycle": 120,
            "progress": 145,
            "status": 100,
            "reason": 190,
        }
        for column in columns:
            self.table.heading(column, text=headings[column])
            self.table.column(column, width=widths[column], anchor="center")
        y_scroll = ttk.Scrollbar(
            table_frame, orient="vertical", command=self.table.yview
        )
        x_scroll = ttk.Scrollbar(
            table_frame, orient="horizontal", command=self.table.xview
        )
        self.table.configure(
            yscrollcommand=y_scroll.set,
            xscrollcommand=x_scroll.set,
        )
        self.table.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)
        self.table.tag_configure("failed", foreground="#b00020")
        self.table.tag_configure("succeeded", foreground="#087f23")
        self.table.tag_configure("stopped", foreground="#087f23")

        charts = ttk.Frame(root, padding=(10, 0, 10, 10))
        charts.pack(fill="both", expand=True)
        episode_frame = ttk.LabelFrame(
            charts,
            text="Episode timeline (click an action to inspect)",
            padding=5,
        )
        episode_frame.pack(fill="both", expand=True, pady=(0, 5))
        detail_frame = ttk.LabelFrame(
            charts,
            text="Selected action: RGB-D collection and inference detail",
            padding=5,
        )
        detail_frame.pack(fill="both", expand=True, pady=(5, 0))
        self.episode_timeline_canvas = self.tk.Canvas(
            episode_frame, height=155, background="#ffffff"
        )
        self.action_detail_canvas = self.tk.Canvas(
            detail_frame, height=155, background="#ffffff"
        )
        self.episode_timeline_canvas.pack(fill="both", expand=True)
        self.action_detail_canvas.pack(fill="both", expand=True)
        self.episode_timeline_canvas.bind(
            "<Configure>", self._timeline_resized
        )
        self.action_detail_canvas.bind(
            "<Configure>", self._timeline_resized
        )
        self.episode_timeline_canvas.bind(
            "<Button-1>", self._episode_timeline_clicked
        )

        ttk.Label(
            root,
            text="CSV: {}    JSONL: {}".format(
                self.recorder.csv_path,
                self.recorder.jsonl_path,
            ),
            anchor="w",
        ).pack(fill="x", padx=12, pady=(0, 8))

    def _refresh(self):
        if rospy.is_shutdown():
            self._finish()
            return
        self.recorder.process_pending()
        if self.recorder.changed:
            self._redraw()
            self.recorder.changed = False
        self.root.after(100, self._refresh)

    def _redraw(self):
        records = self.recorder.public_records()
        for item in self.table.get_children():
            self.table.delete(item)
        for record in records:
            progress = "-"
            if record["progress_value"] is not None:
                if record["target_value"] is None:
                    progress = "{:.2f} {}".format(
                        record["progress_value"],
                        record["target_unit"],
                    )
                else:
                    progress = "{:.2f}/{:.2f} {}".format(
                        record["progress_value"],
                        record["target_value"],
                        record["target_unit"],
                    )
            status = record["status"] or "pending"
            self.table.insert(
                "",
                "end",
                values=(
                    record["sequence"] if record["sequence"] is not None else "-",
                    record["action"],
                    record["control_mode"] or "-",
                    _display_value(
                        record["previous_action_end_to_start_ms"]
                    ),
                    _display_value(record["result_to_inference_start_ms"]),
                    _display_value(record["image_conversion_ms"]),
                    _display_value(record["preprocess_ms"]),
                    _display_value(record["model_ms"]),
                    _display_value(record["inference_total_ms"]),
                    _display_value(record["execution_seconds"], 3),
                    _display_value(record["command_to_result_seconds"], 3),
                    progress,
                    status,
                    record["reason"] or "-",
                ),
                tags=(status,),
            )
        if self.table.get_children():
            self.table.see(self.table.get_children()[-1])

        completed = [r for r in records if r["result_received_at"]]
        successful = [
            r for r in completed if r["status"] in ("succeeded", "stopped")
        ]
        inference_values = [
            r["inference_total_ms"]
            for r in records
            if r["inference_total_ms"] is not None
        ]
        execution_values = [
            r["execution_seconds"]
            for r in completed
            if r["execution_seconds"] is not None
        ]
        gap_values = [
            r["result_to_inference_start_ms"]
            for r in records
            if r["result_to_inference_start_ms"] is not None
        ]
        latest_execution_gap = next(
            (
                r["previous_action_end_to_start_ms"]
                for r in reversed(records)
                if r["previous_action_end_to_start_ms"] is not None
            ),
            None,
        )
        last = records[-1] if records else None
        self.summary_variables["status"].set(
            "RUNNING" if not rospy.is_shutdown() else "STOPPED"
        )
        self.summary_variables["actions"].set(
            "{} total / {} ok".format(len(completed), len(successful))
        )
        self.summary_variables["last_action"].set(
            last["action"] if last else "-"
        )
        self.summary_variables["latest_execution_gap"].set(
            "{:.2f} ms".format(latest_execution_gap)
            if latest_execution_gap is not None
            else "-"
        )
        self.summary_variables["avg_inference"].set(
            "{:.2f} ms".format(sum(inference_values) / len(inference_values))
            if inference_values
            else "-"
        )
        self.summary_variables["avg_execution"].set(
            "{:.3f} s".format(sum(execution_values) / len(execution_values))
            if execution_values
            else "-"
        )
        self.summary_variables["avg_gap"].set(
            "{:.2f} ms".format(sum(gap_values) / len(gap_values))
            if gap_values
            else "-"
        )
        self.summary_variables["device"].set(
            last["device"] if last and last["device"] else "-"
        )
        record_ids = [self._record_identity(record) for record in records]
        if self.selected_action_id not in record_ids:
            self.selected_action_id = record_ids[-1] if record_ids else None
        selected_record = next(
            (
                record
                for record in records
                if self._record_identity(record) == self.selected_action_id
            ),
            None,
        )
        self._draw_episode_timeline(self.episode_timeline_canvas, records)
        self._draw_action_detail(self.action_detail_canvas, selected_record)

    def _timeline_resized(self, _event):
        self.recorder.changed = True

    @staticmethod
    def _record_identity(record):
        if record["sequence"] is not None:
            return "sequence:{}".format(record["sequence"])
        return "count:{}".format(record["action_count"])

    @staticmethod
    def _action_color(action):
        return {
            "MOVE_FORWARD": "#2f8f4e",
            "TURN_LEFT": "#d87a16",
            "TURN_RIGHT": "#8e5db7",
            "STOP": "#b00020",
        }.get(action, "#555555")

    @staticmethod
    def _cycle_timeline_phases(record):
        wait_seconds = (record["result_to_inference_start_ms"] or 0.0) / 1000.0
        inference_seconds = (record["inference_total_ms"] or 0.0) / 1000.0
        execution_seconds = record["execution_seconds"] or 0.0
        exact_gap_ms = record["previous_action_end_to_start_ms"]
        if exact_gap_ms is not None:
            dispatch_seconds = max(
                0.0,
                exact_gap_ms / 1000.0 - wait_seconds - inference_seconds,
            )
        else:
            command_seconds = record["command_to_result_seconds"] or 0.0
            dispatch_seconds = max(0.0, command_seconds - execution_seconds)
        return [
            ("Fresh RGB-D", "#b59ad8", wait_seconds),
            ("Inference", "#3267a8", inference_seconds),
            ("ROS dispatch", "#6c757d", dispatch_seconds),
            (
                "Action execution",
                ActionMonitorGui._action_color(record["action"]),
                execution_seconds,
            ),
        ]

    @staticmethod
    def _action_detail_phases(record):
        fresh_wait = record["fresh_rgbd_wait_ms"]
        queue_wait = record["rgbd_queue_ms"]
        if fresh_wait is None and queue_wait is None:
            fresh_wait = record["result_to_inference_start_ms"]
        conversion = record["image_conversion_ms"]
        preprocess = record["preprocess_ms"]
        model = record["model_ms"]
        return [
            ("Fresh RGB-D", "#b59ad8", fresh_wait or 0.0, fresh_wait),
            ("Queue", "#6c757d", queue_wait or 0.0, queue_wait),
            ("Image conversion", "#148ea1", conversion or 0.0, conversion),
            ("Preprocess", "#2f8f4e", preprocess or 0.0, preprocess),
            ("Model", "#3267a8", model or 0.0, model),
        ]

    @staticmethod
    def _format_episode_time(seconds):
        if seconds >= 60.0:
            minutes = int(seconds // 60.0)
            return "{}:{:04.1f}".format(minutes, seconds - 60.0 * minutes)
        return "{:.2f}s".format(seconds)

    def _episode_timeline_clicked(self, event):
        for x_start, x_end, record_id in self.episode_hit_regions:
            if x_start <= event.x <= x_end:
                self.selected_action_id = record_id
                self.recorder.changed = True
                return

    def _draw_episode_timeline(self, canvas, records):
        canvas.delete("all")
        self.episode_hit_regions = []
        width = max(canvas.winfo_width(), 700)
        height = max(canvas.winfo_height(), 145)
        cycles = []
        cursor = 0.0
        for record in records:
            phases = self._cycle_timeline_phases(record)
            start = cursor
            cursor += sum(value for _, _, value in phases)
            cycles.append((record, phases, start, cursor))
        episode_duration = cursor
        if not cycles or episode_duration <= 0.0:
            canvas.create_text(
                width / 2,
                height / 2,
                text="Waiting for action data...",
                fill="#666666",
            )
            return

        margin_left = 55
        margin_right = 20
        legend_y = 13
        bar_top = 53
        bar_bottom = 85
        axis_y = 112
        plot_width = max(1.0, width - margin_left - margin_right)
        legend = [
            ("Fresh RGB-D", "#b59ad8"),
            ("Inference", "#3267a8"),
            ("ROS dispatch", "#6c757d"),
            ("Forward", "#2f8f4e"),
            ("Left", "#d87a16"),
            ("Right", "#8e5db7"),
            ("Stop", "#b00020"),
        ]
        legend_x = margin_left
        for label, color in legend:
            canvas.create_rectangle(
                legend_x,
                legend_y - 5,
                legend_x + 10,
                legend_y + 5,
                fill=color,
                outline="",
            )
            canvas.create_text(
                legend_x + 14,
                legend_y,
                text=label,
                anchor="w",
                fill="#444444",
                font=("TkDefaultFont", 8),
            )
            legend_x += 24 + 6.3 * len(label)

        canvas.create_text(
            width - margin_right,
            legend_y,
            text="Episode {}".format(
                self._format_episode_time(episode_duration)
            ),
            anchor="e",
            fill="#333333",
            font=("TkDefaultFont", 8, "bold"),
        )
        canvas.create_rectangle(
            margin_left,
            bar_top,
            margin_left + plot_width,
            bar_bottom,
            fill="#f2f2f2",
            outline="",
        )
        for tick in range(5):
            fraction = tick / 4.0
            x = margin_left + fraction * plot_width
            canvas.create_line(
                x,
                bar_top,
                x,
                axis_y - 7,
                fill="#e4e4e4",
            )
            canvas.create_text(
                x,
                axis_y,
                text=self._format_episode_time(episode_duration * fraction),
                fill="#555555",
                font=("TkDefaultFont", 8),
            )

        for index, (record, phases, start, end) in enumerate(cycles):
            sequence = (
                record["sequence"]
                if record["sequence"] is not None
                else record["action_count"] or index + 1
            )
            cycle_x_start = margin_left + plot_width * start / episode_duration
            cycle_x_end = margin_left + plot_width * end / episode_duration
            hit_start = cycle_x_start
            hit_end = max(cycle_x_end, cycle_x_start + 4.0)
            self.episode_hit_regions.append(
                (hit_start, hit_end, self._record_identity(record))
            )
            x = cycle_x_start
            action_segment = None
            for phase_label, color, value in phases:
                value = max(0.0, float(value))
                segment_width = plot_width * value / episode_duration
                if segment_width > 0.0:
                    canvas.create_rectangle(
                        x,
                        bar_top,
                        x + segment_width,
                        bar_bottom,
                        fill=color,
                        outline="",
                    )
                if phase_label == "Action execution":
                    action_segment = (x, x + segment_width)
                x += segment_width
            if action_segment is None or action_segment[1] - action_segment[0] < 2:
                marker_x = cycle_x_end
                canvas.create_line(
                    marker_x,
                    bar_top - 3,
                    marker_x,
                    bar_bottom + 3,
                    fill=self._action_color(record["action"]),
                    width=2,
                )
                action_segment = (marker_x - 1, marker_x + 1)
            exact_gap_ms = record["previous_action_end_to_start_ms"]
            if exact_gap_ms is not None and action_segment[0] > cycle_x_start:
                canvas.create_rectangle(
                    cycle_x_start,
                    bar_top - 2,
                    action_segment[0],
                    bar_bottom + 2,
                    outline="#000000",
                    width=2,
                )
                if action_segment[0] - cycle_x_start >= 72.0:
                    canvas.create_text(
                        (cycle_x_start + action_segment[0]) / 2,
                        bar_top - 9,
                        text="END→START {:.1f}ms".format(exact_gap_ms),
                        fill="#000000",
                        font=("TkDefaultFont", 8, "bold"),
                    )
            action_width = action_segment[1] - action_segment[0]
            if action_width >= 34.0:
                short_action = {
                    "MOVE_FORWARD": "FWD",
                    "TURN_LEFT": "LEFT",
                    "TURN_RIGHT": "RIGHT",
                    "STOP": "STOP",
                }.get(record["action"], record["action"])
                canvas.create_text(
                    (action_segment[0] + action_segment[1]) / 2,
                    (bar_top + bar_bottom) / 2,
                    text="#{} {}".format(sequence, short_action),
                    fill="#ffffff",
                    font=("TkDefaultFont", 8),
                )
            if self._record_identity(record) == self.selected_action_id:
                canvas.create_rectangle(
                    max(margin_left, cycle_x_start - 1),
                    bar_top - 4,
                    min(margin_left + plot_width, hit_end + 1),
                    bar_bottom + 4,
                    outline="#111111",
                    width=2,
                )
                canvas.create_text(
                    (cycle_x_start + min(hit_end, margin_left + plot_width)) / 2,
                    bar_bottom + 13,
                    text="selected #{}".format(sequence),
                    fill="#111111",
                    font=("TkDefaultFont", 8, "bold"),
                )

    def _draw_action_detail(self, canvas, record):
        canvas.delete("all")
        width = max(canvas.winfo_width(), 700)
        height = max(canvas.winfo_height(), 145)
        if record is None:
            canvas.create_text(
                width / 2,
                height / 2,
                text="Select an action on the episode timeline.",
                fill="#666666",
            )
            return

        phases = self._action_detail_phases(record)
        total = sum(value for _label, _color, value, _raw in phases)
        sequence = (
            record["sequence"]
            if record["sequence"] is not None
            else record["action_count"] or "-"
        )
        execution_text = (
            "-"
            if record["execution_seconds"] is None
            else "{:.3f}s".format(record["execution_seconds"])
        )
        execution_gap_ms = record["previous_action_end_to_start_ms"]
        execution_gap_text = (
            "END→START -"
            if execution_gap_ms is None
            else "END→START {:.2f} ms".format(execution_gap_ms)
        )
        canvas.create_text(
            12,
            13,
            text="#{} {}    sensing+inference {:.1f}ms    execution {}".format(
                sequence,
                record["action"],
                total,
                execution_text,
            ),
            anchor="w",
            fill="#222222",
            font=("TkDefaultFont", 9, "bold"),
        )
        canvas.create_text(
            width - 12,
            13,
            text=execution_gap_text,
            anchor="e",
            fill="#000000",
            font=("TkDefaultFont", 12, "bold"),
        )
        if total <= 0.0:
            canvas.create_text(
                width / 2,
                height / 2,
                text="Waiting for inference metrics...",
                fill="#666666",
            )
            return

        margin_left = 55
        margin_right = 20
        legend_y = 40
        bar_top = 66
        bar_bottom = 96
        axis_y = 125
        plot_width = max(1.0, width - margin_left - margin_right)
        legend_x = margin_left
        for label, color, _value, raw_value in phases:
            value_text = "-" if raw_value is None else "{:.1f}ms".format(raw_value)
            legend_label = "{} {}".format(label, value_text)
            canvas.create_rectangle(
                legend_x,
                legend_y - 5,
                legend_x + 10,
                legend_y + 5,
                fill=color,
                outline="",
            )
            canvas.create_text(
                legend_x + 14,
                legend_y,
                text=legend_label,
                anchor="w",
                fill="#444444",
                font=("TkDefaultFont", 8),
            )
            legend_x += 24 + 6.3 * len(legend_label)

        x = margin_left
        for label, color, value, _raw_value in phases:
            segment_width = plot_width * value / total
            if segment_width > 0.0:
                canvas.create_rectangle(
                    x,
                    bar_top,
                    x + segment_width,
                    bar_bottom,
                    fill=color,
                    outline="",
                )
                if segment_width >= 62.0:
                    canvas.create_text(
                        x + segment_width / 2,
                        (bar_top + bar_bottom) / 2,
                        text="{}\n{:.1f}ms".format(label, value),
                        fill="#ffffff",
                        font=("TkDefaultFont", 8),
                    )
            x += segment_width
        canvas.create_line(
            margin_left + plot_width,
            bar_top - 4,
            margin_left + plot_width,
            bar_bottom + 4,
            fill="#111111",
            width=2,
        )
        canvas.create_text(
            margin_left + plot_width,
            bar_top - 9,
            text="action published",
            anchor="e",
            fill="#333333",
            font=("TkDefaultFont", 8),
        )
        for tick in range(5):
            fraction = tick / 4.0
            tick_x = margin_left + fraction * plot_width
            canvas.create_line(
                tick_x,
                bar_bottom,
                tick_x,
                axis_y - 8,
                fill="#e4e4e4",
            )
            canvas.create_text(
                tick_x,
                axis_y,
                text="{:.1f}ms".format(total * fraction),
                fill="#555555",
                font=("TkDefaultFont", 8),
            )

    def _close(self):
        rospy.loginfo(
            "Action monitor window closed; continuing CSV/JSONL recording "
            "in headless mode."
        )
        self.root.destroy()

    def _finish(self):
        self.recorder.close()
        try:
            self.root.destroy()
        except self.tk.TclError:
            pass

    def run(self):
        self.root.mainloop()


def resolve_output_directory(path_text):
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def build_argument_parser():
    parser = argparse.ArgumentParser(
        description="Record and visualize VLN inference/action timing metrics."
    )
    parser.add_argument("--action-topic", default="/vln/action")
    parser.add_argument("--action-result-topic", default="/vln/action_result")
    parser.add_argument(
        "--inference-metrics-topic", default="/vln/inference_metrics"
    )
    parser.add_argument(
        "--output-directory", default="~/.ros/vln_action_metrics"
    )
    parser.add_argument(
        "--history-size",
        type=int,
        default=0,
        help="Maximum actions kept in the GUI; use 0 for the full episode.",
    )
    parser.add_argument("--no-gui", action="store_true")
    parser.add_argument("--node-name", default="ros_vln_action_monitor")
    return parser


def validate_arguments(args):
    for name in (
        "action_topic",
        "action_result_topic",
        "inference_metrics_topic",
        "output_directory",
    ):
        value = getattr(args, name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError("{} must be a non-empty string".format(name))
    if args.history_size < 0:
        raise ValueError("history_size must be >= 0")


def main():
    parser = build_argument_parser()
    args = parser.parse_args(rospy.myargv(argv=sys.argv)[1:])
    try:
        validate_arguments(args)
    except ValueError as error:
        parser.error(str(error))

    rospy.init_node(args.node_name, anonymous=False)
    recorder = ActionMetricsRecorder(
        resolve_output_directory(args.output_directory),
        args.history_size,
    )
    subscribers = [
        rospy.Subscriber(
            args.action_topic,
            String,
            recorder.command_callback,
            queue_size=20,
        ),
        rospy.Subscriber(
            args.action_result_topic,
            String,
            recorder.result_callback,
            queue_size=20,
        ),
        rospy.Subscriber(
            args.inference_metrics_topic,
            String,
            recorder.metrics_callback,
            queue_size=20,
        ),
    ]
    rospy.on_shutdown(recorder.close)
    rospy.loginfo(
        "VLN action monitor ready; action=%s result=%s metrics=%s csv=%s jsonl=%s",
        args.action_topic,
        args.action_result_topic,
        args.inference_metrics_topic,
        recorder.csv_path,
        recorder.jsonl_path,
    )

    use_gui = not args.no_gui
    if use_gui and not os.environ.get("DISPLAY"):
        rospy.logwarn("DISPLAY is not set; action monitor is running headless.")
        use_gui = False
    if use_gui:
        try:
            gui = ActionMonitorGui(recorder)
            gui.run()
        except Exception as error:
            rospy.logerr(
                "Cannot start action monitor GUI (%s); continuing headless.",
                error,
            )

    rate = rospy.Rate(10.0)
    while not rospy.is_shutdown():
        recorder.process_pending()
        try:
            rate.sleep()
        except rospy.ROSInterruptException:
            break
    recorder.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
