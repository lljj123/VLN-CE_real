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
        self.root = tk.Tk()
        self.root.title("VLN Action Timing Monitor")
        self.root.geometry("1240x760")
        self.root.minsize(960, 600)
        self.root.protocol("WM_DELETE_WINDOW", self._close)

        self.summary_variables = {
            name: tk.StringVar(value="-")
            for name in (
                "status",
                "actions",
                "last_action",
                "avg_inference",
                "avg_execution",
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
            ("Avg inference", "avg_inference"),
            ("Avg execution", "avg_execution"),
            ("Device", "device"),
        ]
        for index, (title, key) in enumerate(cards):
            card = ttk.LabelFrame(summary, text=title, padding=(12, 8))
            card.grid(row=0, column=index, padx=4, sticky="nsew")
            ttk.Label(
                card,
                textvariable=self.summary_variables[key],
                font=("TkDefaultFont", 11, "bold"),
            ).pack()
            summary.columnconfigure(index, weight=1)

        table_frame = ttk.LabelFrame(root, text="Action records", padding=8)
        table_frame.pack(fill="both", expand=True, padx=10, pady=(0, 8))
        columns = (
            "seq",
            "action",
            "mode",
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

        charts = ttk.Frame(root, padding=(10, 0, 10, 10))
        charts.pack(fill="both")
        inference_frame = ttk.LabelFrame(
            charts, text="Inference total (ms)", padding=5
        )
        inference_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))
        execution_frame = ttk.LabelFrame(
            charts, text="Execution time (s)", padding=5
        )
        execution_frame.pack(side="left", fill="both", expand=True, padx=(5, 0))
        self.inference_canvas = self.tk.Canvas(
            inference_frame, height=190, background="#ffffff"
        )
        self.execution_canvas = self.tk.Canvas(
            execution_frame, height=190, background="#ffffff"
        )
        self.inference_canvas.pack(fill="both", expand=True)
        self.execution_canvas.pack(fill="both", expand=True)

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
        self.summary_variables["device"].set(
            last["device"] if last and last["device"] else "-"
        )
        self._draw_chart(
            self.inference_canvas,
            records,
            "inference_total_ms",
            "#3267a8",
        )
        self._draw_chart(
            self.execution_canvas,
            completed,
            "execution_seconds",
            "#d87a16",
        )

    def _draw_chart(self, canvas, records, field, color):
        canvas.delete("all")
        width = max(canvas.winfo_width(), 300)
        height = max(canvas.winfo_height(), 160)
        margin_left = 45
        margin_bottom = 25
        values = [
            (record["sequence"], record[field])
            for record in records[-20:]
            if record[field] is not None
        ]
        if not values:
            canvas.create_text(
                width / 2,
                height / 2,
                text="Waiting for action data...",
                fill="#666666",
            )
            return
        maximum = max(value for _, value in values)
        maximum = max(maximum, 0.001)
        canvas.create_line(
            margin_left,
            10,
            margin_left,
            height - margin_bottom,
            fill="#888888",
        )
        canvas.create_line(
            margin_left,
            height - margin_bottom,
            width - 5,
            height - margin_bottom,
            fill="#888888",
        )
        canvas.create_text(
            margin_left - 5,
            10,
            text="{:.2f}".format(maximum),
            anchor="e",
            fill="#555555",
        )
        usable_width = width - margin_left - 10
        bar_width = max(4.0, usable_width / max(len(values), 1) * 0.65)
        step = usable_width / max(len(values), 1)
        plot_height = height - margin_bottom - 18
        for index, (sequence, value) in enumerate(values):
            x_center = margin_left + (index + 0.5) * step
            bar_height = plot_height * value / maximum
            canvas.create_rectangle(
                x_center - bar_width / 2,
                height - margin_bottom - bar_height,
                x_center + bar_width / 2,
                height - margin_bottom,
                fill=color,
                outline="",
            )
            canvas.create_text(
                x_center,
                height - margin_bottom + 10,
                text=str(sequence if sequence is not None else index + 1),
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
    parser.add_argument("--history-size", type=int, default=100)
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
    if args.history_size <= 0:
        raise ValueError("history_size must be positive")


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
