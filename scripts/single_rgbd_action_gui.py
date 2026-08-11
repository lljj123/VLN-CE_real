"""Visualize deterministic CMA actions from one RGB-D pair or an episode.

Episode replay preserves recurrent state and, unless explicitly overridden on
a frame, the model's previous predicted action across frames. The recorded
observations still follow the expert trajectory, so this is an open-loop replay
rather than a closed-loop simulator.
"""
import argparse, json, sys, traceback
from pathlib import Path
import cv2, numpy as np, torch
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QComboBox, QFileDialog, QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox, QProgressBar, QPushButton, QSlider, QTextEdit, QVBoxLayout, QWidget
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from vlnce_real.model import CMAPolicy, DEPTH_SIZE, RGB_SIZE, STATE_HIDDEN_SIZE, batch_observation, encode_instruction
from vlnce_real.preprocessing import preprocess_rgbd
ACTION_LABELS = [
 "STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"]
ACTION_VIEW = {
 'STOP': ('■  STOP / 停止', '#ef5350'),
 'MOVE_FORWARD': ('↑  MOVE_FORWARD / 前进', '#26a269'),
 'TURN_LEFT': ('↶  TURN_LEFT / 左转', '#e5a50a'),
 'TURN_RIGHT': ('↷  TURN_RIGHT / 右转', '#3584e4')}

def resolve_path(path_text):
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def default_checkpoint_path():
    candidates = [
     REPO_ROOT / "training/checkpoints/real_cma_0p4m_15deg/best_robot.pth",
     REPO_ROOT / "data/checkpoints/CMA_PM_DA_Aug_robot.pth"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[-1]


def load_rgb(path):
    path = Path(path)
    payload = np.fromfile((str(path)), dtype=(np.uint8))
    bgr = cv2.imdecode(payload, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("无法读取 RGB 图片：{}".format(path))
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_depth_m(path):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        depth = np.load((str(path)), allow_pickle=False)
    elif suffix == ".npz":
        with np.load(str(path), allow_pickle=False) as archive:
            keys = list(archive.files)
            if "depth" in keys:
                depth = archive["depth"]
            elif len(keys) == 1:
                depth = archive[keys[0]]
            else:
                raise ValueError("NPZ 必须只有一个数组，或包含名为 depth 的数组。")
    else:
        raise ValueError("Depth 只支持数据集的 .npy 或 .npz 文件。")
    depth = np.asarray(depth, dtype=(np.float32))
    if depth.ndim == 3:
        if depth.shape[2] == 1:
            depth = depth[:, :, 0]
    if depth.ndim != 2:
        raise ValueError("Depth 应为 [H, W] 或 [H, W, 1]，实际为 {}。".format(depth.shape))
    return depth


def colorize_depth(depth, min_depth, max_depth):
    depth = np.asarray(depth, dtype=(np.float32))
    valid = np.isfinite(depth) & (depth > min_depth) & (depth <= max_depth)
    normalized = np.zeros_like(depth, dtype=(np.float32))
    normalized[valid] = np.clip((depth[valid] - min_depth) / (max_depth - min_depth), 0.0, 1.0)
    color_index = np.rint((1.0 - normalized) * 255.0).astype(np.uint8)
    colored_bgr = cv2.applyColorMap(color_index, cv2.COLORMAP_JET)
    colored_bgr[~valid] = (0, 0, 0)
    return cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)


def array_to_pixmap(rgb):
    rgb = np.ascontiguousarray(rgb, dtype=(np.uint8))
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("显示图像必须是 [H, W, 3] RGB。")
    height, width = rgb.shape[:2]
    image = QImage(rgb.data, width, height, int(rgb.strides[0]), QImage.Format_RGB888).copy()
    return QPixmap.fromImage(image)


def contained_episode_path(episode_root, relative_path):
    episode_root = Path(episode_root).resolve()
    resolved = (episode_root / relative_path).resolve()
    try:
        resolved.relative_to(episode_root)
    except ValueError:
        raise ValueError("Episode sample 路径越出目录：{}".format(relative_path))
    if not resolved.is_file():
        raise FileNotFoundError("Episode sample 不存在：{}".format(resolved))
    return resolved


def load_episode(episode_path):
    manifest_path = Path(episode_path).expanduser().resolve()
    if manifest_path.is_dir():
        manifest_path = manifest_path / "episode.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("请选择包含 episode.json 的目录或 episode.json 文件：{}".format(manifest_path))
    with manifest_path.open("r", encoding="utf-8") as input_file:
        manifest = json.load(input_file)
    instruction = str(manifest.get("instruction", "")).strip()
    samples = manifest.get("samples")
    if not instruction:
        raise ValueError("episode.json 中的 instruction 为空。")
    if not (isinstance(samples, list) and samples):
        raise ValueError("episode.json 中没有 samples。")
    normalized_samples = []
    for expected_index, sample in enumerate(samples):
        if int(sample.get("index", -1)) != expected_index:
            raise ValueError("Episode sample index 在 {} 处不连续。".format(expected_index))
        action_index = int(sample.get("action_index", -1))
        if action_index < 0 or action_index >= len(ACTION_LABELS):
            raise ValueError("Episode sample {} 的 action_index 无效。".format(expected_index))
        expert_action = sample.get("action")
        if expert_action != ACTION_LABELS[action_index]:
            raise ValueError("Episode sample {} 的动作名称/编号不一致。".format(expected_index))
        normalized_samples.append({
            "index": expected_index,
            "rgb_path": contained_episode_path(manifest_path.parent, sample["rgb"]),
            "depth_path": contained_episode_path(manifest_path.parent, sample["depth"]),
            "expert_action": expert_action,
        })
    return {
        "episode_id": manifest.get("episode_id", manifest_path.parent.name),
        "manifest_path": manifest_path,
        "instruction": instruction,
        "samples": normalized_samples,
    }


class CMASequenceRunner:
    __doc__ = "CMA runner whose state transition matches real-robot inference."

    def __init__(self, checkpoint_path, instruction, instruction_length, min_depth, max_depth, force_cpu, initial_previous_action='START'):
        self.checkpoint_path = resolve_path(checkpoint_path)
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError("Checkpoint 不存在：{}".format(self.checkpoint_path))
        checkpoint = torch.load((str(self.checkpoint_path)),
          map_location="cpu")
        if checkpoint.get("format_version") != 1:
            raise ValueError("Checkpoint 不是 robot format version 1。")
        if checkpoint.get("action_labels") != ACTION_LABELS:
            raise ValueError("Checkpoint 动作顺序不是 {}。".format(ACTION_LABELS))
        word_list = checkpoint.get("word_list")
        if not (isinstance(word_list, list) and word_list):
            raise ValueError("Checkpoint 缺少有效的 R2R 词表。")
        self.rgb_size = tuple(checkpoint.get("rgb_size", RGB_SIZE))
        self.depth_size = tuple(checkpoint.get("depth_size", DEPTH_SIZE))
        if self.rgb_size != RGB_SIZE or self.depth_size != DEPTH_SIZE:
            raise ValueError("Checkpoint 输入尺寸 {} / {} 与当前模型 {} / {} 不一致。".format(self.rgb_size, self.depth_size, RGB_SIZE, DEPTH_SIZE))
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.device = torch.device(
            "cpu" if force_cpu or not torch.cuda.is_available() else "cuda:0"
        )
        self.instruction_tokens, self.instruction_stats = encode_instruction(word_list, instruction, instruction_length)
        self.policy = CMAPolicy(vocab_size=(len(word_list)),
          num_actions=(len(ACTION_LABELS)))
        self.policy.load_state_dict((checkpoint["state_dict"]), strict=True)
        self.policy.to(self.device)
        self.policy.eval()
        self.initial_previous_action = initial_previous_action
        self.reset()

    def reset(self):
        if self.initial_previous_action not in ["START"] + ACTION_LABELS:
            raise ValueError("首帧上一动作必须是 START 或 {}，实际为 {}。".format(ACTION_LABELS, self.initial_previous_action))
        self.step_index = 0
        self.rnn_states = torch.zeros(1,
          (self.policy.net.num_recurrent_layers),
          STATE_HIDDEN_SIZE,
          device=(self.device))
        self.previous_actions = torch.zeros(1,
          1, dtype=(torch.long), device=(self.device))
        self.not_done_masks = torch.zeros(1,
          1, dtype=(torch.bool), device=(self.device))
        if self.initial_previous_action != "START":
            self.previous_actions.fill_(ACTION_LABELS.index(self.initial_previous_action))
            self.not_done_masks.fill_(1)

    def predict_files(self, rgb_path, depth_path, previous_action_override=None):
        if previous_action_override is not None:
            if previous_action_override not in ACTION_LABELS:
                raise ValueError("上一动作覆盖必须是 {}，实际为 {}。".format(ACTION_LABELS, previous_action_override))
            self.previous_actions.fill_(ACTION_LABELS.index(previous_action_override))
            self.not_done_masks.fill_(1)
        rgb_path = Path(rgb_path).resolve()
        depth_path = Path(depth_path).resolve()
        rgb = load_rgb(rgb_path)
        depth_m = load_depth_m(depth_path)
        if rgb.shape[:2] != depth_m.shape[:2]:
            raise ValueError("RGB 与 Depth 未对齐：{} vs {}。请选择同一 sample。".format(rgb.shape[:2], depth_m.shape[:2]))
        observations, invalid_fraction = preprocess_rgbd(rgb=rgb,
          depth_m=depth_m,
          depth_encoding="32FC1",
          rgb_size=(self.rgb_size),
          depth_size=(self.depth_size),
          min_depth=(self.min_depth),
          max_depth=(self.max_depth))
        observations["instruction"] = self.instruction_tokens
        batch = batch_observation(observations, self.device)
        previous_action = ACTION_LABELS[int(self.previous_actions[0].item())] if bool(self.not_done_masks[0].item()) else "START"
        with torch.no_grad():
            features, self.rnn_states = self.policy.net(batch, self.rnn_states, self.previous_actions, self.not_done_masks)
            logits = self.policy.action_distribution.linear(features)
            probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
        action_index = int(np.argmax(probabilities))
        self.previous_actions.fill_(action_index)
        self.not_done_masks.fill_(1)
        normalized_depth = observations["depth"][:, :, 0]
        result = {'frame_index':self.step_index,
         'action':ACTION_LABELS[action_index],
         'previous_action':previous_action,
         'previous_action_overridden':previous_action_override is not None,
         'probabilities':{label: float(probabilities[index]) for index, label in enumerate(ACTION_LABELS)},
         'rnn_state_norm':float(self.rnn_states.norm().cpu().item()),
         'device':str(self.device),
         'checkpoint':str(self.checkpoint_path),
         'invalid_depth_fraction':invalid_fraction,
         'instruction_stats':self.instruction_stats,
         'rgb_path':str(rgb_path),
         'depth_path':str(depth_path),
         'rgb_preview':observations["rgb"],
         'depth_preview':colorize_depth(normalized_depth,
           min_depth=0.0, max_depth=1.0)}
        self.step_index += 1
        return result


def run_single_frame_inference(checkpoint_path, rgb_path, depth_path, instruction, instruction_length, min_depth, max_depth, force_cpu, initial_previous_action='START'):
    runner = CMASequenceRunner(checkpoint_path=checkpoint_path,
      instruction=instruction,
      instruction_length=instruction_length,
      min_depth=min_depth,
      max_depth=max_depth,
      force_cpu=force_cpu,
      initial_previous_action=initial_previous_action)
    result = runner.predict_files(rgb_path, depth_path)
    result.update({
     'mode': "single",
     'frame_count': 1,
     'expert_action': None,
     'is_correct': None,
     'cumulative_accuracy': None})
    return result


def run_episode_inference(checkpoint_path, episode_path, instruction, instruction_length, min_depth, max_depth, force_cpu, initial_previous_action='START', previous_action_overrides=None):
    episode = load_episode(episode_path)
    runner = CMASequenceRunner(checkpoint_path=checkpoint_path,
      instruction=instruction,
      instruction_length=instruction_length,
      min_depth=min_depth,
      max_depth=max_depth,
      force_cpu=force_cpu,
      initial_previous_action=initial_previous_action)
    previous_action_overrides = previous_action_overrides or {}
    steps = []
    correct_count = 0
    for frame_index, sample in enumerate(episode["samples"]):
        result = runner.predict_files((sample["rgb_path"]),
          (sample["depth_path"]),
          previous_action_override=(previous_action_overrides.get(frame_index)))
        result["mode"] = "episode"
        result["frame_count"] = len(episode["samples"])
        result["expert_action"] = sample["expert_action"]
        result["is_correct"] = result["action"] == sample["expert_action"]
        correct_count += int(result["is_correct"])
        result["cumulative_accuracy"] = correct_count / float(sample["index"] + 1)
        steps.append(result)
    else:
        return {'mode':"episode",
         'episode_id':episode["episode_id"],
         'manifest_path':str(episode["manifest_path"]),
         'instruction':instruction,
         'steps':steps,
         'accuracy':correct_count / (float(len(steps)))}


class InferenceWorker(QThread):
    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str, str)

    def __init__(self, arguments, parent=None):
        super().__init__(parent)
        self.arguments = dict(arguments)

    def run(self):
        try:
            operation = self.arguments.pop("operation", "single")
            if operation == "episode":
                result = run_episode_inference(**self.arguments)
            else:
                result = run_single_frame_inference(**self.arguments)
            self.succeeded.emit(result)
        except Exception as error:
            self.failed.emit(str(error), traceback.format_exc())


class EpisodePathEdit(QLineEdit):
    episode_dropped = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setPlaceholderText("拖入 episode 文件夹或 episode.json；留空时使用单帧模式")

    @staticmethod
    def accepts_path(path):
        path = Path(path)
        return path.is_dir() or path.name == "episode.json"

    def dragEnterEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            if urls[0].isLocalFile():
                if self.accepts_path(urls[0].toLocalFile()):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            if urls[0].isLocalFile():
                path = urls[0].toLocalFile()
                if self.accepts_path(path):
                    self.setText(path)
                    self.episode_dropped.emit(path)
                    event.acceptProposedAction()
                    return
        event.ignore()


class FileDropBox(QFrame):
    file_selected = pyqtSignal(str)

    def __init__(self, title, hint, extensions, parent=None):
        super().__init__(parent)
        self.extensions = {suffix.lower() for suffix in extensions}
        self.preview_pixmap = None
        self.setAcceptDrops(True)
        self.setMinimumSize(390, 275)
        self.setObjectName("dropBox")
        self.title_label = QLabel(title)
        self.title_label.setObjectName("dropTitle")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.preview_label = QLabel(hint)
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setWordWrap(True)
        self.preview_label.setMinimumHeight(205)
        self.preview_label.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.path_label = QLabel("尚未选择文件")
        self.path_label.setAlignment(Qt.AlignCenter)
        self.path_label.setWordWrap(True)
        self.path_label.setObjectName("pathLabel")
        self.path_label.setAttribute(Qt.WA_TransparentForMouseEvents)
        layout = QVBoxLayout(self)
        layout.addWidget(self.title_label)
        layout.addWidget(self.preview_label, 1)
        layout.addWidget(self.path_label)

    def accepts_path(self, path):
        return Path(path).suffix.lower() in self.extensions

    def select_file(self):
        patterns = " ".join("*{}".format(item) for item in self.extensions)
        path, _ = QFileDialog.getOpenFileName(self, "选择文件", str(REPO_ROOT / "training/data"), "支持的文件 ({})".format(patterns))
        if path:
            self.file_selected.emit(path)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.select_file()
        super().mousePressEvent(event)

    def dragEnterEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            if urls[0].isLocalFile():
                if self.accepts_path(urls[0].toLocalFile()):
                    event.acceptProposedAction()
                    self.setProperty("dragActive", True)
                    self.style().unpolish(self)
                    self.style().polish(self)
                    return
        event.ignore()

    def dragLeaveEvent(self, event):
        self.setProperty("dragActive", False)
        self.style().unpolish(self)
        self.style().polish(self)
        super().dragLeaveEvent(event)

    def dropEvent(self, event):
        self.setProperty("dragActive", False)
        self.style().unpolish(self)
        self.style().polish(self)
        urls = event.mimeData().urls()
        if urls:
            if urls[0].isLocalFile():
                path = urls[0].toLocalFile()
                if self.accepts_path(path):
                    event.acceptProposedAction()
                    self.file_selected.emit(path)
                    return
        event.ignore()

    def set_file_preview(self, path, rgb):
        self.path_label.setText(str(path))
        self.path_label.setToolTip(str(path))
        self.preview_pixmap = array_to_pixmap(rgb)
        self._refresh_preview()

    def _refresh_preview(self):
        if self.preview_pixmap is None:
            return
        size = self.preview_label.size()
        self.preview_label.setPixmap(self.preview_pixmap.scaled(max(1, size.width() - 10), max(1, size.height() - 10), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_preview()


class ActionGui(QMainWindow):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.rgb_path = None
        self.depth_path = None
        self.episode = None
        self.episode_results = None
        self.previous_action_overrides = {}
        self.default_initial_previous_action = getattr(args, "initial_previous_action", "START")
        if self.default_initial_previous_action != "START":
            self.previous_action_overrides[0] = self.default_initial_previous_action
        self.worker = None
        self.setWindowTitle("VLN RGB-D 多帧记忆 Action 可视化")
        self.resize(1180, 960)
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(22, 18, 22, 18)
        root.setSpacing(14)
        heading = QLabel("VLN RGB-D 多帧记忆 Action 可视化")
        heading.setObjectName("heading")
        root.addWidget(heading)
        checkpoint_row = QHBoxLayout()
        checkpoint_row.addWidget(QLabel("Checkpoint"))
        self.checkpoint_edit = QLineEdit(str(resolve_path(args.checkpoint)))
        checkpoint_row.addWidget(self.checkpoint_edit, 1)
        checkpoint_button = QPushButton("选择…")
        checkpoint_button.clicked.connect(self.choose_checkpoint)
        checkpoint_row.addWidget(checkpoint_button)
        root.addLayout(checkpoint_row)
        episode_row = QHBoxLayout()
        episode_row.addWidget(QLabel("Episode"))
        self.episode_edit = EpisodePathEdit()
        self.episode_edit.episode_dropped.connect(self.load_episode_path)
        self.episode_edit.returnPressed.connect(self.load_episode_from_edit)
        episode_row.addWidget(self.episode_edit, 1)
        episode_button = QPushButton("选择文件夹…")
        episode_button.clicked.connect(self.choose_episode)
        episode_row.addWidget(episode_button)
        load_episode_button = QPushButton("加载")
        load_episode_button.clicked.connect(self.load_episode_from_edit)
        episode_row.addWidget(load_episode_button)
        clear_episode_button = QPushButton("单帧模式")
        clear_episode_button.clicked.connect(self.clear_episode)
        episode_row.addWidget(clear_episode_button)
        root.addLayout(episode_row)
        inputs = QHBoxLayout()
        self.rgb_box = FileDropBox("RGB", "把数据集的 RGB 图片拖到这里\n或点击选择 JPG / PNG", {
         ".jpg", ".jpeg", ".png", ".bmp"})
        self.depth_box = FileDropBox("Depth", "把同一 sample 的 Depth 拖到这里\n支持 float32 米制 NPY / NPZ", {
         ".npy", ".npz"})
        self.rgb_box.file_selected.connect(self.set_rgb_path)
        self.depth_box.file_selected.connect(self.set_depth_path)
        inputs.addWidget(self.rgb_box)
        inputs.addWidget(self.depth_box)
        root.addLayout(inputs, 1)
        instruction_label = QLabel("Instruction（使用 checkpoint 的英文词表）")
        instruction_label.setObjectName("sectionTitle")
        root.addWidget(instruction_label)
        self.instruction_edit = QTextEdit()
        self.instruction_edit.setPlaceholderText("例如：Go straight down the hallway, turn left into the office and stop in front of the chair.")
        self.instruction_edit.setPlainText(args.instruction)
        self.instruction_edit.setMaximumHeight(85)
        root.addWidget(self.instruction_edit)
        previous_action_row = QHBoxLayout()
        previous_action_row.addWidget(QLabel("当前帧的上一步动作覆盖"))
        self.initial_previous_action_combo = QComboBox()
        self.initial_previous_action_combo.addItems([
         "AUTO"] + ACTION_LABELS)
        self.initial_previous_action_combo.setCurrentText(self.previous_action_overrides.get(0, "AUTO"))
        self.initial_previous_action_combo.currentTextChanged.connect(self.initial_previous_action_changed)
        previous_action_row.addWidget(self.initial_previous_action_combo)
        previous_action_hint = QLabel("AUTO 使用正常值；动作覆盖只替换本帧 embedding，不清空 GRU")
        previous_action_hint.setObjectName("previousActionHint")
        previous_action_row.addWidget(previous_action_hint, 1)
        root.addLayout(previous_action_row)
        self.timeline_frame = QFrame()
        self.timeline_frame.setObjectName("timelineFrame")
        timeline_layout = QVBoxLayout(self.timeline_frame)
        timeline_top = QHBoxLayout()
        self.previous_frame_button = QPushButton("← 上一帧")
        self.previous_frame_button.clicked.connect(self.previous_frame)
        timeline_top.addWidget(self.previous_frame_button)
        self.frame_label = QLabel("帧 0 / 0")
        self.frame_label.setObjectName("frameLabel")
        self.frame_label.setAlignment(Qt.AlignCenter)
        timeline_top.addWidget(self.frame_label, 1)
        self.next_frame_button = QPushButton("下一帧 →")
        self.next_frame_button.clicked.connect(self.next_frame)
        timeline_top.addWidget(self.next_frame_button)
        timeline_layout.addLayout(timeline_top)
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.valueChanged.connect(self.frame_changed)
        timeline_layout.addWidget(self.frame_slider)
        timeline_details = QHBoxLayout()
        self.expert_label = QLabel("专家动作：—")
        self.memory_label = QLabel("记忆状态：尚未运行")
        self.memory_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        timeline_details.addWidget(self.expert_label)
        timeline_details.addWidget(self.memory_label, 1)
        timeline_layout.addLayout(timeline_details)
        self.timeline_frame.hide()
        root.addWidget(self.timeline_frame)
        controls = QHBoxLayout()
        self.note_label = QLabel("注意：这是单帧诊断，RNN 状态和上一动作会清零；中途帧不包含此前路线记忆。")
        self.note_label.setObjectName("note")
        self.note_label.setWordWrap(True)
        controls.addWidget(self.note_label, 1)
        self.run_button = QPushButton("运行推理")
        self.run_button.setObjectName("runButton")
        self.run_button.setMinimumSize(150, 48)
        self.run_button.clicked.connect(self.run_inference)
        controls.addWidget(self.run_button)
        root.addLayout(controls)
        result_frame = QFrame()
        result_frame.setObjectName("resultFrame")
        result_layout = QHBoxLayout(result_frame)
        self.action_label = QLabel("等待运行")
        self.action_label.setObjectName("actionResult")
        self.action_label.setAlignment(Qt.AlignCenter)
        self.action_label.setMinimumWidth(360)
        result_layout.addWidget(self.action_label)
        probabilities_widget = QWidget()
        probability_grid = QGridLayout(probabilities_widget)
        self.probability_bars = {}
        for row, label in enumerate(ACTION_LABELS):
            text, color = ACTION_VIEW[label]
            name_label = QLabel(text)
            name_label.setMinimumWidth(205)
            bar = QProgressBar()
            bar.setRange(0, 1000)
            bar.setValue(0)
            bar.setFormat("0.00%")
            bar.setStyleSheet("QProgressBar::chunk {{ background: {}; }}".format(color))
            probability_grid.addWidget(name_label, row, 0)
            probability_grid.addWidget(bar, row, 1)
            self.probability_bars[label] = (name_label, bar)
        else:
            probability_grid.setColumnStretch(1, 1)
            result_layout.addWidget(probabilities_widget, 1)
            root.addWidget(result_frame)
            self.status_label = QLabel("请选择一对 RGB/Depth 并填写 instruction。")
            self.status_label.setObjectName("status")
            self.status_label.setWordWrap(True)
            root.addWidget(self.status_label)
            self.setStyleSheet('\n            QMainWindow { background: #1e1e24; }\n            QWidget { color: #eeeeec; font-size: 14px; }\n            QLabel#heading { font-size: 25px; font-weight: 700; }\n            QLabel#sectionTitle { font-size: 16px; font-weight: 600; }\n            QLabel#note { color: #f6d32d; }\n            QLabel#previousActionHint { color: #9a9ca5; }\n            QLabel#status { color: #b8c5d6; }\n            QLineEdit, QTextEdit, QComboBox {\n                background: #2b2b33; border: 1px solid #555761;\n                border-radius: 6px; padding: 7px;\n            }\n            QPushButton {\n                background: #3d3d48; border: 1px solid #686a74;\n                border-radius: 6px; padding: 7px 14px;\n            }\n            QPushButton:hover { background: #50505d; }\n            QPushButton:disabled { color: #777780; background: #303038; }\n            QPushButton#runButton {\n                background: #3584e4; border: 0; font-size: 17px;\n                font-weight: 700;\n            }\n            QPushButton#runButton:hover { background: #4a96ed; }\n            QFrame#dropBox {\n                background: #292930; border: 2px dashed #686a74;\n                border-radius: 10px;\n            }\n            QFrame#dropBox[dragActive="true"] {\n                border-color: #62a0ea; background: #27364a;\n            }\n            QLabel#dropTitle { font-size: 18px; font-weight: 700; }\n            QLabel#pathLabel { color: #9a9ca5; font-size: 12px; }\n            QFrame#resultFrame {\n                background: #292930; border: 1px solid #4b4b55;\n                border-radius: 10px; padding: 10px;\n            }\n            QFrame#timelineFrame {\n                background: #252b35; border: 1px solid #40546f;\n                border-radius: 8px; padding: 6px;\n            }\n            QLabel#frameLabel { font-size: 16px; font-weight: 700; }\n            QLabel#actionResult { font-size: 26px; font-weight: 750; }\n            QProgressBar {\n                border: 1px solid #555761; border-radius: 5px;\n                background: #1e1e24; text-align: center; min-height: 20px;\n            }\n            ')
            if args.episode:
                self.episode_edit.setText(args.episode)
                self.load_episode_path(args.episode)

    def choose_checkpoint(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择 CMA checkpoint", self.checkpoint_edit.text(), "PyTorch checkpoint (*.pth)")
        if path:
            self.checkpoint_edit.setText(path)

    def choose_episode(self):
        path = QFileDialog.getExistingDirectory(self, "选择包含 episode.json 的目录", str(REPO_ROOT / "training/data"))
        if path:
            self.episode_edit.setText(path)
            self.load_episode_path(path)

    def load_episode_from_edit(self):
        path = self.episode_edit.text().strip()
        if not path:
            QMessageBox.warning(self, "缺少 Episode", "请拖入或选择包含 episode.json 的目录。")
            return
        self.load_episode_path(path)

    def load_episode_path(self, path):
        try:
            episode = load_episode(path)
            self.episode = episode
            self.episode_results = None
            self.previous_action_overrides = {}
            if self.default_initial_previous_action != "START":
                self.previous_action_overrides[0] = self.default_initial_previous_action
            self.episode_edit.setText(str(episode["manifest_path"].parent))
            self.instruction_edit.setPlainText(episode["instruction"])
            frame_count = len(episode["samples"])
            self.frame_slider.blockSignals(True)
            self.frame_slider.setRange(0, frame_count - 1)
            self.frame_slider.setValue(0)
            self.frame_slider.blockSignals(False)
            self.timeline_frame.show()
            self.run_button.setText("运行整段（保留记忆）")
            self.note_label.setText("多帧模式：在时间轴选中任意帧，可覆盖该帧读取的上一动作；未覆盖帧使用正常的模型上一动作，RNN 始终按顺序保留。画面仍来自专家录制轨迹，不是闭环仿真。")
            self.display_episode_frame(0)
            self.status_label.setText("已加载 Episode {}：{} 帧。点击运行整段以建立 RNN 记忆。".format(episode["episode_id"], frame_count))
        except Exception as error:
            QMessageBox.critical(self, "Episode 读取失败", str(error))

    def clear_episode(self):
        self.episode = None
        self.episode_results = None
        self.episode_edit.clear()
        self.timeline_frame.hide()
        self.run_button.setText("运行推理")
        self.note_label.setText("注意：这是单帧诊断，RNN 状态会清零；AUTO 表示 START，也可在上方指定一个上一动作；中途帧不包含此前路线记忆。")
        self.status_label.setText("已切换为单帧模式，请选择一对 RGB/Depth。")

    def initial_previous_action_changed(self, action):
        frame_index = self.frame_slider.value() if self.episode is not None else 0
        if action == "AUTO":
            self.previous_action_overrides.pop(frame_index, None)
        else:
            self.previous_action_overrides[frame_index] = action

        self.episode_results = None
        if self.episode is not None:
            try:
                self.display_episode_frame(self.frame_slider.value())
            except Exception as error:
                QMessageBox.critical(self, "帧读取失败", str(error))
                return

        self.status_label.setText(
            "第 {} 帧上一动作{}；当前共有 {} 个覆盖，请重新运行整段。".format(
                frame_index + 1,
                "恢复为 AUTO"
                if action == "AUTO"
                else "已覆盖为 {}".format(action),
                len(self.previous_action_overrides),
            )
        )

    def display_episode_frame(self, index):
        if self.episode is None:
            return
        self.initial_previous_action_combo.blockSignals(True)
        self.initial_previous_action_combo.setCurrentText(self.previous_action_overrides.get(index, "AUTO"))
        self.initial_previous_action_combo.blockSignals(False)
        sample = self.episode["samples"][index]
        self.rgb_path = str(sample["rgb_path"])
        self.depth_path = str(sample["depth_path"])
        if self.episode_results is not None:
            self.show_result(self.episode_results["steps"][index])
            return
        rgb = load_rgb(self.rgb_path)
        depth = load_depth_m(self.depth_path)
        self.rgb_box.set_file_preview(self.rgb_path, rgb)
        self.depth_box.set_file_preview(self.depth_path, colorize_depth(depth, self.args.min_depth, self.args.max_depth))
        self.frame_label.setText("帧 {} / {}".format(index + 1, len(self.episode["samples"])))
        self.expert_label.setText("专家动作：{}".format(sample["expert_action"]))
        self.memory_label.setText("记忆状态：尚未运行")
        self.previous_frame_button.setEnabled(index > 0)
        self.next_frame_button.setEnabled(index + 1 < len(self.episode["samples"]))

    def frame_changed(self, index):
        try:
            self.display_episode_frame(index)
        except Exception as error:
            QMessageBox.critical(self, "帧读取失败", str(error))

    def previous_frame(self):
        self.frame_slider.setValue(max(0, self.frame_slider.value() - 1))

    def next_frame(self):
        self.frame_slider.setValue(min(self.frame_slider.maximum(), self.frame_slider.value() + 1))

    def set_rgb_path(self, path):
        try:
            if self.episode is not None:
                self.clear_episode()
            rgb = load_rgb(path)
            self.rgb_path = str(Path(path).resolve())
            self.rgb_box.set_file_preview(self.rgb_path, rgb)
            self.status_label.setText("RGB 已加载：{}".format(rgb.shape))
        except Exception as error:
            QMessageBox.critical(self, "RGB 读取失败", str(error))

    def set_depth_path(self, path):
        try:
            if self.episode is not None:
                self.clear_episode()
            depth = load_depth_m(path)
            preview = colorize_depth(depth, self.args.min_depth, self.args.max_depth)
            self.depth_path = str(Path(path).resolve())
            self.depth_box.set_file_preview(self.depth_path, preview)
            valid = np.isfinite(depth) & (depth > self.args.min_depth) & (depth <= self.args.max_depth)
            self.status_label.setText("Depth 已加载：{}，有效像素 {:.2%}".format(depth.shape, float(valid.mean())))
        except Exception as error:
            QMessageBox.critical(self, "Depth 读取失败", str(error))

    def run_inference(self):
        instruction = self.instruction_edit.toPlainText().strip()
        if self.episode is None and not self.rgb_path:
            QMessageBox.warning(self, "缺少 RGB", "请拖入或选择 RGB 图片。")
            return
        if self.episode is None and not self.depth_path:
            QMessageBox.warning(self, "缺少 Depth", "请拖入或选择同一 sample 的 Depth。")
            return
        if not instruction:
            QMessageBox.warning(self, "缺少 Instruction", "请填写英文 instruction。")
            return

        arguments = {
            "checkpoint_path": self.checkpoint_edit.text().strip(),
            "instruction": instruction,
            "instruction_length": self.args.instruction_length,
            "min_depth": self.args.min_depth,
            "max_depth": self.args.max_depth,
            "force_cpu": self.args.cpu,
        }
        if self.episode is not None:
            arguments.update({
                "operation": "episode",
                "episode_path": str(self.episode["manifest_path"]),
                "initial_previous_action": "START",
                "previous_action_overrides": dict(self.previous_action_overrides),
            })
        else:
            selected_previous_action = self.initial_previous_action_combo.currentText()
            arguments.update({
                "operation": "single",
                "rgb_path": self.rgb_path,
                "depth_path": self.depth_path,
                "initial_previous_action": (
                    "START" if selected_previous_action == "AUTO" else selected_previous_action
                ),
            })
        self.run_button.setEnabled(False)
        self.run_button.setText("推理中…")
        self.status_label.setText("正在加载 checkpoint 并按时间顺序执行 {} CMA 推理…".format("整段" if self.episode is not None else "单帧"))
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.worker = InferenceWorker(arguments, self)
        self.worker.succeeded.connect(self.handle_inference_result)
        self.worker.failed.connect(self.show_error)
        self.worker.finished.connect(self.inference_finished)
        self.worker.start()

    def handle_inference_result(self, result):
        if result.get("mode") == "episode":
            self.episode_results = result
            self.display_episode_frame(self.frame_slider.value())
        else:
            self.show_result(result)

    def show_result(self, result):
        action = result["action"]
        action_text, action_color = ACTION_VIEW[action]
        self.action_label.setText(action_text)
        self.action_label.setStyleSheet("color: {};".format(action_color))
        for label, probability in result["probabilities"].items():
            name_label, bar = self.probability_bars[label]
            bar.setValue(int(round(probability * 1000.0)))
            bar.setFormat("{:.2%}".format(probability))
            font = QFont(name_label.font())
            font.setBold(label == action)
            name_label.setFont(font)
        else:
            self.rgb_path = result["rgb_path"]
            self.depth_path = result["depth_path"]
            self.rgb_box.set_file_preview(self.rgb_path, result["rgb_preview"])
            self.depth_box.set_file_preview(self.depth_path, result["depth_preview"])
            stats = result["instruction_stats"]
            if result.get("mode") == "episode":
                index = result["frame_index"]
                count = result["frame_count"]
                self.frame_label.setText("帧 {} / {}".format(index + 1, count))
                self.expert_label.setText("专家动作：{}｜{}".format(result["expert_action"], "一致" if result["is_correct"] else "不一致"))
                self.memory_label.setText("模型上一动作：{}{}｜RNN ‖h‖={:.3f}".format(result["previous_action"], "（覆盖）" if result.get("previous_action_overridden") else "", result["rnn_state_norm"]))
                self.previous_frame_button.setEnabled(index > 0)
                self.next_frame_button.setEnabled(index + 1 < count)
                episode_accuracy = self.episode_results["accuracy"]
                self.status_label.setText("整段准确率：{:.2%}｜截至本帧：{:.2%}｜设备：{}｜无效深度：{:.2%}｜未知词：{}｜{}".format(episode_accuracy, result["cumulative_accuracy"], result["device"], result["invalid_depth_fraction"], stats.get("unknown_count", 0), result["checkpoint"]))
            else:
                self.status_label.setText("设备：{}｜无效深度：{:.2%}｜指令 token：{}｜未知词：{}｜{}".format(result["device"], result["invalid_depth_fraction"], stats.get("length", 0), stats.get("unknown_count", 0), result["checkpoint"]))

    def show_error(self, message, details):
        print(details, file=(sys.stderr))
        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Critical)
        dialog.setWindowTitle("推理失败")
        dialog.setText(message)
        dialog.setDetailedText(details)
        dialog.exec_()
        self.status_label.setText("推理失败：{}".format(message))

    def inference_finished(self):
        QApplication.restoreOverrideCursor()
        self.run_button.setEnabled(True)
        self.run_button.setText("运行整段（保留记忆）" if self.episode is not None else "运行推理")
        if self.worker is not None:
            self.worker.deleteLater()
            self.worker = None


def build_parser():
    parser = argparse.ArgumentParser(description="Visualize deterministic CMA actions from one RGB-D pair or a stateful episode replay.")
    parser.add_argument("--checkpoint",
      default=(str(default_checkpoint_path())))
    parser.add_argument("--instruction", default="")
    parser.add_argument("--episode")
    parser.add_argument("--instruction-length", type=int, default=200)
    parser.add_argument("--min-depth", type=float, default=0.0)
    parser.add_argument("--max-depth", type=float, default=10.0)
    parser.add_argument("--initial-previous-action",
      choices=([
     "START"] + ACTION_LABELS),
      default="START",
      help="Previous action embedding used on the first frame. The initial GRU state remains zero.")
    parser.add_argument("--cpu", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    if args.instruction_length <= 0:
        raise ValueError("--instruction-length must be positive.")
    if not args.max_depth > args.min_depth:
        raise ValueError("--max-depth must exceed --min-depth.")
    application = QApplication(sys.argv)
    application.setApplicationName("VLN RGB-D Action Visualizer")
    window = ActionGui(args)
    window.show()
    return application.exec_()


if __name__ == "__main__":
    sys.exit(main())
