# 真实小车专家数据格式（format_version 1）

数据集根目录固定为：

```text
training/data/real_episodes_0p4m_30deg/
├── train/
│   └── hallway_001/
│       ├── episode.json
│       ├── instruction.txt
│       ├── rgb/
│       │   ├── 000000.png
│       │   └── 000001.png
│       └── depth/
│           ├── 000000.npy
│           └── 000001.npy
└── val/
    └── another_route_001/
```

采集指令、默认数据划分、episode 名称前缀和 ROS 话题统一配置在
`config/expert_collection.json`。每次执行
`training/start_expert_collection.sh` 会创建一个新的 episode 目录。

一个目录就是一条连续 episode。同一 episode 内只能有一条英文导航指令，
样本顺序就是小车执行动作的时间顺序。不要把一条轨迹的相邻帧随机拆到
`train` 和 `val`；验证集应使用不同路线或不同场景。

## 图像文件

- `rgb/NNNNNN.png`：动作执行前的原始分辨率 BGR8 无损 PNG。加载后需
  转成 RGB，再使用部署时相同的裁剪/缩放流程。
- `depth/NNNNNN.npy`：动作执行前的原始分辨率、已对齐到 RGB 的
  `float32` 米制深度，形状为 `[H, W]`；无效深度为 `0.0`。
- RGB 和 Depth 是近似时间同步的一对数据，并且必须拥有相同的 `H, W`。
- 采集阶段不保存已经拉伸到 `224×224/256×256` 的图，避免永久写入几何
  畸变；训练加载器再统一执行模型预处理。

## 指令文本

`instruction.txt` 是 UTF-8 单行英文文本，例如：

```text
Go down the hallway, turn left into the office, and stop by the chair.
```

相同文本也保存在 `episode.json` 的 `instruction` 字段中，训练程序以
JSON 字段为准，并使用 checkpoint 自带词表转换为 token。

## episode.json

这是 UTF-8 JSON 文件。动作编号必须保持 CMA 的固定顺序：

```text
0 STOP
1 MOVE_FORWARD
2 TURN_LEFT
3 TURN_RIGHT
```

最小可训练示例：

```json
{
  "format_version": 1,
  "source": "real_robot_ros1",
  "status": "complete",
  "episode_id": "hallway_001",
  "split": "train",
  "instruction": "Go down the hallway and stop by the chair.",
  "action_labels": [
    "STOP",
    "MOVE_FORWARD",
    "TURN_LEFT",
    "TURN_RIGHT"
  ],
  "samples": [
    {
      "index": 0,
      "rgb": "rgb/000000.png",
      "depth": "depth/000000.npy",
      "action": "MOVE_FORWARD",
      "action_index": 1,
      "rgb_stamp": 1785728307.261,
      "depth_stamp": 1785728307.280,
      "action_stamp": 1785728307.300,
      "sync_delta_ms": 19.0,
      "invalid_depth_fraction": 0.2529
    },
    {
      "index": 1,
      "rgb": "rgb/000001.png",
      "depth": "depth/000001.npy",
      "action": "STOP",
      "action_index": 0,
      "rgb_stamp": 1785728310.101,
      "depth_stamp": 1785728310.120,
      "action_stamp": 1785728310.140,
      "sync_delta_ms": 19.0,
      "invalid_depth_fraction": 0.2501
    }
  ]
}
```

实际采集器还会记录相机内参、源图尺寸、动作速度、计划执行时长及动作是否
执行完成。这些是附加元数据，现有 `training/real_dataset.py` 会安全忽略
不参与训练的字段。

只有 `status: "complete"` 的 episode 会进入训练。按 `s` 记录 `STOP`
后正常结束才会写入 `complete`；按 `q` 或 Ctrl-C 会写入 `aborted`，保留
文件供检查，但训练加载器会自动跳过。
