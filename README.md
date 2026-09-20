# VLN-CE Real Robot

这是面向 ROS1 小车的最小 VLN-CE CMA 推理项目。程序订阅已对齐的
RGB/Depth 图像，保留跨帧 RNN 状态，并发布英文离散动作。

小车推理路径不创建仿真环境，也不依赖 Habitat-Lab、Habitat-Baselines、
Habitat-Sim、Gym、TorchVision、Matterport3D 或 VLN 训练数据集。
真实小车数据采集和纯 PyTorch 微调工具放在 `training/`；训练同样不依赖
Habitat 或虚拟环境。

## 从 GitHub 克隆

模型权重超过 GitHub 普通单文件限制，使用 Git LFS 保存。克隆机器需要先
安装 Git LFS，然后执行：

```bash
git lfs install
git clone <repository-url>
```

若只下载 GitHub 自动生成的源码 ZIP，可能不包含完整 LFS 权重，推荐使用
`git clone`。

## 数据流

```text
/camera/depth_registered/image_raw (16UC1 mm 或 32FC1 m)
  -> ros_depth_hole_filler.py
  -> /camera/depth_registered/image_filled (32FC1 m)

/camera/rgb/image_color + /camera/depth_registered/image_filled
  -> 近似时间同步
  -> RGB/Depth Tensor
  -> 纯 PyTorch CMA
  -> /vln/action (std_msgs/String)
  -> ros_action_to_cmd_vel.py
  -> /cmd_vel (geometry_msgs/Twist)
  -> /vln/action_result (std_msgs/String)
  -> 完成后使用新的 RGB-D 进行下一次推理
```

动作内容只有：

```text
STOP
MOVE_FORWARD
TURN_LEFT
TURN_RIGHT
```

## 必要依赖

- ROS Noetic：`rospy`、`rostopic`、`rosnode`
- ROS 消息：`sensor_msgs`、`std_msgs`、`geometry_msgs`、`nav_msgs`
- ROS 图像：`cv_bridge`、`message_filters`
- Python：NumPy、PyTorch、OpenCV
- 可视化监控：Tk（Ubuntu/ROS Noetic 通常安装 `python3-tk`）

验证环境为 Python 3.6.15、PyTorch 1.10.2、NumPy 1.19.5、
OpenCV 4.5.5。小车若为 Jetson，应安装与其 JetPack 匹配的 PyTorch，
不要安装普通 x86 CUDA wheel。

## 启动

先启动 ROS Master 和深度相机驱动，然后执行：

```bash
cd /path/to/VLN-CE_real
./scripts/start_vln_real.sh
```

VLN 推理的英文指令、checkpoint、RGB/Depth 输入话题、动作输出话题、
`cmd_vel` 底盘话题、同步容差、动作间隔和深度范围统一保存在：

```text
config/vln_inference.json
```

通常直接修改该 JSON 后运行启动脚本即可。临时替换指令：

```bash
VLN_INSTRUCTION="Go forward and turn left." \
  ./scripts/start_vln_real.sh
```

如果 Python 环境不在默认位置：

```bash
VLN_PYTHON=/path/to/python ./scripts/start_vln_real.sh
```

测试另一个已导出的微调权重：

```bash
VLN_CHECKPOINT=data/checkpoints/CMA_finetuned_robot.pth \
  ./scripts/start_vln_real.sh
```

查看微调权重内部保存的模型信息、R2R词表和参数结构：

```bash
python3 scripts/inspect_checkpoint.py
python3 scripts/inspect_checkpoint.py --find-word hallway --find-word chair
python3 scripts/inspect_checkpoint.py --show-vocab
python3 scripts/inspect_checkpoint.py --list-tensors
```

默认检查 `training/checkpoints/real_cma_0p4m_15deg/best_robot.pth`，也可以
把其他 `.pth` 路径作为第一个参数传入。检查程序只读取文件，不启动ROS或模型
推理，也不会修改权重。

默认采用动作完成事件驱动并持续运行，不使用固定动作间隔：

```text
inference.max_actions = 0
inference.min_action_interval_seconds = 0.0
inference.wait_for_action_result = true
inference.action_result_timeout_seconds = 8.0
```

推理节点一次只发布一个带序号的动作。转换节点到达里程计目标后在
`/vln/action_result` 返回同一序号；推理节点随后丢弃运动期间的旧图像，等待
完成后的第一对新 RGB-D，再立即执行下一次推理。执行失败、里程计失效或结果
超时会终止推理，而不是让模型状态继续前进。

环境变量仍可临时覆盖配置，例如 `VLN_RGB_TOPIC`、
`VLN_DEPTH_RAW_TOPIC`、`VLN_ACTION_TOPIC` 和 `VLN_CHECKPOINT`。也可以把
`ros_vln_inference.py` 的参数直接放在启动脚本后面，参数优先级最高。

## 驱动底盘

首次测试应架空车轮或断开电机，并把速度设得很低。确认底盘订阅的话题确实是
`/cmd_vel` 后，可一键启动深度处理、VLN 推理和动作转换：

```bash
cd /path/to/VLN-CE_real
./scripts/start_vln_with_base.sh
```

转换节点订阅 `/vln/action` 的英文动作或带序号 JSON 命令，并以 20 Hz 连续发布
`geometry_msgs/Twist`。速度和动作尺度集中保存在
[`config/action_to_cmd_vel.json`](config/action_to_cmd_vel.json)，左右转可以
分别标定。默认映射如下：

```text
STOP          -> linear.x = 0,    angular.z = 0
MOVE_FORWARD  -> linear.x = 0.20, angular.z = 0，名义执行 0.40 m（约 2 秒）后停止
TURN_LEFT     -> linear.x = 0,    angular.z = +0.50，名义执行 15° 后停止
TURN_RIGHT    -> linear.x = 0,    angular.z = -0.50，名义执行 15° 后停止
```

正 `angular.z` 表示左转，负值表示右转。默认订阅 `/odom`
（`nav_msgs/Odometry`），以动作开始时的位姿为基准闭环测量前进距离和偏航角；
前进时根据初始偏航角做小幅方向纠偏，接近目标时自动降速。名义时长只作为
里程计失效时的安全超时依据。通常修改：

```text
MOVE_FORWARD.linear_speed_mps   前进线速度，单位 m/s
MOVE_FORWARD.distance_m         每个前进动作的距离，单位 m
TURN_LEFT.angular_speed_radps   左转角速度，单位 rad/s
TURN_LEFT.angle_deg             每个左转动作的角度，单位度
TURN_RIGHT.angular_speed_radps  右转角速度绝对值，单位 rad/s
TURN_RIGHT.angle_deg            每个右转动作的角度，单位度
topics.odom                     里程计话题
control.distance_tolerance_m    前进停止容差，单位 m
control.angle_tolerance_deg     转向停止容差，单位度
control.forward_heading_kp      前进航向纠偏比例系数
```

`/odom` 缺失、超过 `odom_stale_timeout_s` 未更新或在动作超时前没有达到目标
时，节点会立即发布零速度，不会退回不精确的开环动作。只有显式配置
`control.use_odom: false` 才启用旧的速度乘时间模式。

联合脚本默认读取该文件。以下环境变量仍可在本次启动时临时覆盖配置：

```bash
VLN_CMD_VEL_TOPIC=/mobile_base/cmd_vel \
VLN_ODOM_TOPIC=/odom \
VLN_LINEAR_SPEED=0.05 \
VLN_ANGULAR_SPEED=0.15 \
VLN_FORWARD_DISTANCE=0.20 \
VLN_TURN_ANGLE_DEG=10 \
  ./scripts/start_vln_with_base.sh
```

只启动转换节点进行独立测试：

```bash
cd /path/to/VLN-CE_real
python3 scripts/ros_action_to_cmd_vel.py
```

它会自动读取默认配置文件。也可以指定另一份配置：

```bash
python3 scripts/ros_action_to_cmd_vel.py \
  --config config/action_to_cmd_vel.json
```

单独启动这个 Python 节点不会启动相机和 CMA；需要 ROS Master 已运行，而且
`/vln/action` 已由 VLN 推理节点或其他程序发布。可用下面的命令手动测试：

```bash
rostopic pub -1 /vln/action std_msgs/String 'data: "MOVE_FORWARD"'
```

手工发布的纯英文动作仍然兼容。新动作会抢占旧动作；`STOP`、未知动作、
里程计过期、动作超时和节点退出都会发布零速度。联合启动时，推理节点会等待
当前动作结果，因此正常情况下不会用新动作抢占尚未完成的动作。

## Action 耗时监控

监控已和导航启动脚本分离。先保证 ROS Master 已启动，然后分别在两个终端运行：

```bash
# 终端 1：监控界面（建议先启动，以免漏掉第一个 action）
./scripts/start_vln_action_monitor.sh

# 终端 2：VLN 推理和底盘
./scripts/start_vln_with_base.sh
```

退出 `start_vln_with_base.sh` 不会关闭监控；监控需要在自己的终端按 `Ctrl+C`
退出。窗口实时显示：

- 图像消息转换、RGB-D 预处理、模型 forward 和推理总耗时；
- 底盘实际执行时间、命令到结果的端到端时间；
- 上一个 action 结果到下一次推理真正开始的间隔（首个 action 显示 `-`）；
- 里程计控制模式、目标距离/角度、实际完成进度和结果原因；
- 最近动作的端到端横向时间轴，以及单独放大的推理阶段时间轴。

每次运行会把完整记录写到：

```text
~/.ros/vln_action_metrics/action_metrics_YYYYMMDD_HHMMSS.csv
~/.ros/vln_action_metrics/action_metrics_YYYYMMDD_HHMMSS.jsonl
```

监控配置位于 `config/vln_inference.json` 的 `monitor` 段。没有桌面
`DISPLAY`、Tk 无法初始化或显式关闭 GUI 时，监控节点会以无界面模式继续写
CSV/JSONL，不影响底盘控制。常用临时覆盖：

```bash
# 不显示窗口，只记录文件
VLN_MONITOR_GUI=0 ./scripts/start_vln_action_monitor.sh

# 更换输出目录
VLN_MONITOR_OUTPUT_DIRECTORY=/data/vln_metrics \
  ./scripts/start_vln_action_monitor.sh
```

也可以绕过配置启动底层 Python 监控节点：

```bash
python3 scripts/ros_vln_action_monitor.py
```

## RGB-D Action 可视化（单帧/多帧记忆）

从真实数据集中选择一张 RGB（JPG/PNG）和同一 sample 的米制 Depth
（float32 NPY/NPZ），填写英文 instruction，然后查看确定性 action 和四类动作
概率：

```bash
./scripts/start_single_rgbd_action_gui.sh
```

单帧模式会把 GRU 状态清零，适合检查某一帧的视觉输出以及首帧上一动作输入。
也可以拖入包含 `episode.json` 的文件夹，按真实推理顺序运行完整 episode：

```bash
./scripts/start_single_rgbd_action_gui.sh \
  --episode training/data/real_episodes_0p4m_15deg/train/EPISODE_DIR
```

多帧模式会显示时间轴、模型动作概率、专家动作、累计准确率、模型读取的上一动作
以及 RNN 状态范数。在时间轴选中任意一帧后，可将“当前帧的上一步动作覆盖”改为
四种动作之一；重新运行整段时，该帧只替换上一动作 embedding，不清空此前 GRU
状态，之后的记忆会继续从修改后的结果演化。切回 `AUTO` 即撤销该帧覆盖。

可用 `--checkpoint` 切换权重，用 `--initial-previous-action` 指定首帧上一动作。
该工具不依赖 ROS，但需要 PyQt5。episode 画面来自专家采集轨迹，因此这是带记忆
的开环回放，不是模型动作驱动新画面的闭环仿真。

## 保留文件

```text
data/checkpoints/CMA_PM_DA_Aug_robot.pth  权重、R2R 词表和动作元数据
config/vln_inference.json                 VLN 指令、RGB-D 输入和推理配置
config/action_to_cmd_vel.json              底盘速度、距离和转角配置
vlnce_real/                               独立 PyTorch CMA 网络
scripts/ros_depth_hole_filler.py          深度单位转换与小孔洞填充
scripts/ros_vln_inference.py              RGB-D 同步、推理和动作发布
scripts/inspect_checkpoint.py             查看 pth 元数据、词表和参数结构
scripts/start_vln_real.sh                 一键启动
scripts/ros_action_to_cmd_vel.py          英文动作到 Twist 的安全转换
scripts/ros_vln_action_monitor.py         Action 耗时记录和实时可视化
scripts/start_vln_action_monitor.sh       单独启动 Action 耗时监控
scripts/start_vln_with_base.sh             推理与底盘控制一键启动
scripts/single_rgbd_action_gui.py          单帧/episode 记忆与上一动作调试 GUI
scripts/start_single_rgbd_action_gui.sh    启动 RGB-D Action 调试 GUI
training/ros_dagger_collector.py           人工门控 DAgger 数据采集
training/start_dagger_collection.sh        一键启动 DAgger 采集
```

## 可选训练区

使用真实小车 RGB-D、英文指令和人工/遥控专家动作微调时，查看
[`training/README.md`](training/README.md)。采集发生在真实小车，训练
计算可以放到 GPU 训练机；两边都不需要 Habitat。
