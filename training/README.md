# 真实小车数据微调

这里的采集和训练流程完全不使用 Habitat、Habitat-Sim、Matterport3D
或虚拟环境。模型训练与小车推理共用 `vlnce_real/` 中同一套纯 PyTorch
CMA 网络和同一个 checkpoint 词表。

## 正确的数据含义

每条训练样本必须是：

```text
英文导航指令
+ 动作执行前的同步 RGB
+ 动作执行前的 32FC1 米制 Depth
+ 人工/遥控器/可信控制器给出的专家动作
```

专家动作只能是：

```text
STOP
MOVE_FORWARD
TURN_LEFT
TURN_RIGHT
```

不能把模型发布的 `/vln/action` 当标签，否则模型只是在学习自己的错误。
采集器默认只接受独立的 `/vln/expert_action`。

## 1. 专家键盘采集并自动驱动底盘（推荐）

先启动 ROS Master、深度相机和底盘驱动，并停止 VLN 推理节点及
`ros_action_to_cmd_vel.py`，避免多个节点同时控制 `/cmd_vel`。先编辑：

```text
config/expert_collection.json
```

其中 `instruction` 是本批 episode 使用的英文导航指令，`topics` 配置
RGB、Depth、CameraInfo、专家动作和底盘话题。然后每条轨迹只需执行：

```bash
cd /path/to/VLN-CE_real
./training/start_expert_collection.sh
```

启动器会依次寻找 `vlnce_real`、`vlnce` Mamba 环境，然后回退到当前
`PATH` 中的 `python3`。小车没有 Mamba 环境时通常会自动使用
`/usr/bin/python3`。也可以显式指定：

```bash
VLN_PYTHON=/usr/bin/python3 ./training/start_expert_collection.sh
```

一次启动采集一个 episode，按 `s` 正常结束。采集同一路线多遍时，再次执行
同一条脚本即可；它会用配置中的 `episode_prefix` 和当前时间自动生成新目录，
不会覆盖前一次数据。当前大动作数据默认统一保存在
`training/data/real_episodes_0p4m_30deg`，不会与旧的 `0.25m/15°` episode
混合。临时改变设置时仍可使用 `--instruction`、`--episode-id` 或
`--split val` 覆盖配置文件。

程序收到同步 RGB-D 并检测到 `/cmd_vel` 有底盘订阅者后，会显示：

```text
w = 保存当前帧和 MOVE_FORWARD 标签，然后底盘前进一步
a = 保存当前帧和 TURN_LEFT 标签，然后底盘左转一次
d = 保存当前帧和 TURN_RIGHT 标签，然后底盘右转一次
s = 保存当前帧和 STOP 标签，停车并正常完成 episode
q = 紧急停车并放弃 episode
```

每次输入需要按 Enter。保存图像成功之后才会启动底盘；动作执行完并发送
零速度后才允许输入下一步。速度、前进距离和转向角度直接读取
`config/action_to_cmd_vel.json`。当前名义动作是前进 `0.40m @ 0.20m/s`
（约 2 秒）以及左右转 `30° @ 0.30rad/s`（约 1.745 秒）。

专家交互时，启动器默认用 `--log-every 0` 关闭深度节点的逐帧统计，避免日志
插入 `expert>` 输入行。需要诊断深度质量时可以临时恢复，例如：

```bash
VLN_DEPTH_LOG_EVERY=30 ./training/start_expert_collection.sh
```

第一次应使用架空轮或空旷安全区域验证。只测试采集、不让底盘运动：

```bash
./training/start_expert_collection.sh \
  --episode-id dry_run_001 --split train --dry-run
```

完整格式见 `training/DATASET_FORMAT.md`。采集器保存原始分辨率 RGB 和
Depth，不会把当前可能畸变的正方形 resize 结果永久写入数据集。

## 2. 通过外部专家动作话题采集

先启动 ROS Master、深度相机和真实小车的人工/遥控控制程序，然后执行：

```bash
cd /path/to/VLN-CE_real
VLN_INSTRUCTION="Go forward and turn left." \
VLN_EPISODE_ID="room01_run01" \
VLN_DATA_SPLIT="train" \
./training/start_record_real_episode.sh
```

小车每准备执行一个人工动作时，在动作执行前发布对应标签：

```bash
rostopic pub -1 /vln/expert_action std_msgs/String "data: 'MOVE_FORWARD'"
rostopic pub -1 /vln/expert_action std_msgs/String "data: 'TURN_LEFT'"
rostopic pub -1 /vln/expert_action std_msgs/String "data: 'STOP'"
```

每个标签会保存它之前最近的一对同步 RGB-D。`STOP` 会保存最后一帧并结束
该 episode。标签发布和真正的小车控制应由同一个人工控制程序协调；
上面的 `rostopic pub` 只演示标签格式，本身不会驱动电机。

采集结果：

```text
training/data/real_episodes_0p4m_30deg/train/room01_run01/
├── episode.json
├── rgb/000000.jpg
└── depth/000000.npy
```

`episode.json` 保存指令、动作顺序、时间戳、RGB/Depth 同步误差和无效深度
比例。RGB 保存为 BGR JPEG，Depth 保存为 float32 米制 NPY。

## 3. 单独采集验证集

不要把同一条路线的相邻片段随机拆成训练和验证数据。换一个 episode，
最好换路线或场景：

```bash
VLN_INSTRUCTION="Turn right and stop by the chair." \
VLN_EPISODE_ID="room02_val01" \
VLN_DATA_SPLIT="val" \
./training/start_record_real_episode.sh
```

## 4. 离线微调

建议把采集目录复制到有 NVIDIA GPU 的训练机上运行；训练数据来自真实
小车，但训练计算不必占用小车。训练机也不需要 Habitat：

```bash
cd /path/to/VLN-CE_real
./training/start_finetune_real.sh
```

这个命令默认执行的关系是：

```text
CMA_PM_DA_Aug_robot.pth（初始权重、词表、动作顺序）
+ training/data/real_episodes_0p4m_30deg/train/（真实专家数据）
-> training/checkpoints/real_cma_0p4m_30deg/best_robot.pth（新权重）
```

训练不会覆盖 `CMA_PM_DA_Aug_robot.pth`。每次采集脚本创建的是一个
episode；同一数据集根目录下所有 `status: complete` 的 train episode
会一起送入微调。

默认参数：

```text
10 epochs
batch size 2
连续 8 步序列
learning rate 1e-5
冻结 RGB/Depth ResNet 和词嵌入
训练 CMA 注意力、RNN 和动作分类头
```

默认冻结视觉编码器是为了降低显存和小数据过拟合风险。数据足够多后可：

```bash
./training/start_finetune_real.sh --train-visual-encoders
```

可选类别平衡：

```bash
./training/start_finetune_real.sh --class-balance
```

输出：

```text
training/checkpoints/real_cma_0p4m_30deg/
├── best_robot.pth
├── latest_robot.pth
└── latest_training.pth
```

`best_robot.pth` 可直接被小车加载；`latest_training.pth` 额外保存优化器
状态，用于继续训练：

```bash
VLN_TRAIN_EPOCHS=20 ./training/start_finetune_real.sh \
  --resume training/checkpoints/real_cma_0p4m_30deg/latest_training.pth
```

`VLN_TRAIN_EPOCHS` 表示最终 epoch 编号，不是额外增加的轮数；例如已经完成
10 轮，要再训练 10 轮就设为 20。

## 5. 测试微调权重

```bash
VLN_CHECKPOINT=training/checkpoints/real_cma_0p4m_30deg/best_robot.pth \
./scripts/start_vln_with_base.sh
```

联合启动脚本会加载新权重，并继续使用同一个
`config/action_to_cmd_vel.json` 执行 `0.40m/30°` 动作。先在架空轮、低速
或安全区域测试。确认验证路线效果优于原权重后，再替换默认 checkpoint。

## 文件职责

```text
ros_record_real_episode.py  ROS RGB-D/专家动作轨迹采集
ros_expert_drive_collector.py  键盘专家采集并自动执行底盘动作
real_dataset.py             episode 校验、预处理和连续序列装载
finetune_real_cma.py        纯 PyTorch 行为克隆微调
start_record_real_episode.sh  一键启动深度处理与采集
start_expert_collection.sh  一键启动键盘专家采集与底盘控制
start_finetune_real.sh        一键启动离线微调
```
