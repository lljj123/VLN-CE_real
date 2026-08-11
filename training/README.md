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
不会覆盖前一次数据。新的闭环数据默认统一保存在
`training/data/real_episodes_0p4m_15deg`，不会与旧的 `0.40m/30°` episode
混合。临时改变设置时仍可使用 `--instruction`、`--episode-id` 或
`--split val` 覆盖配置文件。

程序收到同步 RGB-D 并检测到 `/cmd_vel` 有底盘订阅者后，会显示：

```text
w = 保存当前帧和 MOVE_FORWARD 标签，然后底盘前进一步
a = 保存当前帧和 TURN_LEFT 标签，然后底盘左转一次
d = 保存当前帧和 TURN_RIGHT 标签，然后底盘右转一次
s = 保存当前帧和 STOP 标签，停车并正常完成 episode
e = 停车并正常完成 episode，但不新增 STOP 标签
q = 紧急停车并放弃 episode
```

每次输入需要按 Enter。保存图像成功之后才会启动底盘；动作执行完并发送
零速度后才允许输入下一步。短转弯片段尚未到达导航终点时使用 `e`，不能用
错误的 `STOP` 标签结束。速度、前进距离和转向角度直接读取
`config/action_to_cmd_vel.json`。当前名义动作是前进 `0.40m @ 0.20m/s`
（约 2 秒）以及左右转 `15° @ 0.50rad/s`（名义约 0.52 秒）。

采集器默认订阅配置的 `/odom`，按实际里程计位移/偏航角达到 `0.40m/15°`
后停止，接近目标时自动降速。里程计缺失、过期或动作超时会立即停车，并把
当前episode标为错误以防错误动作序列进入训练。采集与推理验证共用同一个
`config/action_to_cmd_vel.json`，因此动作尺度和闭环容差一致。

定点强化转弯时，每次物理重置小车都重新启动一个episode。推荐从转角前约
0.8m开始，按实际需要输入：

```text
w, w, a ... a, w, w, e
```

也就是转弯前至少两步直行、转到对正、转弯后至少两步直行，再用 `e` 保存。
不要把搬回起点后的多组数据继续写进同一个episode。

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

### 1.1 人工门控 DAgger 采集

要采集模型实际运行时遇到的起点偏差、打滑和偏航状态，先编辑
`config/dagger_collection.json`，确认其中的 `checkpoint` 正是准备部署并改进的
权重，然后运行：

```bash
./training/start_dagger_collection.sh
```

这个节点自身完成 CMA 推理、人工确认、数据保存和离散底盘动作，不能同时运行
`start_vln_with_base.sh`、`ros_vln_inference.py` 或
`ros_action_to_cmd_vel.py`。默认仍写入
`training/data/real_episodes_0p4m_15deg`，因此旧专家完整轨迹和新 DAgger 轨迹会
在下次微调时共同使用；episode 前缀不同，不会覆盖旧数据。

每一步小车保持停止，程序显示模型建议、置信度和四个动作概率，然后等待：

```text
Enter = 专家认可模型建议，执行模型动作
w/a/d/s = 专家覆盖模型建议，执行专家动作
e = 正常结束但不增加 STOP
q 或 Ctrl-C = 停车并放弃本 episode
```

默认是安全门控：错误模型动作会在执行前被专家替换。起点扰动、车轮打滑以及
此前已接受动作造成的偏差仍会产生真实恢复画面。实验性的
`m w / m a / m d` 会“保存指定专家标签，但故意执行模型建议”，只有设置
`VLN_DAGGER_ALLOW_MODEL_MISTAKE=1` 才会开放；这会让小车主动进入模型错误状态，
必须在空旷区域、低速并有人随时 Ctrl-C 急停时使用。

需要临时换权重或仅验证流程时：

```bash
VLN_DAGGER_CHECKPOINT=training/checkpoints/my_run/best_robot.pth \
  ./training/start_dagger_collection.sh

./training/start_dagger_collection.sh \
  --episode-id dagger_dry_run_001 --dry-run --cpu
```

`--dry-run` 结束的 episode 会写成 `status: "dry_run"`，训练加载器会跳过，
不会把“画面不动但动作标签变化”的调试样本混进训练集。

采集器不会把 GRU 张量当作训练数据保存。它保存每一步的模型建议、专家标签、
真实执行动作及 GRU 范数诊断；训练时用专家动作计算 loss，用真实执行动作作为
下一帧的上一动作输入，从而用新权重重新构建正确的记忆。

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
training/data/real_episodes_0p4m_15deg/train/room01_run01/
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
+ training/data/real_episodes_0p4m_15deg/train/（真实专家数据）
-> training/checkpoints/real_cma_0p4m_15deg/best_robot.pth（新权重）
```

训练不会覆盖 `CMA_PM_DA_Aug_robot.pth`。每次采集脚本创建的是一个
episode；同一数据集根目录下所有 `status: complete` 的 train episode
会一起送入微调。

默认参数：

```text
100 epochs
batch size 2
连续 16 步序列
learning rate 1e-5
冻结 RGB/Depth ResNet、词嵌入和语言双向 LSTM
训练 CMA 注意力、RNN 和动作分类头
```

“连续16步序列”表示离线训练时，CMA的RNN一次按时间顺序处理最多16组
`RGB-D + 上一步动作 + 当前专家动作`，不代表小车会一次执行16个动作，也不改变
单步的0.40m/15°尺度。15°动作下一个90°转弯通常需要约6个转向动作，16步窗口
更容易同时包含转弯前直行、连续转向和转弯后直行。

默认冻结视觉编码器是为了降低显存和小数据过拟合风险。数据足够多后可：

```bash
./training/start_finetune_real.sh --train-visual-encoders
```

当前真实数据的导航指令缺少多样性，因此默认同时冻结官方预训练的语言
双向 LSTM，避免它反复拟合同一句指令。以后拥有大量不同英文指令后，可以
显式解冻语言 LSTM；词嵌入仍保持冻结：

```bash
./training/start_finetune_real.sh --train-instruction-lstm
```

可选类别平衡：

```bash
./training/start_finetune_real.sh --class-balance
```

输出：

```text
training/checkpoints/real_cma_0p4m_15deg/
├── best_robot.pth
├── latest_robot.pth
├── latest_training.pth
├── metrics_history.json
├── metrics_history.csv
├── loss_curve.png
└── accuracy_curve.png
```

`best_robot.pth` 可直接被小车加载；`latest_training.pth` 额外保存优化器
状态和历史指标，用于继续训练。每个 epoch 完成后会更新两张曲线图；存在
验证集时图中同时显示训练集与验证集曲线，没有验证集时只显示训练曲线。
JSON/CSV 文件保存绘图所用的原始数值：

```bash
VLN_TRAIN_EPOCHS=20 ./training/start_finetune_real.sh \
  --resume training/checkpoints/real_cma_0p4m_15deg/latest_training.pth
```

`VLN_TRAIN_EPOCHS` 表示最终 epoch 编号，不是额外增加的轮数；例如已经完成
10 轮，要再训练 10 轮就设为 20。

## 5. 从收敛模型进行 Scheduled Sampling 二阶段微调

不需要从官方预训练权重重新训练。下面的启动脚本默认读取已经收敛的 sequence-64
权重，使用新的优化器和较小学习率，在独立目录进行 100 轮二阶段微调：

```bash
./training/start_scheduled_sampling_finetune.sh
```

默认设置为前 5 轮只用专家上一动作，随后 45 轮把“采用模型自己上一动作”的概率
从 0 线性增加到 20%，余下轮次保持 20%。模型动作使用确定性 `argmax`，与小车
测试时一致。训练仍会保存 teacher-forcing 的 `train_accuracy`，并额外记录：

```text
wrong_model_previous_fraction  被采用的模型上一动作中，错误动作所占比例
free_running_loss              整段全部使用模型上一动作时的开环 loss
free_running_accuracy          整段全部使用模型上一动作时的开环准确率
```

可通过环境变量覆盖默认值，例如：

```bash
VLN_SS_MAX_PROB=0.30 \
VLN_SS_WARMUP_EPOCHS=5 \
VLN_SS_RAMP_EPOCHS=45 \
VLN_SS_EPOCHS=100 \
  ./training/start_scheduled_sampling_finetune.sh
```

默认输出到
`training/checkpoints/real_cma_seq64_scheduled_sampling/`，不会覆盖原来的
`real_cma_seq64_frozen_language` 权重。

## 6. 测试微调权重

```bash
VLN_CHECKPOINT=training/checkpoints/real_cma_0p4m_15deg/best_robot.pth \
./scripts/start_vln_with_base.sh
```

联合启动脚本会加载新权重，并继续使用同一个
`config/action_to_cmd_vel.json` 执行 `0.40m/15°` 动作。先在架空轮、低速
或安全区域测试。联合脚本会先等待 `/odom`，再启动VLN推理；验证动作同样按
里程计闭环结束。确认验证路线效果优于原权重后，再替换默认 checkpoint。

## 文件职责

```text
ros_record_real_episode.py  ROS RGB-D/专家动作轨迹采集
ros_expert_drive_collector.py  键盘专家采集并自动执行底盘动作
ros_dagger_collector.py       人工门控、记忆一致的 DAgger 采集
real_dataset.py             episode 校验、预处理和连续序列装载
finetune_real_cma.py        纯 PyTorch 行为克隆微调
start_record_real_episode.sh  一键启动深度处理与采集
start_expert_collection.sh  一键启动键盘专家采集与底盘控制
start_dagger_collection.sh  一键启动 DAgger 推理、接管与采集
start_finetune_real.sh        一键启动离线微调
start_scheduled_sampling_finetune.sh  从收敛权重进行 scheduled sampling 微调
```
