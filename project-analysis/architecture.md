# VLN-CE Real Robot — 网络与记忆架构

## 项目定位

这是一个面向 ROS1 小车的 VLN-CE（连续环境视觉语言导航）部署与真实数据微调项目。它从 Habitat 生态中剥离了 CMA 策略的单步循环推理路径，仅依赖 PyTorch、OpenCV 和 ROS：接收英文指令与同步 RGB-D，维持跨决策步的循环状态，输出四类离散动作，再由底盘控制节点把动作变成 `cmd_vel`。

当前基础 checkpoint 为 `VLN-CE CMA_PM_DA_Aug`，共 36,909,194 个状态值（float32 约 140.8 MiB），词表大小 2,504。

## 模块概览

| 模块 | 职责 | 关键文件 | 依赖/被依赖 |
|---|---|---|---|
| 策略网络 | 指令、RGB、Depth、上一动作的跨模态融合与循环决策 | `vlnce_real/model.py` | 调用视觉编码器；被推理、GUI、训练使用 |
| 视觉编码 | 提取 RGB/Depth 的 4×4 空间特征 | `vlnce_real/resnet.py` | 被 `CMANet` 使用 |
| 输入预处理 | 对齐训练与部署的 resize、深度过滤和归一化 | `vlnce_real/preprocessing.py` | 被 ROS 推理和数据集使用 |
| 在线推理 | 同步 ROS RGB-D，维护循环状态并发布动作 | `scripts/ros_vln_inference.py` | 调用策略与预处理 |
| 序列回放 | 展示逐帧概率、上一动作和 RNN 范数 | `scripts/single_rgbd_action_gui.py` | 模拟在线状态转移，但后续画面来自专家轨迹 |
| 数据装载 | 校验 episode，切连续窗口并补齐 batch | `training/real_dataset.py` | 被微调脚本使用 |
| 行为克隆微调 | 逐步展开 RNN，以专家动作做交叉熵训练 | `training/finetune_real_cma.py` | 调用数据集和同一策略网络 |
| 动作执行 | 把四类动作映射为里程计闭环速度控制 | `scripts/ros_action_to_cmd_vel.py`, `vlnce_real/odom_control.py` | 消费策略动作，不参与神经网络 |

模块图见 [`modules.png`](modules.png)，核心数据流见 [`dataflow.png`](dataflow.png)。

## 网络总览

一次策略步的输入与输出是：

```text
英文指令 token [B,L]
+ RGB [B,224,224,3]
+ Depth [B,256,256,1]
+ 上一步动作 [B,1]
+ 两层旧 GRU 状态 [B,2,512]
    ↓
CMA 跨模态注意力 + 两级循环更新
    ↓
动作 logits [B,4] + 两层新 GRU 状态 [B,2,512]
```

四个动作的固定顺序是 `STOP=0`、`MOVE_FORWARD=1`、`TURN_LEFT=2`、`TURN_RIGHT=3`。checkpoint 以 `strict=True` 加载，代码中的模块名、形状和动作顺序必须与原权重完全一致。

### 各分支的张量变化

| 分支 | 变换 | 输出 |
|---|---|---|
| 指令 | 2,504 词表 → 50 维冻结词嵌入 → 128×2 双向 LSTM | `[B,256,L]` token 特征 |
| RGB | 224×224 RGB → ImageNet 风格 ResNet-50 → 4×4 → 拼接 64 通道位置嵌入 | `[B,2112,16]` 空间特征 |
| Depth | 256×256 归一化深度 → 2×均值池化 → GroupNorm ResNet-50 → 128 通道压缩 → 拼接 64 通道位置嵌入 | `[B,192,16]` 空间特征 |
| 上一动作 | `START` 或四类动作 → 5×32 embedding | `[B,32]` |
| 动作头 | 第二层 GRU 输出 → Linear(512,4) | `[B,4]` logits |

RGB 和 Depth 都保留 4×4 空间网格，而不是只留下单个全局向量。额外的 64 通道 learned spatial embedding 给网络提供“这是画面中哪个位置”的信息。

## 记忆机制详解

### 1. 记忆到底存在哪里

运行时唯一显式的跨步记忆是：

```text
rnn_states[:, 0, :]  # 第一层 GRU，512 维
rnn_states[:, 1, :]  # 第二层 GRU，512 维
prev_actions         # 上一步动作，下一步映射为 32 维向量
```

它没有显式地图、访问历史列表、图节点、图像缓存或检索式长期记忆。两个 512 维向量是压缩后的隐式历史；无法直接把某一维解释成“走了几步”或“已经过了门”，这些信息若存在，是由训练自动编码进去的。

### 2. START 与 reset

`prev_action_embedding` 有 5 行：第 0 行是 START，后四行分别对应四个动作。实现不是给 START 单独定义动作编号，而是计算：

```text
embedding_index = (previous_action + 1) * mask
```

- `mask=False`：索引为 0，表示 episode 开始，同时两个 GRU 的旧状态会被清零。
- `mask=True`：索引为 `action+1`，表示使用真实的上一动作，并保留旧 GRU 状态。

在线 runner 初始化时将 `rnn_states` 置零、`mask` 置零；第一步预测后保存模型动作，并把 `mask` 设为一。此后每次 `predict()` 都把新状态和模型上一动作带到下一次决策。

### 3. 第一层记忆：为注意力提供历史上下文

先把 RGB 空间特征平均/线性投影到 256 维，把 Depth 展平/线性投影到 128 维，再拼上 32 维上一动作：

```text
[global RGB 256, global Depth 128, previous action 32]
                         ↓
                    416-dimensional
                         ↓
                 GRU-1 hidden 512
```

第一层 GRU 因而主要聚合“到目前为止看到了什么、刚做了什么”。它的 512 维输出投影成 query，在整句指令的 token 特征上做注意力。换句话说，历史状态决定当前更应该关注指令中的哪一段，例如从 “go down the hallway” 转到 “turn left”。

### 4. 跨模态注意力：语言反过来选择视觉位置

被第一层状态选中的指令向量再形成 query，分别在 RGB 的 16 个空间位置和 Depth 的 16 个空间位置上注意：

```text
GRU-1 state → 选择当前指令片段
当前指令片段 → 选择 RGB 位置、Depth 位置
```

注意力是 scaled dot-product，key/query 为 256 维，缩放因子是 `1/sqrt(256)`。RGB value 为 256 维，Depth value 为 128 维。

这就是 CMA（Cross-Modal Attention）的关键：记忆先影响语言焦点，语言焦点再影响视觉焦点，而不是简单把三种模态一次性拼接。

### 5. 第二层记忆：形成直接用于动作的导航状态

以下信息拼成 1,184 维：

```text
GRU-1 state 512
+ attended instruction 256
+ attended RGB 256
+ attended Depth 128
+ previous action 32
= 1184
```

它先被压缩到 512 维，再送入第二层 GRU。第二层的 512 维新状态既被保存为下一步记忆，也直接送入 `Linear(512,4)` 动作头。因此第一层偏“控制注意力”，第二层偏“融合后决策”，但两层都会跨步保留。

`progress_monitor: Linear(512,1)` 是原 checkpoint 保留的辅助头；当前 `forward()`、在线推理和真实数据微调均未调用它，微调时也明确冻结。

## 在线记忆生命周期

`CMARunner` 的状态转移如下：

```text
构造 runner
  → 指令只编码并缓存一次
  → reset(): h1=h2=0, mask=0, previous_action 占位为 0
  → 第一帧: 使用 START embedding，更新 h1/h2，预测 a0
  → 保存 a0, mask=1
  → 第二帧: 使用 h1/h2 和 a0，更新状态，预测 a1
  → ...
```

ROS 相机帧本身不等于记忆步。推理队列只保留最新同步 RGB-D，动作间隔内的帧会被丢弃；只有真正调用 `predict()` 时记忆才推进一步。

当前 runner 只在构造时调用 `reset()`。如果配置 `keep_running_after_stop=true`，模型预测 STOP 后节点仍继续运行，下一步会保留旧 GRU 状态，并把 STOP 当上一动作。若要在同一进程开始一条新指令/新 episode，应显式调用 `reset()`，且当前实现还需要重建 runner 才能替换缓存的指令 token。

## 训练如何学习记忆

数据集按 `sequence_length` 把真实 episode 切成连续窗口。Python 参数解析器的默认值是 16，但当前一键训练脚本 `start_finetune_real.sh` 会显式覆盖为 64。训练循环在一个窗口内逐步调用 `policy.net()`，不 detach 递归状态，因此 loss 可以沿窗口时间反传。

每一步的上一动作不是模型预测，而是前一帧的专家标签：

```text
step t 输入 expert_action[t-1]
step t 监督 expert_action[t]
```

这是 teacher forcing。它让训练稳定，但在线时上一动作来自模型自己；一次错误会改变后续动作 embedding 和观察轨迹，训练数据没有直接覆盖这种误差恢复情形。

### 当前窗口边界的含义

每个 batch/window 开始都重新创建全零 `rnn_states`、全零 `previous_actions`，并令第一步 mask 为零。窗口即使从 episode 的第 16、32、48 步开始，也会被模型视作全新 episode。因此：

- 梯度和状态都不会跨窗口延续；
- 直接运行 Python 训练器且不传参数时，`sequence_length=16, sequence_stride=16`；通过当前一键脚本训练时，两者均为 64；
- 中途窗口的第一帧丢失真实前序状态和上一专家动作；
- 调小 stride 会产生重叠窗口，增加上下文覆盖，但仍不会传递边界之前的状态。

这不影响原始预训练权重已经拥有的循环能力，但会限制真实数据微调对长时记忆的校正。

## 当前本地数据对记忆训练的约束

扫描到 6 条完整 train episode、共 137 步；动作数为：MOVE_FORWARD 89、TURN_LEFT 41、TURN_RIGHT 4、STOP 3，未发现 val episode。当前最长 episode 为 54 步，所以一键脚本实际使用的 64 步窗口能让这 6 条轨迹各自完整进入一个窗口；只有直接采用 Python 参数默认值 16 时，才会形成 12 个窗口，其中 6 个从 episode 中途开始。

这意味着当前一键微调的主要风险是数据量小、右转与停止极少、没有独立验证集，以及 teacher forcing 与部署自回归之间的差异。若绕过一键脚本直接使用 Python 默认的 16 步窗口，还会额外引入中途记忆重置边界。

## 关键实现位置

| 主题 | 位置 |
|---|---|
| 两层 GRU 的定义与状态维度 | `vlnce_real/model.py:149`, `vlnce_real/model.py:221`, `vlnce_real/model.py:266` |
| START/上一动作编码 | `vlnce_real/model.py:204`, `vlnce_real/model.py:308` |
| 第一层状态选择指令 | `vlnce_real/model.py:321-333` |
| 指令选择 RGB/Depth | `vlnce_real/model.py:335-345` |
| 第二层融合和状态更新 | `vlnce_real/model.py:347-361` |
| 在线 reset/状态保存 | `scripts/ros_vln_inference.py:154-184` |
| GUI 整段回放的记忆逻辑 | `scripts/single_rgbd_action_gui.py:297-375` |
| 序列窗口切分 | `training/real_dataset.py:157-170` |
| 训练逐步展开与 teacher forcing | `training/finetune_real_cma.py:234-290` |

## 已理解、待确认与关键决策

### 已理解

- 推理和训练共用同一个纯 PyTorch CMA 网络和 RGB-D 预处理。
- 在线记忆由两层 512 维 GRU 状态与上一动作组成。
- 第一层状态控制语言注意力，语言结果控制视觉注意力，第二层状态直接驱动动作。
- 实机推理保留跨动作状态；GUI 的 episode 模式复现相同状态更新，但画面是专家轨迹的开环回放。
- 真实数据训练在窗口内做 BPTT，窗口外完全重置，并采用专家上一动作 teacher forcing。

### 待确认/风险点

- 没有自动化测试验证 checkpoint 等价性、mask/reset 行为或训练/推理的一致性。
- `keep_running_after_stop=true` 时 STOP 不代表 episode reset；若上层把它当任务边界，会产生状态污染。
- ROS 推理用固定最小时间间隔近似“上一动作已经执行完”，策略节点本身没有消费底盘动作完成确认。
- `detach_state` 仅在验证路径中传入 true，而验证本身已处于 `torch.no_grad()`；该参数当前没有承担训练中的截断 BPTT 边界控制。
- 当前微调没有独立 val 数据，最优 checkpoint 实际依据 train loss 选择。

### 建议的下一步

1. 若目标是改善长路线记忆，优先让数据集返回窗口起点，并对中途窗口做前缀 warm-up，或按 episode 做有状态 truncated BPTT。
2. 增加 reset 单元测试：首帧必须使用 START，第二帧必须保留两层状态，episode 结束必须清零。
3. 增加 teacher-forcing 与 free-running 回放对比，量化上一动作误差累积。
4. 补充右转、STOP 和独立路线验证集，再判断是否需要延长 sequence length。
