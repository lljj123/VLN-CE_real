# VLN-CE 与 VLN-CE_real：网络与记忆一致性检查

## 结论

1. **网络拓扑和权重一致。** `VLN-CE_real` 是原 CMA 推理路径的等价摘录。原 checkpoint 与 robot checkpoint 都有 520 个张量，逐张量 `torch.equal`，520/520 全部完全相等。robot checkpoint 记录的源 SHA256 也等于 `../VLN-CE/data/checkpoints/CMA_PM_DA_Aug.pth` 的实际 SHA256。
2. **单步记忆更新一致。** 给原模型和 standalone 模型相同 RGB、Depth、instruction、上一动作、mask 和两层状态，连续4步比较的 feature、两层新状态和 logits 最大绝对误差均为 0。
3. **序列实现方式不同但递归数学等价。** 原版 Habitat `RNNStateEncoder` 能一次处理 `[T*N,...]` packed sequence；standalone `GRUStateEncoder` 只接受单步 batch，真实训练用 Python 循环展开。独立16步、3 batch GRU 对比的输出最大误差为 `4.17e-7`，最终状态误差为 `3.58e-7`。
4. **当前训练和测试的“状态转移公式”一致，但“上一动作来源”不一致。** 训练使用专家上一动作，测试使用模型上一预测动作。预测一旦出错，下一步起两条记忆就会分叉。
5. **当前一键训练的时间边界已经对齐这批数据。** 一键脚本使用64步窗口，而最长 episode 为54步，因此每条训练轨迹都从起点到终点连续展开。直接运行 Python 训练器的默认值仍是16，会产生中途 reset。
6. **episode 结束处理尚未完全对齐。** 原版在 `done` 后把 mask 设0并重置下一 episode；当前 ROS 配置 `keep_running_after_stop=true`，STOP 后不会自动 reset。

## 代码对应关系

| 功能 | 原 VLN-CE | VLN-CE_real | 判断 |
|---|---|---|---|
| 两层 GRU | `vlnce_baselines/models/cma_policy.py:126,172` | `vlnce_real/model.py:221,266` | 相同 |
| START/上一动作 | `cma_policy.py:233-235` | `model.py:308-310` | 相同 |
| 状态选择指令注意力 | `cma_policy.py:248-263` | `model.py:317-333` | 相同 |
| 指令选择 RGB/Depth | `cma_policy.py:265-274` | `model.py:335-345` | 相同 |
| 第二层融合记忆 | `cma_policy.py:276-294` | `model.py:347-360` | 相同 |
| RNN 序列执行 | Habitat packed sequence | 单步 GRU + Python loop | 数学等价，调用方式不同 |
| 训练初始状态 | 每条完整 trajectory 置零 | 每个 sequence window 置零 | 64步脚本下当前数据等价 |
| 训练上一动作 | 采集轨迹实际执行动作 | 专家动作 | 当前真实数据为专家轨迹时一致 |
| 测试上一动作 | 模型上一预测动作 | 模型上一预测动作 | 相同 |
| episode 结束 | `done mask=0` | runner 需显式 `reset()` | 当前 ROS 自动化不一致 |

## 原版训练与测试的记忆

原 DAgger 训练把完整变长轨迹 pad 后整理成 `[T,N,...]`，只在 `t=0` 把 mask 置零。`RNNStateEncoder.seq_forward()` 根据 mask 构造 packed sequence，一次完成整条轨迹的两层 GRU 展开。动作 loss 对 padding 权重置零。

训练样本中保存的 `prev_action` 是产生下一观察时实际执行的动作；监督标签是 oracle action。DAgger 数据收集可能按 beta 混合 oracle 与模型动作，因此观察、上一动作在轨迹内部是一致的，同时能逐渐覆盖模型自身访问的状态。

原版测试从全零状态、mask=0 开始，之后保存模型预测动作和新 RNN 状态；环境 `done` 时 mask 变0，并重置该环境的上一动作。于是训练和测试使用相同的记忆公式和 episode 边界，差异主要是 IL/DAgger 设计允许的动作分布差异。

## 当前真实数据训练与测试的记忆

真实数据是专家驾驶轨迹。训练窗口内：

```text
observation[t] + expert_action[t-1] + hidden[t]
    -> hidden[t+1] + supervised expert_action[t]
```

实机测试中：

```text
observation[t] + predicted_action[t-1] + hidden[t]
    -> hidden[t+1] + predicted_action[t]
```

在真实闭环运行时，小车确实执行 predicted action，下一观察也来自该动作后的真实位置，所以“观察与上一动作”是自洽的。但训练数据只覆盖专家轨迹；一旦模型走偏，它遇到的观察和记忆状态可能不在训练分布内。

GUI 的整段 episode 回放更严格地说是开环：它保留 predicted previous action，但下一帧仍来自专家轨迹。因此它适合检查 RNN/上一动作敏感性，不等同于闭环导航成功率。

## 54步实际对照

使用当前 `training/checkpoints/real_cma_0p4m_15deg/best_robot.pth`（epoch 600）和最长54步训练 episode：

| 模式 | 上一动作来源 | 在专家画面上的动作正确率 |
|---|---|---:|
| 训练式 teacher forcing | 专家动作 | 53/54 = 98.15% |
| 测试式 free-running 开环回放 | 模型预测 | 33/54 = 61.11% |

第0步模型预测 `TURN_RIGHT`，专家标签是 `MOVE_FORWARD`。第0步的两层状态仍相同，因为两种模式都从 START 开始；第1步开始，两边分别输入 `TURN_RIGHT` 与 `MOVE_FORWARD`，两层记忆立即分叉。最终两份 RNN 状态的最大逐元素差为2.0。

这个结果不能解释为“实车成功率61.11%”，因为模型动作没有改变回放的后续画面；它能证明的是：**当前策略对上一动作非常敏感，teacher-forced 训练指标不能直接代表测试时的自回归记忆质量。**

对全部6条、137步数据复核后，teacher-forced 为136/137（99.27%），free-running 开环为116/137（84.67%）。其中5条 episode 的预测始终等于专家动作，所以两种记忆完全一致；54步 episode 在起点发生唯一一次 teacher-forced 分类错误，随后整条自回归记忆分叉，并贡献了后续大部分动作差异。这说明问题不是每条轨迹都会发生，而是具有明显的级联特征：没有早期错误时两者对齐，一次早期错误可能影响很长的后续序列。

## 其他影响一致性的差异

### 动作物理尺度

原 checkpoint 配置是前进0.25 m、转弯15°；当前实车配置是前进0.40 m、转弯15°。网络虽然接收同一个 `MOVE_FORWARD` token，但一步所代表的位移扩大了60%。基础 checkpoint 的记忆动态因此与实车尺度不完全一致；真实0.40 m数据微调是在校正这个差异。

### Progress Monitor

原 `CMA_PM_DA_Aug` 训练启用了 progress-monitor 辅助 loss，它从第二层512维状态预测路线进度。当前真实微调不计算该辅助 loss，并冻结该头。它不改变测试 forward 的动作与状态公式，但真实微调比原训练少了一项直接约束导航进度记忆的监督。

### 指令长度

原始 checkpoint 的旧配置记录最大指令长度153，当前训练/推理默认200。长度不超过153时，多出的全零 padding 会被 attention mask 屏蔽；超过153的长指令会与原训练分布不同。

### Checkpoint 选择

- GUI 默认选择 `training/checkpoints/real_cma_0p4m_15deg/best_robot.pth`，即当前微调权重。
- ROS `config/vln_inference.json` 默认仍选择基础 `data/checkpoints/CMA_PM_DA_Aug_robot.pth`。
- `data/checkpoints/best_robot.pth` 是另一个较旧的 epoch-400 副本，哈希也不同于训练目录中的 epoch-600 best。

因此若要验证真实微调后的记忆，ROS 启动时必须显式指定训练目录下的 best checkpoint；否则测试的不是刚训练的模型。

## 最终判断

### 对得上的部分

- 网络层、参数、状态布局、mask 清零规则、上一动作 embedding 和两层 GRU 更新公式。
- 当前64步一键训练下，这批最长54步数据从起点到终点的连续记忆。
- 训练逐步展开与测试逐步推理的数值递归。

### 没完全对上的部分

- 训练喂专家上一动作，测试喂模型上一动作；实际数据已显示一步错误即可使记忆分叉。
- 原 DAgger 会收集部分模型轨迹，当前纯专家行为克隆没有覆盖走偏后的恢复状态。
- 原版 `done` 自动 reset，当前 ROS 在 keep-running 模式下 STOP 不自动 reset。
- ROS 默认加载基础 checkpoint，不是当前微调 checkpoint。
- 原预训练动作尺度0.25 m，当前实车前进尺度0.40 m。

## 建议优先级

1. 测试微调模型时统一 checkpoint 路径，并在新 episode 开始前强制 `reset()`。
2. 给 ROS 推理增加明确的 episode start/end 或底盘动作完成协议，不把 `STOP` 与“继续保留状态”混在一起。
3. 在训练中加入 scheduled sampling、DAgger 或模型动作扰动后的恢复数据，缩小 expert-prev-action 与 predicted-prev-action 的差距。
4. 将 Python 参数默认16与启动脚本64统一，避免绕过脚本时悄悄改变记忆长度。
5. 增加 teacher-forced 与 free-running 两套验证指标；前者看分类拟合，后者看自回归记忆稳定性。
