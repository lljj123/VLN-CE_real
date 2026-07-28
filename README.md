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
- ROS 消息：`sensor_msgs`、`std_msgs`、`geometry_msgs`
- ROS 图像：`cv_bridge`、`message_filters`
- Python：NumPy、PyTorch、OpenCV

验证环境为 Python 3.6.15、PyTorch 1.10.2、NumPy 1.19.5、
OpenCV 4.5.5。小车若为 Jetson，应安装与其 JetPack 匹配的 PyTorch，
不要安装普通 x86 CUDA wheel。

## 启动

先启动 ROS Master 和深度相机驱动，然后执行：

```bash
cd /path/to/VLN-CE_real
./scripts/start_vln_real.sh
```

默认英文指令已保存在启动脚本中。临时替换指令：

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

默认每 5 秒最多推理并发布一次动作，持续运行：

```text
VLN_MAX_ACTIONS=0
VLN_MIN_ACTION_INTERVAL=5.0
```

## 驱动底盘

首次测试应架空车轮或断开电机，并把速度设得很低。确认底盘订阅的话题确实是
`/cmd_vel` 后，可一键启动深度处理、VLN 推理和动作转换：

```bash
cd /path/to/VLN-CE_real
./scripts/start_vln_with_base.sh
```

转换节点订阅 `/vln/action` 的英文动作，并以 20 Hz 连续发布
`geometry_msgs/Twist`。默认映射如下：

```text
STOP          -> linear.x = 0,    angular.z = 0
MOVE_FORWARD  -> linear.x = 0.10, angular.z = 0，执行到约 0.25 m 后停止
TURN_LEFT     -> linear.x = 0,    angular.z = +0.30，执行到约 15° 后停止
TURN_RIGHT    -> linear.x = 0,    angular.z = -0.30，执行到约 15° 后停止
```

正 `angular.z` 表示左转，负值表示右转。距离和角度目前使用“速度 × 时间”开环
计算，轮胎打滑和底盘加减速会产生误差，因此必须在真车上标定。例如：

```bash
VLN_CMD_VEL_TOPIC=/mobile_base/cmd_vel \
VLN_LINEAR_SPEED=0.05 \
VLN_ANGULAR_SPEED=0.15 \
VLN_FORWARD_DISTANCE=0.20 \
VLN_TURN_ANGLE_DEG=10 \
  ./scripts/start_vln_with_base.sh
```

只启动转换节点进行独立测试：

```bash
./scripts/ros_action_to_cmd_vel.py --linear-speed 0.05
rostopic pub -1 /vln/action std_msgs/String 'data: "MOVE_FORWARD"'
```

新动作会抢占旧动作；`STOP`、未知动作、动作超时和节点退出都会发布零速度。

## 保留文件

```text
data/checkpoints/CMA_PM_DA_Aug_robot.pth  权重、R2R 词表和动作元数据
vlnce_real/                               独立 PyTorch CMA 网络
scripts/ros_depth_hole_filler.py          深度单位转换与小孔洞填充
scripts/ros_vln_inference.py              RGB-D 同步、推理和动作发布
scripts/start_vln_real.sh                 一键启动
scripts/ros_action_to_cmd_vel.py          英文动作到 Twist 的安全转换
scripts/start_vln_with_base.sh             推理与底盘控制一键启动
```

## 可选训练区

使用真实小车 RGB-D、英文指令和人工/遥控专家动作微调时，查看
[`training/README.md`](training/README.md)。采集发生在真实小车，训练
计算可以放到 GPU 训练机；两边都不需要 Habitat。
