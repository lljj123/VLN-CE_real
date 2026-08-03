# robot_sim_bundle

这是一个可复制的 TurtleBot 2（Kobuki + hexagons 支架 + Xbox 360 Kinect）Gazebo 11 / ROS Noetic 仿真包。
模型已经从 xacro 展开为独立 URDF，网格和两个 Gazebo 传感器/底盘插件也放在本目录内，不依赖原来的 `/home/oamr/turtlebot_ws/src` 路径。

## 包含内容

- `urdf/turtlebot_kinect_sim.urdf`：完整机器人模型。
- `meshes/`：Kobuki、平台、支撑柱和 Kinect 网格。
- `lib/`：Kobuki 底盘与 Kinect Gazebo 插件。
- `launch/robot_world.launch`：启动环境并生成机器人。
- `launch/spawn_robot.launch`：向已经运行的 Gazebo 中生成机器人。
- `worlds/empty.world`：最小测试环境。

## 在当前设备启动

先检查依赖：

```bash
cd /home/oamr/robot
./check.sh
```

使用包内空环境：

```bash
./start_world.sh
```

使用你自己的 Gazebo world/SDF（文件必须包含 `<world>`）：

```bash
./start_world.sh world_file:=/绝对路径/your_environment.world
```

指定初始位姿或关闭 GUI：

```bash
./start_world.sh world_file:=/绝对路径/map.world x:=1.0 y:=2.0 yaw:=1.57 gui:=false
```

如果你的 Gazebo 环境已经启动，只生成机器人：

```bash
./start_spawn.sh x:=0.0 y:=0.0 z:=0.05 yaw:=0.0
```

## 控制和主要话题

仿真底盘直接接收标准速度话题 `/cmd_vel`。例如：

```bash
rostopic pub -r 10 /cmd_vel geometry_msgs/Twist \
  "{linear: {x: 0.2}, angular: {z: 0.0}}"
```

主要输出包括：

- `/odom`、`/joint_states`、`/tf`：底盘里程计、关节和坐标变换。
- `/camera/rgb/image_raw`、`/camera/rgb/camera_info`：RGB 图像和内参。
- `/camera/depth/image_raw`、`/camera/depth/camera_info`：深度图和内参。
- `/camera/depth/points`：点云。
- `/mobile_base/sensors/imu_data`、`/mobile_base/sensors/core`：仿真 IMU 和底盘状态。
- `/mobile_base/events/bumper`、`/mobile_base/events/cliff`：碰撞与悬崖事件。

## 复制到另一台设备

```bash
scp -r /home/oamr/robot user@另一台设备:/home/user/robot
ssh user@另一台设备
cd /home/user/robot
./check.sh
./start_world.sh world_file:=/绝对路径/your_environment.world
```

目标设备需要 Ubuntu 20.04、ROS Noetic、Gazebo 11，并安装运行依赖：

```bash
sudo apt update
sudo apt install ros-noetic-gazebo-ros-pkgs ros-noetic-robot-state-publisher
```

本包内 `.so` 来自当前 x86_64/Ubuntu 20.04/ROS Noetic/Gazebo 11 环境；不同 Ubuntu、ROS、Gazebo 版本或 CPU 架构不能直接复用，需要在目标设备重新编译对应插件。

为避免与真实小车节点重名，仿真建议使用独立 ROS master，或者不要同时启动真实底盘和此仿真包。

## 来源

模型与配置来自当前设备的 TurtleBot、Kobuki 和 turtlebot_simulator ROS 包。模型与插件的来源及许可证类型记录在本包的 `package.xml` 和本说明中。
