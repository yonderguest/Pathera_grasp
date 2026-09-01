# Panthera ROS2 分离工作区

> 当前项目在同一台 IQ9075 上运行，现场主线已切回根目录 `grasp_demo.py`。本工作区暂停使用，仅为未来跨进程或跨机器部署保留。

这是 `grasp_demo.py` 的候选分离式入口。它包含四个节点和一个 bringup 包：

| 节点 | 包 | 职责 |
|---|---|---|
| 语音 | `panthera_voice` | ASR listen、TTS say、半双工录音 |
| 视觉 | `panthera_vision` | RealSense、YOLOE、RGB-D/检测发布 |
| 推流 | `panthera_stream` | 独立 MJPEG 服务 |
| 抓取大脑 | `panthera_grasp_brain` | 独占机械臂，消费同步快照并复用 `GraspPlanner` |
| 启动 | `grasp_bringup` | launch 参数和节点装配 |

当前工作区还未纳入 Git。它不得与根目录单体 `grasp_demo.py` 同时运行，否则会争用相机、语音和机械臂。

## Topic 兼容性与帧同步

以下原有 topic 未改名、未改消息类型：

```text
/vision/image_raw        sensor_msgs/Image
/vision/depth_image      sensor_msgs/Image
/vision/annotated        sensor_msgs/Image
/vision/detections       std_msgs/String（JSON list）
/vision/camera_info      std_msgs/String（JSON）
/voice/listen_request    std_msgs/Bool
/voice/command           std_msgs/String
/voice/say               std_msgs/String
/arm/status              std_msgs/String
```

为兼容地修复 RGB、depth、检测错帧，新增：

```text
/vision/detections_stamped  std_msgs/String
{
  "frame_seq": 123,
  "capture_timestamp_ns": 123456789,
  "detections": [ ... ]
}
```

vision node 为同一帧的 `image_raw`、`depth_image` 和 `annotated` 写入相同的 `Image.header.stamp`。grasp brain 只在两个图像和 `detections_stamped` 的 `capture_timestamp_ns` 完全匹配时才交给规划器；旧 `/vision/detections` 保留给既有订阅者。

`/vision/camera_info` 使用 reliable + transient-local QoS，并在启动后每五秒重发，因此 grasp brain 晚启动不会再因错过一次性内参消息而在机械臂 HOME/开夹爪后失败。

## 生命周期和语音

grasp brain 在拥有机器人前先等待 CameraInfo、手眼文件和可选 GraspNet 依赖。机械臂创建后，正常、异常与 Ctrl-C 路径均由 worker `finally` 调用有限时的 `safe_shutdown()`；主线程请求退出后会 join worker，避免 daemon 线程在回零途中被解释器直接丢弃。

`use_voice:=false` 会同时传给 `panthera_voice` 的 `voice_enabled` 和 grasp brain。此时 brain 不发布 listen request、不等待语音命令，headless 环境会安全退出，TTY 环境仅接受终端输入。

## 构建

```bash
source /opt/ros/humble/setup.bash
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate pathera_grasp

cd /home/ubuntu/A_shen_arm/pathera_grasp/ros2_ws
colcon build --symlink-install \
  --packages-select panthera_voice panthera_vision panthera_stream \
  panthera_grasp_brain grasp_bringup
source install/setup.bash
bash patch_shebangs.sh
```

TASK-003 已在不接硬件的前提下完成上述五个包的构建。构建输出中的 `setuptools/easy_install` 弃用警告来自系统 ROS 打包链，不阻断构建。

`bash patch_shebangs.sh` 不是可选步骤：本机 `/usr/bin/python3` 是 3.12，ROS Humble `rclpy` 使用 3.10 ABI。脚本会验证 `pathera_grasp` 环境确实是 Python 3.10，并可同时导入 `rclpy`、RealSense、OpenCV、Torch 和 Ultralytics，然后修正每个生成入口。每次重新构建后都要再次执行。

## 启动边界

理论启动命令：

```bash
ros2 launch grasp_bringup grasp_system.launch.py \
  stream_port:=8080 \
  voice_prompt_duration:=3.5 \
  use_voice:=false \
  use_npu:=false \
  use_graspnet:=false
```

已使用 `ros2 launch grasp_bringup grasp_system.launch.py --show-args` 成功加载该 launch 文件；这只验证参数解析，不会启动节点或硬件。真实节点联调仍须取得真机授权并按阶段执行。

即使环境就绪，首次真机运行前仍必须确认：

- `hand_eye_calibration.json` 对应当前相机、TCP、基座和末端工具；当前项目标定时间为 `2026-08-31 07:18:17`。
- CAN/串口权限、急停、HOME/PUT/ZERO 轨迹和工作空间均由现场负责人检查。
- 只运行一个机械臂拥有者：单体或 ROS2 grasp brain 二选一。
- NPU、相机、声卡和机械臂硬件测试均另行授权。

## 已知范围

ROS 通讯已具备 durable CameraInfo、带时间戳的帧关联、`use_voice` 一致性和可 join 的抓取 worker。它不等于已经完成真机验收：没有运行实际相机、机械臂、NPU 或声卡，也没有经验证的环境/自碰撞规划。
