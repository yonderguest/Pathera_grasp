# Panthera ROS2 备选工作区（暂停使用）

当前正式入口是仓库根目录 `grasp_demo.py`。本工作区保存早期将语音、视觉、推流和抓取大脑拆成 ROS2 节点的兼容代码，仅供未来跨进程或跨机器部署参考；不得与单进程入口同时启动，也不属于当前实机验收范围。

## 重要兼容性警告

当前单进程规划器已经采用 `FULL_LOAD → 操作者点击放置 → PUT2/PUT1` 的两阶段流程，而 `panthera_grasp_brain` 仍未实现相应的 ROS 放置命令/回调。它不能完成现行持物待放状态机。

**在补齐放置接口、停止语义和回归测试前，不要启动 ROS2 grasp brain 驱动真实机械臂。** 离线测试只验证消息契约、帧同步、QoS 和生命周期片段，不代表 ROS2 实机链已恢复。

## 包结构

| 包 | 职责 | 文档 |
|---|---|---|
| `panthera_voice` | ASR listen、TTS say 与语音状态 | [说明](src/panthera_voice/README.md) |
| `panthera_vision` | RealSense、YOLOE、RGB-D/检测发布 | [说明](src/panthera_vision/README.md) |
| `panthera_stream` | 订阅图像并提供独立 MJPEG 页面 | [说明](src/panthera_stream/README.md) |
| `panthera_grasp_brain` | 消费同步快照并拥有机械臂 | [说明](src/panthera_grasp_brain/README.md) |
| `grasp_bringup` | launch 参数与节点装配 | [说明](src/grasp_bringup/README.md) |

## Topic

```text
/vision/image_raw           sensor_msgs/Image
/vision/depth_image         sensor_msgs/Image
/vision/annotated           sensor_msgs/Image
/vision/detections          std_msgs/String   # 旧 JSON list
/vision/detections_stamped  std_msgs/String   # 带 frame_seq/timestamp 的 JSON
/vision/camera_info         std_msgs/String   # JSON，transient-local
/voice/listen_request       std_msgs/Bool
/voice/command              std_msgs/String
/voice/say                  std_msgs/String
/arm/status                 std_msgs/String
```

同一 RGB、depth、annotated 消息使用同一个 `header.stamp`。`detections_stamped` 载荷示例：

```json
{
  "frame_seq": 123,
  "capture_timestamp_ns": 123456789,
  "detections": []
}
```

grasp brain 只有在 RGB、depth 与 stamped detections 时间戳完全匹配时才组装快照。`camera_info` 使用 reliable + transient-local，并周期重发，避免晚加入订阅者错过内参。

## 构建（仅源码/接口检查）

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

目标机 `/usr/bin/python3` 与 ROS Humble `rclpy` ABI/项目视觉依赖并不一致。每次 `colcon build` 后都要运行 `patch_shebangs.sh`；脚本会先验证 Conda Python 3.10 能导入 ROS 与视觉依赖，再修改生成入口。

可以只解析 launch 参数而不启动节点：

```bash
ros2 launch grasp_bringup grasp_system.launch.py --show-args
```

不要执行完整 launch 进行实机测试，除非已另行完成 FULL_LOAD 放置接口升级并获得现场授权。

## 离线回归

```bash
cd /home/ubuntu/A_shen_arm/pathera_grasp
source /opt/ros/humble/setup.bash
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate pathera_grasp
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -p 'test_ros_transport.py'
```

覆盖：帧时间戳不串帧、CameraInfo durable QoS、`use_voice` 参数一致性、Python 解释器修补脚本和 worker 可终止性。

## 恢复开发待办

- 定义 ROS2 侧 `FULL_LOAD`、`PLACING` 与显式放置请求的消息或 service/action。
- 保证放置意图只由 robot-owning worker 串行消费。
- 为持物状态下取消、SIGINT、节点异常和反馈丢失定义载物处置策略。
- 同步单进程网页的互斥状态机、安全校验和瓶子/盒子 fail-closed profile。
- 增加 fake-robot 测试，验证抓取成功后不会直接进入统一 shutdown。
- 完成后再进行 camera-only、fake-robot、低速真机三级验收。
