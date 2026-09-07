# panthera_vision

ROS2 视觉节点，负责初始化 RealSense、选择 CPU/NPU YOLOE 后端，并发布同帧 RGB、depth、标注图、检测 JSON 和相机内参。

## 入口

```bash
ros2 run panthera_vision vision_node --ros-args -p use_npu:=false
```

该命令会打开相机；只能在 `grasp_demo.py` 和其他 D405 进程停止后使用。`use_npu:=true` 还会占用 QNN runtime。

## 输出

- `/vision/image_raw`
- `/vision/depth_image`
- `/vision/annotated`
- `/vision/detections`
- `/vision/detections_stamped`
- `/vision/camera_info`

三幅图和 stamped detections 必须使用同一采集时间戳；下游禁止把不同帧的 RGB、depth 和检测拼接。CameraInfo 使用 durable QoS 并周期发布。

## 状态

源码保留、当前非正式主线。可做独立相机联调，但不能据此启动 grasp brain 驱动实机。
