# grasp_bringup

ROS2 launch 包，负责装配 voice、vision、stream 和 grasp brain，并传递端口、语音、NPU 与 GraspNet 参数。

仅解析参数：

```bash
ros2 launch grasp_bringup grasp_system.launch.py --show-args
```

完整 launch 会启动相机节点并可能启动机器人节点。由于 `panthera_grasp_brain` 尚未适配 FULL_LOAD 显式放置流程，当前禁止把完整 launch 用于实机。

修改 launch 参数时运行 `tests/test_ros_transport.py`，重点确认 `use_voice` 同时传入 voice 与 brain，避免 brain 在语音关闭时仍等待命令。
