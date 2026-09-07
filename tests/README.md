# 离线测试

此目录验证项目的软件契约，不打开相机、NPU、声卡或真实机械臂。测试主要使用 fake robot、合成 RGB-D、固定截图和临时本地 HTTP 端口。

## 测试文件

| 文件 | 覆盖范围 |
|---|---|
| `test_calibration.py` | 831 手眼标定副本与 Base→joint6→TCP→Camera 坐标链 |
| `test_cartesian_execution.py` | 稀疏轨迹加密、50 Hz 控制与关节不越界 |
| `test_config_and_voice.py` | 点位、对象 profile、目标语义、否定词、语音时序 |
| `test_demo_follow_integration.py` | 主程序随动按需加载、取消、线程和进程锁 |
| `test_generic_target_flow.py` | 瓶子/盒子 plan-only 流程不触发运动 |
| `test_hand_follow.py` | 单/多手门控、深度、20 mm 步进、IK、速度和安全距离 |
| `test_ik_seed.py` | 有界显式 seed 与可重复逆解 |
| `test_npu_postprocess.py` | 类别解码、边界框、类别感知 NMS |
| `test_npu_timeout.py` | NPU 子进程读取超时 |
| `test_perception_snapshot.py` | 同帧 RGB-D、深度、颜色、OBB 中心和采集/推理解耦 |
| `test_planner_safety.py` | 扫描、预抓取、复检、轨迹、重试、FULL_LOAD、放置与停机 |
| `test_recognition_profile.py` | object3 资产、类别、SHA-256 与 Lab 标定 |
| `test_ros_transport.py` | ROS2 帧时间戳、CameraInfo QoS、launch 和 Python ABI 脚本 |
| `test_streamer_ui.py` | Cookie、同源/JSON、命令槽、互斥状态机和网页控件 |
| `test_tts_safety.py` | TTS 私有临时目录、输出路径语义与播放后清理 |
| `fixtures/blocks_scene.png` | 颜色边界与视觉回归固定图 |

## 运行

完整回归：

```bash
source /opt/ros/humble/setup.bash
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate pathera_grasp
cd /home/ubuntu/A_shen_arm/pathera_grasp
PYTHONDONTWRITEBYTECODE=1 python tools/run_offline_tests.py
```

单文件：

```bash
export PYTHONPATH=Panthera-HT_SDK/panthera_python/scripts
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -p 'test_planner_safety.py'
```

## 编写规则

- 新运动门限必须同时有“允许边界内”和“拒绝边界外”测试。
- 测试不得实例化真实 `hightorque_robot` 连接、RealSense pipeline 或 QNN server。
- 涉及线程时要验证机械臂方法只在主线程被调用。
- 涉及网页时要验证未认证、跨源、错误 Content-Type、命令覆盖和错误状态均被拒绝。
- 涉及视觉时必须保持 RGB、depth、mask、detections 的同帧语义。
- 涉及失败恢复时要断言末端状态和动作顺序，不只检查返回值。

## 解释边界

测试全部通过不代表实机验收完成。关节摩擦、夹爪载荷、线缆、标定漂移、障碍物和急停只能通过受控现场流程验证。
