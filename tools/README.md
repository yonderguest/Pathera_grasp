# 诊断与验证工具

本目录的脚本分为“完全离线”“只启动网页”“相机/NPU”和“离线 GraspNet”四类。运行前先看下表，避免误占用硬件资源。

| 脚本 | 用途 | 相机 | NPU | 机械臂 |
|---|---|---:|---:|---:|
| `run_offline_tests.py` | 编译全部自研 Python 并运行 `tests/` | 否 | 否 | 否 |
| `streamer_ui_fixture.py` | 用合成 RGB-D 启动网页，检查布局/API | 否 | 否 | 否 |
| `analyze_candidate_geometry.py` | 离线分析候选抓取几何和补偿 | 否 | 否 | 否 |
| `test_graspnet_offline.py` | 读取保存的 RGB-D/掩膜，输出 GraspNet 候选 | 否 | 否 | 否 |
| `diagnose_npu_models.py` | 对给定图片扫描 QNN context、类别和阈值 | 否 | 是 | 否 |
| `test_recognition_camera.py` | D405 + NPU 物体/颜色烟雾测试 | 是 | 是 | 否 |
| `test_hand_follow_camera.py` | D405 + CPU YOLOE 手部烟雾测试 | 是 | 否 | 否 |

## 推荐命令

统一离线回归：

```bash
source /opt/ros/humble/setup.bash
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate pathera_grasp
cd /home/ubuntu/A_shen_arm/pathera_grasp
PYTHONDONTWRITEBYTECODE=1 python tools/run_offline_tests.py
```

网页夹具：

```bash
python tools/streamer_ui_fixture.py
# 打开终端打印的 http://<IP>:8091/
```

相机/NPU 识别：

```bash
YOLO_NPU=1 python tools/test_recognition_camera.py
```

CPU 手部检测：

```bash
python tools/test_hand_follow_camera.py \
  --frames 15 \
  --prompt hand \
  --confidence 0.10 \
  --output-dir /tmp/pathera_hand_smoke
```

GraspNet 离线输入目录应包含 `color.png`、`depth.png`、`intrinsic.json`，可选 `mask.png`：

```bash
python tools/test_graspnet_offline.py \
  --data-dir /path/to/saved_frame \
  --checkpoint third_party/graspnet-baseline/checkpoint-rs.tar
```

## 注意

- 名称包含 `camera` 或 `npu` 的脚本不会初始化机械臂，但会独占对应设备。
- 不要在 `grasp_demo.py` 或 ROS2 vision node 运行时启动相机工具。
- 输出到 `/tmp` 的诊断图片属于临时文件，不应提交。
- `run_offline_tests.py` 使用内存编译并设置时不会生成 `__pycache__`；运行其他脚本时建议设置 `PYTHONDONTWRITEBYTECODE=1`。
- 工具输出的候选或 IK 成功不等于允许真实运动，实机仍由 `ObjectGraspProfile` 与规划器安全门控制。
