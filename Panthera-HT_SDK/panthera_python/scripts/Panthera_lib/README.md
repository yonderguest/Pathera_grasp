# Panthera_lib 项目核心库

此目录位于厂商 Python SDK 内，但其中列出的模块构成 `pathera_grasp` 的项目适配层。修改这里可能直接影响真实机械臂、相机、NPU 或网页控制，必须同时更新离线测试和根目录运行说明。

## 模块职责

| 模块 | 职责 | 是否接触硬件 |
|---|---|---|
| `Panthera.py` | 厂商机器人类适配、反馈刷新、FK/IK、MoveJ/MoveL、连续 MIT 轨迹、夹爪与停稳检查 | 是：CAN/电机 |
| `grasp_config.py` | `GraspConfig`、`ObjectGraspProfile`、目标文本解析、限位与标定常量校验 | 否 |
| `vision_pipeline.py` | D405 初始化、采集/推理解耦、RGB-D 同帧快照、颜色/深度/抓取几何 | 是：相机；可调用模型 |
| `npu_inference.py` | 管理 `npu_server` 子进程、QNN 输入输出、解码、NMS、超时和关闭 | 是：NPU runtime |
| `lab_color.py` | 标定 Lab 特征提取与颜色分类 | 否 |
| `grasp_planner.py` | 扫描、预抓取、复检、IK、笛卡尔接近、夹持、重试、FULL_LOAD、放置和停机 | 是：经机器人对象 |
| `hand_follow.py` | CPU YOLOE 手部检测、三帧门控、深度审核、安全小步、HOLD 与回 HOME | 由主线程间接驱动机器人 |
| `vision_streamer.py` | MJPEG、网页 UI、Cookie/同源/JSON 校验、互斥状态机与命令队列 | 不直接驱动机器人 |
| `graspnet_pipeline.py` | 从 RGB-D/掩膜构建点云并生成 GraspNet 候选 | 可用 GPU/CPU，不直接驱动机器人 |
| `__init__.py` | 延迟导出，避免导入包时提前打开硬件依赖 | 否 |

## 关键接口契约

### 线程边界

- 机器人、FK/IK、MoveJ/MoveL 和夹爪调用只在主控制线程执行。
- `CameraFeed` 使用采集线程和推理线程，但只发布快照。
- HTTP 线程只校验请求、更新状态、把命令放入单槽队列。
- CPU 手部推理可在视觉工作线程进行，是否执行运动仍由机器人主线程决定。

### 坐标链

```text
Base → joint6（FK）→ TCP（tcp_in_joint6）→ Camera（hand_eye_calibration）
```

`object_base_position()`、`grasp_rotation_from_mask()` 和 `grasp_geometry()` 必须使用同一链路。当前固定 `grasp_offset_base` 为零；近距离误差由新鲜 RGB-D 动态修正。

### 状态机

主要网页控制状态：

```text
IDLE → GRASPING → FULL_LOAD → PLACING → IDLE
IDLE → FOLLOWING → IDLE
IDLE → JOGGING → IDLE
任意允许状态 → STOPPING
```

同一时刻只能有一种运动意图。新增按钮或 API 时必须先定义状态许可、队列覆盖规则、主线程消费点和停止行为。

### 新对象接入

新对象不能仅加入 YOLO prompt。必须新增/完成 `ObjectGraspProfile` 的尺寸、开度、闭合力矩、载荷门限、接近量、中心、姿态和深度策略；只有 `motion_ready` 为真才允许运动。积木的 `preserve_legacy_pipeline` 不得复用于瓶子或盒子。

## 验证

```bash
cd /home/ubuntu/A_shen_arm/pathera_grasp
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate pathera_grasp
export PYTHONPATH=Panthera-HT_SDK/panthera_python/scripts
PYTHONDONTWRITEBYTECODE=1 python tools/run_offline_tests.py
```

变更映射：

- 配置/解析：`test_config_and_voice.py`、`test_recognition_profile.py`
- 视觉/深度/坐标：`test_calibration.py`、`test_perception_snapshot.py`
- NPU：`test_npu_postprocess.py`、`test_npu_timeout.py`
- 运动/状态机：`test_planner_safety.py`、`test_cartesian_execution.py`、`test_ik_seed.py`
- 随动：`test_hand_follow.py`、`test_demo_follow_integration.py`
- 网页：`test_streamer_ui.py`

离线通过只证明软件契约，不证明真实轨迹、碰撞安全或抓取精度。
