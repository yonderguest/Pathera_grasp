# Panthera-HT 彩色积木视觉抓取

这个项目在 **Panthera-HT 六轴机械臂** 上实现一个最小可运行的视觉抓取工作流：

**语言输入 → 相机识别 → 机械臂抓取 → 放置 → 安全回零**

## 工作流

1. 启动后先让机械臂回到 `HOME` 并打开夹爪。
2. 用户在终端输入要抓取的颜色，例如 `红色积木`、`黄色积木`、`蓝色积木`，也支持 `任意颜色`。
3. J1 轴从 `+2.30 rad` 扫描到 `-2.30 rad`，每个扫描位置刷新眼在手相机的手眼变换并检测目标。
4. 使用 YOLOE 分割模型识别积木，再通过 HSV 颜色投票确认颜色。
5. 将目标像素反投影到机械臂基座坐标系，计算夹爪接近方向、开合方向和抓取姿态。
6. 通过工作空间保护和逆运动学检查后，直接执行一次抓取。
7. 抓取成功后回到 `HOME`，在 `PUT1` 释放物体，再移动到 `PUT2`。
8. 退出时回到 `ZERO`，确认关节位置和速度稳定后停止电机。

夹爪闭合后会检查一次反馈力矩。如果低于 `grasp_min_force`（默认 `0.5`），程序会回到识别到目标时的扫描位姿重新识别并抓取一次；第二次仍低于阈值则回到 `HOME` 并重新询问颜色。

## 网页视觉推流

程序启动后会开启一个局域网 MJPEG 推流服务，默认地址：

```text
http://<本机IP>:8080/
```

如果 `8080` 已被占用，程序会自动尝试 `8081`、`8082` 等端口，并在终端打印实际使用的地址。

浏览器页面左侧显示原始 RGB 画面，右侧显示 YOLO 识别画面，并叠加：

- 目标类别
- HSV 判断出的颜色
- YOLO 置信度
- 目标距离，例如 `0.42 m`

终端会优先显示自动检测到的本机局域网 IP，例如 `http://10.62.12.57:8080/`，而不是不可直接访问的 `0.0.0.0`。

相机启动前网页会显示 `Waiting for camera ...`；相机启动后，在等待用户输入颜色的阶段就会开始推送实时 RGB 预览画面，不再长时间黑屏。

夹爪闭合时，右侧 YOLO 画面左上角会实时显示夹爪力矩反馈，例如：

```text
GRIPPER FORCE | 0.350 Nm
```

推流参数可以通过环境变量覆盖：

```bash
export VISION_STREAM_HOST=0.0.0.0
export VISION_STREAM_PORT=8080
export VISION_STREAM_JPEG_QUALITY=85
```

`VISION_STREAM_HOST=0.0.0.0` 表示局域网内其他计算机都可以访问；如果只允许本机访问，可改为 `127.0.0.1`。

## 目录结构

```text
pathera_grasp/
├── grasp_demo.py                                      # 项目入口
├── hand_eye_calibration.json                          # 手眼标定结果
├── mobileclip2_b.ts                                   # YOLOE MobileCLIP 文本编码器
├── models/
│   └── yoloe-26s-seg.pt                               # YOLOE 分割权重
├── third_party/
│   ├── graspnet-baseline/                             # GraspNet 模型与 CPU PointNet++ 实现
│   ├── graspnetAPI/                                   # GraspGroup、NMS、碰撞过滤
│   └── qnn/                                           # QNN HTP npu_server 与 YOLOE 上下文
├── tools/
│   └── test_graspnet_offline.py                       # GraspNet 候选离线验证
└── Panthera-HT_SDK/
    └── panthera_python/
        ├── Panthera-HT_description/                   # URDF 与 mesh，运动学加载依赖
        ├── robot_param/
        │   ├── Leader.yaml                            # 机械臂参数
        │   └── motor_param/                           # 电机参数
        ├── hightorque_robot/                          # 电机/机器人底层 Python 包
        └── scripts/
            └── Panthera_lib/
                ├── Panthera.py                # 机械臂底层控制
                ├── grasp_config.py            # 抓取工作流配置
                ├── vision_pipeline.py         # 相机、检测、位姿计算
                ├── vision_streamer.py         # 网页 MJPEG 推流与可视化
                ├── npu_inference.py           # QNN HTP NPU 检测器封装
                ├── graspnet_pipeline.py       # 可选：GraspNet 候选生成与坐标转换
                └── grasp_planner.py           # 高层规划、执行、安全停机
```

## 模块说明

### `grasp_demo.py`

项目入口。负责：

- 构造 `GraspConfig`
- 启动机械臂并回到 HOME
- 处理语言/终端颜色输入
- 组装视觉、规划和推流服务
- 调用 `GraspPlanner.run_grasp_loop`
- 处理 `SIGINT`/`SIGTERM` 安全退出

### `Panthera.py`

Panthera 机械臂的底层库，继承 `hightorque_robot.Robot`。保留：

- 电机初始化和状态读取
- 关节位置/速度控制
- 夹爪控制
- 正运动学、逆运动学
- 动力学和轨迹规划接口
- 本次抓取工作流所需的夹爪检查、等待稳定等底层方法

### `grasp_config.py`

集中管理抓取流程的所有参数，包括：

- HOME / PUT1 / PUT2 / ZERO 关节位置
- 夹爪开合参数
- 扫描范围
- 工作空间保护范围
- HSV 颜色判断阈值
- 颜色语言命令解析
- GraspNet 开关、checkpoint 路径、NMS/碰撞过滤和抓取候选参数

这样上层逻辑不再散落大量魔法数字。

`use_graspnet` 默认是 `False`，因此项目仍默认使用稳定的 OBB/Seeed 几何抓取路径；只有显式打开后才会切换到 GraspNet 候选生成。

### `vision_pipeline.py`

负责视觉侧逻辑：

- RealSense 相机初始化和深度/彩色对齐
- YOLOE 分割模型加载
- HSV 颜色投票
- 手眼矩阵加载
- 深度点反投影和基座坐标变换
- OBB 短轴到夹爪姿态的几何计算
- `CameraFeed` 持续采集相机画面，并与 YOLO 推理分离，保持推流帧率

### `vision_streamer.py`

负责把当前相机帧和检测结果推送到网页端：

- 原始 RGB 画面 MJPEG 推流
- YOLO 分割 mask、检测框和文字标注
- 局域网 `0.0.0.0:8080` 访问

### `npu_inference.py`

封装 QNN HTP 上的 YOLOE NPU 检测器：

- 管理 `npu_server` 常驻进程
- 输入 `1x3x640x640` 浮点图像
- 输出检测框、类别、置信度和实例 mask
- 配合 `vision_pipeline.detect_targets_npu()` 转换为项目统一的检测格式

### `graspnet_pipeline.py`

可选模块，仅在 `use_graspnet=True` 时加载。它只负责视觉侧候选生成：

- 从 YOLO 目标 mask 提取 RGB-D 点云
- 调用 `graspnet-baseline` 生成 6-DOF 抓取候选
- 使用 `graspnetAPI` 做 NMS、分数排序和碰撞过滤
- 将相机系抓取位姿转换到机械臂基座系

不直接控制机械臂。抓取候选的 workspace、IK、关节跳变等验证仍由 `grasp_planner.py` 完成。

### `grasp_planner.py`

负责高层规划与执行：

- 逆运动学求解和抓取姿态验证
- J1 扫描和候选目标筛选
- 夹爪开合、夹紧判断
- `run_grasp_loop`：颜色询问、扫描、抓取、受力判断、重试
- 抓取执行：OPEN → GRASP → CLOSE → HOME → PUT1 → PUT2
- ZERO 确认和安全停机

## 运行准备

运行前需要确认以下文件和路径：

- `Panthera-HT_SDK/panthera_python/robot_param/Leader.yaml` 存在且与当前机械臂一致。
- 手眼标定文件 `hand_eye_calibration.json` 存在，并包含有效的 `T_tcp_camera`。
- YOLOE 模型可用。默认路径已经放在项目内部：

  `models/yoloe-26s-seg.pt`

  如果需要换用其他权重，可以通过环境变量覆盖：

  ```bash
  export YOLOE_MODEL_PATH=/path/to/yoloe-26s-seg.pt
  ```

  MobileCLIP 文本编码器默认位于项目根目录 `mobileclip2_b.ts`。如果放置到其他位置，同样可以通过环境变量指定：

  ```bash
  export YOLOE_TEXT_ENCODER_PATH=/path/to/mobileclip2_b.ts
  ```

- 已安装 `hightorque_robot`、`pyrealsense2`、`ultralytics`、`opencv-python`、`numpy`、`pinocchio`、`scipy`、`pyyaml` 等依赖。

## 启动

```bash
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate pathera_grasp

cd pathera_grasp
python grasp_demo.py
```

启动后按提示输入颜色即可。退出输入 `q`，或者按 `Ctrl+C`。

## 启用 NPU 目标识别

项目已经接入 Qualcomm QNN HTP NPU。开启后 YOLOE 目标检测从 CPU 切换到 NPU：

```bash
YOLO_NPU=1 python grasp_demo.py
```

NPU 使用的资源：

```text
third_party/qnn/npu_server
third_party/qnn/yoloe-26s-seg_640_iq9075_qnn_brick6.bin
```

当前 NPU 模型是 6 类积木封闭集模型，HSV 颜色投票仍然在 CPU 完成。默认 NPU 置信度阈值为 `0.05`，可以在 `grasp_config.py` 中通过 `npu_confidence_threshold` 调整。

同时启用 NPU 和 GraspNet：

```bash
YOLO_NPU=1 GRASPNET_USE=1 python grasp_demo.py
```

## 启用 GraspNet 候选

当前默认关闭。项目已经内置两个第三方仓库：

```text
third_party/
├── graspnet-baseline/
└── graspnetAPI/
```

本机板端是 `aarch64 + CPU-only torch`，官方 `pointnet2` CUDA 算子无法使用。项目内已经把 `third_party/graspnet-baseline/pointnet2/pointnet2_utils.py` 替换为 CPU 版实现，`knn` 也已编译 CPU 版本。

### 依赖安装状态

当前 `pathera_grasp` 环境已安装：

- `graspnetAPI`
- `open3d`
- `trimesh`
- `transforms3d`
- `grasp_nms`
- `knn_pytorch` CPU 版
- 其余 graspnetAPI 依赖

Python 环境仍保持：

```text
numpy 1.26.4
torch 2.13.0+cpu
```

为了降低 CPU 推理耗时，GraspNet 输入点数已从原来的 `20000` 调整为：

```text
graspnet_num_point = 1024
```

如果实际抓取中发现 1024 点导致候选质量不足，可以再逐步提高到 `2048`。

### checkpoint

官方 RealSense checkpoint 已放入：

```text
third_party/graspnet-baseline/checkpoint-rs.tar
```

当前机器已成功使用该 checkpoint 加载 GraspNet 模型并生成候选。

### 验证

可先在不连接机械臂的情况下用保存好的 RGB-D 图片检查候选：

```bash
python tools/test_graspnet_offline.py \
  --data-dir /path/to/saved_frame \
  --checkpoint third_party/graspnet-baseline/checkpoint-rs.tar
```

然后在 `grasp_config.py` 中设置：

```python
use_graspnet = True
```

或从环境变量覆盖 checkpoint：

```bash
export GRASPNET_CHECKPOINT_PATH=/path/to/checkpoint-rs.tar
```

> GraspNet 的夹爪坐标系可能与 Panthera 末端坐标系不完全一致。应先离线可视化候选位姿，调整 `graspnet_gripper_fix_rotation`，再上真机低速验证。

`pathera_grasp` 是当前板端专用 Conda 环境，已经包含：

- 机械臂：`hightorque_robot`、`pinocchio`、`scipy`、`numpy<2`
- 相机：`pyrealsense2`、`opencv-python`
- 视觉：`torch` CPU、`torchvision`、`ultralytics`、`clip`

如果换一台机器迁移项目，建议按同样方式克隆已有 `panthera` 环境，再安装上述视觉依赖，避免直接改动系统 ROS 环境。

## 设计原则

代码按职责拆成四个模块，保持 **高内聚、低耦合**：

- `Panthera.py` 只负责机械臂底层能力。
- `vision_pipeline.py` 只负责视觉和几何计算。
- `npu_inference.py` 只负责 QNN HTP 检测器封装。
- `graspnet_pipeline.py` 只负责 GraspNet 候选生成和坐标转换。
- `grasp_planner.py` 只负责抓取任务编排。
- `grasp_demo.py` 只负责入口和用户交互。
