# pathera_grasp MuJoCo 使用教程

本文是 `pathera_grasp` 的 MuJoCo 主教程，说明怎样准备独立环境、启动仿真、查看三维场景、执行识别与抓取、重新训练 YOLO，以及这套代码为什么能够形成闭环。文中的 `/home/ros/A_shen_arm/pathera_grasp` 是 WSL2 示例路径，实际可以换成自己的仓库路径。

## 1. 先理解当前仿真的运行方式

当前项目提供两种互不混用的入口：

- 真机入口：`grasp_demo.py`。它属于 IQ9075 板端 Panthera、RealSense、QNN/NPU 和语音链路，只能由现场操作者在完成安全检查后启动。
- WSL2 仿真入口：`run_mujoco.py`。它只延迟导入 `sim/`，不会初始化真机 SDK、CAN、RealSense、NPU、ROS2 或声卡。

仿真默认使用 EGL 离屏渲染。因此执行识别、单次抓取或完整分拣时，终端会打印结果，但不会像 Gazebo 一样自动弹出三维窗口。这并不表示仿真没有启动：MuJoCo 仍在内存中加载模型、推进物理、渲染腕部 RGB-D、求解 IK，并检查接触力、物体抬升和 PUT1 放置。

需要实时查看完整自动任务时，使用第 4.5 节的 `--viewer`；只检查静态场景时，才使用第 5 节的独立 Viewer。EGL 仍用于无窗口批量测试与录像。

## 2. 项目路径和关键文件

```text
/home/ros/A_shen_arm/pathera_grasp/
├── run_mujoco.py                        # WSL2/PC 专用 MuJoCo 入口
├── requirements-sim.txt                 # WSL 仿真 Python 依赖
├── hand_eye_calibration.json            # 原项目 TCP 到相机的手眼外参
├── sim/
│   ├── assets/panthera_follower.urdf    # Pinocchio 使用的六轴运动学模型
│   ├── assets/panthera_ht_stl/           # 官方本体 STL（link2_visual 为 MuJoCo 兼容降面版）
│   ├── assets/realsense_d405/             # 官方 D405 STL、Apache-2.0 许可与来源说明
│   ├── models/panthera_grasp_scene.xml  # MuJoCo 机械臂、夹爪、相机和工位
│   ├── models/block_detector.pt         # 已训练好的合成场景 YOLO 权重
│   ├── mujoco_backend.py                # 模型加载、控制步进、RGB-D 和接触接口
│   ├── recording.py                     # 外部视角 MP4 录像（不影响腕部 RGB-D）
│   ├── detection.py                     # colour/YOLO 检测及深度反投影
│   ├── ik.py                            # Pinocchio 6D IK
│   ├── grasp_pipeline.py                # 识别到抓取、夹持和抬升闭环
│   └── scripts/
│       ├── run_sim_demo.py               # recognition/grasp 两种模式
│       ├── train_synthetic_yolo.py       # 合成数据生成与 CPU 微调
│       └── decimate_stl_visual.py        # STL 视觉降面工具（仅 link2 需要）
└── reports/mujoco_sim/                  # 生成帧和训练结果，Git 忽略
```

## 3. 每次打开新终端后的准备

首次安装时新建独立环境，不要复用 IQ9075 真机环境：

```bash
source /home/ros/miniconda3/etc/profile.d/conda.sh
conda create -n pathera_mujoco python=3.10 -y
conda activate pathera_mujoco
conda install -c conda-forge pinocchio -y
cd /home/ros/A_shen_arm/pathera_grasp
export PYTHONNOUSERSITE=1
python -m pip install -r requirements-sim.txt
```

以后每次打开终端时激活该环境：

```bash
source /home/ros/miniconda3/etc/profile.d/conda.sh
conda activate pathera_mujoco
cd /home/ros/A_shen_arm/pathera_grasp

export MUJOCO_GL=egl
export PYTHONNOUSERSITE=1
export PYTHONPATH=$PWD
```

提示符显示 `(base)` 也不是错误。只要像下面这样明确使用 `pathera_mujoco` 的 Python，并设置三个环境变量，运行的仍是正确环境：

```bash
cd /home/ros/A_shen_arm/pathera_grasp
export MUJOCO_GL=egl
export PYTHONNOUSERSITE=1
export PYTHONPATH=$PWD

python --version
```

三个变量的作用：

- `MUJOCO_GL=egl`：使用离屏 OpenGL 上下文，不要求打开桌面窗口。
- `PYTHONNOUSERSITE=1`：阻止 `~/.local` 中旧 cmeel/coal 覆盖 Conda 内的 Pinocchio；缺少它时可能出现 `libboost_atomic.so.1.90.0` 错误。
- `PYTHONPATH=$PWD`：使 Python 能从项目根目录导入 `sim` 包。

也可以完全依赖 Conda 注入已保存的环境变量：

```bash
cd /home/ros/A_shen_arm/pathera_grasp
/home/ros/miniconda3/bin/conda run -n pathera_mujoco \
  env MUJOCO_GL=egl PYTHONNOUSERSITE=1 PYTHONPATH=$PWD python --version
```

## 4. 日常启动仿真的正确命令

以下命令均在项目根目录执行，并假设已经完成第 3 节的环境准备。

### 4.1 只验证 RGB-D 和颜色识别

```bash
python run_mujoco.py \
  --mode recognition \
  --detector colour
```

正常输出示例：

```text
DETECTION_COUNT=1 source=rendered_rgbd_colour
TARGET_WORLD=0.1903,0.0077,0.0250
```

含义：程序从腕部相机渲染图像，用确定性绿色阈值找到积木，再使用 depth 和相机外参计算 Base 坐标。该模式适合检查场景、相机和坐标链，但不属于最终 YOLO 验收。

### 4.2 执行完整三色 PUT1 分拣任务

```bash
python run_mujoco.py \
  --mode sort \
  --detector colour
```

它会在每次未指定 `--scene-seed` 时随机刷新红、绿、蓝三块 3×3×6 cm 积木，打印可复现的 `SCENE_SEED`，再依次完成绿色、红色、蓝色的识别/抓取/PUT1 放置。只有三块在最终检查中仍留在 PUT1，才打印 `TASK_COMPLETE=green,red,blue` 并停止。

### 4.3 使用已有权重进行 YOLO 识别

仓库已经包含 `sim/models/block_detector.pt`，日常使用不需要先训练：

```bash
python run_mujoco.py \
  --mode recognition \
  --detector yolo \
  --weights sim/models/block_detector.pt
```

正确的 YOLO 输出必须包含：

```text
DETECTION_COUNT=1 source=ultralytics_yolo
```

如果显示 `source=rendered_rgbd_colour`，说明执行的是 `--detector colour`，或者终端中看到的是上一条颜色抓取命令的输出。

### 4.4 使用 YOLO 执行完整抓取闭环

这是本项目最重要的验收命令：

```bash
python run_mujoco.py \
  --mode grasp \
  --detector yolo \
  --weights sim/models/block_detector.pt
```

通过时会看到类似结果：

```text
DETECTION_COUNT=1 source=ultralytics_yolo
PREGRASP_IK=reached:True error_m:0.0058
GRASP_IK=reached:True error_m:0.0047 orientation_error_deg:0.04
BILATERAL_CONTACT=True force_n=3.613,3.665
LIFTED_HEIGHT_M=0.0663
FINAL_TCP_WORLD=0.1700,0.0002,0.0911
```

只有检测、两段 IK、双指接触和抬升门限全部通过，进程才返回成功。

### 4.5 在 MuJoCo 窗口实时观看并操作抓取

你刚才的 `MUJOCO_GL=egl` 命令是**离屏**模式：物理、识别和抓取均在运行，但窗口不会出现。要让同一个抓取进程打开可交互的 MuJoCo Viewer，必须先清除 EGL 设置，再加入 `--viewer`：

```bash
cd /home/ros/A_shen_arm/pathera_grasp
source /home/ros/miniconda3/etc/profile.d/conda.sh
conda activate pathera_mujoco

unset MUJOCO_GL
export PYTHONNOUSERSITE=1
export PYTHONPATH=$PWD

python run_mujoco.py \
  --mode sort \
  --detector colour \
  --viewer \
  --camera-view
```

运行后会弹出两个 WSLg 窗口：**MuJoCo** 外部视角和 **Virtual RealSense D405 RGB** 腕部相机视角。二者共享同一份 `MjModel`/`MjData`；你会实时看到绿色、红色、蓝色依次抓取并放入 PUT1。轨迹按控制器的 20 ms 周期实时节流。

- 鼠标左键拖动：旋转视角；右键拖动：平移；滚轮：缩放（MuJoCo 原生操作）。
- 空格：暂停/继续控制与物理推进；终端会打印 `LIVE_VIEWER_PAUSED=True|False`。
- 关闭 Viewer 窗口：中止本次仿真；不会访问真机。
- `--viewer` 不能和 `export MUJOCO_GL=egl` 同时使用；若仍保留 EGL，程序会明确报出如何修正。保留 EGL 则回到下节的无窗口批量/录像模式。

如果 `$DISPLAY` 和 `$WAYLAND_DISPLAY` 都为空，说明当前 WSL 会话没有图形转发。执行 Windows PowerShell 的 `wsl --update` 后重新打开 WSL；在不能使用 WSLg 的机器上，继续使用 EGL 录像即可完成全部自动化验收。

### 4.6 用浏览器控制 MuJoCo 仿真

网页模式不会导入 `grasp_demo.py`、CAN、真机 RealSense 或机械臂 SDK。它只在本机启动一个 HTTP 服务：浏览器按钮向 MuJoCo 主线程排队命令，主线程独占执行物理、IK 和抓取。默认只监听本机回环地址，适合从 Windows 浏览器访问 WSL2 服务。

```bash
cd /home/ros/A_shen_arm/pathera_grasp
export MUJOCO_GL=egl
export PYTHONNOUSERSITE=1
export PYTHONPATH=$PWD

python run_mujoco.py \
  --mode sort --detector colour --web
```

终端显示 `SIM_WEB_URL=http://localhost:8765/` 后，在 Windows 浏览器打开该地址。页面有外部工作区与虚拟 D405 RGB 两路视频流，以及“开始分拣任务、随机重置、暂停/继续、取消”按钮。网页模式保持运行，按终端 `Ctrl+C` 才关闭服务；任务完成、取消或失败后先随机重置，再开始下一轮。

### 4.7 让仿真可见地“动起来”：录制一次完整验证视频

默认 EGL 是离屏仿真，机械臂确实在推进物理但不会自动弹窗。需要留下可复查证据时，在同一条 YOLO 抓取命令末尾加入 `--record-video`：

```bash
python run_mujoco.py \
  --mode grasp \
  --detector yolo \
  --weights sim/models/block_detector.pt \
  --record-video reports/mujoco_sim/three_blocks_green_grasp.mp4
```

这不是把静态图片拼成动画：录像器挂在 MuJoCo 的 20 ms 控制周期上，以外部相机记录机械臂真实执行的回 HOME、预抓取、接近、闭爪和抬升。成功时终端最后会打印：

```text
SIMULATION_VIDEO=reports/mujoco_sim/three_blocks_green_grasp.mp4 frames=186
```

可在 Windows 资源管理器打开 `\\wsl.localhost\Ubuntu-22.04\home\ros\A_shen_arm\pathera_grasp\reports\mujoco_sim\three_blocks_green_grasp.mp4`，或在 WSL 里用已安装的视频播放器播放。视频可在没有 WSLg、没有独立显卡时生成；它是推荐的程序验收证据。

### 4.7 三色积木随机场景与按色抓取

场景不再只有一个固定绿色立方体：MuJoCo 会在桌面上放置红、绿、蓝三块尺寸为 **3×3×6 cm** 的积木。`--scene-seed` 控制布局；相同 seed 的位置、微小朝向和抓取结果可复现，换整数即可重新“洒”一桌积木。

```bash
# 绿色：可使用现有 YOLO 权重，红蓝作为真实视觉干扰物
python run_mujoco.py \
  --mode grasp --detector yolo --target-color green \
  --scene-seed 20260903 \
  --weights sim/models/block_detector.pt \
  --record-video reports/mujoco_sim/three_blocks_green_grasp.mp4

# 红色或蓝色：使用 RGB-D 颜色识别并执行同一套 IK/夹持/抬升
python run_mujoco.py \
  --mode grasp --detector colour --target-color red \
  --scene-seed 20260903

python run_mujoco.py \
  --mode grasp --detector colour --target-color blue \
  --scene-seed 20260903
```

不需要先用 SolidWorks 画规则积木。MJCF `box size="0.030 0.015 0.015"` 表示半尺寸，因此正好是长 6 cm、宽 3 cm、高 3 cm，并能直接参与真实的 MuJoCo 重力、摩擦和夹爪接触。只有需要倒角、凸点、标签或实物纹理时，才需要从 SolidWorks 导出 STL 并替换视觉网格。

当前仓库内的 YOLO 权重由绿色积木数据训练，已在三色干扰场景中通过绿色抓取验证；它不是按红/绿/蓝分类的模型。若需要“说抓蓝色”也走 YOLO，需要生成三色标注数据并重训多类别或“检测 + 颜色分类”模型；不能只改场景颜色。

## 5. Viewer 的工作方式

`--viewer` 不是另开 `python -m mujoco.viewer` 的静态模型浏览器。它由 `sim/live_viewer.py` 在 `run_mujoco.py` 的同一 Python 进程内通过 `mujoco.viewer.launch_passive(model, data)` 创建；每执行一次 20 ms 控制周期，`MujocoRobot.step_control_period()` 先推进物理，再调用查看器的 `sync()`。所以用户调整的是同一个窗口视角，窗口读取的也正是本次 IK 控制、夹爪控制和积木接触产生的实时状态。

只想检查场景外观、而不运行识别/抓取时，才可单独运行下面的静态浏览器。它不会执行自动抓取：

```bash
cd /home/ros/A_shen_arm/pathera_grasp
unset MUJOCO_GL
python -m mujoco.viewer \
  --mjcf=sim/models/panthera_grasp_scene.xml
```

## 6. 什么时候才需要重新训练 YOLO

以下情况才需要训练：

- `sim/models/block_detector.pt` 丢失；
- 修改了相机视角、积木外观或场景光照；
- 希望重新生成可复现权重或评估训练过程。

先检查已有权重：

```bash
ls -lh sim/models/block_detector.pt
```

重新训练命令：

```bash
python \
  -m sim.scripts.train_synthetic_yolo
```

训练脚本会：

1. 在 MuJoCo 中改变积木位置并渲染 80 张图像；
2. 用渲染掩膜生成 YOLO 标签；
3. 首次运行时下载约 5.4 MB 的官方 `yolo11n.pt` 预训练权重；
4. 在 CPU 上训练 20 轮；
5. 将最佳权重复制为 `sim/models/block_detector.pt`。

训练期间出现连续的 `Downloading`、epoch、mAP 日志属于正常现象。训练是前台阻塞命令；在它结束前，同一终端后面输入的 YOLO 抓取命令不会开始。不要重复启动多个训练进程。

训练数据、曲线和临时权重位于：

```text
reports/mujoco_sim/yolo_train/
```

## 7. 为什么这些代码能够跑起仿真

### 7.1 入口隔离

`run_mujoco.py` 没有真机 backend，也不会导入 `grasp_demo.py`。参数解析完成后，它只延迟导入 `sim.scripts.run_sim_demo`。因此仿真入口不会因为加载真机主程序而意外访问 Panthera SDK、相机、CAN、NPU 或声卡。

### 7.2 MuJoCo 模型和物理状态

`MujocoRobot` 使用：

```python
mujoco.MjModel.from_xml_path(...)
mujoco.MjData(model)
```

加载 `panthera_grasp_scene.xml`。`MjModel` 保存固定模型信息，例如关节、质量、惯量、执行器、相机和碰撞几何；`MjData` 保存每次运行变化的关节位置、速度、控制量、接触和传感器数据。模型的 `base_link` 和 `link1..link6` 已挂接 Panthera-HT 官方 STL 作为视觉几何，和 URDF 一样都以各 link 原点为零位姿放置，因此不会改变已验证的关节坐标链。

这些 STL 只承担外观显示，`contype="0" conaffinity="0"`，不参与接触和质量计算；原有 capsule 碰撞代理和 URDF 惯量继续负责稳定的物理接触。官方 `link2.STL` 有 321,432 个面，超过 MuJoCo 单网格 200,000 面限制，项目使用 `sim/scripts/decimate_stl_visual.py` 在项目内生成 176,026 面的 `link2_visual.STL`。这不是随意删面，而是确定性的顶点体素聚类；原始 STL 仍随项目保留，未被修改。

腕部还挂接了 RealSense 官方 ROS 描述包的 `d405.stl`。它以 D405 公开 URDF 的外壳尺寸和 visual transform 放在已有的 `T_tcp_camera` 标定位姿上；没有真实 CAD 依据的临时安装杆已经删除。虚拟 `wrist_rgbd` 相机与 D405 外壳共用安装坐标，但渲染时隐藏自身外壳和机械臂网格，避免相机拍到自己。该视觉资产的来源和 Apache-2.0 文本见 `sim/assets/realsense_d405/NOTICE.md` 与同目录许可证。

六个关节和执行器全部按 `joint1..joint6`、`joint1_act..joint6_act` 的名称映射，而不是依赖容易变化的数组顺序。控制层每 20 ms 更新一次目标，MuJoCo 以 2 ms 步长执行十次 `mj_step`，因此每个上层控制周期内都有连续物理计算。

### 7.3 机械臂为什么会运动

MJCF 为六个关节配置了 position actuator。`move_j()` 将起点到终点插值后写入：

```python
data.ctrl[actuator_ids] = joint_target
mujoco.mj_step(model, data)
```

位置执行器产生力矩，MuJoCo 根据惯量、重力、阻尼、碰撞和摩擦更新 `qpos/qvel`。运动结束后代码比较实际关节反馈和目标，最大误差超过 `0.040 rad` 就失败，不能只凭“命令已经写入”宣称到位。

### 7.4 腕部 RGB-D 为什么能定位物体

MJCF 中的 `wrist_rgbd` 相机使用项目 `hand_eye_calibration.json` 的 `T_tcp_camera`。MuJoCo `Renderer` 先输出 RGB，再切换 depth rendering 输出以米为单位的深度。

检测器得到像素 `(u,v)` 和深度 `z` 后，使用针孔模型恢复相机坐标：

```text
x = (u - cx) × z / fx
y = -(v - cy) × z / fy
point_camera = [x, y, -z]
```

再通过 MuJoCo 给出的相机世界旋转和平移变换到 Base 坐标。由于深度看到的是积木表面，代码结合已知 3×3×6 cm 积木尺寸和桌面高度恢复积木中心；检测输入本身不会读取 MuJoCo 的物体真值位置。

### 7.5 colour 和 YOLO 的区别

- `colour`：从 RGB 像素阈值寻找绿色积木，稳定、快速，适合物理和坐标回归。
- `yolo`：Ultralytics 从渲染图像输出目标框，框内再结合 depth 计算 Base 坐标。它是最终识别验收路径。

两种检测最终都返回同一种 `Detection` 数据结构，因此后面的 IK 和抓取状态机不需要知道检测器实现。

### 7.6 Pinocchio IK 为什么与 MuJoCo 对得上

Pinocchio 从 `panthera_follower.urdf` 读取同一套六轴关节几何和限位。TCP 固定为原项目定义的：

```text
joint6 + [0.165, 0, 0] m
```

IK 同时优化 TCP 位置和姿态，不再是容易让夹爪撞桌的位置-only IK。测试使用 HOME 加 10 组随机关节姿态比较 Pinocchio 和 MuJoCo TCP，位置差小于 `1e-6 m`。

### 7.7 抓取为什么被认为成功

`SimGraspPipeline` 不是在闭爪后直接打印成功，而是要求：

1. 预抓取 6D IK 可达；
2. 实际 TCP 跟踪误差在门限内；
3. 沿工具 X 轴移动到抓取位；
4. 左右手指都与积木接触；
5. 两侧法向接触力均大于 `0.05 N`；
6. 抬升后积木高度增加至少 `0.030 m`。

任何一步失败都会抛出错误并返回非零退出码。

## 8. 测试命令

只运行 MuJoCo/YOLO 测试：

```bash
python \
  -m unittest discover -s sim/tests -v
```

运行统一离线测试：

```bash
python \
  tools/run_offline_tests.py
```

2026-09-07 合入当前主线后的隔离环境检查为 MuJoCo 专项 11/11 通过，且固定种子的 YOLO 识别、单次抓取和三色 PUT1 分拣入口均成功。统一离线入口完成 80 个 Python 文件的内存编译；真机主线动态测试依赖 `pyrealsense2`、ROS2、厂商扩展和已拉取的 Git LFS 模型，不能在只安装仿真依赖的 WSL 环境中计为通过或跳过，应在对应的 IQ9075 环境单独执行。

## 9. 常见问题

### 没有弹出窗口

识别和抓取默认是 EGL 离屏仿真，这是预期行为。需要查看场景时使用第 5 节 Viewer 命令。

### `ImportError: libboost_atomic.so.1.90.0`

当前 Python 读到了 `~/.local` 的旧包。执行：

```bash
export PYTHONNOUSERSITE=1
```

或重新 `conda activate pathera_mujoco`。

### YOLO 输出仍是 `rendered_rgbd_colour`

确认命令含有：

```text
--detector yolo --weights sim/models/block_detector.pt
```

并确认当前看到的不是上一条 colour 命令的输出。

### `no target found in rendered RGB-D frame`

先运行 colour recognition。如果 colour 也失败，检查 MJCF、相机姿态或环境变量；如果 colour 成功而 YOLO 失败，确认权重存在且没有在训练中被中断覆盖。

### 训练下载很慢

下载和训练不影响真机，因为全过程只使用网络、CPU 和 MuJoCo 合成图像。已有 `block_detector.pt` 时可直接跳过训练。

### 是否需要显卡

不需要独立显卡。当前 Torch 是 `2.4.1+cpu`，`torch.cuda.is_available()` 为 `False`；YOLO 训练和推理均已在 CPU 上通过。EGL 用于无窗口渲染，WSLg Viewer 由系统图形栈显示。

## 10. 安全边界

- MuJoCo 验收只运行 `run_mujoco.py`；不要使用真机 `grasp_demo.py` 代替仿真入口。
- 不要同时启动多次 YOLO 训练。
- 当前连杆长度、惯量、关节限位、165 mm TCP、手眼外参和本体 STL 外观与原项目一致；夹爪、D405 支架、桌面和放置区仍是缺少真实 CAD/现场测量时的简化代理，不是毫米级数字孪生。
- 历史完成报告见 [`archive/TASK-005-mujoco-report.md`](archive/TASK-005-mujoco-report.md)，简要入口见 [`../sim/README.md`](../sim/README.md)。
