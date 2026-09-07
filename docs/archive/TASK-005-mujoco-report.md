# TASK-005 pathera_grasp MuJoCo 仿真接入完成报告

> 归档说明：本文记录的是独立原型分支完成时的状态，保留原分支名和验收数据用于追溯。合入正式仓库后，仿真入口已改名为根目录 `run_mujoco.py`，真机 `grasp_demo.py` 不再由仿真代码修改或拦截；当前使用方法以 [`../../sim/README.md`](../../sim/README.md) 和 [`../mujoco_guide.md`](../mujoco_guide.md) 为准。

状态：核心验收已通过；全程仅在 WSL2/CPU 仿真侧执行，未访问或驱动任何真机硬件。

完整启动、可视化、识别与抓取操作见 [MuJoCo 使用教程](../mujoco_guide.md)；本文件保留验收结果和实现边界，便于分支审查。

## 交付内容

- `run_pathera_grasp.py`：独立原型阶段使用的旧入口名；正式集成后对应 `run_mujoco.py`。
- `sim/models/panthera_grasp_scene.xml`：Panthera 六轴、URDF 惯量、关节限位、执行器、官方 `base_link + link1..link6` STL 外观网格、腕部相机、简化夹爪、桌面、积木和放置区。
- `sim/assets/panthera_ht_stl/`：随分支携带的 Panthera-HT 本体 STL；`link2_visual.STL` 为满足 MuJoCo 20 万面上限生成的视觉降面版本。
- `sim/assets/realsense_d405/`：官方 RealSense ROS D405 STL 与 Apache-2.0 许可证/来源声明；仅作非碰撞外观，不加载真机驱动。
- `sim/recording.py`：外部视角 MP4 录像器；`--record-video` 记录真实执行的控制步，而不是后处理动画。
- `--scene-seed` 与 `--target-color`：三块 3×3×6 cm 红/绿/蓝积木可复现散布和按色抓取；绿色保留 YOLO 验收，红/蓝使用颜色识别。
- `sim/mujoco_backend.py`：20 ms 控制周期、2 ms 物理步长、EGL RGB-D、夹爪和接触力接口。
- `sim/detection.py`：确定性颜色回归与真正的 Ultralytics YOLO 检测，两者都由渲染 RGB-D 反投影到 Base 坐标，不读取物体真值作为检测输入。
- `sim/ik.py`：Pinocchio 6D 数值 IK；TCP 使用原项目 `joint6 + [0.165, 0, 0]`。
- `sim/grasp_pipeline.py`：识别、定位、预抓取、沿工具轴接近、闭爪、双指力校验和抬升验证。
- `sim/scripts/train_synthetic_yolo.py`：固定随机种子的 MuJoCo 数据生成和 CPU YOLO 微调脚本。
- `sim/models/block_detector.pt`：5.4 MB 合成场景检测权重。
- `sim/tests/test_sim_backend.py`：RGB-D/定位、11 姿态 FK 对齐、物理抓取抬升测试。
- `sim/README.md`：正式集成后的环境、运行、安全和已知边界说明。

## 环境与依赖

验证环境：Ubuntu-22.04、用户 `ros`、`arm_ws`、Python 3.10。

联合导入结果：

```text
mujoco 3.11.0
pinocchio 4.1.0
cv2 5.0.0
ultralytics 8.4.115
torch 2.4.1+cpu, cuda=None, available=False
torchvision 0.19.1+cpu
numpy 2.2.6
```

Pinocchio 的三个相关包均来自 conda-forge。Torch/torchvision 来自阿里云 CPU wheel，不需要显卡或 CUDA。

## 验收结果

1. MuJoCo EGL 输出 RGB `(480, 640, 3)` 和米制 depth `(480, 640)`。
2. 11 组关节姿态下，MuJoCo 与 Pinocchio TCP 位置误差均小于 `1e-6 m`。
3. Ultralytics 合成训练集为 64 train + 16 val；最终验证 `mAP50=0.792`、CPU 推理约 `9.6 ms/image`。
4. YOLO 识别：

```text
DETECTION_COUNT=1 source=ultralytics_yolo
TARGET_WORLD=0.1902,0.0078,0.0250
```

5. YOLO 驱动的抓取闭环：

```text
DETECTION_COUNT=1 source=ultralytics_yolo
PREGRASP_IK=reached:True error_m:0.0058
GRASP_IK=reached:True error_m:0.0047 orientation_error_deg:0.04
BILATERAL_CONTACT=True force_n=3.613,3.665
LIFTED_HEIGHT_M=0.0663
FINAL_TCP_WORLD=0.1700,0.0002,0.0911
```

6. 五个积木位置 `(x,y)=(0.18/0.20, ±0.015)` 和 `(0.19,0)` 全部完成抓取，抬升 `65.9–66.9 mm`。
7. 2026-09-07 收尾时仿真专项测试 9/9 通过（含红绿蓝三色独立定位、完整分拣和网页状态机）；仿真入口导入后 `pyrealsense2/rclpy/Panthera/voice_controller` 的已加载模块列表为空。
8. 直接执行 `grasp_demo.py` 或统一入口选择 `--backend real` 均在任何厂商硬件模块初始化前拒绝。

## 审计中发现并解决的问题

- TCP 曾比原项目多 40 mm：恢复为唯一 165 mm TCP，抓取中心单独建模。
- 旧位置 IK 会使手指撞桌且闭爪后不验证：改为固定姿态 6D IK、轴向接近、关节/TCP 跟踪门限、双指接触力和抬升判定。
- 初版识别依赖隐藏整个夹爪：重做简化夹爪为开放光路的可见侧轨，仅隐藏会穿过 D405 光路的粗 link6 碰撞代理。
- RGB-D 返回的是物体表面点：根据已知 3×3×6 cm 积木和支撑平面恢复中心；三色随机场景的颜色定位门限为 25 mm，抓取仍以实际接触力和抬升门限验收。
- 方块初始悬空且两个模式状态不同：初始高度改为接触桌面，并在 reset 后统一稳定 0.5 s。
- base/link1 永久自碰撞：用 MuJoCo collision bitmask 排除机械臂内部代理碰撞，同时保留机械臂与环境/物体碰撞。
- `~/.local` 旧 cmeel/coal 污染 Pinocchio：设置 Conda 环境变量 `PYTHONNOUSERSITE=1` 隔离，不删除用户包。
- 阿里云普通 PyPI 的 Torch 不是明确 CPU 构建：改用批准的阿里云 CPU wheel 直链。
- 第一次训练命令继承 Windows 工作目录，生成数据落在项目外：权重复制回 WSL 目标后，三个误生成目录已移入 Windows 回收站；随后使用 WSL 绝对工作目录重新训练并复验。

## 保留问题与边界

- 2026-09-07 统一离线入口内存编译 64 个 Python 文件并运行 62 项测试，7 项因 WSL 仿真环境有意不安装 `pyrealsense2`/ROS2 `rclpy` 而明确跳过，结果为 `OK (skipped=7)`；未安装或伪造硬件模块。本轮 MuJoCo 专项测试 9 项全部通过。
- 双指接触和接触力通过后，后端会将积木同步为受控持有物；PUT1 验证后会锁定已放置积木。这是确定性集成回归机制，不等同于完全依赖摩擦与柔顺接触的自由物理抓取。
- `pip check` 仍会报告旧 `cmeel-boost 1.83.0` 元数据要求 NumPy 1.26；`PYTHONNOUSERSITE=1` 后联合导入和仿真均正常。为避免破坏复用的 `arm_ws`，没有卸载旧包。
- 机械臂连杆长度、惯量、关节限位、TCP 和手眼外参来自现有项目/只读参考；夹爪、D405 支架和工位缺少真实 CAD 与现场测量，因此仍是明确标注的简化代理，不应把当前模型称为毫米级数字孪生。
- 高层仿真流水线与现有项目共享坐标和规划语义，但没有直接实例化原 `GraspPlanner`，因为其顶层依赖 RealSense/真机模块。后续可先拆出纯几何核心，再通过适配器复用完整状态机。

## 原型归档时的 Git 状态

- 基线分支：`main`，仅含导入的原项目快照。
- 开发分支：`feature/mujoco-sim`，仅在该分支提交仿真功能。
- 未配置远端、未 push、未 merge。
