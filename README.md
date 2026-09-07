# Panthera-HT 视觉抓取系统

本仓库是部署在 IQ9075/AArch64 板端的 Panthera-HT 六轴机械臂视觉抓取项目。当前正式运行入口是根目录的 `grasp_demo.py`：RealSense D405 提供对齐 RGB-D，Qualcomm QNN HTP NPU 上的 YOLOE context 完成物体分割，Lab 标定完成积木颜色判别，规划器执行预抓取、近距离复检、直线接近、夹持判断及失败恢复。网页同时提供双画面预览、目标输入、J1 观察角调整、随动模式、持物放置和安全结束请求。

项目已经从“抓到物体后立即自动放置”改为两阶段流程：抓取成功后保持夹爪闭合并返回 HOME，进入 `FULL_LOAD`（持物待放）；操作者在网页点击“放置”后，机械臂才执行 PUT2/PUT1 放置序列。

> 安全结论：软件按钮、状态机、力矩判断和轨迹检查都不是硬件急停，也不能替代现场清场。项目没有完整环境建图或自碰撞规划。所有实机动作必须由现场操作者启动，并确保有人处于硬件急停可达位置。

## 第一次接触项目，先理解这五件事

1. **这是一个“相机看、算法判断、机械臂执行”的完整程序。** 正式入口只有根目录的 `grasp_demo.py`；正常抓取不需要另外启动 ROS2 节点。
2. **启动程序不等于立刻抓取。** 程序完成自检后先移动到 HOME 并等待目标；操作者可以在网页输入“绿色积木”等目标。
3. **抓取和放置已经拆开。** 抓取成功后物体仍被夹紧，机械臂回 HOME 等待；只有点击网页“放置”按钮才会前往放置区并松爪。
4. **人手随动默认不运行。** 只有点击网页“随动模式”后，CPU 手部模型才会加载并开始计算；它不会在普通抓取过程中占用识别流程。
5. **“结束程序”与“回 HOME”不是一回事。** HOME 是工作待机点；程序结束时会尽量回到本次启动前记录的真实关节姿态，再停止电机。

普通操作者通常只需完成以下流程：现场清场并准备硬件急停 → 启动 `grasp_demo.py` → 打开终端给出的网页 → 输入抓取目标 → 抓取成功后按需点击“放置” → 最后点击“结束程序”。首次部署、重新标定或更换物体类型时，则应完整阅读本文的环境、标定、测试和限制章节。

## 常用术语

| 术语 | 通俗解释 |
|---|---|
| HOME | 抓取任务的工作待机姿态；机械臂在这里等待目标，也在持物时回到这里等待放置 |
| PUT2 | 从 HOME 前往放置区或从放置区返回时经过的安全过渡点，不在这里松爪 |
| PUT1 | 实际放置点；机械臂保持夹紧到达这里，然后张开夹爪 |
| 启动前位姿 | 程序初始化机械臂时读取并记录的关节姿态；正常关机优先回到此处，它不等于 HOME |
| TCP / 夹爪末端 | 规划器使用的工具中心点，可理解为夹爪执行抓取动作时的参考位置 |
| RGB-D | 同时包含彩色图像和逐像素深度的相机数据，本项目由 RealSense D405 提供 |
| NPU | 板端用于加速物体识别的专用计算单元；当前只加速普通物体识别，不加速人手随动 |
| IK | 逆运动学：把期望的夹爪位置和姿态换算成六个关节应到达的角度 |
| HOLD | 当前数据或动作不满足安全条件时保持不动，等待下一轮有效数据或人工处理 |
| FULL_LOAD | “持物待放”状态：已经抓住物体、夹爪保持闭合、机械臂位于 HOME，并等待放置命令 |

如果只想知道某项内容在哪里：运行方法见“环境与启动”，抓取/放置顺序见“系统工作流”，网页按钮见“网页端操作”，识别参数见“识别与标定”，报错排查见“日志与故障定位”，交付前检查见“归档检查清单”。

## 1. 当前能力与边界

| 功能 | 当前状态 | 说明 |
|---|---|---|
| 彩色积木识别与抓取 | 主线功能 | `object3` NPU 分割 + Lab 颜色判别；红、黄、绿、蓝等颜色命令映射到积木 |
| 通用文本目标 | 已接入 | 支持“瓶子”“盒子”“绿色积木”等单目标输入 |
| 瓶子/盒子识别 | 可展示、可计划预览 | 默认禁止真实运动；物理尺寸、夹爪开度、负载和姿态未完成专用标定 |
| 预抓取与近距复检 | 已接入 | 先按当前夹爪到目标的有符号抓取轴间隙生成自适应预抓取点，再用新 RGB-D 帧修正 X/Z |
| FULL_LOAD 持物待放 | 已实现、待完整实机回归 | 抓取成功后沿本次抓取轨迹反向退离并闭爪回 HOME，等待网页“放置” |
| 网页控制与推流 | 已接入 | 左侧原始/深度画面，右侧 YOLO 画面；控制请求使用同源 JSON 与会话 Cookie |
| J1 观察角微调 | 已接入 | 空闲且机械臂接近 HOME 时，每次左/右调整 0.5 rad，并受关节限位约束 |
| 人手随动 | 实验功能 | 默认关闭；网页显式启用后按需加载 CPU YOLOE，单次 TCP 物理步进不超过 20 mm |
| 语音 ASR/TTS | 保留、默认关闭 | SenseVoice + sherpa-onnx VITS；当前不属于正式抓取主线 |
| GraspNet | 可选实验后端 | 默认关闭，仅用于离线候选评估或明确设置 `GRASPNET_USE=1` 的实验 |
| ROS2 | 暂停使用 | 代码保留供未来跨进程/跨机器部署；当前不得与单进程入口同时启动 |

## 2. 系统工作流

### 2.1 启动与待机

1. 进程先取得 `/tmp/pathera_grasp.lock`，防止第二个实例重复占用相机和 CAN。
2. 启动网页服务和 D405，加载手眼标定以及物体识别后端。
3. 手部模型此时不加载、不推理；随动默认关闭。
4. 初始化机械臂并记录“程序启动前位姿”。
5. 机械臂移动到 HOME，张开夹爪，进入目标输入状态。

程序正常结束、网页结束或捕获 SIGINT/SIGTERM 后，会尝试返回“程序启动前位姿”并停止电机；这个姿态不是 HOME。若反馈丢失，安全停机逻辑不会盲目重放过期关节值。

### 2.2 积木抓取

```text
目标输入
  → 当前视角快速检查
  → 必要时扫描 J1
  → 多帧颜色/位置确认
  → 眼在手上的观察位移动
  → 多帧远距复检
  → 自适应预抓取点
  → 短笛卡尔对齐
  → 近距离 X/Z 复检（保持远距 Y 与抓取姿态）
  → 直线接近
  → 闭合夹爪并判断负载
```

预抓取距离不是固定 5 cm，也不是三维欧氏距离的一半。程序取“夹爪末端到目标沿抓取轴方向的有符号间隙”的 50%，再限制到 15–40 mm；横向偏差和姿态偏差由独立走廊约束处理。这样不会把横向误差误当成前进距离。

固定 Base XYZ 补偿当前为 `[0, 0, 0]`。相机、joint6 和 TCP 的位置关系通过完整刚体链组合；近距离修正来自新鲜 RGB-D 数据，不持久化成固定补偿。

### 2.3 抓取成功与放置

抓取成功后：

```text
抓取位（闭合）
  → 沿本次已验证接近轨迹反向退离（闭合）
  → HOME（闭合）
  → FULL_LOAD：等待网页“放置”
```

`FULL_LOAD` 不保存一个写死的末端或物体坐标。不同物体的夹爪闭合位置不同，返回使用本次 `tool_target`、`tool_rotation`、接近轨迹和实时关节反馈重新构造。

点击“放置”后：

```text
HOME（闭合）
  → PUT2（闭合）
  → PUT1（闭合）
  → 在 PUT1 张开夹爪
  → PUT2（张开）
  → HOME（张开）
  → 等待下一目标
```

进入放置前必须从实时反馈确认机械臂位于 HOME 容差内。`FULL_LOAD` 和放置期间，目标输入、J1 点动与随动入口均被互斥锁定。

### 2.4 抓取失败

第一次夹取未达到负载阈值时，机械臂反向返回预抓取/识别区域，获取新帧并重新规划。重试耗尽后张开夹爪并回 HOME，不结束程序，继续等待下一目标。

### 2.5 人手随动

随动仅在网页点击“随动模式”后启用：

```text
HOME + 夹爪张开
  → 按需加载 CPU YOLOE hand 模型
  → 等待唯一手连续稳定 3 帧
  → 停稳
  → 获取新 RGB-D
  → 检测、深度审核、重算 IK
  → 最多移动 20 mm
  → 重复
  → 关闭随动后返回 HOME
```

关键安全门：

- 语义请求位移超过 100 mm：立即 HOLD。
- 单轮实际 TCP 步进：最多 20 mm。
- TCP 到手的最小安全距离：200 mm。
- 关节速度：不超过 0.12 rad/s。
- 多手、边界手、深度离散过大、旧帧、IK 失败或轨迹越界：HOLD。
- 成功跟踪后持续丢手约 1.25 s：进入暂停，需人工关闭并重新启用。
- 当前随动使用 CPU YOLOE；仓库没有可用的 hand QNN context，不得用物体 context 冒充 NPU 手部模型。

## 3. 目录结构

| 路径 | 作用 | 详细说明 |
|---|---|---|
| `grasp_demo.py` | 正式单进程入口 | 本文 |
| `Panthera-HT_SDK/panthera_python/scripts/Panthera_lib/` | 机器人、视觉、规划、NPU、随动核心库 | [核心库说明](Panthera-HT_SDK/panthera_python/scripts/Panthera_lib/README.md) |
| `config/` | 识别 profile 与颜色标定 | [配置说明](config/README.md) |
| `tools/` | 离线回归、相机/NPU诊断、网页夹具 | [工具说明](tools/README.md) |
| `tests/` | 无硬件单元与契约测试 | [测试说明](tests/README.md) |
| `models/` | CPU YOLOE、ASR、TTS 模型资产 | [模型说明](models/README.md) |
| `third_party/` | GraspNet、GraspNetAPI、QNN 运行资产 | [第三方说明](third_party/README.md) |
| `iq9075_speech/` | 离线 ASR 封装 | [ASR 说明](iq9075_speech/README.md) |
| `iq9075_tts/` | 离线 TTS 封装 | [TTS 说明](iq9075_tts/README.md) |
| `voice_demo/` | 独立语音演示/下载脚本 | [语音示例](voice_demo/README.md) |
| `ros2_ws/` | 暂停使用的 ROS2 备选架构 | [ROS2 说明](ros2_ws/README.md) |
| `Panthera-HT_SDK/` | 厂商 SDK、URDF 与底层驱动 | 使用上游 README；除 `Panthera_lib` 项目适配层外不要随意改动 |

根目录资产：

- `hand_eye_calibration.json`：当前项目手眼标定副本，运行时只读取此文件。
- `mobileclip2_b.ts`：CPU YOLOE 文本编码器。
- `voice_controller.py`：主程序使用的语音门面，语音不可用时回退到终端/网页。
- `VOICE_SETUP.md`：语音环境的历史安装与验证记录。
- `requirements_asr.txt`、`requirements_tts.txt`：语音增量依赖，不是整个项目的锁定环境。

## 4. 运行环境

已部署环境：

- 目标机：IQ9075/AArch64，Ubuntu。
- Conda 环境：`/home/ubuntu/miniconda3/envs/pathera_grasp`，Python 3.10。
- 相机：Intel RealSense D405，默认 640×480、30 FPS。
- ROS：Humble，仅用于兼容测试或未来备选入口。
- 机器人配置：`Panthera-HT_SDK/panthera_python/robot_param/Leader.yaml`。

本仓库没有完整、可从零复现的统一依赖锁文件。归档恢复时优先保留现有 Conda 环境导出、厂商 SDK、QNN runtime、模型大文件及设备权限；`requirements_asr.txt` 和 `requirements_tts.txt` 只能补充语音依赖，不能替代整套环境。

## 5. 实机启动

### 5.1 启动前检查

1. 清空机械臂工作空间，确认线缆不会进入轨迹。
2. 确认硬件急停有效且有人值守。
3. 确认 D405、末端工具、基座与 `hand_eye_calibration.json` 对应同一次标定。
4. 确认串口/CAN 权限正常。
5. 确认没有另一实例或 ROS2 grasp brain 占用硬件：

```bash
pgrep -af 'grasp_demo.py|panthera_grasp_brain|panthera_tq.cli'
```

6. 确认 HOME、PUT1、PUT2 和启动前位姿周围无障碍物。

### 5.2 启动命令

```bash
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate pathera_grasp
cd /home/ubuntu/A_shen_arm/pathera_grasp
python grasp_demo.py
```

默认值：物体识别走 NPU，抓取走 OBB/Seeed 几何规划器，语音关闭，手部随动关闭，网页端口 8080。

终端出现以下信息后，从浏览器打开打印的地址：

```text
[STREAM] operator page ready at http://<板端IP>:8080/
```

网页控制凭据使用 `HttpOnly; SameSite=Strict` 会话 Cookie，不放在 URL 中。重新启动程序后应刷新/重新打开页面以建立新会话。8080 仅适合受信任局域网，不应映射到公网。

### 5.3 目标输入示例

```text
红色积木
黄色积木
green block
瓶子
盒子
```

一次只能输入一个对象和一种颜色。纯颜色默认解释为对应颜色的积木；否定或多目标歧义请求会被拒绝。瓶子/盒子默认只识别展示，不产生机械臂运动。

### 5.4 停止程序

- 优先使用网页“结束程序”或终端 `Ctrl-C` 请求受控退出。
- 受控退出会尝试返回程序启动前位姿，然后停止电机。
- 若动作异常、碰撞风险或人员进入工作区，应直接使用硬件急停；网页按钮不是急停。
- `FULL_LOAD` 携物状态下的中断处置仍需现场谨慎：当前全局退出沿统一启动姿态停机链执行，不保证物体能被安全放置。

## 6. 网页功能

| 控件 | 可用状态 | 行为 |
|---|---|---|
| 目标输入/预设 | `IDLE` | 排队一个抓取请求，禁止覆盖尚未消费的请求 |
| 左/右箭头 | `IDLE` 且接近 HOME | J1 分别 +0.5/-0.5 rad，其他关节保持目标形状 |
| 随动模式 | `IDLE` | 请求主线程按需加载/运行手部随动；再次关闭后回 HOME |
| 放置 | `FULL_LOAD` | 请求主线程执行 PUT2→PUT1 松开→PUT2→HOME |
| 结束程序 | 非关闭状态 | 锁存全局停止请求，交由主线程和 finally 安全停机 |

HTTP 线程只负责校验和排队，不直接调用机械臂。控制接口要求：有效会话 Cookie、同源请求、`application/json`、正确请求结构和状态机许可。

推流端点：

- `/stream/raw`：相机原始彩色图。
- `/stream/depth`：伪彩深度图。
- `/stream/yolo`：与检测结果同帧的标注图。
- `/api/status`：当前控制模式和按钮许可。

## 7. 主要配置

常用环境变量：

| 变量 | 默认值 | 作用 |
|---|---|---|
| `YOLO_NPU` | `1` | `1` 使用 QNN NPU；`0` 使用较慢的 CPU YOLOE 回退 |
| `VISION_MODEL_PROFILE` | `object3` | 选择 `config/recognition_profiles.json` 中的识别 profile |
| `NPU_CONFIDENCE_THRESHOLD` | profile 值，当前 0.20 | 临时覆盖 NPU 置信度门槛 |
| `COLOR_CLASSIFIER` | `lab` | `lab` 使用标定颜色；`hsv` 使用旧回退算法 |
| `GRASPNET_USE` | `0` | 显式启用实验 GraspNet 候选后端 |
| `GRASPNET_CHECKPOINT_PATH` | 仓库内默认路径 | 指定 GraspNet 权重 |
| `VISION_STREAM_HOST` | `0.0.0.0` | 网页监听地址 |
| `VISION_STREAM_PORT` | `8080` | 网页端口 |
| `VISION_STREAM_JPEG_QUALITY` | `92` | JPEG 质量 |
| `VISION_STREAM_PREVIEW_FPS` | `15` | 网页预览节奏，允许 10–30 |
| `REALSENSE_SERIAL` | 空 | 多相机时锁定序列号 |
| `VOICE_INPUT` | `0` | 显式打开离线语音入口 |
| `IQ9075_ASR_MODEL_DIR` | `models/sensevoice` | ASR 模型路径 |
| `IQ9075_SHERPA_TTS_MODEL_DIR` | `models/sherpa_tts/vits-melo-tts-zh_en` | TTS 模型路径 |
| `VOICE_PROMPT_DURATION` | `3.5` | 语音提示录音时长 |
| `YOLOE_MODEL_PATH` | `models/yoloe-26s-seg.pt` | CPU YOLOE 权重 |
| `YOLOE_TEXT_ENCODER_PATH` | `mobileclip2_b.ts` | CPU YOLOE 文本编码器 |

位置、限位、运动时长、深度门限和抓取参数集中在 `GraspConfig`。实机标定值不应通过网页或未审计环境变量任意放宽。

## 8. 识别与标定资产

当前 `object3` 类别顺序必须与 QNN context 输出一致：

```text
bottle / box / toy building block
```

`config/recognition_profiles.json` 保存 context 路径、类别顺序、输入输出张量和阈值；`config/color_calibration.json` 保存运行时 Lab 分类模型；`config/color_calibration_samples.json` 保存原始标定样本。任何 context 替换都要同时复核类别顺序、输出规格与 SHA-256。

手眼标定使用根目录 `hand_eye_calibration.json`。运行时会校验 4×4 刚体矩阵的形状、有限值、末行和旋转合法性。更换相机、相机安装、末端工具或基座后必须重新标定，不应仅叠加固定 XYZ 补偿掩盖坐标链问题。

## 9. 离线验证

统一回归不会打开相机、NPU、声卡或机械臂：

```bash
source /opt/ros/humble/setup.bash
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate pathera_grasp
cd /home/ubuntu/A_shen_arm/pathera_grasp
PYTHONDONTWRITEBYTECODE=1 python tools/run_offline_tests.py
```

它会在内存中编译以下自研代码：主程序、语音门面、ASR/TTS 包、语音示例、工具、测试、`Panthera_lib` 项目核心以及 ROS2 源码，然后运行 `tests/` 下全部单元测试。第三方仓库和厂商生成/依赖目录不纳入本项目静态编译。

针对性命令：

```bash
export PYTHONPATH=Panthera-HT_SDK/panthera_python/scripts
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -p 'test_planner_safety.py'
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -p 'test_streamer_ui.py'
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -p 'test_hand_follow.py'
```

相机/NPU 工具会占用 D405 或 NPU，但不初始化机械臂，仍须先确认没有其他进程占用：

```bash
python tools/test_recognition_camera.py
python tools/test_hand_follow_camera.py --frames 15 --prompt hand --confidence 0.10
```

## 10. 日志定位

| 日志前缀 | 含义 |
|---|---|
| `[VISION-PERF]` | 采集 FPS、检测 FPS、推理耗时、候选数量和后端 |
| `[COLOR]` | 多帧颜色证据、类别与置信差 |
| `[REFINE]` / `[CLOSE-REFINE]` | 远距/近距位置复检与修正量 |
| `[PREGRASP]` | 自适应退距、走廊和笛卡尔对齐 |
| `[PLAN]` / `[IK]` | 路径与逆解尝试 |
| `[GRASP]` / `[ACCURACY]` | 闭爪力矩、目标与实际末端误差 |
| `[RETRY]` / `[RECOVERY]` | 失败退离、重识别与回 HOME |
| `[FULL_LOAD]` | 持物返回 HOME 与等待放置 |
| `[PUT1]` / `[PUT2]` / `[READY]` | 放置序列 |
| `[FOLLOW]` / `[HOLD]` | 随动模型、门控和保持原因 |
| `[STREAM-AUTH]` | 网页会话或请求校验失败 |
| `[SHUTDOWN]` / `[SAFETY FAULT]` | 受控退出及降级停机 |

## 11. 已知限制

- 无环境障碍物地图与完整自碰撞检查；`avoid_collisions=False` 的笛卡尔规划不能证明现场安全。
- 瓶子和盒子没有真实抓取参数，保持 fail-closed。
- CPU 手部识别现场约 1–1.5 FPS，随动是低速“停—看—小步”实验功能。
- FULL_LOAD/网页放置状态机已有离线回归，但仍需最终完整实机流程确认。
- FULL_LOAD 中途全局退出的载物处置策略没有单独的“安全放置区”定义。
- ROS2 仅保留兼容源码，当前生产入口仍为单进程 `grasp_demo.py`。
- 仓库包含约 GB 级模型与第三方资产；Git 归档之外还应记录大文件完整性和可恢复来源。
- 当前依赖环境不是锁定式可复现构建，归档时应额外导出 Conda 包清单与 Python 包清单。

## 12. 归档检查清单

归档或打标签前建议完成：

- [ ] 运行 `python tools/run_offline_tests.py` 并保存完整输出。
- [ ] 执行 `git diff --check`，确认无空白错误。
- [ ] 确认 `git status --short` 中仅有计划纳入的文件。
- [ ] 核对 `hand_eye_calibration.json`、`recognition_profiles.json` 与 QNN context SHA-256。
- [ ] 保存当前 Conda 环境与 pip 包列表。
- [ ] 由现场操作者完成一次积木“抓取 → FULL_LOAD HOME → 放置 → HOME”回归。
- [ ] 记录 D405 序列号、机械臂固件/SDK 版本、HOME/PUT 点位和夹爪标定。
- [ ] 不把临时日志、缓存、诊断图片、密钥、Cookie 或无关零字节文件加入提交。
- [ ] 经船长确认后再提交、打标签或推送；不要擅自合并 ROS2/MuJoCo 实验分支。

## 13. 维护原则

1. 真实机械臂动作只允许主控制线程调用；网页、视觉与语音线程只传递数据或排队意图。
2. “能识别”与“允许运动”必须分离；新对象只有在 `ObjectGraspProfile` 参数完整且实机验收后才能开启运动。
3. 坐标问题优先修复刚体变换、TCP 定义和标定，不用不断叠加固定补偿。
4. 运动放宽必须增加对应边界测试，不能只修改阈值。
5. 官方 SDK 与第三方源码保持来源可追溯；项目适配尽量集中在 `Panthera_lib`、配置和入口层。
6. 所有硬件测试由现场操作者执行，并保存终端日志作为验收依据。
