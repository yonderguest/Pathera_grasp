# 配置与标定

本目录保存识别层配置，不保存机械臂关节点位或厂商电机参数。运行入口 `grasp_demo.py` 会通过 `GraspConfig.apply_recognition_profile()` 读取识别 profile，并通过 `load_color_model()` 读取颜色模型。

## 文件

| 文件 | 用途 | 修改要求 |
|---|---|---|
| `recognition_profiles.json` | QNN context、类别顺序、输入输出张量、置信度与颜色后端入口 | context 变化时必须同步类别顺序和张量规格 |
| `color_calibration.json` | 运行时 Lab 颜色中心、尺度与判别门限 | 应由真实 D405 场景标定结果生成或审计后修改 |
| `color_calibration_samples.json` | 生成颜色模型所用的原始样本 | 保留用于复算和来源追踪 |

## 识别 profile

当前默认 profile 为 `object3`，类别顺序固定为：

```text
bottle / box / toy building block
```

QNN 输出中的类别索引没有自描述能力，因此 JSON 中的 `classes` 必须与编译 context 时的 prompt 顺序完全一致。错误顺序会导致框位置看似正常、类别却整体错位。

`block4` 和其他旧 context 仅用于回退/诊断。切换 profile 后至少执行：

```bash
VISION_MODEL_PROFILE=object3 python tools/test_recognition_camera.py
```

相机烟雾测试不初始化机械臂，但会独占 D405 与 NPU。

## 颜色标定

默认 `COLOR_CLASSIFIER=lab`。判别流程对掩膜核心做白平衡增益估计，再以 Lab 中位特征和标定中心分类；最近类距离以及最近/次近类间隔都必须通过。`COLOR_CLASSIFIER=hsv` 仅作为旧算法回退。

修改颜色模型后运行：

```bash
export PYTHONPATH=Panthera-HT_SDK/panthera_python/scripts
python -m unittest discover -s tests -p 'test_recognition_profile.py'
python -m unittest discover -s tests -p 'test_perception_snapshot.py'
```

## 边界

- 手眼标定不在本目录，正式副本是根目录 `hand_eye_calibration.json`。
- HOME/PUT、夹爪、工作空间和运动门限在 `Panthera_lib/grasp_config.py`。
- `recognition_profiles.json` 只说明识别能力，不授予真实运动权限；瓶子/盒子的运动许可由 `ObjectGraspProfile` 单独控制。
- 不要把现场临时绝对路径、密钥或会话信息写入配置。
