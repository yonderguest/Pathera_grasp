# MuJoCo 模型资产

## `panthera_grasp_scene.xml`

MuJoCo 场景，包含 Panthera-HT 六轴结构、代理碰撞体、虚拟 D405、代理夹爪、桌面、三色积木和 PUT1 放置区。机械臂常规 STL 直接复用本项目已有 Panthera SDK，仅 `../assets/panthera_ht_stl/link2_visual.STL` 是为实时显示保留的降面副本；D405 外观的来源和 Apache-2.0 文本位于 `../assets/realsense_d405/`。

## `block_detector.pt`

这是仿真专用的单类别 Ultralytics YOLO 权重，只用于合成场景中的绿色积木回归。它由 `../scripts/train_synthetic_yolo.py` 生成：脚本用固定种子的 MuJoCo RGB 图像构建数据集，再从 Ultralytics `yolo11n.pt` 预训练权重微调。

该权重不能替代 IQ9075 上的 QNN YOLOE context，也不应宣称支持真实相机、红蓝颜色分类或通用物体识别。重新训练产生的数据、日志和临时权重必须保存在 `reports/mujoco_sim/`，不得提交。

此权重继承上游 Ultralytics 模型与软件的许可约束；公开分发或商业交付前，维护者需要再次核对所用 Ultralytics 版本及许可，不应把该文件视为无约束的自有模型。
