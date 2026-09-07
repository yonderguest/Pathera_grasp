# 第三方依赖与实验后端

本目录保存外部项目或板端运行资产。除项目自有的轻量适配文件外，不应在归档整理时机械性重写第三方源码。

| 路径 | 内容 | 当前使用状态 |
|---|---|---|
| `graspnet-baseline/` | GraspNet 基线实现及可选 checkpoint | 实验候选后端，默认关闭 |
| `graspnetAPI/` | GraspNet 数据结构、NMS 和可视化 API | GraspNet 实验依赖 |
| `qnn/` | IQ9075 QNN runner、C 源码与 YOLOE context | 默认物体识别主线使用 |

## 维护规则

- 保留各上游 README、许可证、版本/提交来源。
- 不在第三方包内混入业务逻辑；适配代码放在 `Panthera_lib/graspnet_pipeline.py`、`npu_inference.py` 或配置层。
- 上游升级后运行完整离线回归，并单独执行对应相机/NPU/GraspNet 烟雾测试。
- checkpoint、context 和编译二进制属于平台相关资产，归档时记录 SHA-256 与目标架构。
- GraspNet 的候选成功不等于可执行运动；最终仍必须通过项目工作空间、IK、姿态、关节跳变和路径检查。

## GraspNet

默认 `GRASPNET_USE=0`。离线检查参见 `tools/test_graspnet_offline.py`。只有在依赖、checkpoint、点云尺度、夹爪坐标修正和候选过滤都经过验证后，才允许在受控环境显式启用。

## QNN

QNN 资产与诊断详见 [qnn/README.md](qnn/README.md)。
