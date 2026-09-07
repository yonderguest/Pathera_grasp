# IQ9075 QNN 物体识别资产

此目录提供 `NpuYoloDetector` 使用的本地子进程和 QNN context。`npu_server` 从标准输入接收预处理后的张量，执行 QNN HTP 推理并把固定输出写回标准输出；Python 层负责超时、解码、类别映射与 NMS。

## 文件

| 文件 | 说明 |
|---|---|
| `npu_server` | AArch64/IQ9075 可执行 runner |
| `npu_server.c` | runner 源码 |
| `yoloe-26s-seg_640_iq9075_qnn_object3.bin` | 默认：bottle / box / toy building block |
| `yoloe-26s-seg_640_iq9075_qnn_block4.bin` | 旧四提示积木 context |
| `yoloe-26s-seg_640_iq9075_qnn_brick6.bin` | 旧积木实验 context |

## 强约束

- context 类别顺序必须与 `config/recognition_profiles.json` 一致。
- 输入尺寸、输出名和输出形状必须与 profile 一致。
- `object3` 不能用于人手识别；当前人手随动仍使用 CPU YOLOE。
- `npu_server` 必须有执行权限，并能找到目标机上的 Qualcomm QNN/HTP 运行库。
- Python 端读取有超时和 stderr 上限；不要去掉这些保护来掩盖 NPU 卡死。

## 验证

离线解码/超时测试：

```bash
export PYTHONPATH=Panthera-HT_SDK/panthera_python/scripts
python -m unittest discover -s tests -p 'test_npu_*.py'
```

对静态图片比较 context：

```bash
python tools/diagnose_npu_models.py /path/to/image.jpg
```

D405 实时烟雾测试：

```bash
YOLO_NPU=1 VISION_MODEL_PROFILE=object3 python tools/test_recognition_camera.py
```

后两项会使用 NPU，其中实时测试还会独占相机，但不会初始化机械臂。

## 归档

至少记录：runner/context 的 SHA-256、文件大小、QNN SDK/runtime 版本、目标 SoC、编译输入输出张量以及 prompt 顺序。仅保存 `.bin` 而不保存这些元数据会显著增加后续恢复风险。
