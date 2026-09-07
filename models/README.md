# 模型资产

此目录保存运行时大模型文件。模型通常占用数百 MB 到 1 GB，归档时不能只保存源代码而遗漏模型，也不应假定所有 Git 托管方式都能可靠承载大文件。

## 当前资产

| 路径 | 使用者 | 作用 |
|---|---|---|
| `yoloe-26s-seg.pt` | CPU YOLOE、手部随动、物体 CPU 回退 | 通用分割权重 |
| `sensevoice/model.onnx` | `iq9075_speech` | 离线 ASR |
| `sensevoice/tokens.txt` | `iq9075_speech` | ASR token 表 |
| `sensevoice/meta.json` | 归档/模型信息 | SenseVoice 元数据 |
| `sherpa_tts/vits-melo-tts-zh_en/` | `iq9075_tts` | 中文/英文离线 VITS TTS |

QNN 物体识别 context 不在本目录，位于 `third_party/qnn/`。YOLOE 文本编码器 `mobileclip2_b.ts` 位于仓库根目录。

## 完整性

归档前建议记录：

```bash
sha256sum models/yoloe-26s-seg.pt mobileclip2_b.ts
sha256sum models/sensevoice/model.onnx models/sensevoice/tokens.txt
sha256sum models/sherpa_tts/vits-melo-tts-zh_en/model.onnx
```

同时记录文件大小、来源版本和许可证。恢复后先做 import/模型加载测试，再连接任何硬件。

## 边界

- 不要提交推理生成图片、缓存或临时音频到本目录。
- 替换模型可能改变类别顺序、掩膜质量、输入尺寸和内存占用，必须同步配置与测试。
- 第三方模型许可证由模型发布方决定，本 README 不替代原许可证。
