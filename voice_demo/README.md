# 语音示例

本目录用于独立验证 `iq9075_speech` 和 `iq9075_tts`，不初始化相机或机械臂。麦克风与播放示例会占用声卡。

## 示例

识别已有 WAV：

```bash
python voice_demo/demo_transcribe_file.py /path/to/sample_16k_mono.wav
```

录音三秒后识别：

```bash
python voice_demo/demo_transcribe_mic.py
```

默认离线 VITS 播报：

```bash
python voice_demo/demo_say.py "检测到绿色积木"
```

其他播报后端：

```bash
python voice_demo/demo_say.py --sherpa "离线播报"
python voice_demo/demo_say.py --edge "在线播报"
python voice_demo/demo_say.py --offline /path/to/fallback.wav
```

## 模型下载

```bash
bash voice_demo/download_sensevoice.sh
bash voice_demo/download_melo_tts.sh
```

脚本会下载并解压较大的第三方模型，属于有网络和磁盘写入的操作；归档恢复时先检查 `models/`，已有完整模型时不要重复下载。下载来源、许可证和哈希应随归档记录。

## 与主程序的关系

主程序默认 `VOICE_INPUT=0`。只有显式设置 `VOICE_INPUT=1` 才创建 `VoiceInterface`；语音不可用时会回退到终端/网页目标输入。不要让独立语音示例与主程序同时争用声卡。
