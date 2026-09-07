# 语音功能说明

项目把离线语音组件放在仓库内，并通过 `voice_controller.py` 接入单体入口和 ROS2 voice node。

```text
iq9075_speech/     SenseVoice ASR 组件
iq9075_tts/        sherpa-onnx VITS、edge/QNN 等 TTS 后端实现
voice_controller.py 主流程使用的半双工胶水层
voice_demo/        独立 ASR/TTS 演示和模型下载脚本
models/sensevoice/ ASR 模型
models/sherpa_tts/ VITS/MeloTTS 模型
```

## 当前集成后端

主抓取流程固定使用项目内、离线、CPU 路径：

- ASR：`sherpa-onnx + SenseVoice`
- TTS：`sherpa-onnx + VITS/MeloTTS`

即使语音包内部存在 edge-tts 或 QNN 选择，`grasp_demo.py` 和 ROS2 `panthera_voice` 当前不会自动切换到它们。可选 QNN ASR 的历史路径不属于主流程，也不应作为迁移依赖。

## 半双工行为

`VoiceInterface` 现在保证以下顺序：

```text
TTS 播报入队 → 等待队列和播放器结束 → 短暂声学消退 → ASR 录音
```

录音期间 `say()` 会与录音共用锁，新的状态播报会等待，不会播放到正在录音的麦克风中。ROS2 voice node 还在 listen request 后使用短暂 debounce，确保前一个 `voice/say` topic 回调有机会先入队。

这减少提示词自回声，但不替代真实声学环境中的 AEC、唤醒词或人工确认；真机仍应在环境噪声下验证。

## 颜色命令

支持红、黄、蓝、绿、白、黑和任意颜色，中英文均可。解析器会识别直接否定：

```text
“不要红色”          → 不接受，不自动抓取
“不要红色，要蓝色”  → 选择蓝色
“不是黄色，抓绿色”  → 选择绿色
```

多颜色、复杂否定或自然语言歧义没有大模型语义消解；不确定时会回退终端输入，而不是猜测抓取目标。

## 运行参数

```bash
# 关闭语音，始终使用终端选择颜色
VOICE_INPUT=0 python grasp_demo.py

# 单次录音时长（秒）
VOICE_PROMPT_DURATION=5 python grasp_demo.py

# 覆盖项目内模型目录
IQ9075_ASR_MODEL_DIR=/path/to/sensevoice python grasp_demo.py
IQ9075_SHERPA_TTS_MODEL_DIR=/path/to/vits-melo-tts-zh_en python grasp_demo.py
```

主流程需要系统提供 `arecord`（录音）和播放器（默认 `mplayer`）。本次没有执行这些命令，也没有安装依赖。

## 独立离线检查

这些脚本会触发本机声卡或播放设备，不属于 TASK-003 的无硬件验证范围：

```bash
python voice_demo/demo_say.py "检测到红色方块"
python voice_demo/demo_transcribe_file.py /path/to/audio.wav
python voice_demo/demo_transcribe_mic.py
```

仅验证代码逻辑时使用项目根目录的 `python tools/run_offline_tests.py`；它用 fake recognizer/speaker 测试半双工顺序，不访问声卡。
