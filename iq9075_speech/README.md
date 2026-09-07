# iq9075_speech

可独立复用的离线语音识别包。主入口是 `SpeechRecognizer`，与抓取业务解耦；默认使用 sherpa-onnx SenseVoice CPU 后端，便于离线运行和跨机器恢复。QNN 后端只有在配置和 runner/context 齐全时才应显式启用。

## 模块

| 文件 | 作用 |
|---|---|
| `config.py` | `AsrConfig`、默认模型路径和环境变量 |
| `audio.py` | WAV/PCM 转换与 `arecord` 录音 |
| `backend.py` | CPU SenseVoice、QNN SenseVoice 后端接口与实现 |
| `recognizer.py` | `SpeechRecognizer` 门面和后端选择 |
| `errors.py` | 语音组件异常类型 |
| `__init__.py` | 公共导出 |

## 最小用法

```python
from iq9075_speech import AsrConfig, SpeechRecognizer

recognizer = SpeechRecognizer(AsrConfig())
recognizer.load()
try:
    print(recognizer.transcribe_file("sample.wav"))
finally:
    recognizer.close()
```

输入音频以 16 kHz、单声道 WAV/float32/PCM16 最稳定。麦克风接口调用系统 `arecord`，会占用声卡。

## 环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `IQ9075_ASR_MODEL_DIR` | `models/sensevoice` | 包含 `model.onnx` 与 `tokens.txt` |
| `IQ9075_ASR_THREADS` | `2` | CPU 推理线程数 |
| `IQ9075_ASR_BACKEND` | `sensevoice` | `sensevoice`、`sensevoice_qnn` 或 `auto` |
| `IQ9075_SENSEVOICE_QNN_CONTEXT` | 板端历史路径 | QNN context；归档恢复时要单独确认存在 |
| `IQ9075_SENSEVOICE_QNN_RUNNER` | 板端历史路径 | QNN runner |
| `IQ9075_SENSEVOICE_QNN_MAX_FRAMES` | `512` | QNN 最大帧数 |

`auto` 只依据 context 文件是否存在选择后端；它不证明 runner、库路径和设备权限可用。正式主程序默认 `VOICE_INPUT=0`，语音模块不会自动启动。

## 验证

```bash
python voice_demo/demo_transcribe_file.py /path/to/16k_mono.wav
python voice_demo/demo_transcribe_mic.py
```

第二条会使用麦克风，仅在声卡空闲且现场允许时执行。静态编译由 `tools/run_offline_tests.py` 覆盖。
