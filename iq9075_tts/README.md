# iq9075_tts

可独立复用的非阻塞语音播报包。`Speaker.say()` 只入队，后台线程按顺序合成和播放；默认使用离线 sherpa-onnx VITS，避免主流程依赖外网。

## 模块

| 文件 | 作用 |
|---|---|
| `config.py` | edge、sherpa VITS、MeloTTS QNN 配置 |
| `backend.py` | 在线、离线、QNN、预录 WAV 与回退后端 |
| `speaker.py` | 队列、工作线程、播放器生命周期和临时文件清理 |
| `errors.py` | TTS 异常类型 |
| `__init__.py` | 公共导出 |

## 最小用法

```python
from iq9075_tts import Speaker

speaker = Speaker()
speaker.say("检测到绿色积木")
try:
    while speaker.is_busy():
        pass
finally:
    speaker.close()
```

业务代码应避免忙等，主项目的 `VoiceInterface` 会等待队列和播放器结束，并在录音前保留短暂声学消退时间。

## 后端

通过 `IQ9075_TTS_BACKEND` 选择：

- `sherpa` / `vits` / `offline`：默认，离线 VITS。
- `auto`：先尝试 MeloTTS QNN 服务，失败回退 sherpa VITS。
- `melo_qnn` / `qnn` / `melotts`：只使用本机 HTTP QNN 服务。
- `edge` / `edge-tts` / `online`：需要外网。

常用变量：`IQ9075_SHERPA_TTS_MODEL_DIR`、`IQ9075_SHERPA_TTS_THREADS`、`IQ9075_SHERPA_TTS_SPEED`、`IQ9075_TTS_PLAYER`、`IQ9075_TTS_QUEUE_SIZE`、`IQ9075_TTS_PLAY_TIMEOUT`。

默认模型目录应包含 `model.onnx`、`tokens.txt`、`lexicon.txt`，可选 `date.fst` 与 `number.fst`。默认播放器是 `mplayer`。

## 生命周期

- `say()`：非阻塞入队；队列满时丢弃最新消息。
- `clear()`：清空等待队列，不打断当前播放。
- `stop()`：结束当前播放器并清空队列。
- `close()`：停止并通知后台线程退出。

临时音频使用独占临时文件创建，并在每次播报后清理。静态编译由 `tools/run_offline_tests.py` 覆盖。
