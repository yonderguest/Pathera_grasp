# panthera_voice

ROS2 语音适配节点，把 `iq9075_speech`/`iq9075_tts` 包装为 topic 接口。

## Topic

- 订阅 `/voice/listen_request`：收到 `true` 后录音并识别。
- 订阅 `/voice/say`：异步播报文本。
- 发布 `/voice/command`：识别出的命令。

`voice_enabled=false` 时不初始化或请求语音硬件。录音与播放会占用声卡，且应避免 TTS 回声进入下一轮 ASR。

当前主程序默认关闭语音，ROS2 也不是正式运行入口。独立语音验证优先使用根目录 `voice_demo/`。
