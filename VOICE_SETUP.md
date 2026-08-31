# 语音功能迁移与运行说明

本项目已把 `iq9075_speech`（听）和 `iq9075_tts`（说）复制到仓库内，并接入
`grasp_demo.py` / `grasp_planner.py`。

## 目录

```text
pathera_grasp/
├── iq9075_speech/        # 语音识别组件（sherpa-onnx SenseVoice）
├── iq9075_tts/           # 语音播报组件（sherpa-onnx VITS / edge-tts / QNN MeloTTS）
├── voice_controller.py   # grasp_demo 使用的语音胶水层
├── voice_demo/           # ASR/TTS 自测脚本与模型下载脚本
├── requirements_asr.txt
├── requirements_tts.txt
└── models/
    ├── sensevoice/       # ASR 模型：model.onnx + tokens.txt + meta.json
    └── sherpa_tts/       # 离线 TTS 模型：vits-melo-tts-zh_en
```

## 安装依赖

```bash
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate pathera_grasp
pip install -r requirements_asr.txt -r requirements_tts.txt
```

目标环境还需要系统工具：

- `mplayer`：语音播放。
- `arecord`：麦克风录音（`record_and_transcribe` 使用，可选）。

## 模型

模型文件已放在 `pathera_grasp/models/`，换机器时整个项目目录一起拷贝即可，不需要
再依赖 `/home/ubuntu/work` 下的路径。

如果模型文件丢失，可在项目根目录执行：

```bash
bash voice_demo/download_sensevoice.sh
bash voice_demo/download_melo_tts.sh
```

模型路径也可以手动用环境变量覆盖：

```bash
export IQ9075_ASR_MODEL_DIR=/path/to/sensevoice
export IQ9075_SHERPA_TTS_MODEL_DIR=/path/to/vits-melo-tts-zh_en
```

## 语音后端默认值

为保证换机器也能离线运行，项目默认采用：

- ASR：`sherpa-onnx + SenseVoice`，CPU 推理。
- TTS：`sherpa-onnx + VITS`，CPU 离线合成。

可以通过环境变量切换：

```bash
# 语音输入关闭（回退到终端输入）
VOICE_INPUT=0 python grasp_demo.py

# 单次语音识别录音时长，单位秒
VOICE_PROMPT_DURATION=5 python grasp_demo.py
```

## 自测

```bash
cd pathera_grasp

# TTS：离线合成并播放一句
python voice_demo/demo_say.py "小八在，检测到红色方块"

# ASR：识别一个 16kHz WAV
python voice_demo/demo_transcribe_file.py /path/to/audio.wav

# ASR：arecord 录音 3 秒后识别
python voice_demo/demo_transcribe_mic.py
```

## 接入点

- 语音选择颜色：`grasp_demo.py` 的 `choose_target_at_start(voice)` 会先尝试一次
  语音识别，识别到“红色积木/黄色积木/蓝色积木/任意颜色”后直接执行；识别失败时
  回退到终端输入。
- 语音状态播报：`grasp_planner.py` 会在回 HOME、开始扫描、发现目标、夹紧、放置、
  回零等关键节点通过 `VoiceInterface.say()` 非阻塞播报。
