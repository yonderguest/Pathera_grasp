# -*- coding: utf-8 -*-
"""IQ-9075 语音识别独立组件（解耦自 IQ-9075 智能视检平台）

对外极简接口：
    from iq9075_speech import SpeechRecognizer, AsrConfig
    asr = SpeechRecognizer(AsrConfig(model_dir=...))
    asr.load()
    text = asr.transcribe_file("a.wav")          # 输入音频文件
    text = asr.transcribe_waveform(samples, sr)  # 输入 numpy 波形
    text = asr.transcribe_pcm16(pcm_bytes, sr)   # 输入 PCM16 字节流
    asr.close()

后端：默认使用 sherpa-onnx + SenseVoice（离线 ONNX，CPU），便于跨机器恢复；
可通过配置显式启用 SenseVoice QNN，或用 auto 在 QNN context 存在时优先 NPU。
"""
from .recognizer import SpeechRecognizer
from .config import AsrConfig
from .backend import ASRBackend, SenseVoiceBackend, SenseVoiceQnnBackend, QnnBackend

__all__ = [
    "SpeechRecognizer",
    "AsrConfig",
    "ASRBackend",
    "SenseVoiceBackend",
    "SenseVoiceQnnBackend",
    "QnnBackend",
]
__version__ = "1.0.0"
