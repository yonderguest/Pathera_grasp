"""IQ-9075 语音播报独立组件（iq9075_tts）。

输入文本，输出语音播报。从 D:/IQ-9075 的 backend/tts_manager.py 剥离解耦：
edge-tts 合成 + mplayer 播放，队列/非阻塞；另提供离线预录音频兜底。
"""
from .backend import (
    EdgeTtsBackend,
    FallbackTtsBackend,
    MeloTtsQnnBackend,
    OfflineWavBackend,
    SenseVoiceTtsBackend,
    SherpaTtsBackend,
    TtsBackend,
)
from .config import MeloTtsQnnConfig, SherpaTtsConfig, TtsConfig
from .speaker import Speaker

__all__ = [
    "TtsConfig",
    "Speaker",
    "TtsBackend",
    "EdgeTtsBackend",
    "FallbackTtsBackend",
    "MeloTtsQnnBackend",
    "OfflineWavBackend",
    "SherpaTtsBackend",
    "SenseVoiceTtsBackend",
    "SherpaTtsConfig",
    "MeloTtsQnnConfig",
]
