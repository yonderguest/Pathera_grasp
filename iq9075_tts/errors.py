"""TTS 异常定义。"""
from __future__ import annotations


class TtsError(Exception):
    """TTS 基础异常。"""


class SynthesizeError(TtsError):
    """合成失败。"""


class PlaybackError(TtsError):
    """播放失败。"""
