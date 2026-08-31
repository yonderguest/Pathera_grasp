# -*- coding: utf-8 -*-
"""SpeechRecognizer 门面：对外极简调用接口。

输入：
  - transcribe_file(path)     音频文件（WAV，16k 单声道最佳）
  - transcribe_waveform(...)  numpy float32 波形
  - transcribe_pcm16(...)     PCM16 字节流（实时流式场景）
  - record_and_transcribe(...) 用 arecord 录音后识别（可选，需系统工具）
输出：
  - str 识别文本（空字符串表示未识别到有效语音）
"""
import os
from typing import Optional

import numpy as np

from .audio import read_wav, pcm16_to_float32, record_arecord
from .backend import ASRBackend, SenseVoiceBackend, SenseVoiceQnnBackend
from .config import DEFAULT_ASR_MODEL_DIR, AsrConfig


def _has_qnn_model(config: AsrConfig) -> bool:
    """判断 SenseVoice QNN context binary 是否可用。"""
    return bool(config.qnn_context_binary) and os.path.isfile(config.qnn_context_binary)


def _create_backend(config: AsrConfig) -> ASRBackend:
    """按配置选择后端：auto 时优先 SenseVoice NPU，失败回退 CPU。"""
    name = (config.backend or "auto").strip().lower()
    if name in ("", "auto"):
        name = "sensevoice_qnn" if _has_qnn_model(config) else "sensevoice"
    if name == "sensevoice_qnn":
        return SenseVoiceQnnBackend(
            model_dir=config.model_dir or DEFAULT_ASR_MODEL_DIR,
            tokens_path=config.tokens_path,
            context_binary=config.qnn_context_binary,
            runner=config.qnn_runner,
            max_frames=config.qnn_max_frames,
            language=config.language,
            use_itn=config.use_itn,
        )
    if name == "sensevoice":
        return SenseVoiceBackend(
            model_path=config.model_path,
            tokens_path=config.tokens_path,
            num_threads=config.num_threads,
            language=config.language,
            use_itn=config.use_itn,
        )
    raise ValueError("未知 ASR 后端: " + config.backend)


class SpeechRecognizer:
    """独立可复用的语音识别组件门面，不依赖任何业务代码/全局状态。"""

    def __init__(self, config: Optional[AsrConfig] = None,
                 backend: Optional[ASRBackend] = None):
        self._config = (config or AsrConfig.from_env()).resolve()
        self._backend = backend
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    def load(self) -> "SpeechRecognizer":
        """加载模型（幂等）"""
        if self._loaded:
            return self
        if self._backend is None:
            self._backend = _create_backend(self._config)
        self._backend.load()
        self._loaded = True
        return self

    # ── 对外接口 ──

    def transcribe_file(self, wav_path: str, sample_rate: Optional[int] = None) -> str:
        """识别一个音频文件 -> 文本"""
        samples, sr = read_wav(wav_path)
        if sample_rate is None:
            sample_rate = sr
        return self.transcribe_waveform(samples, sample_rate)

    def transcribe_waveform(self, samples: np.ndarray, sample_rate: int = 16000) -> str:
        """识别一段 float32 波形 -> 文本"""
        self._ensure_loaded()
        return self._backend.transcribe(samples, int(sample_rate))

    def transcribe_pcm16(self, pcm_bytes: bytes, sample_rate: int = 16000) -> str:
        """识别一段 PCM16 字节流 -> 文本（适合实时流）"""
        samples = pcm16_to_float32(pcm_bytes)
        return self.transcribe_waveform(samples, sample_rate)

    def record_and_transcribe(self, duration: float = 3.0) -> str:
        """（可选）arecord 录音后识别 -> 文本"""
        tmp = record_arecord(duration, sample_rate=self._config.sample_rate)
        try:
            return self.transcribe_file(tmp, sample_rate=self._config.sample_rate)
        finally:
            try:
                os.remove(tmp)
            except Exception:
                pass

    def close(self) -> None:
        if self._backend is not None:
            self._backend.close()
        self._loaded = False

    # ── 内部 ──
    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()
