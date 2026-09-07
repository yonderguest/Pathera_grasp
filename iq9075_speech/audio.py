# -*- coding: utf-8 -*-
"""音频工具：WAV / 裸 PCM16 <-> float32；可选 arecord 录音。

说明：
- 与原项目一致，统一使用 16 kHz、单声道、S16_LE 音频格式。
- 本模块不依赖任何业务代码，可在任意 Python 3.10+ 环境使用。
"""
import os
import subprocess
import tempfile

import numpy as np
import soundfile as sf

from .errors import AudioReadError


def read_wav(path: str):
    """读取 WAV 文件 -> (float32 单声道样本, 采样率)"""
    try:
        samples, sr = sf.read(path, dtype="float32")
    except Exception as e:
        raise AudioReadError(f"读取 WAV 失败: {path} ({e})")
    if samples.ndim > 1:
        samples = samples.mean(axis=1)  # 多声道转单声道
    return np.asarray(samples, dtype=np.float32), int(sr)


def pcm16_to_float32(pcm: bytes) -> np.ndarray:
    """PCM16 字节流 -> float32（范围 -1.0 ~ 1.0）"""
    raw = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    return raw / 32768.0


def float32_to_pcm16(samples: np.ndarray) -> bytes:
    """float32 波形 -> PCM16 字节流"""
    s = np.clip(samples, -1.0, 1.0)
    return (s * 32767.0).astype(np.int16).tobytes()


def record_arecord(duration: float = 3.0, sample_rate: int = 16000,
                   device: str = "default") -> str:
    """用系统 arecord 录音（ALSA），返回临时 WAV 路径。

    需要目标环境装有 arecord（Linux/ALSA）。录音失败抛 AudioReadError。
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = tmp.name
    tmp.close()
    try:
        subprocess.run(
            ["arecord", "-D", device, "-d", str(int(duration)),
             "-f", "S16_LE", "-r", str(sample_rate), "-c", "1",
             "-t", "wav", tmp_path],
            capture_output=True, timeout=int(duration) + 2)
    except Exception as e:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        raise AudioReadError(f"arecord 调用失败: {e}")
    if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) < 100:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        raise AudioReadError("arecord 录音失败或音频文件无效")
    return tmp_path
