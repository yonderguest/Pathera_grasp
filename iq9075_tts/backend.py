"""TTS 后端：edge-tts 合成 + mplayer 播放；离线预录音频兜底；sherpa-onnx 离线合成。"""
from __future__ import annotations

import os
import subprocess
from typing import Protocol

from .config import MeloTtsQnnConfig, SherpaTtsConfig, TtsConfig
from .errors import SynthesizeError


class TtsBackend(Protocol):
    """TTS 后端协议：文本 -> 合成音频 -> 播放。"""

    def synthesize(self, text: str, out_path: str) -> bool: ...
    def play(self, audio_path: str) -> subprocess.Popen | None: ...


class EdgeTtsBackend:
    """edge-tts（微软 Azure 语音）合成，mplayer 播放。需联网。"""

    output_suffix = ".mp3"

    def __init__(self, config: TtsConfig | None = None) -> None:
        self.config = config or TtsConfig()
        os.makedirs(self.config.cache_dir, exist_ok=True)

    def synthesize(self, text: str, out_path: str) -> bool:
        cmd = [
            "edge-tts",
            "--voice", self.config.voice,
            "--rate", self.config.rate,
            "--volume", self.config.volume,
            "--text", text,
            "--write-media", out_path,
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=self.config.synth_timeout)
        return result.returncode == 0 and os.path.isfile(out_path)

    def play(self, audio_path: str) -> subprocess.Popen | None:
        return subprocess.Popen(
            [self.config.player, "-ao", "alsa", audio_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


class OfflineWavBackend:
    """离线兜底：直接播放预录音频文件（不联网、不合成）。"""

    output_suffix = ".wav"

    def __init__(self, wav_path: str, config: TtsConfig | None = None) -> None:
        self.wav_path = wav_path
        self.config = config or TtsConfig()

    def synthesize(self, text: str, out_path: str) -> bool:
        return os.path.isfile(self.wav_path)

    def play(self, audio_path: str) -> subprocess.Popen | None:
        target = audio_path if os.path.isfile(audio_path) else self.wav_path
        return subprocess.Popen(
            [self.config.player, "-ao", "alsa", target],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


class SherpaTtsBackend:
    """sherpa-onnx 离线 TTS（VITS，如 vits-melo-tts-zh_en），断网可用。

    与 iq9075_speech.SenseVoiceBackend 同栈：sherpa-onnx + ONNX Runtime，
    CPU 推理，ARM64/aarch64 可直接运行。
    """

    output_suffix = ".wav"

    def __init__(self, config: SherpaTtsConfig | None = None) -> None:
        self.config = (config or SherpaTtsConfig()).resolve()
        self._sherpa = None
        self._tts = None

    def _load(self):
        """惰性加载 sherpa-onnx 模型（幂等）。"""
        if self._tts is not None:
            return self._tts
        missing = [
            p for p in (
                self.config.model_path,
                self.config.tokens_path,
                self.config.lexicon_path,
            ) if not p or not os.path.isfile(p)
        ]
        if missing:
            raise FileNotFoundError(
                "sherpa-tts 模型文件缺失：" + ", ".join(str(p) for p in missing)
                + "。请先运行 demo/download_melo_tts.sh 下载模型。"
            )
        import sherpa_onnx

        self._sherpa = sherpa_onnx
        vits = sherpa_onnx.OfflineTtsVitsModelConfig(
            model=self.config.model_path,
            tokens=self.config.tokens_path,
            lexicon=self.config.lexicon_path,
            data_dir=self.config.data_dir,
            noise_scale=self.config.noise_scale,
            noise_scale_w=self.config.noise_scale_w,
            length_scale=self.config.length_scale,
        )
        model = sherpa_onnx.OfflineTtsModelConfig(
            vits=vits,
            provider=self.config.provider,
            num_threads=self.config.num_threads,
            debug=False,
        )
        tts_config = sherpa_onnx.OfflineTtsConfig(
            model=model,
            rule_fsts=self.config.rule_fsts,
            max_num_sentences=1,
        )
        if not tts_config.validate():
            raise ValueError("sherpa-tts 配置校验失败，请检查模型文件路径")
        self._tts = sherpa_onnx.OfflineTts(tts_config)
        return self._tts

    def synthesize(self, text: str, out_path: str) -> bool:
        try:
            tts = self._load()
            audio = self._generate(tts, text)
            if audio is None or len(audio.samples) == 0:
                print("[sherpa-tts] 合成结果为空")
                return False
            self._save_wave(audio, out_path)
            return os.path.isfile(out_path)
        except Exception as exc:  # noqa: BLE001
            print(f"[sherpa-tts] 合成失败: {exc}")
            return False

    def _generate(self, tts, text: str):
        """兼容新旧版 sherpa-onnx 的 generate 调用方式。"""
        try:
            # 新版：GenerationConfig（含 silence_scale 等参数）
            gen_config = self._sherpa.GenerationConfig()
            gen_config.sid = self.config.sid
            gen_config.speed = self.config.speed
            gen_config.silence_scale = self.config.silence_scale
            return tts.generate(text, gen_config)
        except (AttributeError, TypeError):
            # 旧版：直接传 sid / speed
            return tts.generate(text, sid=self.config.sid, speed=self.config.speed)

    def _save_wave(self, audio, out_path: str) -> None:
        if hasattr(self._sherpa, "write_wave"):
            self._sherpa.write_wave(out_path, audio.samples, audio.sample_rate)
        else:
            import soundfile as sf

            sf.write(out_path, audio.samples, audio.sample_rate, subtype="PCM_16")

    def play(self, audio_path: str) -> subprocess.Popen | None:
        return subprocess.Popen(
            [self.config.player, "-ao", "alsa", audio_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


class MeloTtsQnnBackend:
    """Qualcomm MeloTTS（QNN/HTP，目标 QCS9075）后端。

    推理由 Audio Analytics 服务完成（模型在 NPU 上运行），本后端调用其
    /audio-analytics/v1/api/tts/synthesize 接口，把返回的裸 PCM 转成 WAV。
    """

    output_suffix = ".wav"

    def __init__(self, config: MeloTtsQnnConfig | None = None) -> None:
        self.config = config or MeloTtsQnnConfig()

    @property
    def synthesize_url(self) -> str:
        return (
            self.config.base_url.rstrip("/")
            + "/audio-analytics/v1/api/tts/synthesize"
        )

    def synthesize(self, text: str, out_path: str) -> bool:
        try:
            import requests
        except ImportError as exc:
            raise SynthesizeError(f"未安装 requests: {exc}")
        try:
            resp = requests.post(
                self.synthesize_url,
                json={
                    "text": text,
                    "model": self.config.model,
                    "language": self.config.language,
                    "sample_rate": self.config.sample_rate,
                },
                timeout=self.config.timeout,
            )
        except Exception as exc:  # noqa: BLE001
            raise SynthesizeError(f"MeloTTS 服务不可用: {exc}")
        if resp.status_code != 200:
            raise SynthesizeError(
                f"MeloTTS 返回 {resp.status_code}: {resp.text[:200]}")
        pcm = resp.content
        if not pcm:
            return False
        import wave

        with wave.open(out_path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.config.sample_rate)
            w.writeframes(pcm)
        return os.path.isfile(out_path)

    def play(self, audio_path: str) -> subprocess.Popen | None:
        return subprocess.Popen(
            [self.config.player, "-ao", "alsa", audio_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


class FallbackTtsBackend:
    """主后端失败时自动切换备用后端（如 MeloTTS QNN -> 离线 VITS）。"""

    output_suffix = ".wav"

    def __init__(self, primary: TtsBackend, fallback: TtsBackend) -> None:
        self.primary = primary
        self.fallback = fallback

    def synthesize(self, text: str, out_path: str) -> bool:
        try:
            if self.primary.synthesize(text, out_path):
                return True
        except Exception as exc:  # noqa: BLE001
            print(f"[tts] 主后端(MeloTTS QNN)失败，回退离线 VITS: {exc}")
        return self.fallback.synthesize(text, out_path)

    def play(self, audio_path: str) -> subprocess.Popen | None:
        return self.primary.play(audio_path)


# SenseVoice 是 ASR 模型（"听"），TTS（"说"）用的是 sherpa-onnx 的 VITS 模型。
# 为符合"与 ASR 同栈（sherpa-onnx）"的心智，保留 SenseVoiceTtsBackend 作为别名。
SenseVoiceTtsBackend = SherpaTtsBackend
