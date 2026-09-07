"""TTS 配置（支持环境变量覆盖）。"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


# pathera_grasp 根目录：<项目根>/models/sherpa_tts/vits-melo-tts-zh_en
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TTS_MODEL_DIR = str(
    _PROJECT_ROOT / "models" / "sherpa_tts" / "vits-melo-tts-zh_en"
)


@dataclass
class TtsConfig:
    """edge-tts / 播放器 / 队列相关配置。"""

    voice: str = os.environ.get("IQ9075_TTS_VOICE", "zh-CN-XiaoxiaoNeural")
    rate: str = os.environ.get("IQ9075_TTS_RATE", "+0%")
    volume: str = os.environ.get("IQ9075_TTS_VOLUME", "+0%")
    cache_dir: str = os.environ.get("IQ9075_TTS_CACHE_DIR", "/tmp/iq9075_tts")
    player: str = os.environ.get("IQ9075_TTS_PLAYER", "mplayer")
    queue_size: int = int(os.environ.get("IQ9075_TTS_QUEUE_SIZE", "20"))
    synth_timeout: float = float(os.environ.get("IQ9075_TTS_SYNTH_TIMEOUT", "10"))
    play_timeout: float = float(os.environ.get("IQ9075_TTS_PLAY_TIMEOUT", "20"))


@dataclass
class SherpaTtsConfig:
    """sherpa-onnx 离线 TTS（VITS，如 vits-melo-tts-zh_en）配置。

    与 iq9075_speech.AsrConfig 同栈：model_dir 与 model_path/tokens_path/lexicon_path
    二选一；推荐直接指定 model_dir。
    """

    model_dir: str = os.environ.get(
        "IQ9075_SHERPA_TTS_MODEL_DIR",
        DEFAULT_TTS_MODEL_DIR,
    )
    model_path: str = os.environ.get("IQ9075_SHERPA_TTS_MODEL", "")
    tokens_path: str = os.environ.get("IQ9075_SHERPA_TTS_TOKENS", "")
    lexicon_path: str = os.environ.get("IQ9075_SHERPA_TTS_LEXICON", "")
    # vits-melo-tts-zh_en 用 lexicon+tokens 做中文发音，不需要 espeak-ng 的 data_dir；
    # 其它 piper 英文模型才需要 data_dir（espeak-ng-data）。
    data_dir: str = os.environ.get("IQ9075_SHERPA_TTS_DATA_DIR", "")
    # 逗号分隔的规则 FST（数字/日期归一化），如 date.fst,number.fst；留空表示不启用。
    rule_fsts: str = os.environ.get("IQ9075_SHERPA_TTS_RULE_FSTS", "")
    provider: str = os.environ.get("IQ9075_SHERPA_TTS_PROVIDER", "cpu")
    num_threads: int = int(os.environ.get("IQ9075_SHERPA_TTS_THREADS", "2"))
    sid: int = int(os.environ.get("IQ9075_SHERPA_TTS_SID", "0"))
    speed: float = float(os.environ.get("IQ9075_SHERPA_TTS_SPEED", "1.0"))
    silence_scale: float = float(os.environ.get("IQ9075_SHERPA_TTS_SILENCE_SCALE", "0.2"))
    noise_scale: float = float(os.environ.get("IQ9075_SHERPA_TTS_NOISE_SCALE", "0.667"))
    noise_scale_w: float = float(os.environ.get("IQ9075_SHERPA_TTS_NOISE_SCALE_W", "0.8"))
    length_scale: float = float(os.environ.get("IQ9075_SHERPA_TTS_LENGTH_SCALE", "1.0"))
    player: str = os.environ.get("IQ9075_TTS_PLAYER", "mplayer")

    def resolve(self) -> "SherpaTtsConfig":
        """若指定 model_dir，自动拼出 model_path / tokens_path / lexicon_path / rule_fsts。"""
        if self.model_dir:
            self.model_path = self.model_path or os.path.join(self.model_dir, "model.onnx")
            self.tokens_path = self.tokens_path or os.path.join(self.model_dir, "tokens.txt")
            self.lexicon_path = self.lexicon_path or os.path.join(self.model_dir, "lexicon.txt")
            if not self.rule_fsts:
                fsts = [os.path.join(self.model_dir, n) for n in ("date.fst", "number.fst")]
                self.rule_fsts = ",".join(f for f in fsts if os.path.isfile(f))
        return self

    @classmethod
    def from_env(cls) -> "SherpaTtsConfig":
        """从环境变量读取（IQ9075_SHERPA_TTS_MODEL_DIR / _MODEL / _TOKENS / _LEXICON）。"""
        return cls(
            model_dir=os.environ.get(
                "IQ9075_SHERPA_TTS_MODEL_DIR", DEFAULT_TTS_MODEL_DIR
            ),
            model_path=os.environ.get("IQ9075_SHERPA_TTS_MODEL", ""),
            tokens_path=os.environ.get("IQ9075_SHERPA_TTS_TOKENS", ""),
            lexicon_path=os.environ.get("IQ9075_SHERPA_TTS_LEXICON", ""),
        )


@dataclass
class MeloTtsQnnConfig:
    """Qualcomm MeloTTS（QNN/HTP）配置：调用 Audio Analytics 服务的 TTS HTTP 接口。

    服务在 NPU 上完成 MeloTTS 合成，返回裸 PCM；本后端包一层 WAV 头后播放。
    """

    base_url: str = os.environ.get("IQ9075_MELOTTS_URL", "http://127.0.0.1:8085")
    model: str = os.environ.get("IQ9075_MELOTTS_MODEL", "melo-tts-zh")
    language: str = os.environ.get("IQ9075_MELOTTS_LANGUAGE", "zh")
    sample_rate: int = int(os.environ.get("IQ9075_MELOTTS_SAMPLE_RATE", "16000"))
    timeout: float = float(os.environ.get("IQ9075_MELOTTS_TIMEOUT", "60"))
    player: str = os.environ.get("IQ9075_TTS_PLAYER", "mplayer")
