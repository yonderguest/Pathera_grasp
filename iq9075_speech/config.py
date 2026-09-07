# -*- coding: utf-8 -*-
"""识别组件配置（数据类 + 环境变量）。"""
import os
from dataclasses import dataclass
from pathlib import Path


# pathera_grasp 根目录：<项目根>/models/sensevoice
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASR_MODEL_DIR = str(_PROJECT_ROOT / "models" / "sensevoice")


@dataclass
class AsrConfig:
    """语音识别配置。model_dir 与 model_path/tokens_path 二选一。

    NPU 后端（SenseVoice QNN）由 qnn_* 字段指定；backend 可选：
      - "" / "auto"      : QNN context binary 存在时优先 NPU，否则回退 CPU
      - "sensevoice_qnn" : 强制 SenseVoice QNN（HTP/NPU）
      - "sensevoice"     : 强制 sherpa-onnx SenseVoice（CPU）
    """
    model_dir: str = os.environ.get(
        "IQ9075_ASR_MODEL_DIR", DEFAULT_ASR_MODEL_DIR
    )          # 模型目录（内含 model.onnx + tokens.txt），推荐
    model_path: str = ""         # 或直接指定 model.onnx 路径
    tokens_path: str = ""        # 或直接指定 tokens.txt 路径
    num_threads: int = 2         # sherpa-onnx 推理线程数（CPU）
    language: str = "zh"
    use_itn: bool = True         # 逆文本正则化（数字/标点归一）
    sample_rate: int = 16000     # 期望输入采样率（Hz）
    # 默认使用 CPU 版 sherpa-onnx SenseVoice，跨机器可移植；如需板端 NPU，
    # 可设置 IQ9075_ASR_BACKEND=sensevoice_qnn。
    backend: str = os.environ.get("IQ9075_ASR_BACKEND", "sensevoice")

    # ---- SenseVoice QNN（HTP，目标 QCS9075）----
    qnn_context_binary: str = os.environ.get(
        "IQ9075_SENSEVOICE_QNN_CONTEXT",
        "/home/ubuntu/work/npu_models/sensevoice_encoder_qnn.bin",
    )
    qnn_runner: str = os.environ.get(
        "IQ9075_SENSEVOICE_QNN_RUNNER",
        "/home/ubuntu/work/npu_models/qnn_context_run",
    )
    qnn_max_frames: int = int(os.environ.get(
        "IQ9075_SENSEVOICE_QNN_MAX_FRAMES", "512"))

    def resolve(self) -> "AsrConfig":
        """若指定 model_dir，自动拼出 model_path / tokens_path"""
        if self.model_dir and (not self.model_path or not self.tokens_path):
            self.model_path = os.path.join(self.model_dir, "model.onnx")
            self.tokens_path = os.path.join(self.model_dir, "tokens.txt")
        return self

    @classmethod
    def from_env(cls) -> "AsrConfig":
        """从环境变量读取配置：
           IQ9075_ASR_MODEL_DIR  模型目录（必填）
           IQ9075_ASR_THREADS    线程数（默认 2）
           IQ9075_ASR_BACKEND    后端（auto/sensevoice_qnn/sensevoice）
        """
        return cls(
            model_dir=os.environ.get(
                "IQ9075_ASR_MODEL_DIR", DEFAULT_ASR_MODEL_DIR
            ),
            num_threads=int(os.environ.get("IQ9075_ASR_THREADS", "2")),
            backend=os.environ.get("IQ9075_ASR_BACKEND", "sensevoice"),
        )
