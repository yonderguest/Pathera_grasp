# -*- coding: utf-8 -*-
"""ASR 推理后端抽象与实现。

- ASRBackend         : 抽象接口，便于替换不同推理后端。
- SenseVoiceBackend  : 当前实现，sherpa-onnx + SenseVoice（离线 ONNX，CPU）。
                      与原项目 voice_controller._load_model / _recognize 逻辑一致，
                      已移除对 shared.state / PROJECT_DIR 等全局依赖。
- SenseVoiceQnnBackend: SenseVoice 编码器 ONNX 由 AI Hub 云端编译为 QCS9075 的
                       QNN context binary，本机通过 qnn_context_run 常驻加载后推理，
                       特征提取/CTC 解码在本组件内完成，全程 NPU 推理；
                       QnnBackend 为其历史占位名的兼容别名。
"""
import abc
import json
import os
import re
import struct
import subprocess
import tempfile

import numpy as np

from .errors import ModelNotFoundError, RecognitionError


class ASRBackend(abc.ABC):
    """语音识别后端抽象（输入波形，输出文本）"""

    @abc.abstractmethod
    def load(self) -> None:
        """加载模型，幂等（可重复调用）"""

    @abc.abstractmethod
    def transcribe(self, samples: np.ndarray, sample_rate: int) -> str:
        """识别一段音频波形 -> 文本"""

    @abc.abstractmethod
    def close(self) -> None:
        """释放模型资源"""


class SenseVoiceBackend(ASRBackend):
    """sherpa-onnx SenseVoice 离线识别后端（ARM64 aarch64 可用）"""

    def __init__(self, model_path: str, tokens_path: str,
                 num_threads: int = 2, language: str = "zh", use_itn: bool = True):
        self._model_path = model_path
        self._tokens_path = tokens_path
        self._num_threads = num_threads
        self._language = language
        self._use_itn = use_itn
        self._recognizer = None

    def load(self) -> None:
        if not (os.path.exists(self._model_path) and os.path.exists(self._tokens_path)):
            raise ModelNotFoundError(
                f"模型文件不存在: {self._model_path} / {self._tokens_path}")
        try:
            import sherpa_onnx
        except ImportError as e:
            raise ModelNotFoundError(f"未安装 sherpa-onnx: {e}")
        self._recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=self._model_path, tokens=self._tokens_path,
            num_threads=self._num_threads,
            language=self._language, use_itn=self._use_itn)

    def transcribe(self, samples: np.ndarray, sample_rate: int) -> str:
        if self._recognizer is None:
            raise RecognitionError("模型尚未加载，请先调用 load()")
        samples = np.asarray(samples, dtype=np.float32)
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        stream = self._recognizer.create_stream()
        stream.accept_waveform(int(sample_rate), samples)
        self._recognizer.decode_stream(stream)
        result = stream.result
        return (result.text if result and result.text else "").strip()

    def close(self) -> None:
        self._recognizer = None


class SenseVoiceQnnBackend(ASRBackend):
    """SenseVoice（QNN/HTP，目标 QCS9075）后端。

    - 编码器由 Qualcomm AI Hub 云端编译成 QNN context binary；
    - 本机用 qnn_context_run 常驻加载上下文（不重复加载 500MB 二进制）；
    - fbank/LFR/CMVN 特征、CTC 解码在本组件内完成，与 sherpa-onnx CPU 链路一致。
    """

    def __init__(self, model_dir: str, tokens_path: str = "",
                 context_binary: str = "", runner: str = "",
                 max_frames: int = 512, language: str = "zh",
                 use_itn: bool = True):
        self._model_dir = model_dir
        self._tokens_path = tokens_path or os.path.join(model_dir, "tokens.txt")
        self._context_binary = context_binary or (
            "/home/ubuntu/work/npu_models/sensevoice_encoder_qnn.bin")
        self._runner = runner or "/home/ubuntu/work/npu_models/qnn_context_run"
        self._max_frames = max_frames
        self._language = language
        self._use_itn = use_itn
        self._meta = {}
        self._tokens = []
        self._proc = None
        self._in_specs = ""
        self._out_specs = ""
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        meta_path = os.path.join(self._model_dir, "meta.json")
        if not os.path.isfile(meta_path):
            raise ModelNotFoundError("SenseVoice 元数据缺失: " + meta_path)
        if not os.path.isfile(self._tokens_path):
            raise ModelNotFoundError("SenseVoice tokens 缺失: " + self._tokens_path)
        if not os.path.isfile(self._context_binary):
            raise ModelNotFoundError("QNN context binary 缺失: " + self._context_binary)
        if not os.path.isfile(self._runner):
            raise ModelNotFoundError("QNN 运行器缺失: " + self._runner)

        with open(meta_path, "r", encoding="utf-8") as fh:
            self._meta = json.load(fh)
        with open(self._tokens_path, "r", encoding="utf-8") as fh:
            self._tokens = [line.strip().split()[0] for line in fh if line.strip()]

        in_ids, out_ids = self._dump_context_ids()
        self._in_specs = self._make_specs(in_ids, is_input=True)
        self._out_specs = self._make_specs(out_ids, is_input=False)
        self._start_runner()
        self._loaded = True

    def transcribe(self, samples: np.ndarray, sample_rate: int) -> str:
        if not self._loaded:
            raise RecognitionError("模型尚未加载，请先调用 load()")
        samples = np.asarray(samples, dtype=np.float32)
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        features = self._extract_features(samples, int(sample_rate))
        num_frames = features.shape[0]
        if num_frames > self._max_frames:
            raise RecognitionError(
                f"音频过长: {num_frames} > {self._max_frames} 帧")
        x = np.zeros((1, self._max_frames, features.shape[1]), dtype=np.float32)
        x[0, :num_frames, :] = features

        lang = int(self._meta.get("lang_zh", 3))
        text_norm = int(self._meta.get(
            "with_itn" if self._use_itn else "without_itn", 14))
        logits = self._run_qnn(x, np.int32(lang), np.int32(text_norm))
        return self._ctc_decode(logits)

    def close(self) -> None:
        if self._proc is not None:
            try:
                if self._proc.stdin:
                    self._proc.stdin.close()
                self._proc.terminate()
                self._proc.wait(timeout=3)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
        self._proc = None
        self._loaded = False

    # ---- 内部实现 ----

    def _dump_context_ids(self):
        fd, tmp = tempfile.mkstemp(prefix="sensevoice_ctx_", suffix=".json")
        os.close(fd)
        try:
            subprocess.run(
                ["/usr/bin/qnn-context-binary-utility",
                 "--context_binary", self._context_binary,
                 "--json_file", tmp],
                capture_output=True, timeout=30, check=True)
            with open(tmp, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            graph = data["info"]["graphs"][0]["info"]
            ins = {t["info"]["name"]: t["info"]["id"] for t in graph["graphInputs"]}
            outs = {t["info"]["name"]: t["info"]["id"] for t in graph["graphOutputs"]}
            return ins, outs
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass

    def _make_specs(self, ids: dict, is_input: bool) -> str:
        parts = []
        # 与 qnn_context_run 的 spec 格式一致: name:id:type:dims
        if is_input:
            specs = {
                "x": ("f32", [1, self._max_frames, 560]),
                "language": ("i32", [1]),
                "text_norm": ("i32", [1]),
            }
        else:
            specs = {"output_0": ("f32", [1, self._max_frames + 4,
                                           int(self._meta.get("vocab_size", 25055))])}
        for name, (dtype, dims) in specs.items():
            parts.append("%s:%d:%s:%s" % (name, int(ids[name]), dtype,
                                          ",".join(str(d) for d in dims)))
        return ";".join(parts)

    def _start_runner(self) -> None:
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = "/usr/lib:/usr/lib/dsp/cdsp" + (
            ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
        env["ADSP_LIBRARY_PATH"] = "/usr/lib/dsp/cdsp"
        self._proc = subprocess.Popen(
            [self._runner, self._context_binary, self._in_specs, self._out_specs],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, env=env)

    def _extract_features(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        if sample_rate != 16000:
            target_n = max(1, int(round(samples.size * 16000.0 / sample_rate)))
            xp = np.linspace(0.0, float(samples.size - 1), num=target_n)
            samples = np.interp(xp, np.arange(samples.size), samples)

        import kaldi_native_fbank as knf

        opts = knf.FbankOptions()
        opts.frame_opts.dither = 0.0
        opts.frame_opts.snip_edges = True
        opts.frame_opts.samp_freq = 16000
        opts.frame_opts.frame_shift_ms = 10.0
        opts.frame_opts.frame_length_ms = 25.0
        opts.frame_opts.remove_dc_offset = True
        opts.frame_opts.preemph_coeff = 0.97
        opts.frame_opts.window_type = "hamming"
        opts.frame_opts.round_to_power_of_two = True
        opts.mel_opts.num_bins = 80
        opts.mel_opts.high_freq = 0.0
        opts.mel_opts.low_freq = 20.0
        opts.mel_opts.is_librosa = False
        fbank = knf.OnlineFbank(opts)
        # normalize_samples=0：sherpa 会先乘 32768
        fbank.accept_waveform(16000, (samples * 32768.0).astype(np.float32))
        fbank.input_finished()
        n = fbank.num_frames_ready
        frames = np.stack([fbank.get_frame(i) for i in range(n)]) if n else \
            np.zeros((0, 80), dtype=np.float32)

        window = int(self._meta.get("lfr_window_size", 7))
        shift = int(self._meta.get("lfr_window_shift", 6))
        return self._apply_lfr_cmvn(frames, window, shift)

    def _apply_lfr_cmvn(self, frames: np.ndarray, window: int, shift: int) -> np.ndarray:
        if frames.shape[0] == 0:
            return np.zeros((0, window * frames.shape[1]), dtype=np.float32)
        input_frames = frames.shape[0]
        output_frames = 1 + (input_frames - 1) // shift
        left_context = (window - 1) // 2
        out = np.empty((output_frames, window * frames.shape[1]), dtype=np.float32)
        for i in range(output_frames):
            center = i * shift
            left_padding = max(0, left_context - center)
            first = 0 if center < left_context else center - left_context
            max_offset = input_frames - 1 - first
            row = []
            for j in range(window):
                if j < left_padding:
                    frame = 0
                else:
                    offset = j - left_padding
                    frame = input_frames - 1 if offset > max_offset else first + offset
                row.append(frames[frame])
            out[i] = np.concatenate(row)
        neg_mean = np.asarray(self._meta["neg_mean"], dtype=np.float32)
        inv_stddev = np.asarray(self._meta["inv_stddev"], dtype=np.float32)
        return (out + neg_mean) * inv_stddev

    def _run_qnn(self, x: np.ndarray, lang: np.int32, text_norm: np.int32) -> np.ndarray:
        payload = (x.astype(np.float32).tobytes()
                   + np.array([lang], dtype=np.int32).tobytes()
                   + np.array([text_norm], dtype=np.int32).tobytes())
        assert self._proc is not None and self._proc.stdin is not None
        self._proc.stdin.write(struct.pack("<I", len(payload)) + payload)
        self._proc.stdin.flush()
        header = self._read_exact(4)
        out_len = struct.unpack("<I", header)[0]
        data = self._read_exact(out_len)
        return np.frombuffer(data, dtype=np.float32).reshape(
            1, self._max_frames + 4, int(self._meta.get("vocab_size", 25055)))

    def _read_exact(self, n: int) -> bytes:
        assert self._proc is not None and self._proc.stdout is not None
        buf = b""
        while len(buf) < n:
            chunk = self._proc.stdout.read(n - len(buf))
            if not chunk:
                raise RecognitionError("QNN 运行器提前退出")
            buf += chunk
        return buf

    def _ctc_decode(self, logits: np.ndarray) -> str:
        blank = int(self._meta.get("blank_id", 0))
        ids = logits[0].argmax(axis=1).tolist()
        out = []
        for tok in ids:
            if not out or tok != out[-1]:
                out.append(tok)
        out = [t for t in out if t != blank]
        out = out[4:]  # 跳过 lang/emotion/event 等前 4 个特殊 token
        text = "".join(self._tokens[t] if 0 <= t < len(self._tokens) else ""
                       for t in out)
        return text.replace("\u2581", " ").strip()


# 历史占位名兼容：README 中的 QnnBackend 即 SenseVoiceQnnBackend
QnnBackend = SenseVoiceQnnBackend
