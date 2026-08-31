"""Speaker 门面：非阻塞队列播报（从 tts_manager.py 抽取，去掉业务依赖）。"""
from __future__ import annotations

import os
import queue
import subprocess
import tempfile
import threading

from .backend import (
    EdgeTtsBackend,
    FallbackTtsBackend,
    MeloTtsQnnBackend,
    SherpaTtsBackend,
    TtsBackend,
)
from .config import SherpaTtsConfig, TtsConfig


def _default_backend(config: TtsConfig) -> TtsBackend:
    """按 IQ9075_TTS_BACKEND 选择默认后端。

    - auto（默认）：MeloTTS QNN（NPU）优先，失败回退离线 sherpa VITS；
    - melo_qnn/qnn：仅 MeloTTS QNN；
    - sherpa/vits ：仅离线 sherpa VITS；
    - edge/online ：edge-tts 在线合成。
    """
    # 默认离线 sherpa VITS，跨机器无需依赖 Audio Analytics 服务。
    name = os.environ.get("IQ9075_TTS_BACKEND", "sherpa").strip().lower()
    if name in ("", "auto"):
        return FallbackTtsBackend(
            MeloTtsQnnBackend(),
            SherpaTtsBackend(SherpaTtsConfig()),
        )
    if name in ("melo_qnn", "qnn", "melotts"):
        return MeloTtsQnnBackend()
    if name in ("sherpa", "vits", "offline"):
        return SherpaTtsBackend(SherpaTtsConfig())
    if name in ("edge", "edge-tts", "online"):
        return EdgeTtsBackend(config)
    raise ValueError("未知 TTS 后端: " + name)


class Speaker:
    """非阻塞语音播报：say() 入队，后台线程依次合成并播放。"""

    def __init__(self, backend: TtsBackend | None = None, config: TtsConfig | None = None) -> None:
        self.config = config or TtsConfig()
        self.backend = backend if backend is not None else _default_backend(self.config)
        self._queue: queue.Queue[str | None] = queue.Queue(maxsize=self.config.queue_size)
        self._lock = threading.Lock()
        self._player: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._started = threading.Event()

    def say(self, text: str) -> None:
        """播放语音（非阻塞，自动排队；队列满时丢弃最新消息）。"""
        text = (text or "").strip()
        if not text:
            return
        self._ensure_worker()
        try:
            self._queue.put_nowait(text)
        except queue.Full:
            print(f"[tts] 队列已满，丢弃: {text[:20]}")

    def stop(self) -> None:
        """中断当前播放并清空队列。"""
        self._kill_player()
        self.clear()

    def clear(self) -> None:
        """只清空等待队列，不中断当前播放。"""
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def close(self) -> None:
        """停止并退出后台线程。"""
        self.stop()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass

    def is_busy(self) -> bool:
        """当前是否还有待播报内容（队列非空或正在播放）。"""
        with self._lock:
            playing = self._player is not None
        return playing or not self._queue.empty()

    def _ensure_worker(self) -> None:
        if self._started.is_set() and self._thread is not None and self._thread.is_alive():
            return
        self._started.set()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self) -> None:
        while True:
            try:
                text = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if text is None:
                break
            tmp_path = tempfile.mktemp(suffix=getattr(self.backend, "output_suffix", ".mp3"))
            try:
                if not self.backend.synthesize(text, tmp_path):
                    print(f"[tts] 合成失败: {text[:20]}")
                    continue
                with self._lock:
                    self._player = self.backend.play(tmp_path)
                player = self._player
                if player is not None:
                    try:
                        player.wait(timeout=self.config.play_timeout)
                    except subprocess.TimeoutExpired:
                        self._kill_player()
            except Exception as exc:  # noqa: BLE001 - 单条播报失败不影响后续
                print(f"[tts] 播报失败: {exc}")
            finally:
                with self._lock:
                    if self._player is not None:
                        try:
                            self._player.kill()
                        except Exception:
                            pass
                        self._player = None
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    def _kill_player(self) -> None:
        with self._lock:
            if self._player is not None:
                try:
                    self._player.kill()
                    self._player.wait(timeout=2)
                except Exception:
                    pass
                self._player = None
