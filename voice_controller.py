"""Voice glue for the Panthera grasping demo.

This module wraps the copied ``iq9075_speech`` and ``iq9075_tts`` packages
into a small, non-blocking interface used by ``grasp_demo.py`` and
``grasp_planner.py``:

- ASR: sherpa-onnx + SenseVoice (offline CPU by default)
- TTS: sherpa-onnx + VITS (offline, project-local model by default)

Voice is optional.  If model files or audio hardware are unavailable, the
class reports ``available=False`` and the demo falls back to terminal input.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path


class VoiceInterface:
    """Lazy-ish voice assistant with graceful degradation."""

    def __init__(
        self,
        project_root: Path,
        model_dir: str | None = None,
        tts_model_dir: str | None = None,
        prompt_duration: float = 3.5,
        enabled: bool = True,
    ) -> None:
        self.project_root = Path(project_root)
        self.prompt_duration = float(prompt_duration)
        self.enabled = bool(enabled)
        self.available = False
        self._recognizer = None
        self._speaker = None
        self._audio_lock = threading.Lock()
        self._playback_settle_seconds = 0.25

        if not self.enabled:
            return

        try:
            from iq9075_speech import AsrConfig, SpeechRecognizer
            from iq9075_tts import SherpaTtsBackend, SherpaTtsConfig, Speaker

            asr_model_dir = model_dir or str(
                self.project_root / "models" / "sensevoice"
            )
            tts_dir = tts_model_dir or str(
                self.project_root / "models" / "sherpa_tts" / "vits-melo-tts-zh_en"
            )

            asr_config = AsrConfig(
                model_dir=asr_model_dir,
                backend="sensevoice",
                num_threads=2,
            )
            self._recognizer = SpeechRecognizer(asr_config)
            self._recognizer.load()

            tts_config = SherpaTtsConfig(model_dir=tts_dir).resolve()
            self._speaker = Speaker(backend=SherpaTtsBackend(tts_config))
            self.available = True
        except Exception as exc:  # noqa: BLE001 - voice is optional
            print(f"[VOICE] voice interface disabled: {exc!r}")
            self.close()

    def say(self, text: str) -> None:
        """Queue a spoken announcement.  Never raises."""
        if not self.available or self._speaker is None or not text:
            return
        try:
            # A recognition session owns this lock until recording ends, so a
            # status announcement can never play into the microphone.
            with self._audio_lock:
                self._speaker.say(text)
        except Exception as exc:  # noqa: BLE001
            print(f"[VOICE] announcement failed: {exc!r}")

    def wait_until_idle(self, timeout: float = 30.0) -> bool:
        """Wait for queued TTS and the audio player to drain."""
        if self._speaker is None:
            return True
        deadline = time.monotonic() + max(0.0, timeout)
        while self._speaker.is_busy():
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.05)
        return True

    def listen_for_command(self) -> str | None:
        """Record one voice command and return its text, or None on failure."""
        if not self.available or self._recognizer is None:
            return None
        try:
            with self._audio_lock:
                if not self.wait_until_idle():
                    print("[VOICE] TTS drain timed out; stopping playback before ASR.")
                    if self._speaker is not None:
                        self._speaker.stop()
                # Player exit normally drains ALSA; this short guard interval
                # lets residual acoustic echo decay before opening arecord.
                time.sleep(self._playback_settle_seconds)
                text = self._recognizer.record_and_transcribe(
                    duration=self.prompt_duration
                )
            return (text or "").strip() or None
        except Exception as exc:  # noqa: BLE001
            print(f"[VOICE] listen failed: {exc!r}")
            return None

    def close(self) -> None:
        """Release ASR model and stop the TTS worker."""
        if self._speaker is not None:
            try:
                with self._audio_lock:
                    self.wait_until_idle(timeout=2.0)
                    self._speaker.close()
            except Exception:
                pass
            self._speaker = None
        if self._recognizer is not None:
            try:
                self._recognizer.close()
            except Exception:
                pass
            self._recognizer = None
        self.available = False
