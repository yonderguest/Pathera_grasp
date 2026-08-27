"""Qualcomm QNN HTP NPU client for the brick-only YOLOE segmentation model.

This module wraps the prebuilt ``npu_server`` binary.  The server loads the QNN
context once and keeps running, while Python sends preprocessed 1x3x640x640
float32 NCHW frames over stdin and reads outputs from a FIFO.

The compiled end-to-end model emits:

    output_0: [1, 300, 38]  -> x1,y1,x2,y2, class_prob, class_idx, 32 mask coeffs
    output_1: [1, 32, 160, 160] -> mask prototypes
"""

from __future__ import annotations

import fcntl
import os
import struct
import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from .grasp_config import GraspConfig


BRICK_NAMES = (
    "red brick",
    "orange brick",
    "yellow brick",
    "green brick",
    "blue brick",
    "pink brick",
)


class NpuYoloDetector:
    """Persistent QNN HTP YOLOE detector for 6 brick classes."""

    def __init__(self, config: GraspConfig, confidence: float = 0.30) -> None:
        project_root = Path(config.project_root)
        self.server = str(project_root / config.npu_server_path)
        self.ctx = str(project_root / config.npu_context_path)
        self.in_dims = f"1,3,{config.npu_input_size},{config.npu_input_size}"
        self.out_specs = config.npu_output_specs
        self.input_size = int(config.npu_input_size)
        self.in_bytes = 1 * 3 * self.input_size * self.input_size * 4
        self.confidence = float(confidence)
        self.names = BRICK_NAMES

        self.fifo = f"/tmp/npu_resp_{os.getpid()}.fifo"
        try:
            os.unlink(self.fifo)
        except OSError:
            pass
        os.mkfifo(self.fifo)
        self._rfd = os.open(self.fifo, os.O_RDONLY | os.O_NONBLOCK)
        flags = fcntl.fcntl(self._rfd, fcntl.F_GETFL)
        fcntl.fcntl(self._rfd, fcntl.F_SETFL, flags & ~os.O_NONBLOCK)

        self.proc: subprocess.Popen | None = None
        self.stderr_lines: list[str] = []
        self._spawn()

    def _spawn(self) -> None:
        if not Path(self.server).is_file():
            raise FileNotFoundError(f"npu_server binary not found: {self.server}")
        if not Path(self.ctx).is_file():
            raise FileNotFoundError(f"QNN context binary not found: {self.ctx}")

        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = "/usr/lib/dsp/cdsp:/usr/lib"
        env["ADSP_LIBRARY_PATH"] = "/usr/lib/dsp/cdsp"
        self.proc = subprocess.Popen(
            [self.server, self.ctx, self.fifo, self.in_dims, self.out_specs],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            env=env,
        )
        self.stderr_lines = []
        threading.Thread(target=self._drain_stderr, daemon=True).start()

        deadline = time.time() + 30.0
        while time.time() < deadline:
            if "NPU_SERVER_READY" in "".join(self.stderr_lines):
                return
            if self.proc.poll() is not None:
                break
            time.sleep(0.05)
        raise RuntimeError(
            "npu_server failed: " + "".join(self.stderr_lines)[-2000:]
        )

    def _drain_stderr(self) -> None:
        assert self.proc is not None
        for line in self.proc.stderr:
            self.stderr_lines.append(line.decode(errors="replace"))

    @staticmethod
    def _read_exact(fd: int, size: int) -> bytes:
        buf = b""
        while len(buf) < size:
            chunk = os.read(fd, size - len(buf))
            if not chunk:
                raise RuntimeError("npu_server pipe closed")
            buf += chunk
        return buf

    @staticmethod
    def _letterbox(bgr: np.ndarray, input_size: int):
        height, width = bgr.shape[:2]
        ratio = min(input_size / height, input_size / width)
        new_w = max(1, round(width * ratio))
        new_h = max(1, round(height * ratio))
        resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
        dx = (input_size - new_w) // 2
        dy = (input_size - new_h) // 2
        canvas[dy : dy + new_h, dx : dx + new_w] = resized
        blob = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        return blob, height, width, ratio, dx, dy

    def infer(self, bgr: np.ndarray) -> list[dict]:
        assert self.proc is not None
        blob, height, width, ratio, dx, dy = self._letterbox(
            bgr, self.input_size
        )
        self.proc.stdin.write(struct.pack("<I", self.in_bytes) + blob.tobytes())
        self.proc.stdin.flush()

        n_out = struct.unpack("<I", self._read_exact(self._rfd, 4))[0]
        outs: list[bytes] = []
        for _ in range(n_out):
            size = struct.unpack("<I", self._read_exact(self._rfd, 4))[0]
            outs.append(self._read_exact(self._rfd, size))
        return self._decode(outs, height, width, ratio, dx, dy)

    def _decode(
        self,
        outs: list[bytes],
        height: int,
        width: int,
        ratio: float,
        dx: int,
        dy: int,
    ) -> list[dict]:
        pred = np.frombuffer(outs[0], dtype=np.float32).reshape(300, 38)
        proto = np.frombuffer(outs[1], dtype=np.float32).reshape(32, 160, 160)
        boxes = pred[:, 0:4]
        confidence = pred[:, 4]
        labels = pred[:, 5].astype(int)
        coeffs = pred[:, 6:38]

        detections: list[dict] = []
        for index in np.where(confidence >= self.confidence)[0]:
            x1 = max(0.0, (boxes[index, 0] - dx) / ratio)
            y1 = max(0.0, (boxes[index, 1] - dy) / ratio)
            x2 = min(float(width), (boxes[index, 2] - dx) / ratio)
            y2 = min(float(height), (boxes[index, 3] - dy) / ratio)

            raw_mask = 1.0 / (1.0 + np.exp(-(coeffs[index] @ proto.reshape(32, -1))))
            raw_mask = raw_mask.reshape(160, 160)
            raw_mask = cv2.resize(
                raw_mask, (self.input_size, self.input_size),
                interpolation=cv2.INTER_LINEAR,
            )
            lx1 = max(0, min(self.input_size, int(round(boxes[index, 0]))))
            ly1 = max(0, min(self.input_size, int(round(boxes[index, 1]))))
            lx2 = max(0, min(self.input_size, int(round(boxes[index, 2]))))
            ly2 = max(0, min(self.input_size, int(round(boxes[index, 3]))))
            crop = raw_mask[ly1:ly2, lx1:lx2]
            box_w = max(1, int(round(x2 - x1)))
            box_h = max(1, int(round(y2 - y1)))
            if crop.size == 0:
                crop = np.zeros((box_h, box_w), dtype=np.float32)
            else:
                crop = cv2.resize(
                    crop, (box_w, box_h), interpolation=cv2.INTER_LINEAR
                )

            detections.append(
                {
                    "box": [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))],
                    "cls": int(labels[index]),
                    "conf": float(confidence[index]),
                    "mask": (crop > 0.5).astype(np.uint8),
                }
            )
        return detections

    def close(self) -> None:
        if self.proc is not None:
            try:
                self.proc.stdin.close()
            except Exception:
                pass
            try:
                self.proc.terminate()
            except Exception:
                pass
            self.proc = None
        try:
            os.close(self._rfd)
        except OSError:
            pass
        try:
            os.unlink(self.fifo)
        except OSError:
            pass

