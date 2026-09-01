"""Qualcomm QNN HTP NPU client for the block-only YOLOE segmentation model.

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
import select
import struct
import subprocess
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from .grasp_config import GraspConfig


class NpuYoloDetector:
    """Persistent QNN HTP YOLOE detector for four generic block prompts."""

    def __init__(self, config: GraspConfig, confidence: float = 0.30) -> None:
        project_root = Path(config.project_root)
        self.server = str(project_root / config.npu_server_path)
        self.ctx = str(project_root / config.npu_context_path)
        self.in_dims = f"1,3,{config.npu_input_size},{config.npu_input_size}"
        self.out_specs = config.npu_output_specs
        self.input_size = int(config.npu_input_size)
        self.in_bytes = 1 * 3 * self.input_size * self.input_size * 4
        self.confidence = float(confidence)
        self.iou_threshold = float(config.npu_iou_threshold)
        self.pre_nms_top_k = int(config.npu_pre_nms_top_k)
        self.max_detections = int(config.npu_max_detections)
        self.names = tuple(config.npu_class_names)
        self.response_timeout = float(config.npu_response_timeout)
        self._io_lock = threading.Lock()
        self.last_decode_stats: dict[str, int] = {}

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
        self.stderr_lines: deque[str] = deque(maxlen=config.npu_stderr_max_lines)
        self._stderr_thread: threading.Thread | None = None
        self._healthy = True
        try:
            self._spawn()
        except Exception:
            self.close()
            raise

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
        self.stderr_lines.clear()
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr,
            name="npu-stderr",
            daemon=False,
        )
        self._stderr_thread.start()

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
        proc = self.proc
        if proc is None or proc.stderr is None:
            return
        for line in proc.stderr:
            self.stderr_lines.append(line.decode(errors="replace"))

    def _read_exact(self, fd: int, size: int) -> bytes:
        buf = bytearray()
        deadline = time.monotonic() + self.response_timeout
        while len(buf) < size:
            if self.proc is None or self.proc.poll() is not None:
                raise RuntimeError("npu_server exited while waiting for output")
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise TimeoutError(
                    f"npu_server response timed out after {self.response_timeout:.1f}s"
                )
            readable, _, _ = select.select([fd], [], [], remaining)
            if not readable:
                raise TimeoutError(
                    f"npu_server response timed out after {self.response_timeout:.1f}s"
                )
            chunk = os.read(fd, size - len(buf))
            if not chunk:
                raise RuntimeError("npu_server pipe closed")
            buf.extend(chunk)
        return bytes(buf)

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
        with self._io_lock:
            if (
                not self._healthy
                or
                self.proc is None
                or self.proc.poll() is not None
                or self.proc.stdin is None
            ):
                raise RuntimeError("npu_server is not healthy")
            try:
                blob, height, width, ratio, dx, dy = self._letterbox(
                    bgr, self.input_size
                )
                self.proc.stdin.write(
                    struct.pack("<I", self.in_bytes) + blob.tobytes()
                )
                self.proc.stdin.flush()
                n_out = struct.unpack("<I", self._read_exact(self._rfd, 4))[0]
                if n_out != 2:
                    raise RuntimeError(f"unexpected npu_server output count: {n_out}")
                outs: list[bytes] = []
                for _ in range(n_out):
                    size = struct.unpack("<I", self._read_exact(self._rfd, 4))[0]
                    outs.append(self._read_exact(self._rfd, size))
            except Exception as exc:
                # The FIFO protocol has no request id.  Reusing it after any
                # timeout/short read could pair a late response with a newer
                # RGB-D frame, so poison and close this detector fail-closed.
                self._healthy = False
                self.close()
                raise RuntimeError(
                    "NPU transport failed; detector was closed to prevent cross-frame output"
                ) from exc
        return self._decode(outs, height, width, ratio, dx, dy)

    @staticmethod
    def _nms_indices(boxes, scores, iou_threshold, pre_top_k, max_detections):
        """Class-agnostic NMS before expensive prototype-mask decoding."""
        boxes = np.asarray(boxes, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32)
        order = np.argsort(scores)[::-1][: int(pre_top_k)]
        kept: list[int] = []
        while order.size and len(kept) < int(max_detections):
            current = int(order[0])
            kept.append(current)
            if order.size == 1:
                break
            rest = order[1:]
            x1 = np.maximum(boxes[current, 0], boxes[rest, 0])
            y1 = np.maximum(boxes[current, 1], boxes[rest, 1])
            x2 = np.minimum(boxes[current, 2], boxes[rest, 2])
            y2 = np.minimum(boxes[current, 3], boxes[rest, 3])
            intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
            area_current = max(
                0.0,
                float(boxes[current, 2] - boxes[current, 0]),
            ) * max(0.0, float(boxes[current, 3] - boxes[current, 1]))
            area_rest = np.maximum(0.0, boxes[rest, 2] - boxes[rest, 0]) * np.maximum(
                0.0, boxes[rest, 3] - boxes[rest, 1]
            )
            union = np.maximum(area_current + area_rest - intersection, 1e-9)
            order = rest[(intersection / union) <= float(iou_threshold)]
        return kept

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
        raw_labels = pred[:, 5]
        labels = np.full(raw_labels.shape, -1, dtype=int)
        finite_labels = np.isfinite(raw_labels)
        labels[finite_labels] = np.rint(raw_labels[finite_labels]).astype(int)
        coeffs = pred[:, 6:38]

        finite = np.all(np.isfinite(pred), axis=1)
        valid_label = (
            finite_labels
            & (np.abs(raw_labels - labels) <= 1e-3)
            & (labels >= 0)
            & (labels < len(self.names))
        )
        valid_source_box = (
            (boxes[:, 2] - boxes[:, 0] >= 2.0)
            & (boxes[:, 3] - boxes[:, 1] >= 2.0)
        )
        mapped_boxes = np.column_stack(
            (
                np.clip((boxes[:, 0] - dx) / ratio, 0.0, float(width)),
                np.clip((boxes[:, 1] - dy) / ratio, 0.0, float(height)),
                np.clip((boxes[:, 2] - dx) / ratio, 0.0, float(width)),
                np.clip((boxes[:, 3] - dy) / ratio, 0.0, float(height)),
            )
        )
        valid_mapped_box = (
            (mapped_boxes[:, 2] - mapped_boxes[:, 0] >= 2.0)
            & (mapped_boxes[:, 3] - mapped_boxes[:, 1] >= 2.0)
        )
        valid_box = valid_source_box & valid_mapped_box
        eligible = np.where(
            finite
            & valid_label
            & valid_box
            & (confidence >= self.confidence)
        )[0]
        selected = self._nms_indices(
            mapped_boxes[eligible],
            confidence[eligible],
            self.iou_threshold,
            self.pre_nms_top_k,
            self.max_detections,
        )
        self.last_decode_stats = {
            "raw": int(pred.shape[0]),
            "invalid_nonfinite": int(np.count_nonzero(~finite)),
            "invalid_label": int(np.count_nonzero(finite & ~valid_label)),
            "invalid_box": int(np.count_nonzero(finite & valid_label & ~valid_box)),
            "above_threshold": int(eligible.size),
            "after_nms": int(len(selected)),
        }
        detections: list[dict] = []
        for relative_index in selected:
            index = int(eligible[relative_index])
            x1, y1, x2, y2 = mapped_boxes[index]

            mask_logits = coeffs[index] @ proto.reshape(32, -1)
            raw_mask = 1.0 / (1.0 + np.exp(-np.clip(mask_logits, -60.0, 60.0)))
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
        proc = self.proc
        if proc is not None:
            try:
                if proc.stdin is not None:
                    proc.stdin.close()
            except Exception:
                pass
            try:
                proc.terminate()
                proc.wait(timeout=2.0)
            except Exception:
                try:
                    proc.kill()
                    proc.wait(timeout=2.0)
                except Exception:
                    pass
            self.proc = None
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=2.0)
            self._stderr_thread = None
        try:
            os.close(self._rfd)
        except OSError:
            pass
        try:
            os.unlink(self.fifo)
        except OSError:
            pass
