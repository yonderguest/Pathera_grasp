"""Lightweight MJPEG streaming for the Panthera visual grasp demo."""

from __future__ import annotations

import threading
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import cv2
import numpy as np


_COLOR_MAP = {
    "red": (0, 0, 255),
    "yellow": (0, 255, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "white": (255, 255, 255),
    "black": (50, 50, 50),
    "unknown": (128, 128, 128),
}


_INDEX_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Panthera Vision Stream</title>
  <style>
    body {
      margin: 0;
      background: #111;
      color: #eee;
      font-family: "Segoe UI", "Microsoft YaHei", Arial, sans-serif;
    }
    header {
      padding: 14px 18px;
      border-bottom: 1px solid #2c2c2c;
    }
    header h1 {
      margin: 0;
      font-size: 22px;
      font-weight: 600;
    }
    header p {
      margin: 5px 0 0;
      color: #aaa;
      font-size: 13px;
    }
    .panels {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      padding: 12px;
    }
    .panel {
      flex: 1 1 480px;
      min-width: 320px;
      background: #181818;
      border: 1px solid #2c2c2c;
      border-radius: 8px;
      overflow: hidden;
    }
    .panel h2 {
      margin: 0;
      padding: 10px 12px;
      font-size: 16px;
      font-weight: 600;
      background: #202020;
      border-bottom: 1px solid #2c2c2c;
    }
    .panel img {
      display: block;
      width: 100%;
      height: auto;
      background: #000;
    }
  </style>
</head>
<body>
  <header>
    <h1>Panthera Vision Stream</h1>
    <p>左：原始 RGB 画面 &nbsp;|&nbsp; 右：YOLO 识别画面（类别、颜色、置信度、距离）</p>
  </header>
  <div class="panels">
    <div class="panel">
      <h2>原始画面</h2>
      <img src="/stream/raw" alt="raw camera">
    </div>
    <div class="panel">
      <h2>YOLO 识别</h2>
      <img src="/stream/yolo" alt="yolo detections">
    </div>
  </div>
</body>
</html>
"""


def draw_detections(
    color_image: np.ndarray,
    detections: list[dict[str, Any]],
    selected_color: str | None = None,
    force_feedback: str | None = None,
) -> np.ndarray:
    """Draw segmentation masks, boxes and text on a copy of the frame."""
    image = np.asarray(color_image, dtype=np.uint8).copy()
    for detection in detections:
        color_name = str(detection.get("color") or "unknown")
        bbox_color = _COLOR_MAP.get(color_name, (128, 128, 128))

        mask = detection.get("mask")
        if mask is not None and np.asarray(mask).shape == image.shape[:2]:
            mask_bool = np.asarray(mask, dtype=bool)
            alpha = 0.42
            image[mask_bool] = (
                image[mask_bool].astype(np.float32) * alpha
                + np.asarray(bbox_color, dtype=np.float32) * (1.0 - alpha)
            ).astype(np.uint8)

        x1, y1, x2, y2 = [int(value) for value in detection["bbox"]]
        is_target = selected_color is None or selected_color == color_name
        thickness = 3 if is_target else 2
        cv2.rectangle(image, (x1, y1), (x2, y2), bbox_color, thickness)

        class_name = str(detection.get("class_name") or "object")
        confidence = float(detection.get("confidence") or 0.0)
        depth_m = detection.get("depth_m")
        depth_text = f"{float(depth_m):.3f} m" if depth_m is not None else "N/A"
        label = (
            f"{class_name} | {color_name} | "
            f"{confidence:.2f} | {depth_text}"
        )

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.55
        text_thickness = 2
        text_size, baseline = cv2.getTextSize(
            label, font, scale, text_thickness
        )
        text_y = max(y1 - 8, text_size[1] + 6)
        cv2.rectangle(
            image,
            (x1, text_y - text_size[1] - 8),
            (x1 + text_size[0] + 8, text_y + baseline + 4),
            (35, 35, 35),
            -1,
        )
        cv2.putText(
            image,
            label,
            (x1 + 4, text_y),
            font,
            scale,
            (255, 255, 255),
            text_thickness,
            cv2.LINE_AA,
        )

    if force_feedback:
        force_label = f"GRIPPER FORCE | {force_feedback}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.75
        thickness = 2
        text_size, baseline = cv2.getTextSize(
            force_label, font, scale, thickness
        )
        top = 12
        left = 12
        cv2.rectangle(
            image,
            (left, top),
            (left + text_size[0] + 18, top + text_size[1] + baseline + 16),
            (20, 20, 20),
            -1,
        )
        cv2.putText(
            image,
            force_label,
            (left + 9, top + text_size[1] + 8),
            font,
            scale,
            (0, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
    return image


class VisionStreamer:
    """Publish camera frames and detections as browser-friendly MJPEG streams."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8080,
        jpeg_quality: int = 85,
    ) -> None:
        self.host = host
        self.port = port
        self.jpeg_quality = int(jpeg_quality)

        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._raw_frame: np.ndarray | None = None
        self._detections: list[dict[str, Any]] = []
        self._selected_color: str | None = None
        self._force_feedback: str | None = None
        self._generation = 0
        self._closed = False

        self._cached_generation = -1
        self._jpeg_cache: dict[str, bytes] = {}
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def is_closed(self) -> bool:
        return self._closed

    @property
    def url(self) -> str:
        host = self.host
        if host in ("0.0.0.0", "::", ""):
            host = self._guess_lan_ip()
        return f"http://{host}:{self.port}/"

    @staticmethod
    def _guess_lan_ip() -> str:
        candidates: list[str] = []
        try:
            candidates.append(socket.gethostbyname(socket.gethostname()))
        except OSError:
            pass
        try:
            probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            probe.connect(("8.8.8.8", 80))
            candidates.append(str(probe.getsockname()[0]))
            probe.close()
        except OSError:
            pass
        for candidate in candidates:
            if candidate and not candidate.startswith("127."):
                return candidate
        return "0.0.0.0"

    def set_selected_color(self, selected_color: str | None) -> None:
        with self._condition:
            self._selected_color = selected_color
            self._condition.notify_all()

    def set_force_feedback(self, force_feedback: str | None) -> None:
        with self._condition:
            self._force_feedback = force_feedback
            self._generation += 1
            self._condition.notify_all()

    def publish(self, color_image: np.ndarray, detections: list[dict[str, Any]]) -> None:
        with self._condition:
            self._raw_frame = np.asarray(color_image, dtype=np.uint8).copy()
            self._detections = list(detections)
            self._generation += 1
            self._condition.notify_all()

    def start(self) -> bool:
        handler_class = type(
            "_VisionStreamHandler",
            (_VisionStreamHandler,),
            {"streamer": self},
        )
        requested_port = int(self.port)
        candidate_ports = [requested_port]
        if requested_port:
            candidate_ports.extend(
                range(requested_port + 1, min(requested_port + 20, 65536))
            )
        candidate_ports.append(0)

        server = None
        last_error: OSError | None = None
        for candidate_port in candidate_ports:
            try:
                server = ThreadingHTTPServer(
                    (self.host, candidate_port), handler_class
                )
                self.port = int(server.server_address[1])
                break
            except OSError as exc:
                last_error = exc
                continue

        if server is None:
            print(
                f"[STREAM] failed to start web stream on port {requested_port}: "
                f"{last_error}",
                flush=True,
            )
            return False

        if requested_port and self.port != requested_port:
            print(
                f"[STREAM] port {requested_port} is busy; using {self.port} instead.",
                flush=True,
            )

        self._server = server
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(
            placeholder,
            "Waiting for camera ...",
            (150, 245),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (220, 220, 220),
            2,
            cv2.LINE_AA,
        )
        self.publish(placeholder, [])

        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="vision-stream-server",
            daemon=True,
        )
        self._thread.start()
        print(f"[STREAM] web preview ready at {self.url}", flush=True)
        return True

    def stop(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        if self._server is not None:
            try:
                self._server.shutdown()
                self._server.server_close()
            except Exception:
                pass

    def _wait_for_next_frame(
        self,
        previous_generation: int,
        timeout: float = 1.0,
    ):
        with self._condition:
            while (
                not self._closed
                and self._raw_frame is not None
                and self._generation == previous_generation
            ):
                self._condition.wait(timeout)
                if self._closed or self._generation != previous_generation:
                    break
            if self._closed and self._raw_frame is None:
                return None
            return (
                self._raw_frame,
                list(self._detections),
                self._generation,
                self._selected_color,
            )

    def _jpeg_for_kind(
        self,
        kind: str,
        raw_frame: np.ndarray,
        detections: list[dict[str, Any]],
        generation: int,
    ) -> bytes:
        with self._lock:
            if generation == self._cached_generation and kind in self._jpeg_cache:
                return self._jpeg_cache[kind]

            raw_jpeg = self._encode_jpeg(raw_frame)
            annotated = draw_detections(
                raw_frame,
                detections,
                self._selected_color,
                self._force_feedback,
            )
            annotated_jpeg = self._encode_jpeg(annotated)
            combined = np.hstack((raw_frame, annotated))
            combined_jpeg = self._encode_jpeg(combined)

            self._jpeg_cache = {
                "raw": raw_jpeg,
                "yolo": annotated_jpeg,
                "combined": combined_jpeg,
            }
            self._cached_generation = generation
            return self._jpeg_cache[kind]

    def _encode_jpeg(self, image: np.ndarray) -> bytes:
        ok, encoded = cv2.imencode(
            ".jpg",
            image,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not ok:
            raise RuntimeError("failed to encode JPEG frame")
        return encoded.tobytes()


class _VisionStreamHandler(BaseHTTPRequestHandler):
    """HTTP handler bound to a VisionStreamer instance."""

    streamer: VisionStreamer

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self._send_index()
        elif path in ("/stream/raw", "/stream/yolo", "/stream/combined"):
            self._send_stream(path.rsplit("/", 1)[-1])
        else:
            self.send_error(404)

    def _send_index(self) -> None:
        content = _INDEX_HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _send_stream(self, kind: str) -> None:
        self.send_response(200)
        self.send_header(
            "Content-Type", "multipart/x-mixed-replace; boundary=frame"
        )
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Pragma", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        previous_generation = -1
        while not self.streamer.is_closed:
            snapshot = self.streamer._wait_for_next_frame(
                previous_generation, timeout=1.0
            )
            if snapshot is None:
                continue
            raw_frame, detections, generation, _ = snapshot
            if raw_frame is None:
                continue
            previous_generation = generation
            try:
                jpeg = self.streamer._jpeg_for_kind(
                    kind, raw_frame, detections, generation
                )
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                break
            except Exception:
                break

    def log_message(self, format: str, *args) -> None:
        return
