#!/usr/bin/env python3
"""ROS2 MJPEG stream node.

Subscribes to /vision/image_raw and /vision/annotated and serves a small web
page with two MJPEG streams.
"""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


_INDEX_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>Panthera ROS Vision Stream</title>
  <style>
    body { margin: 0; background: #111; color: #eee; font-family: sans-serif; }
    header { padding: 14px 18px; border-bottom: 1px solid #2c2c2c; }
    .panels { display: flex; flex-wrap: wrap; gap: 12px; padding: 12px; }
    .panel { flex: 1 1 480px; background: #181818; border-radius: 8px; overflow: hidden; }
    .panel h2 { margin: 0; padding: 10px 12px; background: #202020; }
    .panel img { width: 100%; height: auto; }
  </style>
</head>
<body>
  <header><h1>Panthera Vision Stream</h1></header>
  <div class="panels">
    <div class="panel"><h2>原始画面</h2><img src="/stream/raw"></div>
    <div class="panel"><h2>YOLO 识别</h2><img src="/stream/yolo"></div>
  </div>
</body>
</html>
"""


class PantheraStreamNode(Node):
    def __init__(self) -> None:
        super().__init__("panthera_stream")
        self.declare_parameter("host", "0.0.0.0")
        self.declare_parameter("port", 8080)

        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._raw_jpeg = b""
        self._annotated_jpeg = b""
        self._raw_generation = 0
        self._annotated_generation = 0
        self._closing = False
        self._server = None
        self._thread = None

        self._sub_raw = self.create_subscription(
            Image, "vision/image_raw", self._cb_raw, 10
        )
        self._sub_annotated = self.create_subscription(
            Image, "vision/annotated", self._cb_annotated, 10
        )

        self._start_server()

    def _image_to_bgr(self, msg: Image) -> np.ndarray | None:
        if msg.encoding not in ("bgr8", "rgb8"):
            self.get_logger().warn(f"unsupported image encoding: {msg.encoding}")
            return None
        image = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, 3
        )
        if msg.encoding == "rgb8":
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image

    def _cb_raw(self, msg: Image) -> None:
        image = self._image_to_bgr(msg)
        if image is None:
            return
        ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            with self._condition:
                self._raw_jpeg = jpeg.tobytes()
                self._raw_generation += 1
                self._condition.notify_all()

    def _cb_annotated(self, msg: Image) -> None:
        image = self._image_to_bgr(msg)
        if image is None:
            return
        ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            with self._condition:
                self._annotated_jpeg = jpeg.tobytes()
                self._annotated_generation += 1
                self._condition.notify_all()

    def _start_server(self) -> None:
        host = str(self.get_parameter("host").value)
        port = int(self.get_parameter("port").value)
        node = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                return

            def do_GET(self):
                if self.path == "/":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(_INDEX_HTML.encode("utf-8"))
                    return
                if self.path == "/stream/raw":
                    self._send_mjpeg("raw")
                    return
                if self.path == "/stream/yolo":
                    self._send_mjpeg("annotated")
                    return
                self.send_error(404)

            def _send_mjpeg(self, kind: str) -> None:
                self.send_response(200)
                self.send_header(
                    "Content-Type", "multipart/x-mixed-replace; boundary=frame"
                )
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                previous_generation = -1
                while not node._closing:
                    with node._condition:
                        while not node._closing:
                            if kind == "raw":
                                jpeg = node._raw_jpeg
                                generation = node._raw_generation
                            else:
                                jpeg = node._annotated_jpeg
                                generation = node._annotated_generation
                            if jpeg and generation != previous_generation:
                                break
                            node._condition.wait(timeout=0.5)
                        if node._closing:
                            return
                    previous_generation = generation
                    try:
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                        self.wfile.flush()
                    except Exception:
                        return

        try:
            self._server = ThreadingHTTPServer((host, port), Handler)
            self._server.daemon_threads = True
        except OSError as exc:
            self.get_logger().error(f"failed to bind {host}:{port}: {exc}")
            return
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="ros-mjpeg-server",
            daemon=False,
        )
        self._thread.start()
        self.get_logger().info(f"MJPEG stream listening on http://{host}:{port}/")

    def destroy_node(self) -> None:
        with self._condition:
            self._closing = True
            self._condition.notify_all()
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PantheraStreamNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
