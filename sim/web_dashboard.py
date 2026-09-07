"""Local browser dashboard for the hardware-isolated MuJoCo simulation.

The HTTP threads never mutate MuJoCo state.  They only enqueue commands; the
simulation-owning Python thread consumes them and remains the sole caller of
``mj_step`` / IK / grasp code.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from collections.abc import Callable
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import cv2
import mujoco
import numpy as np

from sim.mujoco_backend import MujocoRobot


class WebTaskCancelled(RuntimeError):
    """A browser cancel request observed from the MuJoCo control thread."""


class SimWebDashboard:
    """Publish two virtual cameras and serialize browser commands for one robot."""

    def __init__(
        self,
        robot: MujocoRobot,
        *,
        host: str = "127.0.0.1",
        port: int = 8765,
    ) -> None:
        self.robot = robot
        self.host = host
        self.port = int(port)
        self._lock = threading.Lock()
        self._commands: queue.Queue[str] = queue.Queue()
        self._closed = False
        self._cancel_requested = False
        self._paused = False
        self._state: dict[str, Any] = {
            "phase": "ready",
            "message": "仿真已就绪：可开始 PUT1 分拣任务。",
            "scene_seed": robot.scene_seed,
            "completed_colours": [],
            "last_error": None,
        }
        self._frames = {"external": b"", "wrist": b""}
        self._frame_versions = {"external": 0, "wrist": 0}
        # The controller runs at 50 Hz; rendering two 640x480 JPEG streams at
        # that rate wastes CPU without improving a browser operator's view.
        self._control_steps_since_capture = 0
        self._frame_ready = threading.Condition(self._lock)
        self._renderer = mujoco.Renderer(robot.model, height=480, width=640)
        self._external_camera = mujoco.MjvCamera()
        self._external_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self._external_camera.lookat[:] = (0.04, 0.0, 0.08)
        self._external_camera.distance = 0.86
        self._external_camera.azimuth = 128.0
        self._external_camera.elevation = -22.0
        self._server: ThreadingHTTPServer | None = None
        self._server_thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        if self._server is None:
            raise RuntimeError("dashboard has not started")
        bound_host, bound_port = self._server.server_address[:2]
        display_host = "localhost" if bound_host == "127.0.0.1" else str(bound_host)
        return f"http://{display_host}:{bound_port}/"

    def start(self) -> None:
        """Start the HTTP service; the caller keeps ownership of MuJoCo."""
        dashboard = self

        class Handler(_DashboardHandler):
            app = dashboard

        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        self._server.daemon_threads = True
        self._server_thread = threading.Thread(
            target=self._server.serve_forever,
            name="mujoco-sim-web",
            daemon=True,
        )
        self._server_thread.start()
        self.capture(self.robot.data)

    def close(self) -> None:
        self._closed = True
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._server_thread is not None:
            self._server_thread.join(timeout=2.0)
            self._server_thread = None
        self._renderer.close()

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                **self._state,
                "paused": self._paused,
                "cancel_requested": self._cancel_requested,
                "web_url": self.url if self._server is not None else None,
            }

    def request_start(self) -> tuple[bool, str]:
        with self._lock:
            phase = self._state["phase"]
            if phase != "ready":
                if phase == "completed":
                    return False, "请先点击“随机重置场景”，再开始下一轮。"
                if phase == "error":
                    return False, "上轮任务失败，请先随机重置场景。"
                if phase == "cancelled":
                    return False, "上轮任务已取消，请先随机重置场景。"
                return False, "已有命令排队或任务正在运行。"
            self._commands.put("start")
            self._state["phase"] = "queued"
            self._state["message"] = "已排队，等待 MuJoCo 主循环执行。"
            return True, self._state["message"]

    def request_reset(self) -> tuple[bool, str]:
        with self._lock:
            if self._state["phase"] in {"queued", "running"}:
                return False, "已有命令排队或任务正在运行；请等待或先取消。"
            self._commands.put("reset")
            self._state["phase"] = "queued"
            self._state["message"] = "已排队重置随机场景。"
            return True, self._state["message"]

    def request_pause(self) -> tuple[bool, str]:
        with self._lock:
            if self._state["phase"] != "running":
                return False, "只有任务运行时才能暂停或继续。"
            self._paused = not self._paused
            state = "已暂停" if self._paused else "已继续"
            self._state["message"] = state
            return True, state

    def request_cancel(self) -> tuple[bool, str]:
        with self._lock:
            if self._state["phase"] != "running":
                return False, "当前没有可取消的任务。"
            self._cancel_requested = True
            self._paused = False
            self._state["message"] = "正在由 MuJoCo 主循环安全取消。"
            return True, self._state["message"]

    def on_control_step(self, data: mujoco.MjData) -> None:
        """Main-thread observer: publish frames and honour pause/cancel requests."""
        while True:
            with self._lock:
                if self._cancel_requested:
                    raise WebTaskCancelled("browser cancelled the simulation task")
                paused = self._paused
            if not paused:
                break
            time.sleep(0.05)
        self._control_steps_since_capture += 1
        if self._control_steps_since_capture >= 2:
            self._control_steps_since_capture = 0
            self.capture(data)

    def capture(self, data: mujoco.MjData) -> None:
        """Encode externally rendered and virtual-D405 images for browser readers."""
        self._renderer.update_scene(data, camera=self._external_camera)
        external = self._renderer.render()
        wrist = self.robot.render_wrist_rgbd().rgb
        encoded = {
            "external": self._jpeg(external),
            "wrist": self._jpeg(wrist),
        }
        with self._frame_ready:
            for name, image in encoded.items():
                self._frames[name] = image
                self._frame_versions[name] += 1
            self._frame_ready.notify_all()

    @staticmethod
    def _jpeg(rgb: np.ndarray) -> bytes:
        ok, encoded = cv2.imencode(
            ".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 82]
        )
        if not ok:
            raise RuntimeError("could not JPEG-encode MuJoCo frame")
        return encoded.tobytes()

    def stream_frame(self, name: str, version: int) -> tuple[bytes, int]:
        with self._frame_ready:
            self._frame_ready.wait_for(
                lambda: self._closed or self._frame_versions[name] != version,
                timeout=1.0,
            )
            return self._frames[name], self._frame_versions[name]

    def run(self, run_task: Callable[[], dict[str, Any]]) -> None:
        """Serve commands on the MuJoCo-owning thread until Ctrl+C terminates."""
        while not self._closed:
            try:
                command = self._commands.get(timeout=0.10)
            except queue.Empty:
                continue
            if command == "reset":
                self.robot.randomize_scene()
                self.capture(self.robot.data)
                with self._lock:
                    self._paused = False
                    self._cancel_requested = False
                    self._state.update(
                        phase="ready",
                        message="随机场景已重置。",
                        scene_seed=self.robot.scene_seed,
                        completed_colours=[],
                        last_error=None,
                    )
                continue
            if command != "start":
                continue
            with self._lock:
                self._cancel_requested = False
                self._state.update(
                    phase="running",
                    message="正在执行识别、抓取与 PUT1 放置。",
                    completed_colours=[],
                    last_error=None,
                )
            try:
                result = run_task()
            except WebTaskCancelled:
                with self._lock:
                    self._cancel_requested = False
                    self._paused = False
                    self._state.update(
                        phase="cancelled", message="任务已取消，请重置场景后重试。"
                    )
            except Exception as exc:  # State is shown to the operator; no hardware is involved.
                with self._lock:
                    self._paused = False
                    self._cancel_requested = False
                    self._state.update(
                        phase="error", message="任务失败。", last_error=str(exc)
                    )
            else:
                with self._lock:
                    self._paused = False
                    self._cancel_requested = False
                    self._state.update(
                        phase="completed",
                        message="分拣任务完成，所有请求颜色均已放入 PUT1。",
                        completed_colours=result["completed_colours"],
                    )


class _DashboardHandler(BaseHTTPRequestHandler):
    """HTTP adapter; all control requests call thread-safe dashboard methods."""

    app: SimWebDashboard

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/":
            self._send(HTTPStatus.OK, "text/html; charset=utf-8", _PAGE.encode())
        elif self.path == "/api/status":
            self._send_json(self.app.status())
        elif self.path == "/stream/external.mjpg":
            self._stream("external")
        elif self.path == "/stream/wrist.mjpg":
            self._stream("wrist")
        else:
            self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        action = self.path.removeprefix("/api/")
        actions = {
            "start": self.app.request_start,
            "reset": self.app.request_reset,
            "pause": self.app.request_pause,
            "cancel": self.app.request_cancel,
        }
        if action not in actions:
            self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            return
        ok, message = actions[action]()
        self._send_json({"ok": ok, "message": message}, HTTPStatus.OK if ok else HTTPStatus.CONFLICT)

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        self._send(status, "application/json; charset=utf-8", json.dumps(payload).encode())

    def _send(self, status: HTTPStatus, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _stream(self, name: str) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        version = -1
        try:
            while not self.app._closed:
                frame, version = self.app.stream_frame(name, version)
                if not frame:
                    continue
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode())
                self.wfile.write(frame + b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            return


_PAGE = """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>pathera_grasp MuJoCo 仿真控制台</title>
<style>body{margin:0;background:#101820;color:#eaf2f7;font:16px system-ui;padding:24px}main{max-width:1180px;margin:auto}h1{margin-top:0}button{padding:10px 14px;margin:0 8px 12px 0;border:0;border-radius:6px;background:#168aad;color:white;font-weight:600;cursor:pointer}button.warn{background:#d97706}button.stop{background:#c2410c}.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}.card{background:#172a3a;padding:14px;border-radius:10px}.card img{width:100%;background:#000;border-radius:6px}pre{white-space:pre-wrap;background:#0b1219;padding:12px;border-radius:6px}@media(max-width:760px){.grid{grid-template-columns:1fr}}</style>
<main><h1>pathera_grasp · MuJoCo 仿真控制台</h1><p>仅控制本地 MuJoCo；不会初始化真机、相机、CAN 或 NPU。</p>
<button onclick="post('start')">开始分拣任务</button><button onclick="post('reset')">随机重置场景</button><button class="warn" onclick="post('pause')">暂停 / 继续</button><button class="stop" onclick="post('cancel')">取消当前任务</button>
<pre id="status">连接中…</pre><section class="grid"><div class="card"><h2>MuJoCo 外部视角</h2><img src="/stream/external.mjpg"></div><div class="card"><h2>虚拟 RealSense D405 RGB</h2><img src="/stream/wrist.mjpg"></div></section></main>
<script>async function post(a){const r=await fetch('/api/'+a,{method:'POST'});const d=await r.json();if(!d.ok)alert(d.message)}async function tick(){try{const d=await (await fetch('/api/status',{cache:'no-store'})).json();document.querySelector('#status').textContent=JSON.stringify(d,null,2)}catch(e){document.querySelector('#status').textContent='状态连接失败'}setTimeout(tick,500)}tick()</script></html>"""
