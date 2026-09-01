"""Lightweight MJPEG streaming for the Panthera visual grasp demo."""

from __future__ import annotations

import json
import threading
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable

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

_DEPTH_DISPLAY_MIN_M = 0.07
_DEPTH_DISPLAY_MAX_M = 0.60


_LEGACY_INDEX_HTML = """<!doctype html>
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
    .control {
      margin: 12px;
      padding: 14px;
      background: #181818;
      border: 1px solid #2c2c2c;
      border-radius: 8px;
    }
    .control-row {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
    }
    select, input, button {
      min-height: 38px;
      border-radius: 6px;
      border: 1px solid #555;
      background: #262626;
      color: #eee;
      padding: 0 10px;
      font-size: 15px;
    }
    input { flex: 1 1 220px; }
    button {
      background: #1769aa;
      border-color: #2584cf;
      cursor: pointer;
      font-weight: 600;
    }
    .safety {
      display: block;
      margin: 12px 0 8px;
      color: #ffd166;
    }
    #control-status { color: #9bd3ff; min-height: 22px; }
  </style>
</head>
<body>
  <header>
    <h1>Panthera Vision Stream</h1>
    <p>左：原始 RGB 画面 &nbsp;|&nbsp; 右：YOLO 识别画面（类别、颜色、置信度、距离）</p>
  </header>
  <section class="control">
    <div class="control-row">
      <select id="target-preset" aria-label="目标颜色">
        <option value="红色积木">红色积木</option>
        <option value="黄色积木">黄色积木</option>
        <option value="绿色积木">绿色积木</option>
        <option value="蓝色积木">蓝色积木</option>
        <option value="白色积木">白色积木</option>
        <option value="黑色积木">黑色积木</option>
        <option value="任意颜色">任意颜色</option>
      </select>
      <input id="target-text" maxlength="64" placeholder="也可以输入：不要红色，要绿色积木">
      <button id="send-target" type="button">确认目标</button>
    </div>
    <label class="safety">
      <input id="safety-confirm" type="checkbox">
      我已确认机械臂扫描、预抓取、抓取和放置路径内无人且无障碍物
    </label>
    <div id="control-status">等待目标输入。提交后机械臂会自动开始扫描。</div>
  </section>
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
  <script>
    const statusNode = document.getElementById("control-status");
    async function refreshStatus() {
      try {
        const response = await fetch("/api/status", {cache: "no-store"});
        const data = await response.json();
        statusNode.textContent = data.message || "等待目标输入";
      } catch (_) {
        statusNode.textContent = "控制状态连接失败";
      }
    }
    document.getElementById("send-target").addEventListener("click", async () => {
      const confirmed = document.getElementById("safety-confirm").checked;
      if (!confirmed) {
        statusNode.textContent = "请先勾选现场安全确认";
        return;
      }
      const typed = document.getElementById("target-text").value.trim();
      const command = typed || document.getElementById("target-preset").value;
      if (!window.confirm(`确认提交“${command}”？提交后会自动开始机械臂扫描。`)) return;
      try {
        const response = await fetch("/api/target", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({command, confirmed: true}),
        });
        const data = await response.json();
        statusNode.textContent = data.message || (response.ok ? "目标已提交" : "提交失败");
      } catch (_) {
        statusNode.textContent = "目标提交失败";
      }
    });
    refreshStatus();
    setInterval(refreshStatus, 1000);
  </script>
</body>
</html>
"""


# Operator page: aligned depth on the left and YOLO annotations on the right.
_INDEX_HTML = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Panthera 抓取</title>
<style>
:root{color-scheme:dark;--bg:#0b1220;--card:#111c2e;--line:#26354d;--text:#edf4ff;--muted:#9eb0c8;--blue:#3b82f6;--red:#dc3545}
*{box-sizing:border-box}body{margin:0;background:linear-gradient(145deg,#08101d,#111c2e);color:var(--text);font-family:"Segoe UI","Microsoft YaHei",sans-serif}
.shell{width:min(1480px,100%);margin:auto;padding:18px}header{display:flex;justify-content:space-between;align-items:center;gap:12px;margin-bottom:14px}h1{margin:0;font-size:clamp(21px,3vw,30px)}
.badge{padding:7px 11px;border:1px solid #2d4b70;border-radius:999px;color:#9bd3ff;background:#10233c;font-size:13px}.card{background:#111c2ef5;border:1px solid var(--line);border-radius:14px;box-shadow:0 12px 35px #0005;overflow:hidden}
.vision-stage{display:grid;grid-template-columns:68px minmax(0,1fr) 68px;gap:12px;align-items:center}.video-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}.video h2{margin:0;padding:10px 14px;font-size:16px;border-bottom:1px solid var(--line)}.video img{display:block;width:100%;aspect-ratio:4/3;object-fit:contain;background:#03070d}.controls{margin-top:14px;padding:16px}.controls h2{margin:0 0 12px;font-size:17px}.row{display:grid;grid-template-columns:180px 1fr auto;gap:10px}
select,input,button{min-height:43px;border-radius:9px;border:1px solid #3a4b64;padding:0 12px;font:inherit}select,input{color:var(--text);background:#0b1525}button{color:white;border:0;font-weight:700;cursor:pointer}button:disabled{opacity:.45;cursor:not-allowed}#send-target{background:var(--blue)}#stop-program{background:var(--red)}
.jog-button{height:96px;padding:0;border:1px solid #31547d;background:#102b4d;color:#d9ebff;font-size:40px;box-shadow:0 10px 25px #0004}.jog-button:hover:not(:disabled){background:#174574}.jog-button span{display:block;font-size:11px;font-weight:600;margin-top:-5px}.actions{display:flex;gap:10px;margin-top:12px;align-items:center;justify-content:flex-end;flex-wrap:wrap}#control-status{margin-top:12px;padding:11px 13px;border-radius:9px;background:#0b1525;color:#b9d9ff;min-height:43px}
@media(max-width:900px){.vision-stage{grid-template-columns:48px minmax(0,1fr) 48px;gap:7px}.video-grid{grid-template-columns:1fr}.jog-button{height:76px;font-size:31px}.jog-button span{display:none}.row{grid-template-columns:1fr}.actions button{width:100%}header{align-items:flex-start;flex-direction:column}.shell{padding:10px}}
</style></head><body><main class="shell">
<header><h1>Panthera 抓取</h1><span class="badge">NPU 实时视觉</span></header>
<section class="vision-stage">
<button id="jog-left" class="jog-button" type="button" aria-label="一号关节向左转动零点五弧度" title="J1 向左 +0.5 rad">←<span>J1 左转 0.5 rad</span></button>
<section class="video-grid">
<div class="card video"><h2>深度画面</h2><img src="/stream/depth" alt="对齐深度画面"></div>
<div class="card video"><h2>YOLO 识别</h2><img src="/stream/yolo" alt="积木识别画面"></div>
</section>
<button id="jog-right" class="jog-button" type="button" aria-label="一号关节向右转动零点五弧度" title="J1 向右 -0.5 rad">→<span>J1 右转 0.5 rad</span></button>
</section>
<section class="card controls"><h2>选择抓取目标</h2><div class="row">
<select id="target-preset" aria-label="目标颜色"><option value="红色积木">红色积木</option><option value="黄色积木">黄色积木</option><option value="绿色积木">绿色积木</option><option value="蓝色积木">蓝色积木</option><option value="白色积木">白色积木</option><option value="黑色积木">黑色积木</option><option value="任意颜色">任意颜色</option></select>
<input id="target-text" maxlength="64" placeholder="可选：直接输入目标，例如“抓绿色积木”"><button id="send-target" type="button">开始抓取</button></div>
<div class="actions"><button id="stop-program" type="button">结束程序并回启动姿态</button></div>
<div id="control-status">正在连接程序状态…</div></section>
</main><script>
const statusNode=document.getElementById("control-status"),sendButton=document.getElementById("send-target"),stopButton=document.getElementById("stop-program"),jogButtons=[document.getElementById("jog-left"),document.getElementById("jog-right")];
async function refreshStatus(){try{const r=await fetch("/api/status",{cache:"no-store"}),d=await r.json();statusNode.textContent=d.message||"等待目标";sendButton.disabled=!d.accepting_targets||d.jog_active||d.stop_requested;jogButtons.forEach(b=>b.disabled=!d.accepting_jog||d.stop_requested);stopButton.disabled=!!d.stop_requested}catch(_){statusNode.textContent="控制状态连接失败"}}
sendButton.addEventListener("click",async()=>{const typed=document.getElementById("target-text").value.trim(),command=typed||document.getElementById("target-preset").value;if(!confirm(`确认开始抓取“${command}”？请确保机械臂路径内无人、无障碍物。`))return;try{const r=await fetch("/api/target",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({command,confirmed:true})}),d=await r.json();statusNode.textContent=d.message}catch(_){statusNode.textContent="目标提交失败"}});
jogButtons.forEach(button=>button.addEventListener("click",async()=>{const direction=button.id==="jog-left"?"left":"right";jogButtons.forEach(b=>b.disabled=true);sendButton.disabled=true;try{const r=await fetch("/api/joint1",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({direction})}),d=await r.json();statusNode.textContent=d.message}catch(_){statusNode.textContent="一号关节转动请求失败"}}));
stopButton.addEventListener("click",async()=>{if(!confirm("确认结束程序？机械臂将返回程序接管前的启动姿态，再停止电机。"))return;try{const r=await fetch("/api/stop",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({confirmed:true})}),d=await r.json();statusNode.textContent=d.message;stopButton.disabled=true;sendButton.disabled=true}catch(_){statusNode.textContent="结束请求发送失败"}});
refreshStatus();setInterval(refreshStatus,1000);
</script></body></html>
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
        color_confidence = float(detection.get("color_confidence") or 0.0)
        color_frames = int(detection.get("color_frames") or 1)
        depth_m = detection.get("depth_m")
        depth_text = f"{float(depth_m):.3f} m" if depth_m is not None else "N/A"
        depth_spread = detection.get("depth_spread_m")
        if depth_spread is not None:
            depth_text += f" spread={float(depth_spread) * 1000.0:.1f}mm"
        label = (
            f"{class_name} det={confidence:.2f} | {color_name} "
            f"color={color_confidence:.2f} f={color_frames} | {depth_text}"
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
        stop_callback: Callable[[], None] | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.jpeg_quality = int(jpeg_quality)
        self._stop_callback = stop_callback

        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._cache_lock = threading.Lock()
        self._raw_frame: np.ndarray | None = None
        self._depth_frame: np.ndarray | None = None
        self._depth_scale: float = 0.001
        self._detections: list[dict[str, Any]] = []
        self._selected_color: str | None = None
        self._has_selected_color = False
        self._pending_target_command: str | None = None
        self._pending_joint1_jog: str | None = None
        self._joint1_jog_active = False
        self._accepting_targets = False
        self._stop_requested = False
        self._control_message = "等待目标输入。"
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
            self._has_selected_color = True
            label = selected_color if selected_color is not None else "任意颜色"
            self._control_message = f"当前目标：{label}积木；机械臂流程已接收。"
            self._condition.notify_all()

    def submit_target_command(self, command: str) -> None:
        """Queue one browser command; a newer submission replaces an old one."""
        normalized = str(command).strip()
        if not normalized:
            raise ValueError("目标不能为空")
        if len(normalized) > 64:
            raise ValueError("目标命令不能超过 64 个字符")
        with self._condition:
            if self._closed:
                raise RuntimeError("推流服务已经关闭")
            if not self._accepting_targets:
                raise RuntimeError("主程序当前不在等待目标，未接受本次提交")
            if self._joint1_jog_active:
                raise RuntimeError("一号关节正在转动，请等待动作完成")
            self._pending_target_command = normalized
            self._control_message = f"已提交：{normalized}；等待主程序确认。"
            self._condition.notify_all()

    def poll_target_command(self) -> str | None:
        """Consume the latest browser target command without blocking."""
        with self._condition:
            command = self._pending_target_command
            self._pending_target_command = None
            if command is not None:
                self._control_message = f"主程序正在解析：{command}"
            return command

    def submit_joint1_jog(self, direction: str) -> None:
        """Queue one idle-state J1 jog for execution by the main robot thread."""
        normalized = str(direction).strip().lower()
        if normalized not in {"left", "right"}:
            raise ValueError("一号关节方向必须是 left 或 right")
        with self._condition:
            if self._closed:
                raise RuntimeError("推流服务已经关闭")
            if not self._accepting_targets or self._stop_requested:
                raise RuntimeError("机械臂当前不在等待目标，不能手动转动")
            if self._pending_target_command is not None:
                raise RuntimeError("抓取目标已经提交，不能手动转动")
            if self._joint1_jog_active:
                raise RuntimeError("一号关节转动请求正在处理中")
            self._pending_joint1_jog = normalized
            self._joint1_jog_active = True
            label = "左" if normalized == "left" else "右"
            self._control_message = f"已请求一号关节向{label}转动 0.5 rad。"
            self._condition.notify_all()

    def poll_joint1_jog(self) -> str | None:
        with self._condition:
            direction = self._pending_joint1_jog
            self._pending_joint1_jog = None
            return direction

    def finish_joint1_jog(self, message: str) -> None:
        with self._condition:
            self._joint1_jog_active = False
            if not self._stop_requested:
                self._control_message = str(message)
            self._condition.notify_all()

    def set_accepting_targets(self, accepting: bool) -> None:
        """Only permit browser commands while the operator prompt is active."""
        with self._condition:
            self._accepting_targets = bool(accepting)
            if not self._accepting_targets:
                self._pending_target_command = None
                self._pending_joint1_jog = None
                self._joint1_jog_active = False
            self._condition.notify_all()

    def set_control_message(self, message: str) -> None:
        with self._condition:
            self._control_message = str(message)
            self._condition.notify_all()

    def request_stop(self) -> bool:
        """Latch one web stop request and notify the owning application."""
        with self._condition:
            if self._closed:
                raise RuntimeError("推流服务已经关闭")
            if self._stop_requested:
                return False
            self._stop_requested = True
            self._accepting_targets = False
            self._pending_target_command = None
            self._pending_joint1_jog = None
            self._joint1_jog_active = False
            self._control_message = "结束请求已接收：机械臂正在返回程序启动姿态。"
            callback = self._stop_callback
            self._condition.notify_all()
        if callback is not None:
            callback()
        return True

    def control_status(self) -> dict[str, Any]:
        with self._condition:
            selected = None
            if self._has_selected_color:
                selected = self._selected_color or "any"
            return {
                "selected_color": selected,
                "accepting_targets": self._accepting_targets,
                "accepting_jog": (
                    self._accepting_targets
                    and not self._joint1_jog_active
                    and not self._stop_requested
                ),
                "jog_active": self._joint1_jog_active,
                "pending": self._pending_target_command is not None,
                "stop_requested": self._stop_requested,
                "message": self._control_message,
            }

    def set_force_feedback(self, force_feedback: str | None) -> None:
        with self._condition:
            self._force_feedback = force_feedback
            self._generation += 1
            self._condition.notify_all()

    def publish(
        self,
        color_image: np.ndarray,
        detections: list[dict[str, Any]],
        depth_image: np.ndarray | None = None,
        depth_scale: float | None = None,
    ) -> None:
        with self._condition:
            self._raw_frame = np.asarray(color_image, dtype=np.uint8).copy()
            if depth_image is not None:
                self._depth_frame = np.asarray(depth_image).copy()
            if depth_scale is not None:
                self._depth_scale = float(depth_scale)
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
                server.daemon_threads = True
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
            daemon=False,
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
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

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
                self._depth_frame,
                self._depth_scale,
                list(self._detections),
                self._generation,
                self._selected_color,
            )

    def _jpeg_for_kind(
        self,
        kind: str,
        raw_frame: np.ndarray,
        depth_frame: np.ndarray | None,
        depth_scale: float,
        detections: list[dict[str, Any]],
        generation: int,
    ) -> bytes:
        # JPEG work is isolated from the frame-publication lock. A slow or
        # failing browser encoder cannot block CameraFeed.publish().
        with self._condition:
            selected_color = self._selected_color
            force_feedback = self._force_feedback
        with self._cache_lock:
            if generation != self._cached_generation:
                self._jpeg_cache = {}
                self._cached_generation = generation
            if kind in self._jpeg_cache:
                return self._jpeg_cache[kind]

            if kind == "raw":
                image = raw_frame
            elif kind == "depth":
                image = self._render_depth(depth_frame, depth_scale, raw_frame.shape)
            else:
                annotated = draw_detections(
                    raw_frame,
                    detections,
                    selected_color,
                    force_feedback,
                )
                image = annotated if kind == "yolo" else np.hstack((raw_frame, annotated))
            encoded = self._encode_jpeg(image)
            self._jpeg_cache[kind] = encoded
            return encoded

    def _encode_jpeg(self, image: np.ndarray) -> bytes:
        ok, encoded = cv2.imencode(
            ".jpg",
            image,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not ok:
            raise RuntimeError("failed to encode JPEG frame")
        return encoded.tobytes()

    @staticmethod
    def _render_depth(depth_frame, depth_scale, fallback_shape) -> np.ndarray:
        """Render aligned depth without altering the metric frame used for grasping."""
        if depth_frame is None:
            image = np.zeros(fallback_shape, dtype=np.uint8)
            cv2.putText(
                image,
                "Waiting for depth ...",
                (145, image.shape[0] // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (220, 220, 220),
                2,
                cv2.LINE_AA,
            )
            return image

        meters = np.asarray(depth_frame, dtype=np.float32) * float(depth_scale)
        valid = np.isfinite(meters) & (meters > 0.0)
        span = _DEPTH_DISPLAY_MAX_M - _DEPTH_DISPLAY_MIN_M
        normalized = np.zeros(meters.shape, dtype=np.uint8)
        normalized[valid] = np.clip(
            (meters[valid] - _DEPTH_DISPLAY_MIN_M) / span * 255.0,
            0.0,
            255.0,
        ).astype(np.uint8)
        image = cv2.applyColorMap(255 - normalized, cv2.COLORMAP_TURBO)
        image[~valid] = 0
        median = float(np.median(meters[valid])) if np.any(valid) else float("nan")
        label = (
            f"Depth {_DEPTH_DISPLAY_MIN_M:.2f}-{_DEPTH_DISPLAY_MAX_M:.2f} m"
            f" | median={median:.3f} m"
        )
        cv2.rectangle(image, (8, 8), (410, 42), (20, 20, 20), -1)
        cv2.putText(
            image,
            label,
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return image


class _VisionStreamHandler(BaseHTTPRequestHandler):
    """HTTP handler bound to a VisionStreamer instance."""

    streamer: VisionStreamer

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self._send_index()
        elif path == "/api/status":
            self._send_json(200, self.streamer.control_status())
        elif path in ("/stream/raw", "/stream/depth", "/stream/yolo", "/stream/combined"):
            self._send_stream(path.rsplit("/", 1)[-1])
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        if path not in ("/api/target", "/api/stop", "/api/joint1"):
            self._send_json(404, {"message": "接口不存在"})
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0 or content_length > 4096:
                raise ValueError("请求长度无效")
            payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
            if path == "/api/stop":
                if payload.get("confirmed") is not True:
                    raise ValueError("必须确认结束程序")
                accepted = self.streamer.request_stop()
                message = (
                    "结束请求已接收，机械臂将返回程序启动姿态后停止。"
                    if accepted
                    else "结束请求已经在处理中。"
                )
                self._send_json(202, {"message": message})
                return
            if path == "/api/joint1":
                direction = payload.get("direction")
                if not isinstance(direction, str):
                    raise ValueError("一号关节方向必须是字符串")
                self.streamer.submit_joint1_jog(direction)
                label = "左" if direction.strip().lower() == "left" else "右"
                self._send_json(
                    202,
                    {"message": f"一号关节向{label}转动请求已提交。"},
                )
                return
            if payload.get("confirmed") is not True:
                raise ValueError("必须确认现场安全")
            command = payload.get("command")
            if not isinstance(command, str):
                raise ValueError("目标命令必须是字符串")
            self.streamer.submit_target_command(command)
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._send_json(400, {"message": str(exc)})
            return
        except RuntimeError as exc:
            self._send_json(409, {"message": str(exc)})
            return
        self._send_json(202, {"message": f"目标已提交：{command.strip()}"})

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        content = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

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
            raw_frame, depth_frame, depth_scale, detections, generation, _ = snapshot
            if raw_frame is None:
                continue
            previous_generation = generation
            try:
                jpeg = self.streamer._jpeg_for_kind(
                    kind,
                    raw_frame,
                    depth_frame,
                    depth_scale,
                    detections,
                    generation,
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
