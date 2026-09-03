"""Lightweight MJPEG streaming for the Panthera visual grasp demo."""

from __future__ import annotations

import json
import secrets
import threading
import socket
import time
from enum import Enum
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable
from urllib.parse import urlsplit

import cv2
import numpy as np

from .grasp_config import canonical_object_name, normalize_target_request


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

_CONTROL_TOKEN_HEADER = "X-Panthera-Control-Token"
_CONTROL_COOKIE = "panthera_control"


class ControlMode(str, Enum):
    """Mutually exclusive web-control modes; robot calls remain main-thread only."""

    IDLE = "idle"
    GRASPING = "grasping"
    JOGGING = "jogging"
    FOLLOW_ARMING = "follow_arming"
    FOLLOWING = "following"
    RETURNING = "returning"
    STOPPING = "stopping"


# Operator page: aligned depth left, YOLO right, and mutually exclusive controls.
_INDEX_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Panthera 抓取</title>
<style>
:root{color-scheme:dark;--bg:#061019;--surface:#0d1b27e8;--surface2:#102433;--line:#244052;--text:#ecf8f8;--muted:#91a8b3;--cyan:#35d0c5;--cyan2:#159eaa;--violet:#8b7cf6;--amber:#eeb957;--red:#ef5d6c;--shadow:0 24px 70px #0008}
*{box-sizing:border-box}body{margin:0;min-height:100vh;color:var(--text);font-family:Inter,"Segoe UI","Microsoft YaHei",sans-serif;background:radial-gradient(circle at 15% -10%,#153e4a 0,transparent 34%),radial-gradient(circle at 95% 0,#272252 0,transparent 28%),linear-gradient(145deg,#050b12,#091722 62%,#071019)}
button,select,input{font:inherit}.shell{width:min(1500px,100%);margin:auto;padding:22px}header{display:flex;justify-content:space-between;align-items:center;gap:16px;margin-bottom:18px}.brand small{display:block;color:var(--cyan);font-size:12px;letter-spacing:.18em;text-transform:uppercase}.brand h1{margin:3px 0 0;font-size:clamp(24px,3vw,34px);font-weight:720;letter-spacing:-.03em}.status-cluster{display:flex;gap:8px;flex-wrap:wrap;justify-content:flex-end}.pill{padding:7px 11px;border:1px solid var(--line);border-radius:999px;color:#b9cbd2;background:#0b1b26cc;font-size:12px}.pill strong{color:var(--text);font-weight:650}.pill[data-mode="following"]{color:#b9fff4;border-color:#248f89;background:#0b3639}.pill[data-mode="stopping"]{color:#ffd4d8;border-color:#9c3d49;background:#3b1720}
.card{background:linear-gradient(155deg,#102433eb,#091721f4);border:1px solid var(--line);border-radius:18px;box-shadow:var(--shadow);overflow:hidden;backdrop-filter:blur(16px)}.vision-stage{display:grid;grid-template-columns:72px minmax(0,1fr) 72px;grid-template-areas:"left videos right";gap:14px;align-items:center}.video-grid{grid-area:videos;display:grid;grid-template-columns:1fr 1fr;gap:14px}.video-head{display:flex;align-items:center;justify-content:space-between;padding:11px 15px;border-bottom:1px solid var(--line)}.video h2{margin:0;font-size:15px;font-weight:650}.video small{color:var(--muted)}.video img{display:block;width:100%;aspect-ratio:4/3;object-fit:contain;background:#02070b}
.jog-button{height:108px;padding:0 5px;border:1px solid #2b6671;border-radius:16px;background:linear-gradient(160deg,#123747,#0c2230);color:#c9fffa;font-size:38px;box-shadow:0 15px 35px #0006;cursor:pointer;transition:.18s transform,.18s border-color,.18s background}.jog-button:first-of-type{grid-area:left}.jog-button:last-of-type{grid-area:right}.jog-button:hover:not(:disabled){transform:translateY(-2px);border-color:var(--cyan);background:#124453}.jog-button span{display:block;margin-top:-4px;font-size:10px;font-weight:650;color:#9ddbd7}
.control-grid{display:grid;grid-template-columns:minmax(0,2fr) minmax(290px,1fr);gap:14px;margin-top:14px}.panel{padding:17px}.panel-title{display:flex;align-items:center;justify-content:space-between;gap:10px;margin-bottom:13px}.panel h2{margin:0;font-size:16px}.panel p{margin:4px 0 0;color:var(--muted);font-size:12px;line-height:1.6}.target-row{display:grid;grid-template-columns:190px minmax(180px,1fr) auto;gap:10px}.mode-actions{display:grid;grid-template-columns:1fr;gap:10px}.system-actions{display:flex;gap:10px;margin-top:12px;justify-content:flex-end}
select,input,button.action{min-height:46px;border-radius:11px;border:1px solid #315064;padding:0 13px}select,input{min-width:0;color:var(--text);background:#07151f;outline:none}select:focus,input:focus,button:focus-visible{border-color:var(--cyan);box-shadow:0 0 0 3px #35d0c529;outline:none}button.action{color:white;border:0;font-weight:700;cursor:pointer;transition:.18s transform,.18s filter}button.action:hover:not(:disabled){transform:translateY(-1px);filter:brightness(1.08)}button:disabled{opacity:.38;cursor:not-allowed;transform:none!important}#send-target{background:linear-gradient(135deg,var(--cyan2),#2473c4)}#follow-mode{background:linear-gradient(135deg,#6758dd,var(--violet))}#follow-mode[aria-pressed="true"]{background:linear-gradient(135deg,#137f79,var(--cyan))}#stop-program{background:linear-gradient(135deg,#b93449,var(--red))}
.feedback{display:grid;grid-template-columns:auto 1fr;gap:10px 12px;align-items:center;margin-top:14px;padding:13px 14px;border:1px solid #203d4e;border-radius:13px;background:#06131d}.feedback-label{color:var(--muted);font-size:12px}.feedback-value{color:#cfe9ed;font-size:13px;overflow-wrap:anywhere}.auth-note{display:none;margin-top:10px;color:#ffd59a;font-size:12px}.auth-note.visible{display:block}.hand-state{color:var(--muted);font-size:12px}
@media(max-width:980px){.vision-stage{grid-template-columns:1fr 1fr;grid-template-areas:"videos videos" "left right"}.jog-button{height:62px}.jog-button span{display:inline;margin:0 0 0 8px;font-size:11px}.control-grid{grid-template-columns:1fr}.target-row{grid-template-columns:180px 1fr auto}}
@media(max-width:720px){.shell{padding:10px}header{align-items:flex-start;flex-direction:column}.status-cluster{justify-content:flex-start}.video-grid{grid-template-columns:1fr}.target-row{grid-template-columns:1fr}.system-actions{flex-direction:column}.system-actions button{width:100%}.feedback{grid-template-columns:1fr}.panel{padding:14px}}
</style>
</head>
<body><main class="shell">
<header><div class="brand"><small>Firefly Robotics · IQ9075</small><h1>Panthera 抓取</h1></div><div class="status-cluster"><span id="mode-pill" class="pill"><strong>模式</strong> · 连接中</span><span class="pill"><strong>视觉</strong> · NPU / RGB-D</span></div></header>
<section class="vision-stage">
<button id="jog-left" class="jog-button" type="button" aria-label="一号关节向左转动零点五弧度" title="J1 向左 +0.5 rad">←<span>J1 左转 0.5 rad</span></button>
<section class="video-grid">
<div class="card video"><div class="video-head"><h2>实时画面</h2><select id="preview-kind" aria-label="预览类型"><option value="raw">RGB 原图</option><option value="depth">深度图</option></select></div><img id="preview-image" src="/stream/raw" alt="实时相机画面"></div>
<div class="card video"><div class="video-head"><h2>YOLO 识别</h2><small>Object + Colour</small></div><img src="/stream/yolo" alt="YOLO 目标识别画面"></div>
</section>
<button id="jog-right" class="jog-button" type="button" aria-label="一号关节向右转动零点五弧度" title="J1 向右 -0.5 rad">→<span>J1 右转 0.5 rad</span></button>
</section>
<section class="control-grid">
<section class="card panel"><div class="panel-title"><div><h2>抓取目标</h2><p>支持输入瓶子、盒子、绿色积木等自然语言；未通过物理策略校验的目标不会运动。</p></div></div><div class="target-row">
<select id="target-preset" aria-label="目标预设"><option value="红色积木">红色积木</option><option value="黄色积木">黄色积木</option><option value="绿色积木">绿色积木</option><option value="蓝色积木">蓝色积木</option><option value="瓶子">瓶子</option><option value="盒子">盒子</option><option value="任意颜色积木">任意颜色积木</option></select>
<input id="target-text" maxlength="64" autocomplete="off" placeholder="输入：抓绿色积木 / 瓶子 / 盒子"><button id="send-target" class="action" type="button">开始抓取</button></div>
<div id="auth-note" class="auth-note visible">正在建立本机控制会话…</div></section>
<section class="card panel"><div class="panel-title"><div><h2>运行模式</h2><p>随动采用停—看—最多 20 mm 闭环步进；停止随动后返回 HOME。</p></div><span id="hand-state" class="hand-state">手部：未启用</span></div><div class="mode-actions"><button id="follow-mode" class="action" type="button" aria-pressed="false">随动模式</button></div><div class="system-actions"><button id="stop-program" class="action" type="button">结束程序并回启动姿态</button></div></section>
</section>
<section class="feedback" aria-live="polite"><span class="feedback-label">控制状态</span><span id="control-status" class="feedback-value">正在连接程序状态…</span></section>
</main>
<script>
const statusNode=document.getElementById("control-status"),modePill=document.getElementById("mode-pill"),handState=document.getElementById("hand-state"),authNote=document.getElementById("auth-note"),sendButton=document.getElementById("send-target"),followButton=document.getElementById("follow-mode"),stopButton=document.getElementById("stop-program"),targetText=document.getElementById("target-text"),jogButtons=[document.getElementById("jog-left"),document.getElementById("jog-right")],previewKind=document.getElementById("preview-kind"),previewImage=document.getElementById("preview-image");
let lastStatus=null,authValid=false;
const modeNames={idle:"抓取待机",grasping:"抓取中",jogging:"J1 手动",follow_arming:"随动准备",following:"随动中",returning:"返回 HOME",stopping:"安全结束"};
function showAuthError(message){authValid=false;authNote.textContent=message;authNote.classList.add("visible");if(lastStatus)renderStatus(lastStatus)}
async function controlFetch(path,payload){const r=await fetch(path,{method:"POST",credentials:"same-origin",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});let d={};try{d=await r.json()}catch(_){throw new Error(`控制接口返回异常（HTTP ${r.status}）`)}if(!r.ok){if(r.status===403)showAuthError("控制 Cookie 已过期，请刷新当前页面建立新的本机控制会话。");throw new Error(d.message||`请求失败（HTTP ${r.status}）`)}return d}
function renderStatus(d){lastStatus=d;const mode=d.mode||"idle",followActive=!!d.follow_active,canStopFollow=mode==="follow_arming"||mode==="following";modePill.dataset.mode=mode;modePill.textContent=`模式 · ${modeNames[mode]||mode}`;statusNode.textContent=d.message||"等待目标";sendButton.disabled=!authValid||!d.accepting_targets||d.pending||d.stop_requested;jogButtons.forEach(b=>b.disabled=!authValid||!d.accepting_jog||d.pending||d.stop_requested);followButton.disabled=!authValid||d.stop_requested||d.pending||(followActive?!canStopFollow:!d.accepting_follow);followButton.setAttribute("aria-pressed",followActive?"true":"false");followButton.textContent=mode==="returning"?"正在返回 HOME":(followActive?"结束随动":"随动模式");stopButton.disabled=!authValid||!!d.stop_requested;if(followActive){const confidence=Number(d.follow_hand_confidence||0),age=d.follow_last_seen_age_s;handState.textContent=d.follow_hand_visible?`手部：已锁定 ${(confidence*100).toFixed(0)}%`:(age==null?"手部：等待检测":`手部：丢失 ${Number(age).toFixed(1)} s`)}else{handState.textContent="手部：未启用"}}
async function verifyAuth(){try{await controlFetch("/api/auth",{});authValid=true;authNote.classList.remove("visible");if(lastStatus)renderStatus(lastStatus)}catch(e){showAuthError("控制 Cookie 无效，请刷新当前页面后重试。");statusNode.textContent=e.message}}
async function refreshStatus(){try{const r=await fetch("/api/status",{cache:"no-store"}),d=await r.json();renderStatus(d)}catch(_){statusNode.textContent="控制状态连接失败"}}
sendButton.addEventListener("click",async()=>{const command=targetText.value.trim()||document.getElementById("target-preset").value;if(!confirm(`确认开始抓取“${command}”？请确保机械臂路径内无人、无障碍物。`))return;try{sendButton.disabled=true;const d=await controlFetch("/api/target",{command,confirmed:true});statusNode.textContent=d.message}catch(e){statusNode.textContent=e.message}finally{refreshStatus()}});
targetText.addEventListener("keydown",e=>{if(e.key==="Enter"&&!sendButton.disabled)sendButton.click()});
jogButtons.forEach(button=>button.addEventListener("click",async()=>{const direction=button.id==="jog-left"?"left":"right";try{jogButtons.forEach(b=>b.disabled=true);const d=await controlFetch("/api/joint1",{direction});statusNode.textContent=d.message}catch(e){statusNode.textContent=e.message}finally{refreshStatus()}}));
followButton.addEventListener("click",async()=>{const enabled=!(lastStatus&&lastStatus.follow_active);if(enabled&&!confirm("确认启动随动模式？机械臂将从 HOME 开始，以停—看—最多 20 mm 的闭环步进低速跟随唯一手部。"))return;try{followButton.disabled=true;const d=await controlFetch("/api/follow",{enabled,confirmed:true});statusNode.textContent=d.message}catch(e){statusNode.textContent=e.message}finally{refreshStatus()}});
stopButton.addEventListener("click",async()=>{if(!confirm("确认安全结束程序？机械臂将返回程序接管前的启动姿态，再停止电机。此按钮不是急停。"))return;try{stopButton.disabled=true;const d=await controlFetch("/api/stop",{confirmed:true});statusNode.textContent=d.message}catch(e){statusNode.textContent=e.message}finally{refreshStatus()}});
refreshStatus();verifyAuth();setInterval(refreshStatus,500);
previewKind.addEventListener("change",()=>{previewImage.src=`/stream/${previewKind.value}?t=${Date.now()}`});
</script></body></html>
"""


def draw_detections(
    color_image: np.ndarray,
    detections: list[dict[str, Any]],
    selected_color: str | None = None,
    selected_object: str | None = None,
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
        class_name = str(detection.get("class_name") or "object")
        is_target = (
            (selected_color is None or selected_color == color_name)
            and (
                selected_object is None
                or canonical_object_name(class_name) == selected_object
            )
        )
        thickness = 3 if is_target else 2
        cv2.rectangle(image, (x1, y1), (x2, y2), bbox_color, thickness)

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
        control_token: str | None = None,
        preview_fps: float = 15.0,
    ) -> None:
        self.host = host
        self.port = port
        self.jpeg_quality = int(jpeg_quality)
        self.preview_fps = max(1.0, float(preview_fps))
        self._preview_interval_s = 1.0 / self.preview_fps
        self._stop_callback = stop_callback
        if control_token is not None and not str(control_token).strip():
            raise ValueError("control_token cannot be empty")
        self._control_token = (
            str(control_token).strip()
            if control_token is not None
            else secrets.token_urlsafe(24)
        )

        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._cache_lock = threading.Lock()
        self._raw_frame: np.ndarray | None = None
        self._depth_frame: np.ndarray | None = None
        self._depth_scale: float = 0.001
        self._detections: list[dict[str, Any]] = []
        self._selected_color: str | None = None
        self._selected_object: str | None = None
        self._has_selected_color = False
        self._pending_target_command: str | None = None
        self._pending_joint1_jog: str | None = None
        self._pending_follow_command: bool | None = None
        self._joint1_jog_active = False
        self._accepting_targets = False
        self._stop_requested = False
        self._control_mode = ControlMode.IDLE
        self._command_sequence = 0
        self._active_command_id: int | None = None
        self._follow_hand_visible = False
        self._follow_hand_confidence = 0.0
        self._follow_last_seen_at: float | None = None
        self._control_message = "等待目标输入。"
        self._force_feedback: str | None = None
        self._generation = 0
        self._capture_generation = 0
        self._capture_frame: np.ndarray | None = None
        self._capture_depth_frame: np.ndarray | None = None
        self._capture_depth_scale: float = 0.001
        self._last_capture_publish_at = 0.0
        self._closed = False

        self._jpeg_cache: dict[tuple[str, int], bytes] = {}
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def is_closed(self) -> bool:
        with self._condition:
            return self._closed

    @property
    def url(self) -> str:
        host = self.host
        if host in ("0.0.0.0", "::", ""):
            host = self._guess_lan_ip()
        return f"http://{host}:{self.port}/"

    @property
    def control_url(self) -> str:
        """Plain operator URL; the browser receives an HttpOnly session cookie."""
        return self.url

    def control_token_matches(
        self,
        candidate: str | None = None,
        cookie_header: str | None = None,
    ) -> bool:
        """Accept legacy header or the browser's HttpOnly session cookie."""
        values = []
        if candidate:
            values.append(str(candidate))
        if cookie_header:
            cookie = SimpleCookie()
            try:
                cookie.load(cookie_header)
            except Exception:
                cookie = SimpleCookie()
            morsel = cookie.get(_CONTROL_COOKIE)
            if morsel is not None:
                values.append(morsel.value)
        return any(secrets.compare_digest(value, self._control_token) for value in values)

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
        """Backward-compatible colour-only selection (always a block)."""
        self.set_selected_target(selected_color)

    def set_selected_target(self, target) -> None:
        request = normalize_target_request(target)
        with self._condition:
            self._selected_color = request.color
            self._selected_object = canonical_object_name(request.object_name)
            self._has_selected_color = True
            self._control_message = f"当前目标：{request.label}；机械臂流程已接收。"
            self._condition.notify_all()

    def clear_selected_target(self, message: str = "等待目标输入。") -> None:
        """Clear the completed target before accepting the next grasp request."""
        with self._condition:
            self._selected_color = None
            self._selected_object = None
            self._has_selected_color = False
            self._pending_target_command = None
            self._active_command_id = None
            if self._control_mode not in {
                ControlMode.FOLLOW_ARMING,
                ControlMode.FOLLOWING,
                ControlMode.STOPPING,
            }:
                self._control_mode = ControlMode.IDLE
            self._control_message = str(message)
            self._condition.notify_all()

    def submit_target_command(self, command: str) -> int:
        """Queue one browser command without overwriting an accepted request."""
        normalized = str(command).strip()
        if not normalized:
            raise ValueError("目标不能为空")
        if len(normalized) > 64:
            raise ValueError("目标命令不能超过 64 个字符")
        with self._condition:
            if self._closed:
                raise RuntimeError("推流服务已经关闭")
            if not self._accepting_targets or self._control_mode is not ControlMode.IDLE:
                raise RuntimeError("主程序当前不在等待目标，未接受本次提交")
            if self._pending_target_command is not None:
                raise RuntimeError("已有抓取目标等待处理，请勿重复提交")
            self._pending_target_command = normalized
            self._control_mode = ControlMode.GRASPING
            self._command_sequence += 1
            self._active_command_id = self._command_sequence
            self._control_message = f"已提交：{normalized}；等待主程序确认。"
            self._condition.notify_all()
            return self._command_sequence

    def poll_target_command(self) -> str | None:
        """Consume the latest browser target command without blocking."""
        with self._condition:
            command = self._pending_target_command
            self._pending_target_command = None
            if command is not None:
                self._control_message = f"主程序正在解析：{command}"
            return command

    def reject_target_command(self, message: str) -> None:
        """Return to idle after the main thread rejects or cannot parse a target."""
        with self._condition:
            if self._control_mode is ControlMode.GRASPING:
                self._control_mode = ControlMode.IDLE
            self._pending_target_command = None
            self._active_command_id = None
            if not self._stop_requested:
                self._accepting_targets = True
                self._control_message = str(message)
            self._condition.notify_all()

    def submit_joint1_jog(self, direction: str) -> int:
        """Queue one idle-state J1 jog for execution by the main robot thread."""
        normalized = str(direction).strip().lower()
        if normalized not in {"left", "right"}:
            raise ValueError("一号关节方向必须是 left 或 right")
        with self._condition:
            if self._closed:
                raise RuntimeError("推流服务已经关闭")
            if (
                not self._accepting_targets
                or self._stop_requested
                or self._control_mode is not ControlMode.IDLE
            ):
                raise RuntimeError("机械臂当前不在等待目标，不能手动转动")
            self._pending_joint1_jog = normalized
            self._joint1_jog_active = True
            self._control_mode = ControlMode.JOGGING
            self._command_sequence += 1
            self._active_command_id = self._command_sequence
            label = "左" if normalized == "left" else "右"
            self._control_message = f"已请求一号关节向{label}转动 0.5 rad。"
            self._condition.notify_all()
            return self._command_sequence

    def poll_joint1_jog(self) -> str | None:
        with self._condition:
            direction = self._pending_joint1_jog
            self._pending_joint1_jog = None
            return direction

    def finish_joint1_jog(self, message: str) -> None:
        with self._condition:
            self._joint1_jog_active = False
            self._active_command_id = None
            if self._control_mode is ControlMode.JOGGING:
                self._control_mode = ControlMode.IDLE
            if not self._stop_requested:
                self._control_message = str(message)
            self._condition.notify_all()

    def submit_follow_command(self, enabled: bool) -> int:
        """Queue a follow-mode transition; the HTTP thread never moves the robot."""
        if not isinstance(enabled, bool):
            raise ValueError("随动模式 enabled 必须是布尔值")
        with self._condition:
            if self._closed:
                raise RuntimeError("推流服务已经关闭")
            if self._stop_requested:
                raise RuntimeError("程序正在安全结束，不能切换随动模式")
            if enabled:
                if (
                    not self._accepting_targets
                    or self._control_mode is not ControlMode.IDLE
                    or self._pending_follow_command is not None
                ):
                    raise RuntimeError("机械臂当前不是空闲状态，不能启动随动模式")
                self._control_mode = ControlMode.FOLLOW_ARMING
                self._pending_follow_command = True
                self._control_message = "随动请求已接收；等待主程序确认 HOME 与手部模型。"
            else:
                if self._control_mode not in {
                    ControlMode.FOLLOW_ARMING,
                    ControlMode.FOLLOWING,
                }:
                    raise RuntimeError("随动模式当前未运行")
                if self._pending_follow_command is not None:
                    raise RuntimeError("已有随动切换请求等待处理")
                self._pending_follow_command = False
                self._control_message = "正在停止随动；机械臂将返回 HOME。"
            self._command_sequence += 1
            self._active_command_id = self._command_sequence
            self._condition.notify_all()
            return self._command_sequence

    def poll_follow_command(self) -> bool | None:
        """Consume one follow transition in the main robot thread."""
        with self._condition:
            command = self._pending_follow_command
            self._pending_follow_command = None
            if command is False:
                self._control_mode = ControlMode.RETURNING
            return command

    def activate_follow_mode(self, message: str = "随动模式已启动。") -> None:
        """Acknowledge successful HOME/model checks from the main thread."""
        with self._condition:
            if self._stop_requested:
                return
            if self._control_mode is not ControlMode.FOLLOW_ARMING:
                raise RuntimeError("随动模式不在启动确认阶段")
            self._control_mode = ControlMode.FOLLOWING
            self._active_command_id = None
            self._accepting_targets = False
            self._control_message = str(message)
            self._condition.notify_all()

    def finish_follow_mode(self, message: str) -> None:
        """Acknowledge failed/finished follow and HOME return from the main thread."""
        with self._condition:
            self._pending_follow_command = None
            self._active_command_id = None
            self._follow_hand_visible = False
            self._follow_hand_confidence = 0.0
            if not self._stop_requested:
                self._control_mode = ControlMode.IDLE
                self._accepting_targets = True
                self._control_message = str(message)
            self._condition.notify_all()

    def update_follow_feedback(
        self,
        hand_visible: bool,
        confidence: float = 0.0,
        message: str | None = None,
    ) -> None:
        """Publish detector telemetry only; this method never commands motion."""
        with self._condition:
            self._follow_hand_visible = bool(hand_visible)
            self._follow_hand_confidence = float(
                np.clip(float(confidence), 0.0, 1.0)
            )
            if hand_visible:
                self._follow_last_seen_at = time.monotonic()
            if message is not None and not self._stop_requested:
                self._control_message = str(message)
            self._condition.notify_all()

    def set_accepting_targets(self, accepting: bool) -> None:
        """Only permit browser commands while the operator prompt is active."""
        with self._condition:
            requested = bool(accepting)
            self._accepting_targets = (
                requested
                and not self._stop_requested
                and self._control_mode is ControlMode.IDLE
            )
            if not self._accepting_targets:
                self._pending_target_command = None
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
            self._pending_follow_command = None
            self._joint1_jog_active = False
            self._control_mode = ControlMode.STOPPING
            self._active_command_id = None
            self._control_message = "结束请求已接收：机械臂正在返回程序启动姿态。"
            callback = self._stop_callback
            self._condition.notify_all()
        if callback is not None:
            try:
                callback()
            except Exception as exc:
                # The stop latch is authoritative even if an optional notifier
                # fails; never reopen controls after a stop request.
                print(f"[STREAM] stop callback failed: {exc!r}", flush=True)
        return True

    def control_status(self) -> dict[str, Any]:
        with self._condition:
            selected = None
            if self._has_selected_color:
                selected = self._selected_color or "any"
            pending = (
                self._pending_target_command is not None
                or self._pending_joint1_jog is not None
                or self._pending_follow_command is not None
            )
            idle = self._control_mode is ControlMode.IDLE
            now = time.monotonic()
            hand_age = (
                None
                if self._follow_last_seen_at is None
                else max(0.0, now - self._follow_last_seen_at)
            )
            return {
                "selected_color": selected,
                "selected_object": self._selected_object,
                "mode": self._control_mode.value,
                "command_id": self._active_command_id,
                "accepting_targets": self._accepting_targets and idle and not pending,
                "accepting_jog": (
                    self._accepting_targets
                    and idle
                    and not self._joint1_jog_active
                    and not self._stop_requested
                    and not pending
                ),
                "accepting_follow": (
                    self._accepting_targets
                    and idle
                    and not self._stop_requested
                    and not pending
                ),
                "jog_active": self._joint1_jog_active,
                "follow_active": self._control_mode in {
                    ControlMode.FOLLOW_ARMING,
                    ControlMode.FOLLOWING,
                    ControlMode.RETURNING,
                },
                "follow_hand_visible": self._follow_hand_visible,
                "follow_hand_confidence": self._follow_hand_confidence,
                "follow_last_seen_age_s": hand_age,
                "pending": pending,
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

    def publish_capture(
        self,
        color_image: np.ndarray,
        depth_image: np.ndarray | None = None,
        depth_scale: float | None = None,
    ) -> None:
        """Publish a throttled latest-only preview independent of inference.

        The capture thread never waits for HTTP clients and never grows a
        queue. Analysis frames remain on their own generation so YOLO boxes
        cannot be drawn on a newer, unrelated RGB frame.
        """
        now = time.monotonic()
        with self._condition:
            if now - self._last_capture_publish_at < self._preview_interval_s:
                return
            self._last_capture_publish_at = now
            self._capture_frame = np.asarray(color_image, dtype=np.uint8).copy()
            if depth_image is not None:
                self._capture_depth_frame = np.asarray(depth_image).copy()
            if depth_scale is not None:
                self._capture_depth_scale = float(depth_scale)
            self._capture_generation += 1
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
        print(
            f"[STREAM] operator page ready at {self.control_url} "
            "(browser session cookie enabled)",
            flush=True,
        )
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
        kind: str = "yolo",
    ):
        preview_kind = kind in {"raw", "depth"}
        with self._condition:
            while (
                not self._closed
                and (self._capture_frame if preview_kind else self._raw_frame) is not None
                and (self._capture_generation if preview_kind else self._generation)
                == previous_generation
            ):
                self._condition.wait(timeout)
                generation = self._capture_generation if preview_kind else self._generation
                if self._closed or generation != previous_generation:
                    break
            frame = self._capture_frame if preview_kind else self._raw_frame
            if self._closed and frame is None:
                return None
            if preview_kind:
                return (
                    self._capture_frame,
                    self._capture_depth_frame,
                    self._capture_depth_scale,
                    [],
                    self._capture_generation,
                    self._selected_color,
                )
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
            selected_object = self._selected_object
            force_feedback = self._force_feedback
        with self._cache_lock:
            cache_key = (kind, int(generation))
            if cache_key in self._jpeg_cache:
                return self._jpeg_cache[cache_key]

            if kind == "raw":
                image = raw_frame
            elif kind == "depth":
                image = self._render_depth(depth_frame, depth_scale, raw_frame.shape)
            else:
                annotated = draw_detections(
                    raw_frame,
                    detections,
                    selected_color,
                    selected_object,
                    force_feedback,
                )
                image = annotated if kind == "yolo" else np.hstack((raw_frame, annotated))
            encoded = self._encode_jpeg(image)
            self._jpeg_cache[cache_key] = encoded
            # Keep only the newest generation for each stream kind.
            for key in tuple(self._jpeg_cache):
                if key[0] == kind and key != cache_key:
                    del self._jpeg_cache[key]
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
    server_version = "PantheraControl/1.0"
    sys_version = ""

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
        if path not in (
            "/api/auth",
            "/api/target",
            "/api/stop",
            "/api/joint1",
            "/api/follow",
        ):
            self._send_json(404, {"message": "接口不存在"})
            return
        if not self.streamer.control_token_matches(
            self.headers.get(_CONTROL_TOKEN_HEADER),
            self.headers.get("Cookie"),
        ):
            print(
                f"[STREAM-AUTH] rejected {path}; refresh the current operator page "
                "to establish a new session cookie.",
                flush=True,
            )
            self._send_json(403, {"message": "控制令牌无效或缺失"})
            return
        if not self._same_origin():
            self._send_json(403, {"message": "拒绝跨源控制请求"})
            return
        content_type = self.headers.get("Content-Type", "")
        media_type = content_type.split(";", 1)[0].strip().lower()
        if media_type != "application/json":
            self._send_json(415, {"message": "控制接口只接受 application/json"})
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0 or content_length > 4096:
                raise ValueError("请求长度无效")
            payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("JSON 请求体必须是对象")
            if path == "/api/auth":
                self._send_json(
                    200,
                    {"message": "本机控制会话有效"},
                    set_control_cookie=True,
                )
                return
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
            if path == "/api/follow":
                if payload.get("confirmed") is not True:
                    raise ValueError("必须确认随动模式切换")
                enabled = payload.get("enabled")
                if not isinstance(enabled, bool):
                    raise ValueError("随动模式 enabled 必须是布尔值")
                request_id = self.streamer.submit_follow_command(enabled)
                message = (
                    "随动启动请求已提交，正在检查 HOME 和手部模型。"
                    if enabled
                    else "随动停止请求已提交，机械臂将返回 HOME。"
                )
                self._send_json(
                    202,
                    {"message": message, "request_id": request_id},
                )
                return
            if path == "/api/joint1":
                direction = payload.get("direction")
                if not isinstance(direction, str):
                    raise ValueError("一号关节方向必须是字符串")
                request_id = self.streamer.submit_joint1_jog(direction)
                label = "左" if direction.strip().lower() == "left" else "右"
                self._send_json(
                    202,
                    {
                        "message": f"一号关节向{label}转动请求已提交。",
                        "request_id": request_id,
                    },
                )
                return
            if payload.get("confirmed") is not True:
                raise ValueError("必须确认现场安全")
            command = payload.get("command")
            if not isinstance(command, str):
                raise ValueError("目标命令必须是字符串")
            request_id = self.streamer.submit_target_command(command)
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._send_json(400, {"message": str(exc)})
            return
        except RuntimeError as exc:
            self._send_json(409, {"message": str(exc)})
            return
        self._send_json(
            202,
            {"message": f"目标已提交：{command.strip()}", "request_id": request_id},
        )

    def do_OPTIONS(self) -> None:
        # No CORS opt-in: a browser from another origin must never acquire
        # permission to issue physical-control requests.
        self._send_json(403, {"message": "跨源控制未启用"})

    def _same_origin(self) -> bool:
        origin = self.headers.get("Origin")
        if origin is None:
            # Authenticated non-browser clients do not normally send Origin.
            return True
        try:
            parsed = urlsplit(origin)
        except ValueError:
            return False
        host = self.headers.get("Host", "").strip().lower()
        return (
            parsed.scheme in {"http", "https"}
            and bool(host)
            and parsed.netloc.lower() == host
            and parsed.path in {"", "/"}
            and not parsed.query
            and not parsed.fragment
        )

    def _send_security_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' data:; connect-src 'self'; "
            "style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; "
            "frame-ancestors 'none'; base-uri 'none'; form-action 'none'",
        )

    def _send_json(
        self,
        status: int,
        payload: dict[str, Any],
        *,
        set_control_cookie: bool = False,
    ) -> None:
        content = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        if set_control_cookie:
            self.send_header(
                "Set-Cookie",
                f"{_CONTROL_COOKIE}={self.streamer._control_token}; "
                "Path=/; HttpOnly; SameSite=Strict",
            )
        self._send_security_headers()
        self.end_headers()
        self.wfile.write(content)

    def _send_index(self) -> None:
        content = _INDEX_HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        # The URL no longer exposes a token. A fresh page receives a session
        # cookie which JavaScript cannot read and which is sent same-origin.
        self.send_header(
            "Set-Cookie",
            f"{_CONTROL_COOKIE}={self.streamer._control_token}; "
            "Path=/; HttpOnly; SameSite=Strict",
        )
        self._send_security_headers()
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
        self._send_security_headers()
        self.end_headers()

        previous_generation = -1
        while not self.streamer.is_closed:
            snapshot = self.streamer._wait_for_next_frame(
                previous_generation, timeout=1.0, kind=kind
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
