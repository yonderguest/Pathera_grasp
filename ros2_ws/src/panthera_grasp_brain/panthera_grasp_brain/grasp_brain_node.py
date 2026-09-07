#!/usr/bin/env python3
"""ROS2 grasp-brain node.

This node owns the physical Panthera arm and the high-level grasp workflow.
It does not import camera or audio code directly; it consumes:

  /vision/image_raw      sensor_msgs/Image
  /vision/depth_image    sensor_msgs/Image
  /vision/detections     std_msgs/String  (JSON)
  /vision/camera_info    std_msgs/String  (JSON)
  /voice/command         std_msgs/String

and publishes:

  /voice/listen_request  std_msgs/Bool
  /voice/say             std_msgs/String
  /arm/status            std_msgs/String
"""

from __future__ import annotations

import json
import os
import select
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pyrealsense2 as rs
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String


def _project_root() -> Path:
    # <project>/ros2_ws/src/panthera_grasp_brain/panthera_grasp_brain/grasp_brain_node.py
    return Path(__file__).resolve().parents[4]


def _add_paths(root: Path) -> None:
    for path in (
        root,
        root / "Panthera-HT_SDK" / "panthera_python",
        root / "Panthera-HT_SDK" / "panthera_python" / "scripts",
    ):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def _decode_bgr(msg: Image) -> np.ndarray | None:
    if msg.encoding != "bgr8":
        return None
    return np.frombuffer(msg.data, dtype=np.uint8).reshape(
        msg.height, msg.width, 3
    ).copy()


def _decode_depth(msg: Image) -> np.ndarray | None:
    if msg.encoding not in ("16UC1", "mono16"):
        return None
    return np.frombuffer(msg.data, dtype=np.uint16).reshape(
        msg.height, msg.width
    ).copy()


def _read_terminal_command(interrupted: threading.Event) -> str | None:
    print("> ", end="", flush=True)
    while not interrupted.is_set():
        try:
            readable, _, _ = select.select([sys.stdin], [], [], 0.2)
        except (OSError, ValueError):
            return None
        if readable:
            line = sys.stdin.readline()
            return line if line else None
    return None


def _image_timestamp_ns(msg: Image) -> int:
    return int(msg.header.stamp.sec) * 1_000_000_000 + int(
        msg.header.stamp.nanosec
    )


def camera_info_qos() -> QoSProfile:
    return QoSProfile(
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )


def _decode_detections(payload) -> list[dict]:
    detections = []
    for item in payload:
        detections.append(
            {
                "class_name": str(item.get("class_name") or "object"),
                "color": str(item.get("color") or "unknown"),
                "confidence": float(item.get("confidence") or 0.0),
                "color_confidence": float(item.get("color_confidence") or 0.0),
                "depth_m": float(item.get("depth_m") or 0.0),
                "bbox": tuple(float(v) for v in item.get("bbox", [0, 0, 0, 0])),
                "pixel": np.asarray(item.get("pixel", [0.0, 0.0]), dtype=float),
                "short_axis_uv": np.asarray(
                    item.get("short_axis_uv", [1.0, 0.0]), dtype=float
                ),
                "obb_angle_deg": float(item.get("obb_angle_deg") or 0.0),
            }
        )
    return detections


class SynchronizedFrameBuffer:
    """Join colour, aligned depth and detections by capture timestamp."""

    def __init__(self, max_pending: int = 8) -> None:
        self._condition = threading.Condition()
        self._max_pending = max(3, int(max_pending))
        self._colors: dict[int, np.ndarray] = {}
        self._depths: dict[int, np.ndarray] = {}
        self._detections: dict[int, tuple[int, list[dict]]] = {}
        self._latest: dict | None = None

    def _prune(self) -> None:
        timestamps = sorted(
            set(self._colors) | set(self._depths) | set(self._detections)
        )
        for timestamp_ns in timestamps[: -self._max_pending]:
            self._colors.pop(timestamp_ns, None)
            self._depths.pop(timestamp_ns, None)
            self._detections.pop(timestamp_ns, None)

    def _assemble(self, timestamp_ns: int) -> None:
        if (
            timestamp_ns not in self._colors
            or timestamp_ns not in self._depths
            or timestamp_ns not in self._detections
        ):
            return
        frame_seq, detections = self._detections.pop(timestamp_ns)
        snapshot = {
            "color_image": self._colors.pop(timestamp_ns),
            "depth_image": self._depths.pop(timestamp_ns),
            "detections": list(detections),
            "timestamp": time.monotonic(),
            "detections_timestamp": time.monotonic(),
            "capture_timestamp_ns": int(timestamp_ns),
            "frame_seq": int(frame_seq),
        }
        if self._latest is None or frame_seq > self._latest["frame_seq"]:
            self._latest = snapshot
            self._condition.notify_all()

    def put_color(self, timestamp_ns: int, image: np.ndarray) -> None:
        with self._condition:
            self._colors[int(timestamp_ns)] = image
            self._assemble(int(timestamp_ns))
            self._prune()

    def put_depth(self, timestamp_ns: int, image: np.ndarray) -> None:
        with self._condition:
            self._depths[int(timestamp_ns)] = image
            self._assemble(int(timestamp_ns))
            self._prune()

    def put_detections(
        self,
        frame_seq: int,
        timestamp_ns: int,
        detections: list[dict],
    ) -> None:
        with self._condition:
            self._detections[int(timestamp_ns)] = (int(frame_seq), list(detections))
            self._assemble(int(timestamp_ns))
            self._prune()

    def freshness_marker(self) -> int:
        with self._condition:
            return -1 if self._latest is None else int(self._latest["frame_seq"])

    def wait_for_newer(self, marker: int, timeout: float) -> dict | None:
        deadline = time.monotonic() + timeout
        with self._condition:
            while self._latest is None or self._latest["frame_seq"] <= marker:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return None
                self._condition.wait(timeout=remaining)
            snapshot = dict(self._latest)
            snapshot["detections"] = list(snapshot["detections"])
            return snapshot


class RosVisionClient:
    """Subscribes to vision topics and exposes a CameraFeed-like interface."""

    def __init__(self, node: Node) -> None:
        self.node = node
        self._lock = threading.Lock()
        self._frames = SynchronizedFrameBuffer()
        self._legacy_detections: list[dict] = []
        self._camera_info: dict | None = None
        self.depth_scale = 0.001

        node.create_subscription(Image, "vision/image_raw", self._cb_color, 10)
        node.create_subscription(Image, "vision/depth_image", self._cb_depth, 10)
        node.create_subscription(String, "vision/detections", self._cb_detections, 10)
        node.create_subscription(
            String,
            "vision/detections_stamped",
            self._cb_detections_stamped,
            10,
        )
        node.create_subscription(
            String,
            "vision/camera_info",
            self._cb_camera_info,
            camera_info_qos(),
        )

    def _cb_color(self, msg: Image) -> None:
        image = _decode_bgr(msg)
        timestamp_ns = _image_timestamp_ns(msg)
        if image is None or timestamp_ns <= 0:
            return
        self._frames.put_color(timestamp_ns, image)

    def _cb_depth(self, msg: Image) -> None:
        image = _decode_depth(msg)
        timestamp_ns = _image_timestamp_ns(msg)
        if image is None or timestamp_ns <= 0:
            return
        self._frames.put_depth(timestamp_ns, image)

    def _cb_detections(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
            detections = _decode_detections(payload)
        except Exception:
            return
        with self._lock:
            self._legacy_detections = detections

    def _cb_detections_stamped(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
            frame_seq = int(payload["frame_seq"])
            timestamp_ns = int(payload["capture_timestamp_ns"])
            detections = _decode_detections(payload.get("detections", []))
        except Exception as exc:
            self.node.get_logger().warning(
                f"invalid vision/detections_stamped payload: {exc!r}"
            )
            return
        self._frames.put_detections(frame_seq, timestamp_ns, detections)

    def _cb_camera_info(self, msg: String) -> None:
        try:
            info = json.loads(msg.data)
        except Exception:
            return
        with self._lock:
            self._camera_info = info
            self.depth_scale = float(info.get("depth_scale", 0.001))

    def get_intrinsics(self, timeout: float = 10.0):
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                info = self._camera_info
            if info is not None:
                intrinsic = rs.intrinsics()
                intrinsic.width = int(info.get("width", 640))
                intrinsic.height = int(info.get("height", 480))
                intrinsic.fx = float(info.get("fx", 600.0))
                intrinsic.fy = float(info.get("fy", 600.0))
                intrinsic.ppx = float(info.get("ppx", 320.0))
                intrinsic.ppy = float(info.get("ppy", 240.0))
                intrinsic.model = rs.distortion.inverse_brown_conrady
                intrinsic.coeffs = [0.0] * 5
                return intrinsic
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.05)

    def freshness_marker(self) -> int:
        return self._frames.freshness_marker()

    def wait_for_newer(self, after_timestamp: float, timeout: float = 3.0):
        marker = (
            int(after_timestamp)
            if isinstance(after_timestamp, (int, np.integer))
            else self.freshness_marker()
        )
        capture = self._frames.wait_for_newer(marker, timeout)
        if capture is not None:
            capture["intrinsics"] = self.get_intrinsics(timeout=0.0)
        return capture


class RosVoiceClient:
    """Subscribes to voice commands and requests microphone recognition."""

    def __init__(self, node: Node) -> None:
        self.node = node
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._seq = 0
        self._command_seq = -1
        self._command = ""
        self._pub_request = node.create_publisher(Bool, "voice/listen_request", 10)
        node.create_subscription(String, "voice/command", self._cb_command, 10)

    def _cb_command(self, msg: String) -> None:
        with self._cond:
            self._command_seq += 1
            self._command = msg.data
            self._cond.notify_all()

    def request_listen(self) -> None:
        msg = Bool()
        msg.data = True
        self._pub_request.publish(msg)

    def wait_for_command(self, timeout: float = 8.0) -> str | None:
        with self._cond:
            start_seq = self._command_seq
            self.request_listen()
            deadline = time.monotonic() + timeout
            while self._command_seq <= start_seq:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._cond.wait(timeout=remaining)
            return self._command


class RosVoiceAnnouncer:
    def __init__(self, node: Node) -> None:
        self.node = node
        self._pub = node.create_publisher(String, "voice/say", 10)

    def say(self, text: str) -> None:
        if not text:
            return
        msg = String()
        msg.data = text
        self._pub.publish(msg)


class PantheraGraspBrainNode(Node):
    def __init__(self) -> None:
        super().__init__("panthera_grasp_brain")
        default_root = _project_root()
        self.declare_parameter("project_root", str(default_root))
        root = Path(str(self.get_parameter("project_root").value)).expanduser().resolve()
        _add_paths(root)

        from Panthera_lib.grasp_config import GraspConfig, parse_target_command

        self._parse_target_command = parse_target_command
        self._root = root

        self.declare_parameter(
            "robot_config",
            str(
                root
                / "Panthera-HT_SDK"
                / "panthera_python"
                / "robot_param"
                / "Leader.yaml"
            ),
        )
        self.declare_parameter("model_path", str(root / "models" / "yoloe-26s-seg.pt"))
        self.declare_parameter("text_encoder_path", str(root / "mobileclip2_b.ts"))
        self.declare_parameter(
            "calibration_file", str(root / "hand_eye_calibration.json")
        )
        self.declare_parameter("use_voice", True)
        self.declare_parameter("use_npu", False)
        self.declare_parameter("use_graspnet", False)
        self.declare_parameter("voice_prompt_duration", 3.5)

        config = GraspConfig()
        config.project_root = root
        config.robot_config = self.get_parameter("robot_config").value
        config.model_path = self.get_parameter("model_path").value
        config.text_encoder_path = Path(self.get_parameter("text_encoder_path").value)
        config.calibration_file = Path(self.get_parameter("calibration_file").value)
        config.use_voice = bool(self.get_parameter("use_voice").value)
        config.use_npu = bool(self.get_parameter("use_npu").value)
        config.use_graspnet = bool(self.get_parameter("use_graspnet").value)
        config.voice_prompt_duration = float(
            self.get_parameter("voice_prompt_duration").value
        )
        config.voice_asr_model_dir = str(root / "models" / "sensevoice")
        config.voice_tts_model_dir = str(
            root / "models" / "sherpa_tts" / "vits-melo-tts-zh_en"
        )
        config.validate()
        self.config = config

        self.vision = RosVisionClient(self)
        self.voice = RosVoiceClient(self) if config.use_voice else None
        self.announcer = RosVoiceAnnouncer(self) if config.use_voice else None
        self._pub_status = self.create_publisher(String, "arm/status", 10)

        self._shutdown_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="panthera-grasp-worker",
            daemon=False,
        )
        self._thread.start()
        self._status("grasp_brain started")

    def _status(self, text: str) -> None:
        msg = String()
        msg.data = text
        self._pub_status.publish(msg)
        self.get_logger().info(f"[arm] {text}")

    def _select_target(self):
        if self.config.use_voice and self.voice is not None:
            if self.announcer is not None:
                self.announcer.say("请说出要抓取的颜色，例如红色积木")
            text = self.voice.wait_for_command(
                timeout=self.config.voice_prompt_duration + 5.0
            )
            if text:
                color, accepted = self._parse_target_command(text.strip())
                if accepted:
                    selected = color if color is not None else "任意颜色"
                    if self.announcer is not None:
                        self.announcer.say(f"已选择{selected}积木")
                    return color

        if not sys.stdin.isatty():
            return False

        while not self._shutdown_event.is_set():
            try:
                command = _read_terminal_command(self._shutdown_event)
            except (EOFError, KeyboardInterrupt):
                return False
            if command is None:
                return False
            command = command.strip()
            if command.lower() == "q":
                return False
            color, accepted = self._parse_target_command(command)
            if accepted:
                selected = color if color is not None else "任意颜色"
                if self.announcer is not None:
                    self.announcer.say(f"已选择{selected}积木")
                return color
            self._status("无法识别颜色，请输入红/黄/蓝/绿/白/黑积木")
        return False

    def _run(self) -> None:
        from Panthera_lib.Panthera import Panthera
        from Panthera_lib.grasp_planner import GraspPlanner
        from Panthera_lib.graspnet_pipeline import GraspNetCandidateProvider
        from Panthera_lib.vision_pipeline import load_hand_eye

        planner = None
        try:
            tcp_camera = load_hand_eye(self.config)
            intrinsic = self.vision.get_intrinsics(timeout=10.0)
            if intrinsic is None:
                raise RuntimeError("timed out waiting for vision/camera_info")

            graspnet_provider = None
            if self.config.use_graspnet:
                graspnet_provider = GraspNetCandidateProvider(self.config)
                graspnet_provider.load()

            self._status("initializing Panthera robot")
            robot = Panthera(self.config.robot_config)
            planner = GraspPlanner(
                robot,
                self.config,
                self._shutdown_event,
                graspnet_provider=graspnet_provider,
                voice=self.announcer,
            )

            self._status("returning HOME and opening gripper")
            planner.home()
            planner.open_gripper()

            task_complete = planner.run_grasp_loop(
                self.vision,
                intrinsic,
                tcp_camera,
                None,
                self._select_target,
            )
            if task_complete:
                self._status("grasp and placement completed")
        except Exception as exc:  # noqa: BLE001
            self._status(f"grasp_brain exception: {exc!r}")
        finally:
            if planner is not None:
                if planner.safe_shutdown(None):
                    self._status("grasp_brain stopped safely")
                else:
                    self._status("grasp_brain stopped through fault fallback")

    def shutdown_and_join(self, timeout: float | None = None) -> bool:
        """Request worker shutdown and wait for the finite safety sequence."""
        self._shutdown_event.set()
        if timeout is None:
            timeout = (
                self.config.zero_move_timeout
                + self.config.zero_verify_timeout
                + self.config.zero_settle_time
                + 5.0
            )
        self._thread.join(timeout=max(0.0, timeout))
        alive = self._thread.is_alive()
        if alive:
            self.get_logger().error("grasp worker did not finish safety shutdown")
        return not alive


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PantheraGraspBrainNode()
    executor = rclpy.executors.MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        while rclpy.ok() and node._thread.is_alive():
            executor.spin_once(timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown_and_join()
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
