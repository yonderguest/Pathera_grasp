"""Camera, YOLOE detection and eye-in-hand grasp geometry."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

from .grasp_config import GraspConfig


def load_hand_eye(config: GraspConfig) -> np.ndarray:
    """Load and validate T_tcp_camera from the hand-eye calibration file."""
    with config.calibration_file.open("r", encoding="utf-8") as file:
        data = json.load(file)
    transform = np.asarray(data["T_tcp_camera"], dtype=float)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("T_tcp_camera must be a 4x4 matrix")
    rotation = transform[:3, :3]
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError("T_tcp_camera last row must be [0, 0, 0, 1]")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-5
    ):
        raise ValueError("T_tcp_camera rotation is not a valid rigid transform")
    return transform


def init_camera(config: GraspConfig):
    """Start RealSense depth/colour streams and return aligned pipeline objects."""
    pipeline = rs.pipeline()
    stream_config = rs.config()
    stream_config.enable_stream(
        rs.stream.depth, config.width, config.height, rs.format.z16, config.fps
    )
    stream_config.enable_stream(
        rs.stream.color, config.width, config.height, rs.format.bgr8, config.fps
    )
    profile = pipeline.start(stream_config)
    sensor = profile.get_device().first_depth_sensor()
    depth_scale = float(sensor.get_depth_scale())
    color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
    intrinsic = color_profile.get_intrinsics()
    align = rs.align(rs.stream.color)
    for _ in range(30):
        pipeline.wait_for_frames()
    print(
        f"[VISION] camera ready: fx={intrinsic.fx:.2f}, fy={intrinsic.fy:.2f}, "
        f"cx={intrinsic.ppx:.2f}, cy={intrinsic.ppy:.2f}, scale={depth_scale:.5f}"
    )
    return pipeline, align, intrinsic, depth_scale


class CameraFeed:
    """Continuously capture raw frames and run YOLO on a separate cadence."""

    def __init__(
        self,
        pipeline,
        align,
        config: GraspConfig,
        depth_scale: float,
        streamer=None,
        model=None,
    ) -> None:
        self.pipeline = pipeline
        self.align = align
        self.config = config
        self.depth_scale = depth_scale
        self.streamer = streamer
        self.model = model
        self.npu_detector = None

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._capture_thread: threading.Thread | None = None
        self._inference_thread: threading.Thread | None = None

        self._latest_capture: dict | None = None
        self._latest_for_scan: dict | None = None
        self._detections: list = []
        self._detections_timestamp = 0.0
        self._last_inference_capture_timestamp = 0.0
        self._last_inference_wall_time = 0.0
        self._last_error_time = 0.0

    def set_model(self, model) -> None:
        self.model = model

    def set_npu_detector(self, detector) -> None:
        self.npu_detector = detector

    def start(self) -> None:
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            name="camera-capture",
            daemon=True,
        )
        self._inference_thread = threading.Thread(
            target=self._inference_loop,
            name="camera-inference",
            daemon=True,
        )
        self._capture_thread.start()
        self._inference_thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop_event.set()
        for thread in (self._capture_thread, self._inference_thread):
            if thread is not None:
                thread.join(timeout=timeout)

    def latest(self):
        with self._lock:
            return self._latest_for_scan

    def wait_for_newer(self, after_timestamp: float, timeout: float = 2.0):
        deadline = time.monotonic() + timeout
        latest = self.latest()
        while (
            latest is None
            or latest["detections_timestamp"] <= after_timestamp
        ):
            if time.monotonic() >= deadline:
                return latest
            time.sleep(0.01)
            latest = self.latest()
        return latest

    def _capture_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                frames = self.pipeline.wait_for_frames()
                aligned = self.align.process(frames)
                color_frame = aligned.get_color_frame()
                depth_frame = aligned.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue

                color_image = np.asanyarray(color_frame.get_data()).copy()
                depth_image = np.asanyarray(depth_frame.get_data()).copy()
                timestamp = time.monotonic()

                with self._lock:
                    self._latest_capture = {
                        "color_image": color_image,
                        "depth_image": depth_image,
                        "timestamp": timestamp,
                    }
                    self._latest_for_scan = {
                        "color_image": color_image,
                        "detections": list(self._detections),
                        "timestamp": timestamp,
                        "detections_timestamp": self._detections_timestamp,
                    }
                    detections = list(self._detections)

                if self.streamer is not None:
                    self.streamer.publish(color_image, detections)
            except Exception as exc:
                self._report_error(exc)
                time.sleep(0.05)

    def _inference_loop(self) -> None:
        while not self._stop_event.is_set():
            model = self.model
            if model is None and self.npu_detector is None:
                time.sleep(0.02)
                continue

            with self._lock:
                capture = self._latest_capture

            if capture is None or (
                self.npu_detector is None
                and capture["timestamp"] == self._last_inference_capture_timestamp
            ):
                time.sleep(0.01)
                continue

            now = time.monotonic()
            if now - self._last_inference_wall_time < self.config.camera_detection_interval:
                time.sleep(0.01)
                continue

            color_image = capture["color_image"]
            depth_image = capture["depth_image"]
            timestamp = capture["timestamp"]
            try:
                if self.npu_detector is not None:
                    detections = detect_targets_npu(
                        self.npu_detector,
                        color_image,
                        depth_image,
                        self.depth_scale,
                        self.config,
                    )
                else:
                    detections = detect_targets(
                        model,
                        color_image,
                        depth_image,
                        self.depth_scale,
                        self.config,
                    )
            except Exception as exc:
                self._report_error(exc)
                self._last_inference_capture_timestamp = timestamp
                continue

            with self._lock:
                self._detections = detections
                self._detections_timestamp = timestamp
                self._last_inference_capture_timestamp = timestamp
                self._last_inference_wall_time = time.monotonic()

            if self.streamer is not None:
                self.streamer.publish(color_image, detections)

    def _report_error(self, exc: Exception) -> None:
        now = time.monotonic()
        if now - self._last_error_time > 5.0:
            self._last_error_time = now
            print(f"[CAMERA-FEED] frame failed: {exc!r}", flush=True)


def get_base_camera_transform(robot, tcp_camera):
    """Refresh Base<-Camera transform from current forward kinematics."""
    fk = robot.forward_kinematics()
    if fk is None:
        raise RuntimeError("forward_kinematics failed")
    base_tcp = np.eye(4)
    base_tcp[:3, :3] = np.asarray(fk["rotation"], dtype=float)
    base_tcp[:3, 3] = np.asarray(fk["position"], dtype=float)
    return base_tcp @ tcp_camera


def load_target_model(config: GraspConfig):
    """Load the YOLOE segmentation model with block-specific text prompts."""
    from ultralytics.models.yolo.model import YOLOE

    model = YOLOE(config.model_path)
    encoder_path = Path(config.text_encoder_path)
    if not encoder_path.is_file():
        raise FileNotFoundError(f"YOLOE text encoder not found: {encoder_path}")

    # Ultralytics resolves the MobileCLIP TorchScript asset relative to the
    # current working directory. Temporarily chdir to its folder so the demo
    # still works when launched from anywhere, then restore the original cwd.
    previous_cwd = Path.cwd()
    try:
        if previous_cwd.resolve() != encoder_path.parent.resolve():
            os.chdir(encoder_path.parent)
        model.set_classes(list(config.target_prompts))
    finally:
        if Path.cwd().resolve() != previous_cwd.resolve():
            os.chdir(previous_cwd)
    return model


def classify_color(bgr_image, mask, config: GraspConfig):
    """Classify mask-centre pixels by HSV vote, never by a circular H mean."""
    mask_u8 = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)
    core = cv2.erode(mask_u8, np.ones((7, 7), np.uint8), iterations=1)
    if np.count_nonzero(core) < 150:
        core = mask_u8
    pixels = bgr_image[core.astype(bool)]
    if pixels.size == 0:
        return "unknown", 0.0

    hsv = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    h = hsv[:, 0].astype(np.int16)
    s = hsv[:, 1].astype(np.int16)
    v = hsv[:, 2].astype(np.int16)

    dark_ratio = float(np.mean(v < 60))
    if dark_ratio >= config.color_dark_ratio:
        return "black", dark_ratio
    white_ratio = float(np.mean((s < 45) & (v > 180)))
    if white_ratio >= config.color_white_ratio:
        return "white", white_ratio

    valid = (s >= config.color_min_saturation) & (v >= config.color_min_value)
    if not np.any(valid):
        return "unknown", 0.0
    hue = h[valid]
    votes = {
        "red": int(np.count_nonzero((hue < 12) | (hue >= 168))),
        "yellow": int(np.count_nonzero((hue >= 12) & (hue < 45))),
        "green": int(np.count_nonzero((hue >= 45) & (hue < 85))),
        "blue": int(np.count_nonzero((hue >= 85) & (hue < 140))),
    }
    color, count = max(votes.items(), key=lambda item: item[1])
    ratio = count / float(len(hue))
    if ratio < config.color_dominant_ratio:
        return "unknown", ratio
    return color, ratio


def median_mask_depth(depth_frame, mask, depth_scale, config: GraspConfig):
    if hasattr(depth_frame, "get_data"):
        depth = np.asanyarray(depth_frame.get_data()).astype(np.float32)
    else:
        depth = np.asanyarray(depth_frame).astype(np.float32)
    values = depth[np.asarray(mask, dtype=bool)]
    values = values[
        (values >= config.min_depth_m / depth_scale)
        & (values <= config.max_depth_m / depth_scale)
    ]
    if values.size == 0:
        return None
    return float(np.median(values)) * depth_scale


def _detection_from_result(name, conf, bbox, mask, color_image, depth_frame, depth_scale, config):
    x1, y1, x2, y2 = bbox
    full_mask = (np.asarray(mask, dtype=np.float32) > 0.5).astype(np.uint8)
    if full_mask.shape != (config.height, config.width):
        full_mask = cv2.resize(
            full_mask, (config.width, config.height), interpolation=cv2.INTER_NEAREST
        )
    color, color_confidence = classify_color(color_image, full_mask, config)

    inner = cv2.erode(full_mask, np.ones((5, 5), np.uint8), iterations=1)
    if np.count_nonzero(inner) < 50:
        inner = full_mask
    if not np.any(inner):
        return None

    depth_m = median_mask_depth(depth_frame, inner, depth_scale, config)
    if depth_m is None:
        return None

    contours, _ = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 50.0:
        return None
    rect_points = cv2.boxPoints(cv2.minAreaRect(contour.astype(np.float32)))
    edges = np.roll(rect_points, -1, axis=0) - rect_points
    edge_lengths = np.linalg.norm(edges, axis=1)
    short_edge_uv = edges[int(np.argmin(edge_lengths))]
    short_edge_norm = float(np.linalg.norm(short_edge_uv))
    if short_edge_norm < 1e-6:
        return None
    short_axis_uv = short_edge_uv / short_edge_norm

    ys, xs = np.where(inner)
    return {
        "confidence": float(conf),
        "class_name": name,
        "color": color,
        "color_confidence": color_confidence,
        "pixel": np.array([np.median(xs), np.median(ys)], dtype=float),
        "depth_m": depth_m,
        "bbox": (x1, y1, x2, y2),
        "mask": full_mask,
        "short_axis_uv": short_axis_uv.astype(float),
        "obb_angle_deg": float(
            np.degrees(np.arctan2(short_axis_uv[1], short_axis_uv[0]))
        ),
    }


def detect_targets(model, color_image, depth_frame, depth_scale, config: GraspConfig):
    result = model.predict(
        color_image, imgsz=config.width, conf=config.confidence_threshold, verbose=False
    )[0]
    if result.boxes is None or result.masks is None or len(result.boxes) == 0:
        return []

    confs = result.boxes.conf.cpu().numpy()
    order = np.argsort(confs)[::-1]
    detections = []
    for i in order:
        name = result.names[int(result.boxes.cls[i])]
        conf = float(confs[i])
        x1, y1, x2, y2 = [int(v) for v in result.boxes.xyxy[i]]
        mask = result.masks.data[i].cpu().numpy()
        detection = _detection_from_result(
            name,
            conf,
            (x1, y1, x2, y2),
            mask,
            color_image,
            depth_frame,
            depth_scale,
            config,
        )
        if detection is not None:
            detections.append(detection)
    return detections


def detect_targets_npu(
    npu_detector,
    color_image,
    depth_frame,
    depth_scale,
    config: GraspConfig,
):
    """Run the QNN HTP detector and convert its outputs to the same detection
    dictionaries used by the CPU YOLOE path.
    """
    raw_detections = npu_detector.infer(color_image)
    detections = []
    for raw in raw_detections:
        x1, y1, x2, y2 = [int(value) for value in raw["box"]]
        crop = np.asarray(raw["mask"], dtype=np.float32)
        if crop.shape[:2] != (y2 - y1, x2 - x1):
            crop = cv2.resize(
                crop, (max(1, x2 - x1), max(1, y2 - y1)),
                interpolation=cv2.INTER_NEAREST,
            )
        full_mask = np.zeros(color_image.shape[:2], dtype=np.float32)
        y1 = max(0, min(y1, color_image.shape[0] - 1))
        y2 = max(0, min(y2, color_image.shape[0]))
        x1 = max(0, min(x1, color_image.shape[1] - 1))
        x2 = max(0, min(x2, color_image.shape[1]))
        full_mask[y1:y2, x1:x2] = crop[: y2 - y1, : x2 - x1]

        cls = int(raw["cls"])
        name = npu_detector.names[cls] if cls < len(npu_detector.names) else "object"
        detection = _detection_from_result(
            name,
            float(raw["conf"]),
            (x1, y1, x2, y2),
            full_mask,
            color_image,
            depth_frame,
            depth_scale,
            config,
        )
        if detection is not None:
            detections.append(detection)
    return detections


def object_base_position(detection, intrinsic, base_camera):
    u, v = detection["pixel"]
    camera_point = np.asarray(
        rs.rs2_deproject_pixel_to_point(
            intrinsic, [float(u), float(v)], detection["depth_m"]
        ),
        dtype=float,
    )
    base_point = (base_camera @ np.append(camera_point, 1.0))[:3]
    return camera_point, base_point


def _normalized(vector, name):
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        raise RuntimeError(f"{name} has zero length")
    return vector / norm


def _clamp_direction(reference, candidate, max_angle_deg):
    reference = _normalized(reference, "reference approach")
    candidate = _normalized(candidate, "camera approach")
    dot = float(np.clip(np.dot(reference, candidate), -1.0, 1.0))
    angle = float(np.arccos(dot))
    max_angle = float(np.deg2rad(max_angle_deg))
    if angle <= max_angle or angle < 1e-8:
        return candidate, float(np.rad2deg(angle))
    ratio = max_angle / angle
    direction = _normalized(
        np.sin((1.0 - ratio) * angle) * reference
        + np.sin(ratio * angle) * candidate,
        "clamped approach",
    )
    return direction, float(max_angle_deg)


def _project_perpendicular(vector, axis, name):
    axis = _normalized(axis, f"{name} axis")
    projected = np.asarray(vector, dtype=float) - float(np.dot(vector, axis)) * axis
    return _normalized(projected, name)


def _undirected_angle_deg(first, second):
    dot = abs(
        float(
            np.clip(
                np.dot(
                    _normalized(first, "first axis"),
                    _normalized(second, "second axis"),
                ),
                -1.0,
                1.0,
            )
        )
    )
    return float(np.degrees(np.arccos(dot)))


def _image_axis_angle_deg(axis_base, base_from_camera):
    axis_camera = base_from_camera.T @ np.asarray(axis_base, dtype=float)
    image_axis = axis_camera[:2]
    if float(np.linalg.norm(image_axis)) < 1e-8:
        return float("nan")
    return float(np.degrees(np.arctan2(image_axis[1], image_axis[0])))


def grasp_rotation_from_mask(detection, camera_point, intrinsic, base_camera, config):
    """Build Seeed OBB grasp axes in Camera, then map them to Panthera Base."""
    base_from_camera = np.asarray(base_camera[:3, :3], dtype=float)
    manual_approach = config.manual_grasp_rotation[:, 0].copy()

    camera_to_object_camera = _normalized(camera_point, "camera-to-object ray")
    seeed_approach_base = _normalized(
        base_from_camera @ camera_to_object_camera, "Seeed approach in Base"
    )
    if float(np.dot(seeed_approach_base, manual_approach)) < 0.0:
        seeed_approach_base *= -1.0
    if config.approach_policy == "seeed_pure":
        approach = seeed_approach_base
        approach_tilt_deg = _undirected_angle_deg(manual_approach, approach)
    else:
        approach, approach_tilt_deg = _clamp_direction(
            manual_approach,
            seeed_approach_base,
            config.max_dynamic_approach_tilt_deg,
        )

    short_axis_uv = detection["short_axis_uv"]
    short_axis_camera = _normalized(
        np.array(
            [short_axis_uv[0] / intrinsic.fx, short_axis_uv[1] / intrinsic.fy, 0.0],
            dtype=float,
        ),
        "OBB short axis in Camera",
    )
    seeed_open_camera = _project_perpendicular(
        short_axis_camera, camera_to_object_camera, "Seeed OBB opening axis"
    )
    open_axis = _project_perpendicular(
        base_from_camera @ seeed_open_camera, approach, "Panthera opening axis"
    )

    default_open_axis = config.manual_grasp_rotation[:, 1].copy()
    default_open_axis -= float(np.dot(default_open_axis, approach)) * approach
    default_open_axis = _normalized(default_open_axis, "reference opening axis")

    if float(np.dot(open_axis, default_open_axis)) < 0.0:
        open_axis *= -1.0

    offset_rad = np.deg2rad(config.gripper_open_axis_offset_deg)
    open_axis = (
        np.cos(offset_rad) * open_axis
        + np.sin(offset_rad) * np.cross(approach, open_axis)
    )
    open_axis /= np.linalg.norm(open_axis)
    grip_axis = _normalized(np.cross(open_axis, approach), "grasp side axis")
    open_axis = _normalized(np.cross(approach, grip_axis), "orthogonal opening axis")
    tool_side_axis = -grip_axis

    rotation = np.column_stack((approach, open_axis, tool_side_axis))
    relative_open_angle_deg = float(
        np.degrees(
            np.arctan2(
                np.dot(approach, np.cross(default_open_axis, open_axis)),
                np.dot(default_open_axis, open_axis),
            )
        )
    )
    jaw_image_angle_deg = _image_axis_angle_deg(open_axis, base_from_camera)
    short_axis_projection_error_deg = _undirected_angle_deg(
        short_axis_camera, base_from_camera.T @ open_axis
    )
    return (
        rotation,
        approach_tilt_deg,
        relative_open_angle_deg,
        jaw_image_angle_deg,
        short_axis_projection_error_deg,
    )


def grasp_geometry(base_point, tool_rotation, config):
    tool_target = np.asarray(base_point, dtype=float) + config.grasp_offset_base
    tcp_offset = tool_rotation @ config.tcp_in_joint6
    joint6_target = tool_target - tcp_offset
    return tool_target, joint6_target


def workspace_ok(tool_target, joint6_target, config):
    radial = float(np.hypot(tool_target[0], tool_target[1]))
    valid = (
        config.tool_x_range[0] <= tool_target[0] <= config.tool_x_range[1]
        and config.tool_y_range[0] <= tool_target[1] <= config.tool_y_range[1]
        and config.tool_z_range[0] <= tool_target[2] <= config.tool_z_range[1]
        and config.radial_range[0] <= radial <= config.radial_range[1]
        and config.wrist_z_range[0] <= joint6_target[2] <= config.wrist_z_range[1]
    )
    if not valid:
        print(f"[WORKSPACE] rejected: tool={np.round(tool_target, 3)}, radius={radial:.3f}")
    return valid
