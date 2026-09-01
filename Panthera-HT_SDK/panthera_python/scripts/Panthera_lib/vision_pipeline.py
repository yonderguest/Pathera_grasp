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
    timestamp = data.get("timestamp", "unknown")
    num_samples = data.get("num_samples", "unknown")
    print(
        "[VISION] hand-eye calibration loaded: "
        f"timestamp={timestamp}, samples={num_samples}",
        flush=True,
    )
    return transform


def init_camera(config: GraspConfig):
    """Start RealSense depth/colour streams and return aligned pipeline objects."""
    pipeline = rs.pipeline()
    stream_config = rs.config()
    if config.camera_serial:
        stream_config.enable_device(config.camera_serial)
    stream_config.enable_stream(
        rs.stream.depth, config.width, config.height, rs.format.z16, config.fps
    )
    stream_config.enable_stream(
        rs.stream.color, config.width, config.height, rs.format.bgr8, config.fps
    )
    profile = pipeline.start(stream_config)
    device = profile.get_device()
    sensor = device.first_depth_sensor()
    depth_scale = float(sensor.get_depth_scale())
    color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
    intrinsic = color_profile.get_intrinsics()
    align = rs.align(rs.stream.color)
    for _ in range(30):
        pipeline.wait_for_frames()
    product = device.get_info(rs.camera_info.name)
    serial_number = device.get_info(rs.camera_info.serial_number)
    print(
        f"[VISION] camera ready: product={product}, serial={serial_number}, "
        f"fx={intrinsic.fx:.2f}, fy={intrinsic.fy:.2f}, "
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
        intrinsic=None,
    ) -> None:
        self.pipeline = pipeline
        self.align = align
        self.config = config
        self.depth_scale = depth_scale
        self.streamer = streamer
        self.model = model
        self.intrinsic = intrinsic
        self.npu_detector = None

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._capture_thread: threading.Thread | None = None
        self._inference_thread: threading.Thread | None = None

        self._latest_capture: dict | None = None
        self._latest_for_scan: dict | None = None
        self._detections: list = []
        self._detections_timestamp = 0.0
        self._capture_sequence = 0
        self._last_inference_capture_sequence = -1
        self._last_inference_wall_time = 0.0
        self._last_inference_start_time = 0.0
        self._last_error_time = 0.0
        self._fatal_error: Exception | None = None
        self._performance_window_start = time.monotonic()
        self._performance_capture_count = 0
        self._performance_inference_count = 0
        self._last_inference_latency = 0.0

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
            if self._latest_for_scan is None:
                return None
            snapshot = dict(self._latest_for_scan)
            snapshot["detections"] = list(snapshot["detections"])
            return snapshot

    def freshness_marker(self) -> int:
        """Return the last completed inference sequence, or -1 before startup."""
        with self._lock:
            if self._latest_for_scan is None:
                return -1
            return int(self._latest_for_scan["frame_seq"])

    def wait_for_newer(self, after_timestamp: float, timeout: float = 2.0):
        deadline = time.monotonic() + timeout
        latest = self.latest()
        use_sequence = isinstance(after_timestamp, (int, np.integer))
        while latest is None or (
            latest["frame_seq"] <= int(after_timestamp)
            if use_sequence
            else latest["detections_timestamp"] <= float(after_timestamp)
        ):
            if self._fatal_error is not None:
                raise RuntimeError("camera inference stopped after a fatal backend error") from self._fatal_error
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.01)
            latest = self.latest()
        return latest

    def _store_inference_result(
        self,
        capture: dict,
        detections: list,
        inference_latency: float = 0.0,
    ) -> None:
        """Publish one immutable, frame-consistent RGB-D/detection snapshot."""
        snapshot = {
            "color_image": capture["color_image"],
            "depth_image": capture["depth_image"],
            "detections": list(detections),
            "timestamp": capture["timestamp"],
            "detections_timestamp": capture["timestamp"],
            "capture_timestamp_ns": capture["capture_timestamp_ns"],
            "frame_seq": capture["frame_seq"],
            "intrinsics": self.intrinsic,
            "inference_latency_s": float(inference_latency),
            "snapshot_age_s": float(time.monotonic() - capture["timestamp"]),
        }
        with self._lock:
            self._detections = list(detections)
            self._detections_timestamp = capture["timestamp"]
            self._last_inference_capture_sequence = capture["frame_seq"]
            self._last_inference_wall_time = time.monotonic()
            self._last_inference_latency = float(inference_latency)
            self._performance_inference_count += 1
            self._latest_for_scan = snapshot

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
                capture_timestamp_ns = time.time_ns()

                with self._lock:
                    self._capture_sequence += 1
                    self._performance_capture_count += 1
                    self._latest_capture = {
                        "color_image": color_image,
                        "depth_image": depth_image,
                        "timestamp": timestamp,
                        "capture_timestamp_ns": capture_timestamp_ns,
                        "frame_seq": self._capture_sequence,
                    }
                    detections = list(self._detections)

                if self.streamer is not None:
                    self.streamer.publish(
                        color_image,
                        detections,
                        depth_image=depth_image,
                        depth_scale=self.depth_scale,
                    )
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

            if (
                capture is None
                or capture["frame_seq"] == self._last_inference_capture_sequence
            ):
                time.sleep(0.01)
                continue

            now = time.monotonic()
            if now - self._last_inference_start_time < self.config.camera_detection_interval:
                time.sleep(0.01)
                continue
            self._last_inference_start_time = now

            color_image = capture["color_image"]
            depth_image = capture["depth_image"]
            inference_started = time.monotonic()
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
                with self._lock:
                    self._last_inference_capture_sequence = capture["frame_seq"]
                    if self.npu_detector is not None:
                        self._fatal_error = exc
                if self.npu_detector is not None:
                    break
                continue

            inference_latency = time.monotonic() - inference_started
            self._store_inference_result(capture, detections, inference_latency)
            self._report_performance(len(detections))

            if self.streamer is not None:
                self.streamer.publish(
                    color_image,
                    detections,
                    depth_image=depth_image,
                    depth_scale=self.depth_scale,
                )

    def _report_performance(self, detection_count: int) -> None:
        now = time.monotonic()
        elapsed = now - self._performance_window_start
        if elapsed < self.config.camera_performance_log_interval:
            return
        with self._lock:
            capture_count = self._performance_capture_count
            inference_count = self._performance_inference_count
            latency = self._last_inference_latency
            self._performance_capture_count = 0
            self._performance_inference_count = 0
            self._performance_window_start = now
        backend = "NPU" if self.npu_detector is not None else "CPU"
        print(
            f"[VISION-PERF] backend={backend}, capture={capture_count / elapsed:.1f} FPS, "
            f"detect={inference_count / elapsed:.2f} FPS, infer={latency * 1000.0:.0f} ms, "
            f"objects={detection_count}",
            flush=True,
        )

    def _report_error(self, exc: Exception) -> None:
        now = time.monotonic()
        if now - self._last_error_time > 5.0:
            self._last_error_time = now
            print(f"[CAMERA-FEED] frame failed: {exc!r}", flush=True)


def get_base_camera_transform(robot, tcp_camera, joint_position=None):
    """Return Base<-Camera for an explicit, snapshot-bound joint position."""
    fk = robot.forward_kinematics(joint_position)
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


COLOR_VOTE_NAMES = ("red", "yellow", "green", "blue")


def adaptive_color_core(mask, config: GraspConfig):
    """Keep the deepest mask pixels without erasing small segmented blocks."""
    mask_u8 = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)
    area = int(np.count_nonzero(mask_u8))
    if area == 0:
        return mask_u8

    distance = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
    keep = min(
        area,
        max(config.color_min_core_pixels, int(np.ceil(area * config.color_core_fraction))),
    )
    if keep >= area:
        return mask_u8

    distances = distance[mask_u8.astype(bool)]
    threshold = float(np.partition(distances, distances.size - keep)[distances.size - keep])
    core = ((distance >= threshold) & (mask_u8 > 0)).astype(np.uint8)
    return core if np.any(core) else mask_u8


def extract_color_evidence(bgr_image, mask, config: GraspConfig):
    """Return raw HSV counts that can be safely merged across matched frames."""
    core = adaptive_color_core(mask, config)
    pixels = bgr_image[core.astype(bool)]
    evidence = {
        "core_pixels": int(len(pixels)),
        "dark": 0,
        "white": 0,
        "chromatic_pixels": 0,
        "votes": {name: 0 for name in COLOR_VOTE_NAMES},
    }
    if pixels.size == 0:
        return evidence

    hsv = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    h = hsv[:, 0].astype(np.int16)
    s = hsv[:, 1].astype(np.int16)
    v = hsv[:, 2].astype(np.int16)

    evidence["dark"] = int(np.count_nonzero(v < 60))
    evidence["white"] = int(np.count_nonzero((s < 45) & (v > 180)))
    valid = (s >= config.color_min_saturation) & (v >= config.color_min_value)
    hue = h[valid]
    evidence["chromatic_pixels"] = int(len(hue))
    if hue.size:
        evidence["votes"] = {
            "red": int(np.count_nonzero((hue < 12) | (hue >= 168))),
            "yellow": int(
                np.count_nonzero(
                    (hue >= 12) & (hue < config.color_yellow_green_boundary)
                )
            ),
            "green": int(
                np.count_nonzero(
                    (hue >= config.color_yellow_green_boundary) & (hue < 85)
                )
            ),
            "blue": int(np.count_nonzero((hue >= 85) & (hue < 140))),
        }
    return evidence


def merge_color_evidence(items):
    merged = {
        "core_pixels": 0,
        "dark": 0,
        "white": 0,
        "chromatic_pixels": 0,
        "votes": {name: 0 for name in COLOR_VOTE_NAMES},
    }
    for evidence in items:
        merged["core_pixels"] += int(evidence["core_pixels"])
        merged["dark"] += int(evidence["dark"])
        merged["white"] += int(evidence["white"])
        merged["chromatic_pixels"] += int(evidence["chromatic_pixels"])
        for name in COLOR_VOTE_NAMES:
            merged["votes"][name] += int(evidence["votes"][name])
    return merged


def classify_color_evidence(evidence, config: GraspConfig):
    """Return color, confidence, usable samples and top-two vote margin."""
    core_pixels = int(evidence["core_pixels"])
    if core_pixels <= 0:
        return "unknown", 0.0, 0, 0.0

    dark_ratio = evidence["dark"] / float(core_pixels)
    if dark_ratio >= config.color_dark_ratio:
        return "black", dark_ratio, core_pixels, dark_ratio

    white_ratio = evidence["white"] / float(core_pixels)
    if white_ratio >= config.color_white_ratio:
        return "white", white_ratio, core_pixels, white_ratio

    chromatic_pixels = int(evidence["chromatic_pixels"])
    if chromatic_pixels <= 0:
        return "unknown", 0.0, 0, 0.0

    ranked = sorted(evidence["votes"].items(), key=lambda item: item[1], reverse=True)
    color, count = ranked[0]
    runner_up = ranked[1][1]
    ratio = count / float(chromatic_pixels)
    margin = (count - runner_up) / float(chromatic_pixels)
    if ratio < config.color_dominant_ratio or margin < config.color_min_margin:
        return "unknown", ratio, chromatic_pixels, margin
    return color, ratio, chromatic_pixels, margin


def apply_accumulated_color(detection, evidence_items, matched_frames, config):
    """Copy a detection and replace its color with merged matched-frame evidence."""
    merged = merge_color_evidence(evidence_items)
    color, confidence, sample_count, margin = classify_color_evidence(merged, config)
    updated = dict(detection)
    updated.update(
        {
            "color": color,
            "color_confidence": confidence,
            "color_evidence": merged,
            "color_frames": int(matched_frames),
            "color_samples": int(sample_count),
            "color_margin": float(margin),
        }
    )
    return updated


def classify_color(bgr_image, mask, config: GraspConfig):
    """Classify adaptive mask-core pixels using dominant vote and margin."""
    evidence = extract_color_evidence(bgr_image, mask, config)
    color, confidence, _sample_count, _margin = classify_color_evidence(evidence, config)
    return color, confidence


def robust_surface_point(depth_frame, mask, depth_scale, config: GraspConfig):
    """Estimate one coherent near object surface from matching (u, v, z) pixels.

    A global mask median mixes a cube's top, sides and leaked background.  Use
    a robust near-surface percentile as an anchor, keep its local depth band,
    and derive both the image point and depth from that same pixel set.
    """
    if hasattr(depth_frame, "get_data"):
        depth = np.asanyarray(depth_frame.get_data()).astype(np.float32)
    else:
        depth = np.asanyarray(depth_frame).astype(np.float32)
    mask_bool = np.asarray(mask, dtype=bool)
    ys, xs = np.where(mask_bool)
    raw_values = depth[ys, xs]
    valid = (
        (raw_values >= config.min_depth_m / depth_scale)
        & (raw_values <= config.max_depth_m / depth_scale)
    )
    if int(np.count_nonzero(valid)) < config.depth_min_surface_pixels:
        return None
    ys = ys[valid]
    xs = xs[valid]
    values_m = raw_values[valid] * float(depth_scale)

    anchor = float(np.percentile(values_m, config.depth_surface_percentile))
    in_surface = np.abs(values_m - anchor) <= config.depth_surface_band_m
    if int(np.count_nonzero(in_surface)) < config.depth_min_surface_pixels:
        return None
    surface_depths = values_m[in_surface]
    spread = float(np.percentile(surface_depths, 90) - np.percentile(surface_depths, 10))
    if spread > config.depth_max_surface_spread_m:
        return None
    return {
        "pixel": np.array(
            [np.median(xs[in_surface]), np.median(ys[in_surface])],
            dtype=float,
        ),
        "depth_m": float(np.median(surface_depths)),
        "depth_samples": int(surface_depths.size),
        "depth_spread_m": spread,
    }


def median_mask_depth(depth_frame, mask, depth_scale, config: GraspConfig):
    """Compatibility wrapper returning the robust coherent-surface depth."""
    surface = robust_surface_point(depth_frame, mask, depth_scale, config)
    return None if surface is None else surface["depth_m"]


def _detection_from_result(name, conf, bbox, mask, color_image, depth_frame, depth_scale, config):
    x1, y1, x2, y2 = bbox
    full_mask = (np.asarray(mask, dtype=np.float32) > 0.5).astype(np.uint8)
    if full_mask.shape != (config.height, config.width):
        full_mask = cv2.resize(
            full_mask, (config.width, config.height), interpolation=cv2.INTER_NEAREST
        )
    color_evidence = extract_color_evidence(color_image, full_mask, config)
    color, color_confidence, color_samples, color_margin = classify_color_evidence(
        color_evidence,
        config,
    )

    inner = adaptive_color_core(full_mask, config)
    if not np.any(inner):
        return None

    surface = robust_surface_point(depth_frame, inner, depth_scale, config)
    if surface is None:
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

    return {
        "confidence": float(conf),
        "class_name": name,
        "color": color,
        "color_confidence": color_confidence,
        "color_evidence": color_evidence,
        "color_frames": 1,
        "color_samples": color_samples,
        "color_margin": color_margin,
        "pixel": surface["pixel"],
        "depth_m": surface["depth_m"],
        "depth_samples": surface["depth_samples"],
        "depth_spread_m": surface["depth_spread_m"],
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
