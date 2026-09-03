"""Conservative CPU-YOLOE hand following for the Panthera demo.

The first implementation is intentionally a stop--observe--small-step controller.
It never sends robot commands from the camera or HTTP threads: ``run`` must be
called by the application's robot-owning main thread.  Every perception or
planning failure therefore degrades to a position hold.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np


class HandFollowState(str, Enum):
    """Externally visible state of the hand-follow safety controller."""

    IDLE = "idle"
    ARMING = "arming"
    TRACKING = "tracking"
    HOLD_LOST = "hold_lost"
    PAUSED = "paused"
    STOPPING = "stopping"
    FAULT = "fault"


class SemanticStepLimitError(RuntimeError):
    """The requested hand displacement is too large for one observe cycle."""


@dataclass(frozen=True)
class HandFollowSettings:
    """Fixed, conservative limits approved for the first CPU prototype."""

    # Camera-only A/B validation on the deployed D405 showed that the everyday
    # noun "hand" produced three consecutive correct masks, while "human hand"
    # at 0.25 only produced isolated detections and could never arm the gate.
    prompt: str = "hand"
    input_size: int = 320
    confidence_threshold: float = 0.10
    iou_threshold: float = 0.45
    max_detections: int = 4

    min_mask_area_px: int = 300
    max_mask_area_ratio: float = 0.45
    border_margin_px: int = 8
    min_depth_m: float = 0.15
    max_depth_m: float = 0.80
    depth_surface_percentile: float = 25.0
    depth_surface_band_m: float = 0.030
    min_depth_samples: int = 24
    min_surface_fraction: float = 0.60
    max_depth_spread_m: float = 0.020

    stable_frames: int = 3
    reacquire_frames: int = 2
    track_radius_m: float = 0.10
    max_snapshot_age_s: float = 0.75
    frame_timeout_s: float = 1.0
    lost_pause_timeout_s: float = 1.25

    desired_camera_distance_m: float = 0.30
    min_tcp_hand_distance_m: float = 0.20
    lateral_deadband_m: float = 0.015
    depth_deadband_m: float = 0.020
    position_gain: float = 0.50
    # This is an intent/observation gate, not a robot command size.  A target
    # farther than 10 cm from the desired camera pose is held and requires a
    # fresh operator enable. A passed request may execute at most 20 mm, then
    # the controller must stop, take a new RGB-D frame, and plan again.
    max_semantic_step_m: float = 0.10
    max_tcp_step_m: float = 0.020
    max_joint_step_rad: float = 0.035
    max_joint_speed_rad_s: float = 0.12
    min_step_duration_s: float = 0.40
    joint_limit_margin_rad: float = 0.10
    home_tolerance_rad: float = 0.08
    endpoint_position_tolerance_m: float = 0.006
    endpoint_rotation_tolerance_deg: float = 3.0
    joint_path_samples: int = 6
    inference_hold_period_s: float = 0.04

    def __post_init__(self) -> None:
        if self.input_size < 160 or self.input_size % 32:
            raise ValueError("hand detector input_size must be a multiple of 32")
        if not 0.0 < self.confidence_threshold < 1.0:
            raise ValueError("hand confidence threshold must be in (0, 1)")
        if not 0.0 < self.iou_threshold < 1.0:
            raise ValueError("hand IoU threshold must be in (0, 1)")
        if self.max_detections < 1:
            raise ValueError("hand max_detections must be positive")
        if not 0.0 < self.max_mask_area_ratio < 1.0:
            raise ValueError("hand mask area ratio must be in (0, 1)")
        if not 0.0 < self.min_depth_m < self.max_depth_m:
            raise ValueError("invalid hand depth range")
        if not 0.0 < self.depth_surface_percentile < 50.0:
            raise ValueError("invalid hand depth percentile")
        if (
            self.min_depth_samples < 1
            or not 0.5 < self.min_surface_fraction <= 1.0
            or self.max_depth_spread_m <= 0.0
        ):
            raise ValueError("invalid hand depth quality limits")
        if self.stable_frames < 3 or not 1 <= self.reacquire_frames <= self.stable_frames:
            raise ValueError("invalid hand arming/reacquisition frame counts")
        if self.track_radius_m <= 0.0 or self.max_snapshot_age_s <= 0.0:
            raise ValueError("invalid hand tracking freshness limits")
        if not 1.0 <= self.lost_pause_timeout_s <= 1.5:
            raise ValueError("hand loss timeout must remain inside 1.0..1.5 seconds")
        if not np.isclose(self.desired_camera_distance_m, 0.30, atol=1e-12):
            raise ValueError("first hand-follow version is fixed at a 30 cm camera distance")
        if self.min_tcp_hand_distance_m < 0.20:
            raise ValueError("TCP-to-hand clearance must be at least 20 cm")
        if not 0.0 < self.position_gain <= 1.0:
            raise ValueError("hand-follow position gain must be in (0, 1]")
        if not 0.0 < self.max_semantic_step_m <= 0.10:
            raise ValueError("hand-follow semantic step must be in (0, 100 mm]")
        if not 0.0 < self.max_tcp_step_m <= 0.020:
            raise ValueError("hand-follow TCP step must not exceed 20 mm")
        if not 0.0 < self.max_joint_speed_rad_s <= 0.12:
            raise ValueError("hand-follow joint speed must not exceed 0.12 rad/s")
        if self.max_joint_step_rad <= 0.0 or self.joint_limit_margin_rad <= 0.0:
            raise ValueError("invalid hand-follow joint limits")
        if self.min_step_duration_s <= 0.0:
            raise ValueError("hand-follow step duration must be positive")
        if self.joint_path_samples < 5:
            raise ValueError("hand-follow joint path requires at least five FK samples")
        if not 0.0 < self.inference_hold_period_s <= 0.05:
            raise ValueError("hand-follow inference HOLD period must not exceed 50 ms")


@dataclass(frozen=True)
class HandDetection:
    """One hand instance with a robust RGB-D palm/core point."""

    confidence: float
    pixel: np.ndarray
    camera_point: np.ndarray
    depth_m: float
    depth_spread_m: float
    depth_samples: int
    bbox: tuple[int, int, int, int]
    mask: np.ndarray

    def as_stream_detection(self) -> dict:
        """Return a drawing-compatible dictionary for the existing web stream."""
        return {
            "confidence": float(self.confidence),
            "class_name": "human hand",
            "color": "unknown",
            "color_confidence": 0.0,
            "color_frames": 1,
            "pixel": np.asarray(self.pixel, dtype=float).copy(),
            "depth_m": float(self.depth_m),
            "depth_spread_m": float(self.depth_spread_m),
            "depth_samples": int(self.depth_samples),
            "bbox": tuple(self.bbox),
            "mask": np.asarray(self.mask, dtype=np.uint8).copy(),
        }


@dataclass(frozen=True)
class FollowMotion:
    """A fully checked single-step motion proposal."""

    target_joints: np.ndarray
    target_tool: np.ndarray
    hand_base: np.ndarray
    camera_step: np.ndarray
    duration_s: float
    tcp_hand_clearance_m: float


@dataclass(frozen=True)
class HandFollowResult:
    state: HandFollowState
    reason: str
    completed_steps: int


def load_cpu_hand_model(model_path, text_encoder_path, prompt: str = "hand"):
    """Load an independent CPU YOLOE instance configured for one hand prompt.

    The main application invokes this only after the operator explicitly asks
    for follow mode. YOLOE's MobileCLIP loader resolves its encoder relative to
    the current directory, so the temporary directory change mirrors the
    established object loader.
    """
    from ultralytics.models.yolo.model import YOLOE

    model_path = Path(model_path).resolve()
    encoder_path = Path(text_encoder_path).resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"YOLOE hand model not found: {model_path}")
    if not encoder_path.is_file():
        raise FileNotFoundError(f"YOLOE text encoder not found: {encoder_path}")
    model = YOLOE(str(model_path))
    previous_cwd = Path.cwd()
    try:
        os.chdir(encoder_path.parent)
        model.set_classes([str(prompt)])
    finally:
        os.chdir(previous_cwd)
    return model


def _intrinsic_value(intrinsic, name: str) -> float:
    if isinstance(intrinsic, dict):
        value = intrinsic[name]
    else:
        value = getattr(intrinsic, name)
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"camera intrinsic {name} is invalid")
    return value


def _resize_mask(mask, image_shape: tuple[int, int]) -> np.ndarray:
    mask = (np.asarray(mask, dtype=np.float32) > 0.5).astype(np.uint8)
    if mask.shape != image_shape:
        mask = cv2.resize(
            mask,
            (image_shape[1], image_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    return mask


def _hand_core(mask: np.ndarray) -> np.ndarray:
    """Prefer the thick palm/core over moving fingertips and thin wrist edges."""
    binary = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)
    if not np.any(binary):
        return binary
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    maximum = float(np.max(distance))
    if maximum <= 0.0:
        return binary
    core = ((distance >= 0.50 * maximum) & (binary > 0)).astype(np.uint8)
    return core if int(np.count_nonzero(core)) >= 24 else binary


def hand_candidate_from_mask(
    mask,
    bbox,
    confidence: float,
    depth_image,
    depth_scale: float,
    intrinsic,
    settings: HandFollowSettings,
    diagnostics: dict | None = None,
) -> HandDetection | None:
    """Build one hand candidate, rejecting uncertain geometry fail-closed."""
    def reject(reason: str):
        if diagnostics is not None:
            diagnostics["rejection_reason"] = str(reason)
        return None

    depth = np.asarray(depth_image)
    if depth.ndim != 2 or not np.isfinite(depth_scale) or depth_scale <= 0.0:
        return reject("invalid depth image or scale")
    height, width = depth.shape
    full_mask = _resize_mask(mask, (height, width))
    area = int(np.count_nonzero(full_mask))
    if (
        area < settings.min_mask_area_px
        or area > int(settings.max_mask_area_ratio * height * width)
    ):
        return reject(f"mask area outside limits ({area}px)")

    x1, y1, x2, y2 = (int(value) for value in bbox)
    margin = settings.border_margin_px
    if x1 <= margin or y1 <= margin or x2 >= width - margin or y2 >= height - margin:
        return reject("bounding box touches the image safety margin")

    core = _hand_core(full_mask)
    ys, xs = np.nonzero(core)
    if xs.size < settings.min_depth_samples:
        return reject("too few mask-core pixels")
    depths_m = depth[ys, xs].astype(np.float64) * float(depth_scale)
    valid = (
        np.isfinite(depths_m)
        & (depths_m >= settings.min_depth_m)
        & (depths_m <= settings.max_depth_m)
    )
    if int(np.count_nonzero(valid)) < settings.min_depth_samples:
        return reject("too few metric depth samples")
    xs = xs[valid]
    ys = ys[valid]
    depths_m = depths_m[valid]

    near = float(np.percentile(depths_m, settings.depth_surface_percentile))
    surface = np.abs(depths_m - near) <= settings.depth_surface_band_m
    surface_count = int(np.count_nonzero(surface))
    if (
        surface_count < settings.min_depth_samples
        or surface_count / float(depths_m.size) < settings.min_surface_fraction
    ):
        return reject("near-surface depth support is too small")
    surface_depths = depths_m[surface]
    spread = float(
        np.percentile(surface_depths, 90.0) - np.percentile(surface_depths, 10.0)
    )
    if not np.isfinite(spread) or spread > settings.max_depth_spread_m:
        return reject(f"depth spread is too large ({spread:.4f}m)")

    u = float(np.median(xs[surface]))
    v = float(np.median(ys[surface]))
    z = float(np.median(surface_depths))
    fx = _intrinsic_value(intrinsic, "fx")
    fy = _intrinsic_value(intrinsic, "fy")
    ppx = _intrinsic_value(intrinsic, "ppx")
    ppy = _intrinsic_value(intrinsic, "ppy")
    if fx <= 0.0 or fy <= 0.0:
        return reject("invalid camera focal length")
    camera_point = np.array(
        [(u - ppx) * z / fx, (v - ppy) * z / fy, z],
        dtype=float,
    )
    return HandDetection(
        confidence=float(confidence),
        pixel=np.array([u, v], dtype=float),
        camera_point=camera_point,
        depth_m=z,
        depth_spread_m=spread,
        depth_samples=int(surface_depths.size),
        bbox=(x1, y1, x2, y2),
        mask=full_mask,
    )


class CpuYoloHandDetector:
    """Small adapter around an independently prompted CPU YOLOE model."""

    def __init__(self, model, settings: HandFollowSettings | None = None) -> None:
        self.model = model
        self.settings = settings or HandFollowSettings()
        self.last_diagnostics: dict = {}

    def detect(self, color_image, depth_image, depth_scale: float, intrinsic):
        image = np.asarray(color_image, dtype=np.uint8)
        result = self.model.predict(
            image,
            imgsz=self.settings.input_size,
            conf=self.settings.confidence_threshold,
            iou=self.settings.iou_threshold,
            max_det=self.settings.max_detections,
            verbose=False,
            device="cpu",
        )[0]
        if result.boxes is None or result.masks is None or len(result.boxes) == 0:
            self.last_diagnostics = {
                "raw_count": 0,
                "matching_count": 0,
                "raw_candidates": [],
                "label_rejected": 0,
                "geometry_rejected": 0,
                "accepted_count": 0,
            }
            return []
        confidences = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        boxes = result.boxes.xyxy.cpu().numpy()
        masks = result.masks.data.cpu().numpy()
        detections = []
        raw_candidates = []
        label_rejected = 0
        matching_count = 0
        geometry_rejected = 0
        for index in np.argsort(confidences)[::-1]:
            name = str(result.names[int(classes[index])]).strip().lower()
            raw_item = {
                "name": name,
                "confidence": float(confidences[index]),
                "bbox": tuple(float(value) for value in boxes[index]),
            }
            raw_candidates.append(raw_item)
            if name != self.settings.prompt.strip().lower():
                label_rejected += 1
                raw_item["rejection_reason"] = "unexpected class label"
                continue
            matching_count += 1
            candidate_diagnostics = {}
            candidate = hand_candidate_from_mask(
                masks[index],
                boxes[index],
                float(confidences[index]),
                depth_image,
                depth_scale,
                intrinsic,
                self.settings,
                diagnostics=candidate_diagnostics,
            )
            if candidate is not None:
                detections.append(candidate)
            else:
                geometry_rejected += 1
                raw_item.update(candidate_diagnostics)
        self.last_diagnostics = {
            "raw_count": len(raw_candidates),
            "matching_count": matching_count,
            "raw_candidates": raw_candidates,
            "label_rejected": label_rejected,
            "geometry_rejected": geometry_rejected,
            "accepted_count": len(detections),
        }
        return detections


class HandTrackGate:
    """Unique-hand arming and identity gate, independent from robot hardware."""

    def __init__(self, settings: HandFollowSettings | None = None) -> None:
        self.settings = settings or HandFollowSettings()
        self.state = HandFollowState.IDLE
        self.reason = "idle"
        self._stable_count = 0
        self._reacquire_count = 0
        self._last_point: np.ndarray | None = None
        self._last_valid_time = 0.0

    def reset(self, now: float | None = None) -> None:
        now = time.monotonic() if now is None else float(now)
        self.state = HandFollowState.ARMING
        self.reason = "waiting for one stable hand"
        self._stable_count = 0
        self._reacquire_count = 0
        self._last_point = None
        self._last_valid_time = now

    def pause(self, reason: str) -> None:
        self.state = HandFollowState.PAUSED
        self.reason = str(reason)

    def miss(self, reason: str, now: float | None = None) -> None:
        now = time.monotonic() if now is None else float(now)
        if self.state == HandFollowState.PAUSED:
            return
        # A failed ARMING observation must restart the full three-frame proof;
        # it may not take the shorter two-frame TRACKING reacquisition path.
        was_arming = self.state == HandFollowState.ARMING
        if was_arming:
            self._stable_count = 0
            self._last_point = None
        else:
            self.state = HandFollowState.HOLD_LOST
        self.reason = str(reason)
        self._reacquire_count = 0
        # The manual re-enable timeout applies after a tracked hand is lost.
        # During initial arming, waiting without a hand must remain harmless
        # and must not permanently pause the mode before the operator is ready.
        if (
            not was_arming
            and now - self._last_valid_time >= self.settings.lost_pause_timeout_s
        ):
            self.pause(f"{reason}; manual re-enable required")

    def update(
        self,
        candidates: Iterable[HandDetection],
        now: float | None = None,
        observed_count: int | None = None,
    ) -> HandDetection | None:
        now = time.monotonic() if now is None else float(now)
        candidates = list(candidates)
        if self.state == HandFollowState.PAUSED:
            return None
        observation_count = len(candidates) if observed_count is None else int(observed_count)
        if observation_count < len(candidates) or observation_count < 0:
            raise ValueError("observed hand count cannot be smaller than accepted count")
        if observation_count != 1:
            self.miss(
                "no hand"
                if observation_count == 0
                else "multiple hands are ambiguous",
                now,
            )
            return None
        if len(candidates) != 1:
            self.miss("hand geometry is invalid", now)
            return None

        candidate = candidates[0]
        point = np.asarray(candidate.camera_point, dtype=float)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            self.miss("invalid hand point", now)
            return None
        if self._last_point is not None:
            shift = float(np.linalg.norm(point - self._last_point))
            if shift > self.settings.track_radius_m:
                self.miss(f"hand identity jump {shift:.3f} m", now)
                return None

        if self.state == HandFollowState.ARMING:
            self._stable_count += 1
            if self._stable_count < self.settings.stable_frames:
                self.reason = (
                    f"arming {self._stable_count}/{self.settings.stable_frames}"
                )
                self._last_point = point.copy()
                self._last_valid_time = now
                return None
            self.state = HandFollowState.TRACKING
        elif self.state == HandFollowState.HOLD_LOST:
            self._reacquire_count += 1
            if self._reacquire_count < self.settings.reacquire_frames:
                self.reason = (
                    f"reacquiring {self._reacquire_count}/"
                    f"{self.settings.reacquire_frames}"
                )
                self._last_point = point.copy()
                self._last_valid_time = now
                return None
            self.state = HandFollowState.TRACKING

        self.reason = "tracking one stable hand"
        self._last_point = point.copy()
        self._last_valid_time = now
        return candidate


def _rotation_error_deg(first, second) -> float:
    relative = np.asarray(first, dtype=float).T @ np.asarray(second, dtype=float)
    cosine = np.clip((float(np.trace(relative)) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _base_camera_transform(tool_position, tool_rotation, tcp_camera) -> np.ndarray:
    base_tcp = np.eye(4, dtype=float)
    base_tcp[:3, :3] = np.asarray(tool_rotation, dtype=float)
    base_tcp[:3, 3] = np.asarray(tool_position, dtype=float)
    transform = base_tcp @ np.asarray(tcp_camera, dtype=float)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise RuntimeError("invalid Base<-Camera transform for hand following")
    return transform


def _camera_step(camera_point, settings: HandFollowSettings) -> np.ndarray:
    error = np.asarray(camera_point, dtype=float) - np.array(
        [0.0, 0.0, settings.desired_camera_distance_m], dtype=float
    )
    if abs(error[0]) <= settings.lateral_deadband_m:
        error[0] = 0.0
    if abs(error[1]) <= settings.lateral_deadband_m:
        error[1] = 0.0
    if abs(error[2]) <= settings.depth_deadband_m:
        error[2] = 0.0
    step = settings.position_gain * error
    semantic_norm = float(np.linalg.norm(step))
    if not np.isfinite(semantic_norm):
        raise SemanticStepLimitError("hand-follow semantic step is non-finite")
    if semantic_norm > settings.max_semantic_step_m + 1e-9:
        raise SemanticStepLimitError(
            f"hand-follow semantic step {semantic_norm:.3f} m exceeds "
            f"the {settings.max_semantic_step_m:.3f} m HOLD limit"
        )
    norm = float(np.linalg.norm(step))
    if norm > settings.max_tcp_step_m:
        step *= settings.max_tcp_step_m / norm
    return step


def _workspace_contains(tool_position, joint6_position, config) -> bool:
    """Apply the grasp workspace envelope without importing the camera module."""
    tool = np.asarray(tool_position, dtype=float)
    joint6 = np.asarray(joint6_position, dtype=float)
    if tool.shape != (3,) or joint6.shape != (3,):
        return False
    try:
        radial = float(np.hypot(tool[0], tool[1]))
        return bool(
            config.tool_x_range[0] <= tool[0] <= config.tool_x_range[1]
            and config.tool_y_range[0] <= tool[1] <= config.tool_y_range[1]
            and config.tool_z_range[0] <= tool[2] <= config.tool_z_range[1]
            and config.radial_range[0] <= radial <= config.radial_range[1]
            and config.wrist_z_range[0] <= joint6[2] <= config.wrist_z_range[1]
        )
    except (AttributeError, IndexError, TypeError):
        return False


def _validate_interpolated_joint_path(
    planner,
    current_joints,
    solution,
    current_tool,
    fixed_rotation,
    hand_base,
    safe_lower,
    safe_upper,
    settings: HandFollowSettings,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fail closed when a MoveJ interpolation bows outside the safe envelope."""
    tcp_in_joint6 = np.asarray(planner.config.tcp_in_joint6, dtype=float)
    if tcp_in_joint6.shape != (3,) or not np.all(np.isfinite(tcp_in_joint6)):
        raise RuntimeError("invalid TCP offset for hand-follow path validation")

    endpoint_tool = None
    endpoint_rotation = None
    endpoint_clearance = None
    for index, alpha in enumerate(
        np.linspace(0.0, 1.0, settings.joint_path_samples),
        start=1,
    ):
        sample_joints = current_joints + float(alpha) * (solution - current_joints)
        if (
            np.any(sample_joints < safe_lower - 1e-9)
            or np.any(sample_joints > safe_upper + 1e-9)
        ):
            raise RuntimeError(
                f"hand-follow joint path sample {index} entered a limit margin"
            )
        sample_tool, sample_rotation = planner.current_tool_pose(sample_joints)
        sample_tool = np.asarray(sample_tool, dtype=float)
        sample_rotation = np.asarray(sample_rotation, dtype=float)
        if (
            sample_tool.shape != (3,)
            or sample_rotation.shape != (3, 3)
            or not np.all(np.isfinite(sample_tool))
            or not np.all(np.isfinite(sample_rotation))
        ):
            raise RuntimeError(f"hand-follow FK path sample {index} is invalid")

        displacement = float(np.linalg.norm(sample_tool - current_tool))
        if displacement > settings.max_tcp_step_m + 5e-4:
            raise RuntimeError(
                f"hand-follow FK path sample {index} exceeds the 20.5 mm envelope"
            )
        clearance = float(np.linalg.norm(hand_base - sample_tool))
        if clearance < settings.min_tcp_hand_distance_m:
            raise RuntimeError(
                f"hand-follow FK path sample {index} violates human clearance"
            )
        rotation_error = _rotation_error_deg(sample_rotation, fixed_rotation)
        if rotation_error > settings.endpoint_rotation_tolerance_deg:
            raise RuntimeError(
                f"hand-follow FK path sample {index} orientation error is "
                f"{rotation_error:.2f} deg"
            )
        sample_joint6 = sample_tool - sample_rotation @ tcp_in_joint6
        if not _workspace_contains(sample_tool, sample_joint6, planner.config):
            raise RuntimeError(
                f"hand-follow FK path sample {index} leaves the workspace"
            )
        endpoint_tool = sample_tool
        endpoint_rotation = sample_rotation
        endpoint_clearance = clearance

    assert endpoint_tool is not None
    assert endpoint_rotation is not None
    assert endpoint_clearance is not None
    return endpoint_tool, endpoint_rotation, endpoint_clearance


def plan_bounded_follow_step(
    planner,
    current_joints,
    fixed_tool_rotation,
    hand: HandDetection,
    tcp_camera,
    settings: HandFollowSettings | None = None,
) -> FollowMotion | None:
    """Create one <=20 mm, <=0.12 rad/s checked follow motion."""
    settings = settings or HandFollowSettings()
    current_joints = np.asarray(current_joints, dtype=float)
    fixed_rotation = np.asarray(fixed_tool_rotation, dtype=float)
    camera_point = np.asarray(hand.camera_point, dtype=float)
    # Reject an excessive semantic request before FK/IK or any path planning.
    # The caller turns this exception into a latched HOLD state.
    camera_step = _camera_step(camera_point, settings)
    if float(np.linalg.norm(camera_step)) < 1e-9:
        return None
    current_tool, actual_rotation = planner.current_tool_pose(current_joints)
    current_tool = np.asarray(current_tool, dtype=float)
    actual_rotation = np.asarray(actual_rotation, dtype=float)
    if current_joints.shape != (6,) or fixed_rotation.shape != (3, 3):
        raise RuntimeError("invalid current pose for hand following")

    base_camera = _base_camera_transform(current_tool, actual_rotation, tcp_camera)
    hand_base = (base_camera @ np.append(camera_point, 1.0))[:3]
    base_step = base_camera[:3, :3] @ camera_step
    step_norm = float(np.linalg.norm(base_step))
    if step_norm > settings.max_tcp_step_m + 1e-9:
        raise RuntimeError("hand-follow Cartesian step exceeded 20 mm")

    target_tool = current_tool + base_step
    clearance = float(np.linalg.norm(hand_base - target_tool))
    if clearance < settings.min_tcp_hand_distance_m:
        raise RuntimeError(
            f"TCP-to-hand clearance {clearance:.3f} m is below the 0.20 m limit"
        )
    tcp_offset = fixed_rotation @ np.asarray(planner.config.tcp_in_joint6, dtype=float)
    target_joint6 = target_tool - tcp_offset
    solution = planner.validate_candidate(
        target_tool,
        target_joint6,
        fixed_rotation,
        (current_joints,),
        settings.max_joint_step_rad,
        label="hand-follow step",
    )
    if solution is None:
        raise RuntimeError("hand-follow IK or workspace validation failed")
    solution = np.asarray(solution, dtype=float)
    joint_delta = np.abs(solution - current_joints)
    if solution.shape != (6,) or not np.all(np.isfinite(solution)):
        raise RuntimeError("hand-follow IK returned an invalid solution")
    if float(np.max(joint_delta)) > settings.max_joint_step_rad + 1e-9:
        raise RuntimeError("hand-follow IK exceeded the per-step joint limit")

    lower = np.asarray(planner.config.joint_lower, dtype=float)
    upper = np.asarray(planner.config.joint_upper, dtype=float)
    home = np.asarray(planner.config.home, dtype=float)
    # The calibrated HOME J4 is itself only 0.085 rad above its configured
    # lower limit.  Preserve a 0.10 rad margin where physically available and,
    # for such a pre-approved HOME exception, never move farther toward the
    # limit than HOME already is.
    safe_lower = np.minimum(lower + settings.joint_limit_margin_rad, home)
    safe_upper = np.maximum(upper - settings.joint_limit_margin_rad, home)
    if np.any(solution < safe_lower - 1e-9) or np.any(solution > safe_upper + 1e-9):
        raise RuntimeError("hand-follow IK entered the joint-limit safety margin")

    endpoint_tool, endpoint_rotation, endpoint_clearance = (
        _validate_interpolated_joint_path(
            planner,
            current_joints,
            solution,
            current_tool,
            fixed_rotation,
            hand_base,
            safe_lower,
            safe_upper,
            settings,
        )
    )
    endpoint_error = float(np.linalg.norm(endpoint_tool - target_tool))
    rotation_error = _rotation_error_deg(endpoint_rotation, fixed_rotation)
    actual_step = float(np.linalg.norm(endpoint_tool - current_tool))
    if endpoint_error > settings.endpoint_position_tolerance_m:
        raise RuntimeError(f"hand-follow FK endpoint error is {endpoint_error:.3f} m")
    if rotation_error > settings.endpoint_rotation_tolerance_deg:
        raise RuntimeError(
            f"hand-follow FK orientation error is {rotation_error:.2f} deg"
        )
    if actual_step > settings.max_tcp_step_m + 5e-4:
        raise RuntimeError("hand-follow FK step exceeded the 20 mm command")
    if endpoint_clearance < settings.min_tcp_hand_distance_m:
        raise RuntimeError("hand-follow FK endpoint violates human clearance")

    duration = max(
        settings.min_step_duration_s,
        float(np.max(joint_delta)) / settings.max_joint_speed_rad_s,
    )
    if float(np.max(joint_delta / duration)) > settings.max_joint_speed_rad_s + 1e-9:
        raise RuntimeError("hand-follow joint speed limit was not satisfied")
    return FollowMotion(
        target_joints=solution,
        target_tool=endpoint_tool,
        hand_base=hand_base,
        camera_step=camera_step,
        duration_s=float(duration),
        tcp_hand_clearance_m=endpoint_clearance,
    )


class HandFollowController:
    """Robot-owning, stop--observe--step execution loop."""

    def __init__(
        self,
        planner,
        camera_feed,
        intrinsic,
        tcp_camera,
        detector: CpuYoloHandDetector,
        settings: HandFollowSettings | None = None,
        status_callback: Callable[[dict], None] | None = None,
        preview_callback: Callable[[dict, list[HandDetection]], None] | None = None,
    ) -> None:
        self.planner = planner
        self.camera_feed = camera_feed
        self.intrinsic = intrinsic
        self.tcp_camera = np.asarray(tcp_camera, dtype=float)
        self.detector = detector
        self.settings = settings or detector.settings
        self.status_callback = status_callback
        self.preview_callback = preview_callback
        self.gate = HandTrackGate(self.settings)
        self.state = HandFollowState.IDLE
        self.completed_steps = 0

    def _emit(self, message: str, **extra) -> None:
        if self.status_callback is None:
            return
        payload = {
            "follow_state": self.state.value,
            "message": str(message),
            "completed_steps": int(self.completed_steps),
        }
        payload.update(extra)
        self.status_callback(payload)

    def _interrupted(self) -> bool:
        event = getattr(self.planner, "interrupted", None)
        return bool(event is not None and event.is_set())

    def _hold(self, joints, reason: str) -> None:
        result = self.planner.refresh_arm_hold(np.asarray(joints, dtype=float))
        if result is False:
            raise RuntimeError("robot rejected the hand-follow HOLD command")
        self.state = self.gate.state
        self._emit(reason, hand_seen=False)

    def _wait_paused_until_disabled(self, mode_enabled: Callable[[], bool], joints) -> None:
        """Remain stationary; PAUSED never resumes without an off/on cycle."""
        next_hold = 0.0
        while mode_enabled() and not self._interrupted():
            now = time.monotonic()
            if now >= next_hold:
                self._hold(joints, self.gate.reason)
                next_hold = now + 0.10
            time.sleep(0.02)

    def _return_home(self) -> None:
        current = np.asarray(self.planner.current_joint_position(), dtype=float)
        home = np.asarray(self.planner.config.home, dtype=float)
        if float(np.max(np.abs(current - home))) <= 1e-3:
            self.planner.refresh_arm_hold(home)
            return
        self.planner.move_j(
            home,
            self.planner.config.return_home_duration,
            "HAND FOLLOW HOME",
        )

    def _wait_for_fresh_snapshot(self):
        """Prefer a raw post-stop capture; support older CameraFeed versions."""
        capture_marker = getattr(self.camera_feed, "capture_freshness_marker", None)
        wait_capture = getattr(self.camera_feed, "wait_for_new_capture", None)
        if callable(capture_marker) and callable(wait_capture):
            marker = int(capture_marker())
            return wait_capture(marker, timeout=self.settings.frame_timeout_s)

        marker = self.camera_feed.freshness_marker()
        return self.camera_feed.wait_for_newer(
            marker,
            timeout=self.settings.frame_timeout_s,
        )

    def _detect_while_holding(
        self,
        snapshot,
        current,
        mode_enabled: Callable[[], bool],
    ) -> tuple[list[HandDetection], bool]:
        """Run CPU-only vision in a worker while robot HOLD stays on main."""
        outcome = {}
        completed = threading.Event()

        def infer() -> None:
            try:
                outcome["candidates"] = self.detector.detect(
                    snapshot["color_image"],
                    snapshot["depth_image"],
                    self.camera_feed.depth_scale,
                    self.intrinsic,
                )
            except BaseException as exc:  # re-raised on the robot-owning thread
                outcome["error"] = exc
            finally:
                completed.set()

        result = self.planner.refresh_arm_hold(current)
        if result is False:
            raise RuntimeError("robot rejected HOLD before hand inference")
        worker = threading.Thread(
            target=infer,
            name="hand-follow-cpu-vision",
            daemon=True,
        )
        worker.start()
        cancelled = False
        hold_error = None
        while not completed.wait(self.settings.inference_hold_period_s):
            try:
                result = self.planner.refresh_arm_hold(current)
                if result is False and hold_error is None:
                    hold_error = RuntimeError(
                        "robot rejected HOLD during hand inference"
                    )
            except Exception as exc:
                if hold_error is None:
                    hold_error = RuntimeError(
                        f"robot HOLD failed during hand inference: {exc}"
                    )
            if not mode_enabled() or self._interrupted():
                cancelled = True

        worker.join()
        try:
            result = self.planner.refresh_arm_hold(current)
            if result is False and hold_error is None:
                hold_error = RuntimeError("robot rejected HOLD after hand inference")
        except Exception as exc:
            if hold_error is None:
                hold_error = RuntimeError(
                    f"robot HOLD failed after hand inference: {exc}"
                )
        if hold_error is not None:
            raise hold_error
        if "error" in outcome:
            raise outcome["error"]
        candidates = list(outcome.get("candidates", ()))
        return candidates, cancelled

    def run(self, mode_enabled: Callable[[], bool]) -> HandFollowResult:
        """Run in the robot-owning main thread until the operator disables it."""
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("hand-follow controller must run in the main thread")
        current = self.planner.wait_until_stationary()
        if current is None:
            raise RuntimeError("robot is not stationary before hand-follow arming")
        current = np.asarray(current, dtype=float)
        home = np.asarray(self.planner.config.home, dtype=float)
        if float(np.max(np.abs(current - home))) > self.settings.home_tolerance_rad:
            raise RuntimeError("hand-follow mode may only be armed from HOME")
        _, fixed_rotation = self.planner.current_tool_pose(current)
        fixed_rotation = np.asarray(fixed_rotation, dtype=float)

        self.completed_steps = 0
        self.gate.reset()
        self.state = self.gate.state
        self._emit("随动模式正在确认唯一且稳定的人手。", hand_seen=False)
        reason = "operator disabled hand following"
        try:
            while mode_enabled() and not self._interrupted():
                stable = self.planner.wait_until_stationary()
                if stable is None:
                    self.gate.miss("robot did not become stationary")
                    self._hold(current, self.gate.reason)
                    if self.gate.state == HandFollowState.PAUSED:
                        self._wait_paused_until_disabled(mode_enabled, current)
                        reason = self.gate.reason
                        break
                    continue
                current = np.asarray(stable, dtype=float)
                snapshot = self._wait_for_fresh_snapshot()
                if snapshot is None:
                    self.gate.miss("camera frame timed out")
                    self._hold(current, self.gate.reason)
                    if self.gate.state == HandFollowState.PAUSED:
                        self._wait_paused_until_disabled(mode_enabled, current)
                        reason = self.gate.reason
                        break
                    continue
                candidates, cancelled = self._detect_while_holding(
                    snapshot,
                    current,
                    mode_enabled,
                )
                if cancelled or not mode_enabled() or self._interrupted():
                    break
                if self.preview_callback is not None:
                    try:
                        self.preview_callback(snapshot, candidates)
                    except Exception as exc:
                        self._emit(
                            f"hand preview update failed: {exc}",
                            preview_error=True,
                        )
                now = time.monotonic()
                age = now - float(snapshot["timestamp"])
                if age > self.settings.max_snapshot_age_s:
                    self.gate.miss(f"camera frame is stale ({age:.2f}s)", now)
                    self._hold(current, self.gate.reason)
                    if self.gate.state == HandFollowState.PAUSED:
                        self._wait_paused_until_disabled(mode_enabled, current)
                        reason = self.gate.reason
                        break
                    continue
                diagnostics = getattr(self.detector, "last_diagnostics", {})
                observed_count = diagnostics.get("matching_count", len(candidates))
                hand = self.gate.update(
                    candidates,
                    now,
                    observed_count=observed_count,
                )
                self.state = self.gate.state
                if hand is None:
                    self._hold(current, self.gate.reason)
                    if self.gate.state == HandFollowState.PAUSED:
                        self._wait_paused_until_disabled(mode_enabled, current)
                        reason = self.gate.reason
                        break
                    self._emit(
                        self.gate.reason,
                        hand_seen=bool(candidates),
                        snapshot_age_s=age,
                    )
                    continue

                try:
                    motion = plan_bounded_follow_step(
                        self.planner,
                        current,
                        fixed_rotation,
                        hand,
                        self.tcp_camera,
                        self.settings,
                    )
                except Exception as exc:
                    self.gate.pause(f"motion rejected: {exc}")
                    self.state = self.gate.state
                    self._hold(current, self.gate.reason)
                    self._wait_paused_until_disabled(mode_enabled, current)
                    reason = self.gate.reason
                    break
                if motion is None:
                    self.planner.refresh_arm_hold(current)
                    self._emit(
                        "人手已在随动死区内，机械臂保持。",
                        hand_seen=True,
                        snapshot_age_s=age,
                        hand_depth_m=hand.depth_m,
                    )
                    continue

                self.planner.move_j(
                    motion.target_joints,
                    motion.duration_s,
                    "HAND FOLLOW 20MM STEP",
                    position_tolerance=0.02,
                )
                self.completed_steps += 1
                self.state = HandFollowState.TRACKING
                self._emit(
                    "随动小步完成。",
                    hand_seen=True,
                    snapshot_age_s=age,
                    hand_depth_m=hand.depth_m,
                    step_mm=float(np.linalg.norm(motion.camera_step)) * 1000.0,
                    tcp_hand_distance_m=motion.tcp_hand_clearance_m,
                )
            if self._interrupted():
                reason = "global shutdown requested"
                self.state = HandFollowState.STOPPING
                self.planner.refresh_arm_hold(current)
                return HandFollowResult(self.state, reason, self.completed_steps)

            self.state = HandFollowState.STOPPING
            self._emit("随动模式停止，机械臂正在返回 HOME。")
            self._return_home()
            self.state = HandFollowState.IDLE
            self._emit("随动模式已停止，机械臂已回到 HOME。")
            return HandFollowResult(self.state, reason, self.completed_steps)
        except Exception as exc:
            self.state = HandFollowState.FAULT
            try:
                hold = np.asarray(self.planner.current_joint_position(), dtype=float)
                self.planner.refresh_arm_hold(hold)
            except Exception:
                pass
            self._emit(f"随动模式故障并已请求保持：{exc}")
            raise
