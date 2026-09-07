"""Vision-in-the-loop target detection for rendered RGB-D frames."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import cv2

from .mujoco_backend import RgbdFrame


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    pixel_uv: tuple[float, float]
    depth_m: float
    position_world: np.ndarray
    source: str


def detect_coloured_block(
    frame: RgbdFrame,
    *,
    colour: str = "green",
    object_length_m: float = 0.060,
    object_height_m: float = 0.030,
    support_height_m: float = 0.0,
) -> list[Detection]:
    """Detect one known-colour 3×3×6 cm block without reading MuJoCo truth.

    RGB-D observes the camera-facing surface.  The simulated workcell contract
    supplies the block size and support plane, just as a real grasp profile
    supplies target dimensions, so the returned point is the object centre.
    """
    rgb = frame.rgb.astype(np.int16)
    masks = {
        "red": (rgb[..., 0] > 85) & (rgb[..., 1] < 75) & (rgb[..., 2] < 75),
        "green": (rgb[..., 0] < 55) & (rgb[..., 1] > 80) & (rgb[..., 2] < 70),
        "blue": (rgb[..., 0] < 75) & (rgb[..., 1] < 100) & (rgb[..., 2] > 85),
    }
    if colour not in masks:
        raise ValueError(f"colour must be one of {tuple(masks)}")
    mask = masks[colour]
    # The table is more than 20 cm from the wrist camera at HOME.  This rejects
    # red TCP/debug markers and near-end-effector pixels before component search.
    mask &= np.isfinite(frame.depth_m) & (frame.depth_m > 0.20)
    component_count, labels, stats, _centres = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if component_count <= 1:
        return []
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    rows, cols = np.nonzero(labels == largest_label)
    if rows.size < 40:
        return []
    # A single-object scene makes robust connected-components unnecessary here.
    depth_m = float(np.median(frame.depth_m[rows, cols]))
    u = float(np.median(cols))
    v = float(np.median(rows))
    x = (u - frame.intrinsics.cx) * depth_m / frame.intrinsics.fx
    y = -(v - frame.intrinsics.cy) * depth_m / frame.intrinsics.fy
    point_camera = np.array([x, y, -depth_m], dtype=float)
    surface_world = (
        frame.camera_position_world + frame.camera_rotation_world @ point_camera
    )
    point_world = surface_world.copy()
    # The randomized blocks remain within ±0.15 rad of the calibrated X axis.
    # Move from the camera-facing -X surface to the known geometric centre.
    point_world[0] += 0.5 * object_length_m
    point_world[2] = support_height_m + 0.5 * object_height_m
    return [
        Detection(
            label=f"{colour} toy building block",
            confidence=min(0.99, 0.50 + rows.size / 6000.0),
            pixel_uv=(u, v),
            depth_m=depth_m,
            position_world=point_world,
            source="rendered_rgbd_colour",
        )
    ]


def detect_green_block(
    frame: RgbdFrame,
    *,
    object_size_m: float = 0.060,
    support_height_m: float = 0.0,
) -> list[Detection]:
    """Backward-compatible green-target entry point for existing callers."""
    return detect_coloured_block(
        frame,
        colour="green",
        object_length_m=object_size_m,
        object_height_m=0.030,
        support_height_m=support_height_m,
    )


def detect_with_ultralytics(
    frame: RgbdFrame,
    weights: str,
    *,
    confidence: float = 0.20,
) -> list[Detection]:
    """Run Ultralytics and recover a Base-frame centre from its best box.

    The bundled detector is a tiny synthetic-scene regression model.  A 0.20
    threshold retains its valid edge-of-training-distribution boxes; the
    following RGB/depth foreground check rejects unrelated low-confidence boxes.
    """
    from ultralytics import YOLO  # Imported lazily to preserve sim-only startup.

    # Ultralytics treats NumPy camera arrays as OpenCV BGR input.
    bgr = np.ascontiguousarray(frame.rgb[..., ::-1])
    result = YOLO(weights)(bgr, verbose=False, conf=confidence)[0]
    if result.boxes is None or len(result.boxes) == 0:
        return []
    best = int(result.boxes.conf.argmax().item())
    x1, y1, x2, y2 = result.boxes.xyxy[best].cpu().numpy().astype(int)
    x1 = int(np.clip(x1, 0, frame.intrinsics.width - 1))
    x2 = int(np.clip(x2, x1 + 1, frame.intrinsics.width))
    y1 = int(np.clip(y1, 0, frame.intrinsics.height - 1))
    y2 = int(np.clip(y2, y1 + 1, frame.intrinsics.height))
    crop_rgb = frame.rgb[y1:y2, x1:x2].astype(np.int16)
    crop_depth = frame.depth_m[y1:y2, x1:x2]
    foreground = (
        (crop_rgb[..., 0] < 80)
        & (crop_rgb[..., 1] > 65)
        & (crop_rgb[..., 1] > crop_rgb[..., 0] + 20)
        & np.isfinite(crop_depth)
        & (crop_depth > 0.02)
    )
    rows, cols = np.nonzero(foreground)
    if rows.size < 20:
        return []
    depth_m = float(np.median(crop_depth[rows, cols]))
    u = float(x1 + np.median(cols))
    v = float(y1 + np.median(rows))
    x = (u - frame.intrinsics.cx) * depth_m / frame.intrinsics.fx
    y = -(v - frame.intrinsics.cy) * depth_m / frame.intrinsics.fy
    point_camera = np.array([x, y, -depth_m], dtype=float)
    surface_world = (
        frame.camera_position_world + frame.camera_rotation_world @ point_camera
    )
    point_world = surface_world.copy()
    point_world[0] += 0.030
    point_world[2] = 0.015
    return [
        Detection(
            label="toy building block",
            confidence=float(result.boxes.conf[best].item()),
            pixel_uv=(u, v),
            depth_m=depth_m,
            position_world=point_world,
            source="ultralytics_yolo",
        )
    ]
