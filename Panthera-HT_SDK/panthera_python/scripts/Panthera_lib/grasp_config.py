"""Grasp workflow configuration and spoken/text target parsing."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class GraspConfig:
    """Application-specific constants for the visual grasping workflow."""

    sdk_scripts: str = ""
    robot_config: str = ""
    model_path: str = ""
    calibration_file: Path = Path("hand_eye_calibration.json")
    project_root: Path = field(default_factory=lambda: Path.cwd())
    text_encoder_path: Path = Path("mobileclip2_b.ts")
    stream_host: str = "0.0.0.0"
    stream_port: int = 8080
    stream_jpeg_quality: int = 85
    camera_detection_interval: float = 0.25
    camera_detection_timeout: float = 3.0
    central_x_grasp_ratio: float = 0.60

    # Qualcomm QNN HTP NPU detector.  Disabled by default so the CPU YOLOE
    # path remains the safe fallback.
    use_npu: bool = False
    npu_server_path: str = "third_party/qnn/npu_server"
    npu_context_path: str = "third_party/qnn/yoloe-26s-seg_640_iq9075_qnn_brick6.bin"
    npu_input_size: int = 640
    npu_output_specs: str = "output_0:1,300,38;output_1:1,32,160,160"
    npu_confidence_threshold: float = 0.05

    # GraspNet integration. Kept disabled by default so the existing OBB
    # grasp path remains the safety fallback while GraspNet is evaluated.
    use_graspnet: bool = False
    graspnet_repo_path: str = "third_party/graspnet-baseline"
    graspnet_api_path: str = "third_party/graspnetAPI"
    graspnet_checkpoint_path: str = "third_party/graspnet-baseline/checkpoint-rs.tar"
    graspnet_num_point: int = 1024
    graspnet_pc_radius: float = 1.0
    graspnet_score_threshold: float = 0.0
    graspnet_max_candidates: int = 20
    graspnet_nms_translation: float = 0.02
    graspnet_nms_rotation: float = 30.0
    graspnet_collision_voxel_size: float = 0.01
    graspnet_collision_thresh: float = 0.01
    graspnet_collision_approach_dist: float = 0.05
    graspnet_gripper_fix_rotation: np.ndarray = field(
        default_factory=lambda: np.eye(3, dtype=float),
        repr=False,
    )
    graspnet_pre_grasp_offset: float = 0.05
    graspnet_lift_distance: float = 0.08
    graspnet_manipulability_min: float = 0.0
    graspnet_max_joint_jump: float = 2.6
    graspnet_scene_expand_px: int = 80
    graspnet_target_radius_m: float = 0.12
    graspnet_approach_max_angle_deg: float = 35.0

    width: int = 640
    height: int = 480
    fps: int = 30
    confidence_threshold: float = 0.30
    min_depth_m: float = 0.07
    max_depth_m: float = 1.00

    color_min_saturation: int = 60
    color_min_value: int = 50
    color_dominant_ratio: float = 0.55
    color_dark_ratio: float = 0.70
    color_white_ratio: float = 0.70

    auto_grasp_stable_samples: int = 1
    auto_grasp_position_tolerance_m: float = 0.020

    home: np.ndarray = field(
        default_factory=lambda: np.array([0.000, 1.039, 1.787, -1.500, 0.0, 0.0], dtype=float)
    )
    put1: np.ndarray = field(
        default_factory=lambda: np.array([1.600, 1.300, 0.550, -0.300, 0.0, 0.0], dtype=float)
    )
    put2: np.ndarray = field(
        default_factory=lambda: np.array([1.500, 0.500, 0.560, -0.075, 0.0, 0.0], dtype=float)
    )
    zero: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=float))
    max_torque: list[float] = field(
        default_factory=lambda: [21.0, 36.0, 36.0, 21.0, 10.0, 10.0]
    )
    home_velocity: list[float] = field(default_factory=lambda: [0.4] * 6)

    tcp_in_joint6: np.ndarray = field(
        default_factory=lambda: np.array([0.14, 0.0, 0.0], dtype=float)
    )
    manual_grasp_rotation: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                [0.2102, -0.0070, 0.9776],
                [0.1167, 0.9930, -0.0180],
                [-0.9707, 0.1178, 0.2096],
            ],
            dtype=float,
        )
    )
    grasp_offset_base: np.ndarray = field(
        default_factory=lambda: np.array([0.070, 0.03, -0.100], dtype=float)
    )
    approach_policy: str = "seeed_safe"
    max_dynamic_approach_tilt_deg: float = 25.0
    gripper_open_axis_offset_deg: float = 0.0

    direct_grasp_duration: float = 5.0
    direct_grasp_post_command_wait: float = 1.0
    direct_grasp_settle_timeout: float = 2.0
    return_home_duration: float = 5.0
    put1_duration: float = 5.0
    put2_duration: float = 5.0
    max_home_to_grasp_step: float = 2.60

    joint_lower: np.ndarray = field(
        default_factory=lambda: np.array([-2.4, -0.05, -0.05, -1.6, -1.7, -2.5], dtype=float)
    )
    joint_upper: np.ndarray = field(
        default_factory=lambda: np.array([2.4, 3.2, 4.0, 1.6, 1.7, 2.5], dtype=float)
    )
    manual_grasp_ik_seed: np.ndarray = field(
        default_factory=lambda: np.array([0.159, 2.757, 1.915, -0.500, -0.082, 0.174], dtype=float)
    )

    gripper_limit_lower: float = -4.0
    gripper_limit_upper: float = 4.0
    gripper_open_position: float = -0.263
    gripper_open_velocity: float = 0.5
    gripper_open_torque: float = 0.5
    gripper_open_timeout: float = 4.0
    gripper_open_position_tolerance: float = 0.08
    gripper_close_position: float = -1.898
    gripper_close_velocity: float = 0.35
    gripper_close_torque: float = 0.8
    gripper_clamped_position: float = -1.70
    gripper_clamp_torque: float = 0.4
    gripper_close_timeout: float = 6.0
    gripper_close_attempts: int = 2
    grasp_min_force: float = 0.5
    grasp_max_attempts: int = 2

    tool_x_range: tuple[float, float] = (0.10, 0.7)
    tool_y_range: tuple[float, float] = (-0.45, 0.45)
    tool_z_range: tuple[float, float] = (-0.30, 0.10)
    wrist_z_range: tuple[float, float] = (-0.15, 0.6)
    radial_range: tuple[float, float] = (0.10, 0.7)

    zero_duration: float = 4.0
    zero_move_timeout: float = 12.0
    zero_position_tolerance: float = 0.10
    zero_velocity_tolerance: float = 1.0
    zero_stable_samples: int = 3
    zero_verify_timeout: float = 3.0
    zero_settle_time: float = 0.8

    scan_j1_start: float = 2.30
    scan_j1_end: float = -2.30
    scan_j1_step: float = 0.35
    scan_start_duration: float = 6.0
    scan_step_duration: float = 1.5
    scan_camera_settle_time: float = 0.30
    scan_frame_warmup: int = 2

    target_prompts: tuple[str, ...] = (
        "toy building block",
        "plastic building block",
        "wooden block",
        "Lego brick",
    )

    def validate(self) -> None:
        if self.approach_policy not in {"seeed_safe", "seeed_pure"}:
            raise ValueError("approach_policy must be 'seeed_safe' or 'seeed_pure'")
        if not np.all(
            (self.gripper_limit_lower <= np.array([self.gripper_open_position, self.gripper_close_position]))
            & (np.array([self.gripper_open_position, self.gripper_close_position]) <= self.gripper_limit_upper)
        ):
            raise ValueError("configured gripper positions exceed gripper limits")
        if not self.gripper_close_position < self.gripper_open_position:
            raise ValueError("expected closed gripper position to be below open position")
        if not np.allclose(
            self.manual_grasp_rotation.T @ self.manual_grasp_rotation, np.eye(3), atol=2e-3
        ):
            raise ValueError("manual_grasp_rotation is not orthonormal")
        if not np.isclose(np.linalg.det(self.manual_grasp_rotation), 1.0, atol=2e-3):
            raise ValueError("manual_grasp_rotation must be right handed")
        if not np.allclose(
            self.graspnet_gripper_fix_rotation.T @ self.graspnet_gripper_fix_rotation,
            np.eye(3),
            atol=2e-3,
        ):
            raise ValueError("graspnet_gripper_fix_rotation is not orthonormal")
        if not np.isclose(np.linalg.det(self.graspnet_gripper_fix_rotation), 1.0, atol=2e-3):
            raise ValueError("graspnet_gripper_fix_rotation must be right handed")
        for name, pose in (("PUT1", self.put1), ("PUT2", self.put2)):
            if np.any(pose < self.joint_lower) or np.any(pose > self.joint_upper):
                raise ValueError(f"{name} exceeds the configured joint limits")
        if not (
            self.joint_lower[0] <= self.scan_j1_end < self.scan_j1_start <= self.joint_upper[0]
            and self.scan_j1_step > 0.0
        ):
            raise ValueError("invalid J1 scan range or step")


COLOR_ALIASES = {
    "red": ("red", "\u7ea2"),
    "yellow": ("yellow", "\u9ec4"),
    "blue": ("blue", "\u84dd"),
    "green": ("green", "\u7eff"),
    "white": ("white", "\u767d"),
    "black": ("black", "\u9ed1"),
}
ANY_COLOR_ALIASES = ("any", "all", "\u4efb\u610f", "\u6240\u6709", "\u5168\u90e8")


def parse_target_command(command: str):
    """Return ``(color_name_or_None, accepted)`` from a terminal/speech command."""
    normalized = command.strip().lower().replace(" ", "")
    if not normalized:
        return None, False
    if any(token in normalized for token in ANY_COLOR_ALIASES):
        return None, True
    color = next(
        (name for name, aliases in COLOR_ALIASES.items() if any(alias in normalized for alias in aliases)),
        None,
    )
    return color, color is not None
