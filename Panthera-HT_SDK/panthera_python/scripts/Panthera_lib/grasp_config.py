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
    stream_jpeg_quality: int = 92
    use_voice: bool = False
    voice_asr_model_dir: str = ""
    voice_tts_model_dir: str = ""
    voice_prompt_duration: float = 3.5
    camera_detection_interval: float = 0.02
    camera_detection_timeout: float = 3.0
    camera_performance_log_interval: float = 5.0
    detection_stationary_velocity_tolerance: float = 0.05
    detection_stationary_joint_tolerance: float = 0.005
    detection_stationary_stable_samples: int = 3
    detection_stationary_timeout: float = 2.0
    camera_serial: str = ""
    central_x_grasp_ratio: float = 0.80

    # Qualcomm QNN HTP NPU detector.  Disabled by default so the CPU YOLOE
    # path remains the safe fallback.
    use_npu: bool = True
    npu_server_path: str = "third_party/qnn/npu_server"
    npu_context_path: str = "third_party/qnn/yoloe-26s-seg_640_iq9075_qnn_block4.bin"
    npu_input_size: int = 640
    npu_output_specs: str = "output_0:1,300,38;output_1:1,32,160,160"
    # block4 was validated on the current four-block scene.  Its weakest true
    # block scored 0.157, while the obsolete colour-prompt brick6 context
    # returned no candidates even at 0.15.
    npu_confidence_threshold: float = 0.15
    npu_iou_threshold: float = 0.45
    npu_pre_nms_top_k: int = 50
    npu_max_detections: int = 20
    npu_response_timeout: float = 10.0
    npu_stderr_max_lines: int = 200

    # Default grasp backend is the OBB / Seeed geometric planner.
    # GraspNet is opt-in only via the GRASPNET_USE environment switch.
    use_graspnet: bool = False
    graspnet_repo_path: str = "third_party/graspnet-baseline"
    graspnet_api_path: str = "third_party/graspnetAPI"
    graspnet_checkpoint_path: str = "third_party/graspnet-baseline/checkpoint-rs.tar"
    graspnet_num_point: int = 1024
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
    pre_grasp_offset_m: float = 0.05
    pre_grasp_min_distance_m: float = 0.02
    pre_grasp_lateral_tolerance_m: float = 0.015
    pre_grasp_orientation_tolerance_deg: float = 8.0
    pre_grasp_realign_tolerance_m: float = 0.010
    pre_grasp_abort_shift_m: float = 0.030
    approach_path_lateral_tolerance_m: float = 0.005
    approach_endpoint_tolerance_m: float = 0.015
    pre_grasp_duration: float = 3.0
    pre_grasp_camera_settle_time: float = 0.40
    graspnet_z_offset_m: float = -0.20
    graspnet_max_joint_jump: float = 2.6
    graspnet_scene_expand_px: int = 80
    graspnet_target_radius_m: float = 0.12
    graspnet_approach_max_angle_deg: float = 80.0

    width: int = 640
    height: int = 480
    fps: int = 30
    confidence_threshold: float = 0.30
    min_depth_m: float = 0.07
    max_depth_m: float = 1.00

    color_min_saturation: int = 60
    color_min_value: int = 50
    # Screenshot calibration: yellow top H~=27, lime-green top H~=42.
    color_yellow_green_boundary: int = 38
    color_dominant_ratio: float = 0.45
    color_min_margin: float = 0.12
    color_dark_ratio: float = 0.60
    color_white_ratio: float = 0.60
    color_core_fraction: float = 0.30
    color_min_core_pixels: int = 20
    color_accumulation_min_frames: int = 2
    color_accumulation_max_frames: int = 5
    color_accumulation_min_samples: int = 80
    color_single_frame_strong_ratio: float = 0.60
    color_track_position_tolerance_m: float = 0.020

    depth_surface_percentile: float = 30.0
    depth_surface_band_m: float = 0.012
    depth_min_surface_pixels: int = 30
    depth_max_surface_spread_m: float = 0.020

    home: np.ndarray = field(
        default_factory=lambda: np.array([0.000, 0.240, 1.200, -1.515, 0.0, 0.0], dtype=float)
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
        default_factory=lambda: np.array([0.165, 0.0, 0.0], dtype=float)
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
        default_factory=lambda: np.array([0.150, 0.040, -0.070], dtype=float)
    )
    approach_policy: str = "seeed_safe"
    max_dynamic_approach_tilt_deg: float = 25.0
    gripper_open_axis_offset_deg: float = 0.0

    direct_grasp_duration: float = 5.0
    direct_grasp_post_command_wait: float = 1.0
    direct_grasp_settle_timeout: float = 4.0
    ik_position_tolerance_m: float = 0.03
    ik_rotation_tolerance_deg: float = 8.0
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

    # The current gripper zero calibration keeps the vendor closed=0.0 rad
    # convention; grasp_demo15.py measured full-open at 1.8 rad after reset.
    gripper_limit_lower: float = 0.0
    gripper_limit_upper: float = 2.0
    gripper_open_position: float = 1.8
    gripper_open_velocity: float = 0.5
    gripper_open_torque: float = 0.5
    gripper_open_timeout: float = 4.0
    gripper_open_position_tolerance: float = 0.08
    gripper_close_position: float = 0.0
    gripper_close_velocity: float = 0.35
    gripper_close_torque: float = 0.8
    gripper_clamped_position: float = 0.22
    gripper_clamp_torque: float = 0.4
    gripper_close_timeout: float = 6.0
    gripper_close_attempts: int = 2
    grasp_min_force: float = 0.5
    grasp_max_attempts: int = 2

    tool_x_range: tuple[float, float] = (0.10, 0.60)
    tool_y_range: tuple[float, float] = (-0.45, 0.45)
    tool_z_range: tuple[float, float] = (0.00, 0.30)
    wrist_z_range: tuple[float, float] = (-0.15, 0.6)
    radial_range: tuple[float, float] = (0.10, 0.65)

    zero_duration: float = 4.0
    zero_move_timeout: float = 12.0
    zero_position_tolerance: float = 0.10
    zero_velocity_tolerance: float = 1.0
    zero_stable_samples: int = 3
    zero_verify_timeout: float = 3.0
    zero_settle_time: float = 0.8
    shutdown_fault_hold_time: float = 0.25

    scan_j1_start: float = 1.80
    scan_j1_end: float = -1.80
    scan_j1_step: float = 0.30
    scan_start_duration: float = 6.0
    scan_step_duration: float = 2.0
    scan_camera_settle_time: float = 0.30
    scan_frame_warmup: int = 1

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
        if not (
            self.gripper_close_position
            <= self.gripper_clamped_position
            < self.gripper_open_position
        ):
            raise ValueError("gripper clamped threshold must lie between closed and open")
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
        for name, pose in (("HOME", self.home), ("PUT1", self.put1), ("PUT2", self.put2)):
            if np.any(pose < self.joint_lower) or np.any(pose > self.joint_upper):
                raise ValueError(f"{name} exceeds the configured joint limits")
        if not (
            self.joint_lower[0] <= self.scan_j1_end < self.scan_j1_start <= self.joint_upper[0]
            and self.scan_j1_step > 0.0
        ):
            raise ValueError("invalid J1 scan range or step")
        if self.scan_frame_warmup < 1:
            raise ValueError("scan_frame_warmup must be at least 1")
        color_ratios = (
            self.color_dominant_ratio,
            self.color_min_margin,
            self.color_dark_ratio,
            self.color_white_ratio,
            self.color_core_fraction,
            self.color_single_frame_strong_ratio,
        )
        if not all(0.0 < value <= 1.0 for value in color_ratios):
            raise ValueError("color ratios must be in (0, 1]")
        if not (
            1 <= self.color_accumulation_min_frames <= self.color_accumulation_max_frames
            and self.color_min_core_pixels > 0
            and self.color_accumulation_min_samples > 0
            and self.color_track_position_tolerance_m > 0.0
        ):
            raise ValueError("invalid color accumulation settings")
        if not 12 < self.color_yellow_green_boundary < 85:
            raise ValueError("yellow/green hue boundary must be between 12 and 85")
        if (
            self.pre_grasp_offset_m <= 0.0
            or not 0.0 < self.pre_grasp_min_distance_m <= self.pre_grasp_offset_m
            or self.pre_grasp_lateral_tolerance_m <= 0.0
            or self.pre_grasp_orientation_tolerance_deg <= 0.0
            or not 0.0 < self.pre_grasp_realign_tolerance_m < self.pre_grasp_abort_shift_m
            or self.approach_path_lateral_tolerance_m <= 0.0
            or self.approach_endpoint_tolerance_m <= 0.0
            or self.pre_grasp_duration <= 0.0
            or self.pre_grasp_camera_settle_time < 0.0
        ):
            raise ValueError("invalid pre-grasp settings")
        if not (
            0.0 < self.depth_surface_percentile < 50.0
            and self.depth_surface_band_m > 0.0
            and self.depth_min_surface_pixels > 0
            and self.depth_max_surface_spread_m > 0.0
        ):
            raise ValueError("invalid depth surface settings")
        if not (
            self.camera_detection_interval >= 0.0
            and self.camera_performance_log_interval > 0.0
            and self.detection_stationary_velocity_tolerance > 0.0
            and self.detection_stationary_joint_tolerance > 0.0
            and self.detection_stationary_stable_samples > 0
            and self.detection_stationary_timeout > 0.0
        ):
            raise ValueError("invalid camera timing or stationary settings")
        if self.zero_move_timeout <= 0.0:
            raise ValueError("zero_move_timeout must be positive")
        if (
            self.npu_response_timeout <= 0.0
            or self.npu_stderr_max_lines < 10
            or not 0.0 < self.npu_iou_threshold < 1.0
            or self.npu_pre_nms_top_k < self.npu_max_detections
            or self.npu_max_detections < 1
        ):
            raise ValueError("invalid NPU timeout or stderr history limit")


COLOR_ALIASES = {
    "red": ("red", "\u7ea2"),
    "yellow": ("yellow", "\u9ec4"),
    "blue": ("blue", "\u84dd"),
    "green": ("green", "\u7eff"),
    "white": ("white", "\u767d"),
    "black": ("black", "\u9ed1"),
}
ANY_COLOR_ALIASES = ("any", "all", "\u4efb\u610f", "\u6240\u6709", "\u5168\u90e8")
NEGATION_PREFIXES = (
    "不要",
    "别抓",
    "别要",
    "排除",
    "不是",
    "不抓",
    "not",
    "no",
)


def _token_is_negated(normalized: str, token_index: int) -> bool:
    """Return whether a colour token is immediately negated by its prefix."""
    prefix = normalized[max(0, token_index - 6) : token_index]
    return any(prefix.endswith(negation) for negation in NEGATION_PREFIXES)


def parse_target_command(command: str):
    """Return ``(color_name_or_None, accepted)`` from a terminal/speech command."""
    normalized = command.strip().lower().replace(" ", "")
    if not normalized:
        return None, False

    any_positions = [
        normalized.find(token)
        for token in ANY_COLOR_ALIASES
        if token in normalized
    ]
    if any(
        position >= 0 and not _token_is_negated(normalized, position)
        for position in any_positions
    ):
        return None, True

    positive_colors: list[tuple[int, str]] = []
    for name, aliases in COLOR_ALIASES.items():
        for alias in aliases:
            start = 0
            while True:
                index = normalized.find(alias, start)
                if index < 0:
                    break
                if not _token_is_negated(normalized, index):
                    positive_colors.append((index, name))
                start = index + len(alias)

    if not positive_colors:
        return None, False
    positive_colors.sort(key=lambda item: item[0])
    return positive_colors[0][1], True
