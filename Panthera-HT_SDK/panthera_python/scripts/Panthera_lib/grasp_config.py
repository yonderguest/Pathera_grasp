"""Grasp workflow configuration and spoken/text target parsing."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


OBJECT3_CLASSES = ("bottle", "box", "toy building block")
LEGACY_BLOCK4_CLASSES = (
    "toy building block",
    "plastic building block",
    "wooden block",
    "Lego brick",
)
CANONICAL_BLOCK_NAMES = frozenset(name.lower() for name in LEGACY_BLOCK4_CLASSES)


@dataclass(frozen=True)
class TargetRequest:
    """Canonical operator request carried through detection and refinement."""

    object_name: str | None = "toy building block"
    color: str | None = None

    @property
    def label(self) -> str:
        object_label = self.object_name or "any object"
        color_label = self.color or "any colour"
        return f"{color_label} {object_label}"


@dataclass(frozen=True)
class ObjectGraspProfile:
    """Object-specific manipulation contract.

    Recognition support and permission to move are intentionally separate.
    ``bottle`` and ``box`` remain plan-only until their physical dimensions,
    gripper aperture and load limits have been measured on the real hardware.
    ``None`` therefore means "not calibrated", never "unlimited".
    """

    object_name: str
    motion_enabled: bool = False
    plan_only: bool = True
    parameters_confirmed: bool = False
    preserve_legacy_pipeline: bool = False
    min_confidence: float = 0.35
    required_coherent_frames: int = 3
    gripper_open_position_rad: float | None = None
    gripper_opening_m: float | None = None
    gripper_close_torque_limit: float | None = None
    load_min_torque: float | None = None
    max_payload_kg: float | None = None
    approach_inset_m: float | None = None
    center_strategy: str = "unconfirmed"
    orientation_strategy: str = "unconfirmed"
    depth_strategy: str = "unconfirmed"
    max_dimensions_m: tuple[float, float, float] | None = None

    @property
    def missing_motion_parameters(self) -> tuple[str, ...]:
        """Return unconfirmed fields that block a new object motion profile."""
        if self.preserve_legacy_pipeline:
            return ()
        required = {
            "gripper_open_position_rad": self.gripper_open_position_rad,
            "gripper_opening_m": self.gripper_opening_m,
            "gripper_close_torque_limit": self.gripper_close_torque_limit,
            "load_min_torque": self.load_min_torque,
            "max_payload_kg": self.max_payload_kg,
            "approach_inset_m": self.approach_inset_m,
            "max_dimensions_m": self.max_dimensions_m,
        }
        missing = [name for name, value in required.items() if value is None]
        for name, value in (
            ("center_strategy", self.center_strategy),
            ("orientation_strategy", self.orientation_strategy),
            ("depth_strategy", self.depth_strategy),
        ):
            if not value or value == "unconfirmed":
                missing.append(name)
        return tuple(missing)

    @property
    def motion_ready(self) -> bool:
        return (
            self.motion_enabled
            and not self.plan_only
            and self.parameters_confirmed
            and not self.missing_motion_parameters
        )

    def validate(self) -> None:
        canonical = canonical_object_name(self.object_name)
        if canonical not in OBJECT3_CLASSES:
            raise ValueError(f"unsupported object grasp profile: {self.object_name!r}")
        if self.preserve_legacy_pipeline and canonical != "toy building block":
            raise ValueError(
                "preserve_legacy_pipeline is only valid for toy building block"
            )
        if self.motion_enabled and self.plan_only:
            raise ValueError(f"{canonical} cannot be both motion-enabled and plan-only")
        if self.motion_enabled and not self.motion_ready:
            missing = ", ".join(self.missing_motion_parameters) or "confirmation"
            raise ValueError(f"{canonical} profile is not motion-ready: {missing}")
        if not 0.0 < self.min_confidence <= 1.0:
            raise ValueError(f"{canonical} has invalid min_confidence")
        if self.required_coherent_frames < 1:
            raise ValueError(f"{canonical} has invalid required_coherent_frames")
        for name, value in (
            ("gripper_open_position_rad", self.gripper_open_position_rad),
            ("gripper_opening_m", self.gripper_opening_m),
            ("gripper_close_torque_limit", self.gripper_close_torque_limit),
            ("load_min_torque", self.load_min_torque),
            ("max_payload_kg", self.max_payload_kg),
            ("approach_inset_m", self.approach_inset_m),
        ):
            if value is not None and (not np.isfinite(value) or value < 0.0):
                raise ValueError(f"{canonical} has invalid {name}: {value!r}")
        if self.max_dimensions_m is not None:
            dimensions = np.asarray(self.max_dimensions_m, dtype=float)
            if (
                dimensions.shape != (3,)
                or not np.all(np.isfinite(dimensions))
                or np.any(dimensions <= 0.0)
            ):
                raise ValueError(f"{canonical} has invalid max_dimensions_m")


def canonical_object_name(name: str | None) -> str | None:
    if name is None:
        return None
    normalized = str(name).strip().lower().replace("_", " ").replace("-", " ")
    if normalized in CANONICAL_BLOCK_NAMES or normalized == "colour-region fallback":
        return "toy building block"
    return normalized


def normalize_target_request(value) -> TargetRequest:
    """Accept the new request object and legacy colour-string call sites."""
    if isinstance(value, TargetRequest):
        return value
    if value is None or isinstance(value, str):
        return TargetRequest(color=value)
    raise TypeError(f"unsupported target request: {type(value).__name__}")


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
    stream_preview_fps: float = 15.0
    use_voice: bool = False
    voice_asr_model_dir: str = ""
    voice_tts_model_dir: str = ""
    voice_prompt_duration: float = 3.5
    camera_detection_interval: float = 0.02
    camera_detection_timeout: float = 3.0
    camera_performance_log_interval: float = 5.0
    detection_stationary_velocity_tolerance: float = 0.12
    detection_stationary_joint_tolerance: float = 0.010
    detection_stationary_stable_samples: int = 3
    detection_stationary_timeout: float = 3.0
    # Maximum accepted hand-follow intent per observe cycle. The physical MoveJ
    # step is bounded separately; this is a closed-loop ceiling, not an
    # open-loop trajectory length.
    follow_max_semantic_step_m: float = 0.10
    follow_max_tcp_step_m: float = 0.020
    camera_serial: str = ""
    central_x_grasp_ratio: float = 0.80

    # Qualcomm QNN HTP NPU detector.  Disabled by default so the CPU YOLOE
    # path remains the safe fallback.
    use_npu: bool = True
    npu_server_path: str = "third_party/qnn/npu_server"
    recognition_profile: str = "object3"
    npu_context_path: str = "third_party/qnn/yoloe-26s-seg_640_iq9075_qnn_object3.bin"
    npu_input_size: int = 640
    npu_output_specs: str = "output_0:1,300,38;output_1:1,32,160,160"
    # The reference used 0.10, but a camera-only smoke test showed 3.8 FPS and
    # many low-score bottle/box false positives.  0.20 retained both visible
    # blocks while reaching about 7 FPS on the deployed IQ9075.
    npu_confidence_threshold: float = 0.20
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
    # Eye-in-hand observation move. Preserve the scan-camera orientation and
    # correct only part of the measured image/range error so the target remains
    # visible instead of jumping directly to the final grasp orientation.
    observation_centering_gain: float = 0.50
    observation_axial_gain: float = 0.50
    observation_min_advance_m: float = 0.008
    observation_max_advance_m: float = 0.025
    observation_max_lateral_shift_m: float = 0.035
    observation_max_translation_m: float = 0.045
    observation_min_camera_distance_m: float = 0.12
    observation_max_joint_step: float = 0.75

    # Final approach standoff. Half of the current signed axial gap is used,
    # then clamped so neither a tiny nor an excessively remote waypoint is
    # generated. This is intentionally not half of the 3-D Euclidean distance.
    pre_grasp_standoff_ratio: float = 0.50
    pre_grasp_min_distance_m: float = 0.015
    pre_grasp_max_distance_m: float = 0.040
    pre_grasp_lateral_tolerance_m: float = 0.015
    pre_grasp_orientation_tolerance_deg: float = 8.0
    # MoveJ only needs this tighter convergence at the final free-space
    # waypoint.  The vendor default (0.05 rad) can leave 20--30 mm of TCP
    # lateral error, which is too large for the following straight approach.
    # 0.02 rad proved tighter than the real motor can repeat reliably and
    # caused otherwise valid pre-grasp moves to time out.  The Cartesian
    # realignment below handles the remaining TCP error explicitly.
    pre_grasp_joint_tolerance_rad: float = 0.040
    # If real feedback is still outside the straight-approach corridor, first
    # retreat/re-align to the commanded standoff with a short Cartesian path.
    pre_grasp_realign_max_translation_m: float = 0.045
    pre_grasp_realign_max_orientation_deg: float = 12.0
    pre_grasp_realign_endpoint_tolerance_m: float = 0.010
    pre_grasp_realign_duration: float = 2.5
    # Cartesian paths are planned sparsely at eef_step resolution, but the
    # MIT position/velocity controller must be refreshed continuously.  The
    # executor performs shape-preserving interpolation at this period and
    # starts only while feedback still matches the validated first sample.
    trajectory_control_period_s: float = 0.020
    trajectory_start_joint_tolerance_rad: float = 0.025
    trajectory_start_tcp_tolerance_m: float = 0.010
    stationary_hold_period_s: float = 0.050
    stationary_hold_max_drift_rad: float = 0.025
    # Maximum deviation from the commanded start-to-target Cartesian segment.
    # The separate pre-grasp gate still limits lateral error relative to the
    # tool approach axis; do not use this value to reject a path merely because
    # that path is converging from a small residual lateral offset.
    approach_path_lateral_tolerance_m: float = 0.007
    approach_endpoint_tolerance_m: float = 0.015
    pre_grasp_duration: float = 3.0
    pre_grasp_camera_settle_time: float = 0.40
    refine_frame_warmup: int = 3
    refine_required_observations: int = 3
    refine_max_frame_attempts: int = 8
    refine_total_timeout_s: float = 4.0
    refine_frame_timeout_s: float = 0.8
    refine_target_match_radius_m: float = 0.050
    # The observation move is deliberately capped and may not fully centre a
    # far-edge target; Base-coordinate identity remains the primary gate.
    refine_max_center_distance_px: float = 220.0
    refine_fallback_min_area_px: int = 80
    refine_fallback_max_area_px: int = 50000
    refine_max_position_spread_m: float = 0.012
    refine_max_xy_correction_m: float = 0.035
    refine_max_z_correction_m: float = 0.020
    refine_max_total_correction_m: float = 0.035
    refine_max_approach_tilt_deg: float = 30.0
    refine_max_open_axis_change_deg: float = 30.0
    # One final, position-only visual correction at the collision-separated
    # pre-grasp waypoint.  The close image may be magnified or partly occluded,
    # so never recompute wrist orientation there and accept only a small,
    # coherent, fully visible RGB-D correction.  The latest physical check
    # found no repeatable Base-Y bias, so close-range vision may correct Base-X
    # and Base-Z but preserves the far-field Base-Y coordinate.
    close_refine_enabled: bool = True
    close_refine_preserve_base_y: bool = True
    close_refine_min_depth_m: float = 0.075
    close_refine_border_margin_px: int = 6
    close_refine_max_bbox_area_ratio: float = 0.30
    close_refine_max_depth_spread_m: float = 0.015
    close_refine_max_position_spread_m: float = 0.009
    close_refine_max_xy_correction_m: float = 0.035
    close_refine_max_z_correction_m: float = 0.018
    close_refine_max_total_correction_m: float = 0.035
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
    color_accumulation_timeout_s: float = 1.0
    color_accumulation_min_samples: int = 80
    color_single_frame_strong_ratio: float = 0.60
    color_track_position_tolerance_m: float = 0.020
    color_classifier_backend: str = "lab"
    color_calibration_file: Path = Path("config/color_calibration.json")
    color_calibration_model: dict | None = field(default=None, repr=False)
    # Current auto exposure shifts valid red/blue Lab centres outside the
    # calibration's conservative one-frame radius.  Expand only the distance
    # gate; the nearest-class margin still rejects yellow/green ambiguity.
    color_lab_distance_scale: float = 2.5

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
    # Residual physical correction after the calibrated Camera->TCP chain is
    # composed explicitly.  Keep all fixed Base-axis compensation at zero;
    # any accepted close-range correction is measured dynamically from fresh
    # RGB-D frames and is never persisted as a calibration constant.
    grasp_offset_base: np.ndarray = field(
        default_factory=lambda: np.array([0.000, 0.000, 0.000], dtype=float)
    )
    # Optional tool-axis overtravel.  A 5 mm trial moved visibly in the intended
    # direction but enlarged the total miss because the tilted tool axis also
    # introduced a large downward component.  Keep the mechanism available for
    # future calibration, but disable it while close-range visual position
    # refinement supplies the final correction.
    grasp_approach_overtravel_m: float = 0.000
    approach_policy: str = "seeed_safe"
    max_dynamic_approach_tilt_deg: float = 25.0
    gripper_open_axis_offset_deg: float = 0.0

    # 15..40 mm final approach: 5.0 s was visibly too slow.  3.5 s shortens
    # the move by 30% while the dense executor still enforces configured joint
    # velocity and acceleration limits before sending a command.
    direct_grasp_duration: float = 3.5
    grasp_retry_retreat_duration: float = 2.0
    direct_grasp_post_command_wait: float = 1.0
    direct_grasp_settle_timeout: float = 4.0
    direct_grasp_joint_tolerance_rad: float = 0.050
    direct_grasp_tcp_tolerance_m: float = 0.012
    ik_position_tolerance_m: float = 0.03
    ik_rotation_tolerance_deg: float = 8.0
    # The planner already supplies ordered seeds. Avoid starting another
    # eight-seed search inside every outer seed attempt.
    ik_single_seed_max_iterations: int = 600
    ik_max_seed_attempts: int = 3
    move_wait_min_timeout_s: float = 6.0
    move_wait_margin_s: float = 4.0
    # Joint-space moves use the same continuously refreshed MIT transport as
    # Cartesian moves.  A one-shot Joint_Pos_Vel command can be consumed by the
    # motor watchdog before HOME/PUT has made meaningful progress.
    joint_move_settle_timeout_s: float = 1.0
    joint_move_max_endpoint_error_rad: float = 0.20
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
    # Manual web jog is only accepted while the target prompt is idle. Positive
    # J1 is defined as the operator's left arrow; negative J1 is right.
    joint1_jog_step_rad: float = 0.5
    joint1_jog_duration: float = 3.0
    joint1_jog_posture_tolerance_rad: float = 0.20

    target_prompts: tuple[str, ...] = OBJECT3_CLASSES
    # These names are baked into the deployed object3 QNN context in this exact
    # order. Changing them requires recompiling the context, not only Python.
    npu_class_names: tuple[str, ...] = OBJECT3_CLASSES
    # Recognition and manipulation eligibility are deliberately separate.
    # Bottle/box geometry has not yet passed gripper-width/load validation.
    graspable_object_names: tuple[str, ...] = ("toy building block",)
    object_grasp_profiles: dict[str, ObjectGraspProfile] = field(
        default_factory=dict,
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.object_grasp_profiles:
            return
        # The block entry is a read-only description of the already validated
        # path.  Its values are copied from the existing top-level fields so
        # introducing profiles cannot silently change the physical behaviour.
        self.object_grasp_profiles = {
            "toy building block": ObjectGraspProfile(
                object_name="toy building block",
                motion_enabled=True,
                plan_only=False,
                parameters_confirmed=True,
                preserve_legacy_pipeline=True,
                # Metadata only for the legacy block path: one strong object3
                # frame at the existing 0.20 threshold remains sufficient.
                min_confidence=self.npu_confidence_threshold,
                required_coherent_frames=1,
                gripper_open_position_rad=self.gripper_open_position,
                gripper_close_torque_limit=self.gripper_close_torque,
                load_min_torque=self.grasp_min_force,
                approach_inset_m=self.grasp_approach_overtravel_m,
                center_strategy="segmentation_obb_center",
                orientation_strategy="mask_obb_short_axis",
                depth_strategy="coherent_near_surface",
            ),
            # Physical aperture, payload, insertion and size limits have not
            # been supplied.  Keep both generic classes visible to recognition
            # while making accidental motion enablement fail configuration.
            "bottle": ObjectGraspProfile(object_name="bottle"),
            "box": ObjectGraspProfile(object_name="box"),
        }

    def object_grasp_profile(self, object_name: str) -> ObjectGraspProfile:
        canonical = canonical_object_name(object_name)
        try:
            return self.object_grasp_profiles[canonical]
        except KeyError as exc:
            raise KeyError(f"no grasp profile for {object_name!r}") from exc

    def apply_recognition_profile(self, profile: str) -> None:
        """Switch context, class order and threshold as one atomic profile."""
        normalized = profile.strip().lower()
        if normalized == "object3":
            self.recognition_profile = normalized
            self.npu_context_path = (
                "third_party/qnn/yoloe-26s-seg_640_iq9075_qnn_object3.bin"
            )
            self.target_prompts = OBJECT3_CLASSES
            self.npu_class_names = OBJECT3_CLASSES
            self.npu_confidence_threshold = 0.20
            return
        if normalized in {"block4", "legacy_block4"}:
            self.recognition_profile = "block4"
            self.npu_context_path = (
                "third_party/qnn/yoloe-26s-seg_640_iq9075_qnn_block4.bin"
            )
            self.target_prompts = LEGACY_BLOCK4_CLASSES
            self.npu_class_names = LEGACY_BLOCK4_CLASSES
            self.npu_confidence_threshold = 0.15
            return
        raise ValueError("VISION_MODEL_PROFILE must be 'object3' or 'block4'")

    def validate(self) -> None:
        if self.recognition_profile not in {"object3", "block4"}:
            raise ValueError("invalid recognition profile")
        expected_classes = (
            OBJECT3_CLASSES
            if self.recognition_profile == "object3"
            else LEGACY_BLOCK4_CLASSES
        )
        if tuple(self.npu_class_names) != expected_classes:
            raise ValueError(
                "NPU class order does not match the selected recognition profile"
            )
        if self.color_classifier_backend not in {"lab", "hsv"}:
            raise ValueError("COLOR_CLASSIFIER must be 'lab' or 'hsv'")
        if not 1.0 <= self.color_lab_distance_scale <= 2.5:
            raise ValueError("Lab distance scale must be between 1.0 and 2.5")
        if not self.graspable_object_names:
            raise ValueError("at least one graspable object class is required")
        canonical_profiles = {
            canonical_object_name(name): profile
            for name, profile in self.object_grasp_profiles.items()
        }
        if set(canonical_profiles) != set(OBJECT3_CLASSES):
            raise ValueError(
                "object_grasp_profiles must define bottle, box and toy building block"
            )
        for name, profile in canonical_profiles.items():
            if canonical_object_name(profile.object_name) != name:
                raise ValueError(f"object grasp profile key/name mismatch for {name!r}")
            profile.validate()
        graspable = {
            canonical_object_name(name) for name in self.graspable_object_names
        }
        recognised = {
            canonical_object_name(name) for name in self.npu_class_names
        }
        if not graspable.issubset(recognised):
            raise ValueError(
                "graspable objects must be present in the selected recognition profile"
            )
        for name in graspable:
            profile = canonical_profiles.get(name)
            if profile is None or not profile.motion_ready:
                raise ValueError(f"{name} is graspable but its profile is not motion-ready")
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
        if not 0.0 <= self.grasp_approach_overtravel_m <= 0.015:
            raise ValueError("grasp approach overtravel must be between 0 and 15 mm")
        if self.direct_grasp_duration <= 0.0 or self.grasp_retry_retreat_duration <= 0.0:
            raise ValueError("grasp approach and retry retreat durations must be positive")
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
        if not (
            0.0 < self.joint1_jog_step_rad <= self.joint_upper[0] - self.joint_lower[0]
            and self.joint1_jog_duration > 0.0
            and self.joint1_jog_posture_tolerance_rad > 0.0
        ):
            raise ValueError("invalid J1 web-jog settings")
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
            and self.color_accumulation_timeout_s > 0.0
            and self.color_track_position_tolerance_m > 0.0
        ):
            raise ValueError("invalid color accumulation settings")
        if not 12 < self.color_yellow_green_boundary < 85:
            raise ValueError("yellow/green hue boundary must be between 12 and 85")
        if (
            not 0.0 < self.observation_centering_gain <= 1.0
            or not 0.0 < self.observation_axial_gain <= 1.0
            or not 0.0 <= self.observation_min_advance_m <= self.observation_max_advance_m
            or self.observation_max_lateral_shift_m <= 0.0
            or self.observation_max_translation_m <= 0.0
            or self.observation_min_camera_distance_m <= 0.0
            or self.observation_max_joint_step <= 0.0
            or not 0.0 < self.pre_grasp_standoff_ratio <= 1.0
            or not 0.0 < self.pre_grasp_min_distance_m <= self.pre_grasp_max_distance_m
            or self.pre_grasp_lateral_tolerance_m <= 0.0
            or self.pre_grasp_orientation_tolerance_deg <= 0.0
            or self.pre_grasp_joint_tolerance_rad <= 0.0
            or self.pre_grasp_realign_max_translation_m <= 0.0
            or self.pre_grasp_realign_max_orientation_deg
            < self.pre_grasp_orientation_tolerance_deg
            or self.pre_grasp_realign_endpoint_tolerance_m <= 0.0
            or self.pre_grasp_realign_duration <= 0.0
            or not 0.010 <= self.trajectory_control_period_s <= 0.050
            or self.trajectory_start_joint_tolerance_rad <= 0.0
            or self.trajectory_start_tcp_tolerance_m <= 0.0
            or not 0.020 <= self.stationary_hold_period_s <= 0.100
            or self.stationary_hold_max_drift_rad <= 0.0
            or self.approach_path_lateral_tolerance_m <= 0.0
            or self.approach_endpoint_tolerance_m <= 0.0
            or self.pre_grasp_duration <= 0.0
            or self.pre_grasp_camera_settle_time < 0.0
        ):
            raise ValueError("invalid pre-grasp settings")
        if not (
            2 <= self.refine_required_observations <= self.refine_max_frame_attempts
            and self.refine_frame_warmup >= 1
            and self.refine_total_timeout_s > 0.0
            and 0.0 < self.refine_frame_timeout_s <= self.refine_total_timeout_s
            and self.refine_target_match_radius_m > 0.0
            and self.refine_max_center_distance_px > 0.0
            and 0 < self.refine_fallback_min_area_px < self.refine_fallback_max_area_px
            and 0.0 < self.refine_max_position_spread_m <= self.refine_target_match_radius_m
            and 0.0 < self.refine_max_xy_correction_m <= self.refine_target_match_radius_m
            and 0.0 < self.refine_max_z_correction_m <= self.refine_target_match_radius_m
            and 0.0 < self.refine_max_total_correction_m <= self.refine_target_match_radius_m
            and 0.0 < self.refine_max_approach_tilt_deg <= 45.0
            and 0.0 < self.refine_max_open_axis_change_deg <= 45.0
        ):
            raise ValueError("invalid visual-refinement settings")
        if not (
            self.close_refine_min_depth_m >= 0.070
            and self.close_refine_border_margin_px >= 0
            and 0.0 < self.close_refine_max_bbox_area_ratio < 1.0
            and self.close_refine_max_depth_spread_m > 0.0
            and 0.0 < self.close_refine_max_position_spread_m
            <= self.refine_max_position_spread_m
            and 0.0 < self.close_refine_max_xy_correction_m
            <= self.refine_max_xy_correction_m
            and 0.0 < self.close_refine_max_z_correction_m
            <= self.refine_max_z_correction_m
            and 0.0 < self.close_refine_max_total_correction_m
            <= self.refine_max_total_correction_m
        ):
            raise ValueError("invalid close-range refinement settings")
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
        if not 0.0 < self.follow_max_semantic_step_m <= 0.10:
            raise ValueError("follow_max_semantic_step_m must be in (0, 100 mm]")
        if not 0.0 < self.follow_max_tcp_step_m <= 0.020:
            raise ValueError("follow_max_tcp_step_m must be in (0, 20 mm]")
        if not 1.0 <= self.stream_preview_fps <= 30.0:
            raise ValueError("stream_preview_fps must be between 1 and 30")
        if self.zero_move_timeout <= 0.0:
            raise ValueError("zero_move_timeout must be positive")
        if self.ik_single_seed_max_iterations < 50 or self.ik_max_seed_attempts < 1:
            raise ValueError("invalid bounded IK settings")
        if self.move_wait_min_timeout_s <= 0.0 or self.move_wait_margin_s <= 0.0:
            raise ValueError("invalid bounded move timeout settings")
        if (
            self.joint_move_settle_timeout_s <= 0.0
            or not 0.05 <= self.joint_move_max_endpoint_error_rad <= 0.20
        ):
            raise ValueError("invalid dense joint-move feedback limits")
        if (
            self.direct_grasp_joint_tolerance_rad <= 0.0
            or self.direct_grasp_tcp_tolerance_m <= 0.0
        ):
            raise ValueError("invalid direct-grasp settle tolerances")
        if (
            self.npu_response_timeout <= 0.0
            or self.npu_stderr_max_lines < 10
            or not 0.0 < self.npu_iou_threshold < 1.0
            or self.npu_pre_nms_top_k < self.npu_max_detections
            or self.npu_max_detections < 1
        ):
            raise ValueError("invalid NPU timeout or stderr history limit")
        if not self.target_prompts or not self.npu_class_names:
            raise ValueError("YOLOE prompt/class lists must not be empty")


COLOR_ALIASES = {
    "red": ("red", "\u7ea2"),
    "yellow": ("yellow", "\u9ec4"),
    "blue": ("blue", "\u84dd"),
    "green": ("green", "\u7eff"),
    "white": ("white", "\u767d"),
    "black": ("black", "\u9ed1"),
}
ANY_COLOR_ALIASES = ("any", "all", "\u4efb\u610f", "\u6240\u6709", "\u5168\u90e8")
OBJECT_ALIASES = {
    "toy building block": (
        "toybuildingblock",
        "buildingblock",
        "block",
        "legobrick",
        "lego",
        "\u79ef\u6728",
        "\u79ef\u6728\u5757",
        "\u4e50\u9ad8",
    ),
    "bottle": ("bottle", "\u74f6\u5b50", "\u74f6", "\u6c34\u74f6"),
    "box": ("box", "\u76d2\u5b50", "\u76d2", "\u7eb8\u76d2"),
}
ANY_OBJECT_ALIASES = ("anyobject", "anything", "\u4efb\u610f\u7269\u4f53", "\u4efb\u4f55\u7269\u4f53")
NEGATION_PREFIXES = (
    "\u4e0d\u8981",
    "\u522b\u6293",
    "\u522b\u8981",
    "\u6392\u9664",
    "\u4e0d\u662f",
    "\u4e0d\u6293",
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


def _find_alias_matches(normalized: str, aliases_by_name):
    """Return ordered ``(index, canonical_name, negated)`` alias matches."""
    matches: list[tuple[int, str, bool]] = []
    for name, aliases in aliases_by_name.items():
        for alias in aliases:
            start = 0
            while True:
                index = normalized.find(alias, start)
                if index < 0:
                    break
                matches.append((index, name, _token_is_negated(normalized, index)))
                start = index + len(alias)
    matches.sort(key=lambda item: item[0])
    return matches


def _distinct_alias_names(matches, *, negated: bool) -> list[str]:
    names = []
    for _index, name, is_negated in matches:
        if is_negated == negated and name not in names:
            names.append(name)
    return names


def _contains_positive_alias(normalized: str, aliases) -> bool:
    for alias in aliases:
        start = 0
        while True:
            index = normalized.find(alias, start)
            if index < 0:
                break
            if not _token_is_negated(normalized, index):
                return True
            start = index + len(alias)
    return False


def parse_target_request(command: str):
    """Return ``(TargetRequest, accepted)`` from terminal/speech text.

    A colour-only command intentionally means a coloured building block, which
    preserves the old UI while preventing an object3 detection of a same-colour
    bottle or box from becoming a motion target.
    """
    normalized = command.strip().lower().replace(" ", "")
    if not normalized:
        return None, False

    color_matches = _find_alias_matches(normalized, COLOR_ALIASES)
    object_matches = _find_alias_matches(normalized, OBJECT_ALIASES)
    positive_colors = _distinct_alias_names(color_matches, negated=False)
    negative_colors = _distinct_alias_names(color_matches, negated=True)
    positive_objects = _distinct_alias_names(object_matches, negated=False)
    negative_objects = _distinct_alias_names(object_matches, negated=True)
    any_color = _contains_positive_alias(normalized, ANY_COLOR_ALIASES)
    any_object = _contains_positive_alias(normalized, ANY_OBJECT_ALIASES)

    # The control surface accepts exactly one positive object and one positive
    # colour.  Never guess which noun/adjective belongs together in a compound
    # command such as "red bottle and blue box".
    if (
        len(positive_colors) > 1
        or len(positive_objects) > 1
        or (any_color and positive_colors)
        or (any_object and positive_objects)
    ):
        return None, False
    # Exclusions are supported only when the same dimension also has one
    # explicit positive replacement.  Thus "不要红色瓶子" cannot silently turn
    # into an unrestricted bottle request.
    if (
        (negative_colors and not positive_colors and not any_color)
        or (negative_objects and not positive_objects and not any_object)
    ):
        return None, False

    color = positive_colors[0] if positive_colors else None
    object_name = positive_objects[0] if positive_objects else None
    if color is None and object_name is None and not any_color and not any_object:
        return None, False
    if object_name is None and not any_object:
        object_name = "toy building block"
    return TargetRequest(object_name=object_name, color=color), True


def parse_target_command(command: str):
    """Compatibility wrapper returning the legacy ``(colour, accepted)``."""
    request, accepted = parse_target_request(command)
    return (request.color if request is not None else None), accepted
