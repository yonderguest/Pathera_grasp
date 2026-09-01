"""High-level grasp planning, execution and safe shutdown."""

from __future__ import annotations

import contextlib
import signal
import threading
import time
from enum import Enum

import numpy as np

from .grasp_config import GraspConfig
from .vision_pipeline import (
    apply_accumulated_color,
    detect_requested_color_regions,
    get_base_camera_transform,
    grasp_geometry,
    grasp_rotation_from_mask,
    object_base_position,
    workspace_ok,
)


class RobotLifecycleState(str, Enum):
    """Finite lifecycle for the process that owns the physical robot."""

    OWNED = "owned"
    HOMED = "homed"
    ACTIVE = "active"
    STOPPING = "stopping"
    FAULT_HOLD = "fault_hold"
    STOPPED = "stopped"


class GraspPlanner:
    """Coordinate the Panthera robot through the visual grasp workflow."""

    def __init__(
        self,
        robot,
        config: GraspConfig,
        interrupt_event: threading.Event,
        graspnet_provider=None,
        voice=None,
    ):
        self.robot = robot
        self.config = config
        self.interrupted = interrupt_event
        self.graspnet_provider = graspnet_provider
        self.voice = voice
        self.lifecycle_state = RobotLifecycleState.OWNED
        self._last_command: np.ndarray | None = None
        self._sdk_lock = threading.RLock()
        self._shutdown_lock = threading.Lock()
        self._shutdown_complete = False
        self._shutdown_succeeded = False

    @property
    def last_command(self):
        """Return the most recently accepted six-axis position command."""
        if self._last_command is None:
            return None
        return self._last_command.copy()

    def _remember_command(self, joints) -> None:
        command = np.asarray(joints, dtype=float)
        if command.shape != (6,) or not np.all(np.isfinite(command)):
            raise ValueError("joint command must contain six finite values")
        self._last_command = command.copy()

    def _say(self, text: str) -> None:
        """Speak a status message when a voice interface is available."""
        if self.voice is not None:
            self.voice.say(text)

    @contextlib.contextmanager
    def sdk_call(self):
        with self._sdk_lock:
            old_mask = None
            if threading.current_thread() is threading.main_thread() and hasattr(
                signal, "pthread_sigmask"
            ):
                old_mask = signal.pthread_sigmask(
                    signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM}
                )
            try:
                yield
            finally:
                if old_mask is not None:
                    signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)

    @contextlib.contextmanager
    def hold_current_pose(self, label):
        """Refresh a stationary six-axis hold while camera work is blocking."""
        command = getattr(self.robot, "Joint_Pos_Vel", None)
        if not callable(command):
            # Lightweight offline fakes do not expose motor commands.
            yield None
            return

        target = np.asarray(self.current_joint_position(), dtype=float)
        stop_event = threading.Event()
        failures = []
        period = float(self.config.stationary_hold_period_s)

        def refresh_hold():
            next_tick = time.monotonic()
            while not stop_event.is_set() and not self.interrupted.is_set():
                try:
                    with self.sdk_call():
                        result = command(
                            target.tolist(),
                            [0.0] * 6,
                            self.config.max_torque,
                            iswait=False,
                        )
                    if result is False:
                        failures.append("position-hold command was rejected")
                        return
                except Exception as exc:  # pragma: no cover - hardware boundary
                    failures.append(repr(exc))
                    return
                next_tick += period
                stop_event.wait(max(0.0, next_tick - time.monotonic()))

        worker = threading.Thread(
            target=refresh_hold,
            name=f"arm-hold-{label}",
            daemon=True,
        )
        print(f"[HOLD] maintaining {label} pose at {1.0 / period:.1f} Hz ...")
        worker.start()
        try:
            yield target
        finally:
            stop_event.set()
            worker.join(timeout=max(1.0, 4.0 * period))
        if worker.is_alive():
            raise RuntimeError(f"{label} position-hold worker did not stop")
        if failures:
            raise RuntimeError(f"{label} position hold failed: {failures[0]}")
        actual = np.asarray(self.current_joint_position(), dtype=float)
        drift = float(np.max(np.abs(actual - target)))
        print(f"[HOLD] {label} released: joint_drift={drift:.4f} rad")
        if drift > self.config.stationary_hold_max_drift_rad:
            raise RuntimeError(
                f"{label} drifted {drift:.4f} rad while awaiting camera data"
            )

    def sleep_interruptible(self, seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if self.interrupted.is_set():
                return False
            time.sleep(min(0.05, deadline - time.monotonic()))
        return True

    def current_joint_position(self):
        with self.sdk_call():
            return self.robot.current_joint_position()

    def solve_ik(self, target_joint6, tool_rotation, seed, jump_limit):
        started = time.monotonic()
        with self.sdk_call():
            joints = self.robot.inverse_kinematics(
                target_joint6.tolist(),
                tool_rotation,
                seed.tolist(),
                multi_init=False,
                max_iter=self.config.ik_single_seed_max_iterations,
            )
        elapsed = time.monotonic() - started
        if joints is None:
            print(
                f"[IK] explicit seed failed after {elapsed:.2f}s "
                f"({self.config.ik_single_seed_max_iterations} iteration limit).",
                flush=True,
            )
            return None
        joints = np.asarray(joints, dtype=float)
        if joints.shape != (6,) or not np.all(np.isfinite(joints)):
            print(f"[IK] rejected malformed solution: {joints!r}")
            return None
        if np.any(joints < self.config.joint_lower) or np.any(
            joints > self.config.joint_upper
        ):
            print(f"[IK] rejected joint-limit violation: q={np.round(joints, 3)}")
            return None
        joint_jump = float(np.max(np.abs(joints - seed)))
        if joint_jump > jump_limit:
            print(
                f"[IK] rejected branch jump={joint_jump:.3f} rad "
                f"(limit={jump_limit:.3f})."
            )
            return None

        with self.sdk_call():
            fk = self.robot.forward_kinematics(joints)
        if fk is None:
            return None
        actual_position = np.asarray(fk["position"], dtype=float)
        actual_rotation = np.asarray(fk["rotation"], dtype=float)
        position_error = float(np.linalg.norm(actual_position - target_joint6))
        rotation_error = float(
            np.degrees(
                np.arccos(
                    np.clip(
                        (
                            float(np.trace(actual_rotation.T @ tool_rotation)) - 1.0
                        )
                        / 2.0,
                        -1.0,
                        1.0,
                    )
                )
            )
        )
        if (
            position_error > self.config.ik_position_tolerance_m
            or rotation_error > self.config.ik_rotation_tolerance_deg
        ):
            print(
                "[IK] rejected inaccurate solution: "
                f"pos_err={position_error:.4f} m, rot_err={rotation_error:.2f} deg"
            )
            return None
        print(f"[IK] explicit seed accepted in {elapsed:.2f}s.", flush=True)
        return joints

    def validate_candidate(
        self,
        tool_target,
        joint6_target,
        tool_rotation,
        seeds,
        jump_limit,
        label="candidate",
        check_workspace=True,
    ):
        """Apply the same workspace and IK policy to every grasp backend."""
        tool_target = np.asarray(tool_target, dtype=float)
        joint6_target = np.asarray(joint6_target, dtype=float)
        tool_rotation = np.asarray(tool_rotation, dtype=float)
        if check_workspace and not workspace_ok(
            tool_target, joint6_target, self.config
        ):
            print(f"[PLAN] {label} rejected by workspace limits.")
            return None
        unique_seeds = []
        for seed in seeds:
            candidate_seed = np.asarray(seed, dtype=float)
            if candidate_seed.shape != (6,) or not np.all(np.isfinite(candidate_seed)):
                continue
            if not any(np.allclose(candidate_seed, old) for old in unique_seeds):
                unique_seeds.append(candidate_seed.copy())
        unique_seeds = unique_seeds[: self.config.ik_max_seed_attempts]
        print(
            f"[PLAN] {label}: bounded IK with {len(unique_seeds)} explicit seed(s).",
            flush=True,
        )
        for index, seed in enumerate(unique_seeds, start=1):
            print(
                f"[IK] {label}: seed {index}/{len(unique_seeds)} ...",
                flush=True,
            )
            solution = self.solve_ik(
                joint6_target,
                tool_rotation,
                seed,
                jump_limit,
            )
            if solution is not None:
                return solution
        return None

    def plan_grasp(self, joint6_target, tool_rotation):
        current = self.current_joint_position()
        final = self.solve_ik(
            joint6_target,
            tool_rotation,
            self.config.manual_grasp_ik_seed,
            self.config.max_home_to_grasp_step,
        )
        if final is None:
            print("[IK] manual grasp branch failed; trying current pose.")
            final = self.solve_ik(
                joint6_target,
                tool_rotation,
                current,
                self.config.max_home_to_grasp_step,
            )
        if final is None:
            raise RuntimeError("final grasp IK failed")
        if np.max(np.abs(final - current)) > self.config.max_home_to_grasp_step:
            raise RuntimeError("final grasp is too far from the current pose")
        return final

    def move_j(
        self,
        joints,
        duration,
        label,
        wait=True,
        position_tolerance=None,
    ):
        joints = np.asarray(joints, dtype=float)
        timeout = max(
            self.config.move_wait_min_timeout_s,
            float(duration) + self.config.move_wait_margin_s,
        )
        print(
            f"[MOVE] {label}: duration={duration:.1f}s, "
            f"feedback_timeout={timeout:.1f}s ...",
            flush=True,
        )
        started = time.monotonic()
        move_kwargs = {
            "duration": duration,
            "max_torque": self.config.max_torque,
            "label": label,
            "wait": wait,
            "timeout": timeout,
        }
        if position_tolerance is not None:
            move_kwargs["tolerance"] = float(position_tolerance)
        with self.sdk_call():
            result = self.robot.move_j_checked(joints, **move_kwargs)
        if result is False:
            raise RuntimeError(f"{label} move failed")
        print(
            f"[MOVE] {label}: completed in {time.monotonic() - started:.2f}s.",
            flush=True,
        )
        self._remember_command(joints)
        if self.lifecycle_state not in {
            RobotLifecycleState.STOPPING,
            RobotLifecycleState.FAULT_HOLD,
            RobotLifecycleState.STOPPED,
        }:
            self.lifecycle_state = RobotLifecycleState.ACTIVE

    def home(self):
        with self.sdk_call():
            result = self.robot.Joint_Pos_Vel(
                self.config.home.tolist(),
                self.config.home_velocity,
                self.config.max_torque,
                iswait=True,
            )
        if result is False:
            raise RuntimeError("HOME move failed")
        self._remember_command(self.config.home)
        self.lifecycle_state = RobotLifecycleState.HOMED
        self._say("机械臂已回到初始位置")

    def open_gripper(self, ignore_interrupt=False):
        cfg = self.config
        hold_joints = None
        if callable(getattr(self.robot, "Joint_Pos_Vel", None)):
            hold_joints = np.asarray(self.current_joint_position(), dtype=float)
        with self.sdk_call():
            result = self.robot.gripper_control(
                cfg.gripper_open_position,
                cfg.gripper_open_velocity,
                cfg.gripper_open_torque,
            )
        if result is False:
            raise RuntimeError("gripper open rejected")
        deadline = time.monotonic() + cfg.gripper_open_timeout
        last_position = float("nan")
        while time.monotonic() < deadline:
            if self.interrupted.is_set() and not ignore_interrupt:
                return False
            if hold_joints is not None:
                with self.sdk_call():
                    hold_result = self.robot.Joint_Pos_Vel(
                        hold_joints.tolist(),
                        [0.0] * 6,
                        cfg.max_torque,
                        iswait=False,
                    )
                if hold_result is False:
                    raise RuntimeError("arm hold rejected while opening gripper")
            with self.sdk_call():
                last_position, _ = self.robot.gripper_state()
            if abs(last_position - cfg.gripper_open_position) <= cfg.gripper_open_position_tolerance:
                print(
                    f"[GRIPPER] open confirmed: actual={last_position:+.3f}, "
                    f"target={cfg.gripper_open_position:+.3f}"
                )
                return True
            time.sleep(0.05)
        raise RuntimeError(
            f"gripper did not fully open: actual={last_position:+.3f}, "
            f"target={cfg.gripper_open_position:+.3f}"
        )

    def hold_arm(self, joints, duration):
        joints = np.asarray(joints, dtype=float)
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline:
            if self.interrupted.is_set():
                return False
            with self.sdk_call():
                self.robot.Joint_Pos_Vel(
                    joints.tolist(),
                    [0.0] * 6,
                    self.config.max_torque,
                    iswait=False,
                )
            time.sleep(0.05)
        return True

    def close_gripper(self, hold_joints=None, force_callback=None):
        cfg = self.config
        with self.sdk_call():
            result = self.robot.gripper_control(
                cfg.gripper_close_position,
                cfg.gripper_close_velocity,
                cfg.gripper_close_torque,
            )
        if result is False:
            raise RuntimeError("gripper close rejected")
        if hold_joints is not None:
            hold_joints = np.asarray(hold_joints, dtype=float)

        deadline = time.monotonic() + cfg.gripper_close_timeout
        last_position = float("nan")
        last_torque = float("nan")
        while time.monotonic() < deadline:
            if self.interrupted.is_set():
                return False, last_position, last_torque, "interrupted"
            if hold_joints is not None:
                with self.sdk_call():
                    self.robot.Joint_Pos_Vel(
                        hold_joints.tolist(),
                        [0.0] * 6,
                        self.config.max_torque,
                        iswait=False,
                    )
            with self.sdk_call():
                last_position, last_torque = self.robot.gripper_state()
            if force_callback is not None:
                force_callback(last_position, last_torque)
            if last_position <= cfg.gripper_clamped_position:
                return True, last_position, last_torque, "position reached"
            if abs(last_torque) >= cfg.gripper_clamp_torque:
                return True, last_position, last_torque, "torque rise"
            time.sleep(0.1)
        return False, last_position, last_torque, "timeout"

    def settle_at_grasp(self, target_joints, duration=2.0, pos_tol=0.015, vel_tol=0.2):
        """Passively verify the endpoint after the checked trajectory finishes.

        Do not resend ``Joint_Pos_Vel`` with a zero velocity vector here.  Real
        logs showed that command increased endpoint error from roughly
        0.02--0.03 rad to about 0.04 rad instead of correcting the pose.
        """
        target_joints = np.asarray(target_joints, dtype=float)
        deadline = time.monotonic() + duration
        pos_err = float("inf")
        vel_max = float("inf")
        while time.monotonic() < deadline:
            if self.interrupted.is_set():
                return False
            q = self.current_joint_position()
            with self.sdk_call():
                dq = np.asarray(self.robot.get_current_vel(), dtype=float)
            pos_err = float(np.max(np.abs(q - target_joints)))
            vel_max = float(np.max(np.abs(dq)))
            if pos_err <= pos_tol and vel_max <= vel_tol:
                print(
                    f"[GRASP] settle result: pos_err={pos_err:.4f}, "
                    f"vel_max={vel_max:.4f}, settled=True"
                )
                return True
            time.sleep(0.05)
        print(
            f"[GRASP] settle result: pos_err={pos_err:.4f}, "
            f"vel_max={vel_max:.4f}, settled=False"
        )
        return False

    @staticmethod
    def rotation_error_deg(first, second):
        first = np.asarray(first, dtype=float)
        second = np.asarray(second, dtype=float)
        return float(
            np.degrees(
                np.arccos(
                    np.clip((float(np.trace(first.T @ second)) - 1.0) / 2.0, -1.0, 1.0)
                )
            )
        )

    def current_tool_pose(self, joints):
        """Return the configured gripper-tip pose for explicit joints."""
        with self.sdk_call():
            fk = self.robot.forward_kinematics(np.asarray(joints, dtype=float))
        if fk is None:
            raise RuntimeError("forward kinematics failed for pre-grasp")
        rotation = np.asarray(fk["rotation"], dtype=float)
        joint6_position = np.asarray(fk["position"], dtype=float)
        if rotation.shape != (3, 3) or joint6_position.shape != (3,):
            raise RuntimeError("invalid forward-kinematics result for pre-grasp")
        return joint6_position + rotation @ self.config.tcp_in_joint6, rotation

    def current_tool_position(self, joints):
        return self.current_tool_pose(joints)[0]

    def wait_until_stationary(self):
        """Return a stable joint sample, or None when the arm is still moving."""
        cfg = self.config
        deadline = time.monotonic() + cfg.detection_stationary_timeout
        stable = 0
        latest = None
        previous = None
        while time.monotonic() < deadline and not self.interrupted.is_set():
            with self.sdk_call():
                refresh = getattr(self.robot, "refresh_motor_state", None)
                if callable(refresh):
                    refresh()
                latest = np.asarray(self.robot.current_joint_position(), dtype=float)
                velocity = np.asarray(self.robot.get_current_vel(), dtype=float)
            if (
                latest.shape != (6,)
                or velocity.shape != (6,)
                or not np.all(np.isfinite(latest))
                or not np.all(np.isfinite(velocity))
            ):
                raise RuntimeError("invalid joint feedback while waiting for camera stability")
            position_delta = (
                float("inf")
                if previous is None
                else float(np.max(np.abs(latest - previous)))
            )
            if (
                position_delta <= cfg.detection_stationary_joint_tolerance
                and float(np.max(np.abs(velocity)))
                <= cfg.detection_stationary_velocity_tolerance
            ):
                stable += 1
                if stable >= cfg.detection_stationary_stable_samples:
                    return latest
            else:
                stable = 0
            previous = latest.copy()
            time.sleep(0.05)
        print(
            "[VISION] arm did not become stationary before RGB-D capture: "
            f"q={None if latest is None else np.round(latest, 4)}"
        )
        return None

    def pre_grasp_alignment(self, found, current_joints=None):
        """Measure signed approach travel, lateral error and orientation error."""
        if current_joints is None:
            current_joints = self.current_joint_position()
        current_tool, current_rotation = self.current_tool_pose(current_joints)
        tool_target = np.asarray(found["tool_target"], dtype=float)
        tool_rotation = np.asarray(found["tool_rotation"], dtype=float)
        approach = np.asarray(tool_rotation[:, 0], dtype=float)
        approach /= float(np.linalg.norm(approach))
        delta = tool_target - current_tool
        axial = float(np.dot(delta, approach))
        lateral = float(np.linalg.norm(delta - axial * approach))
        orientation = self.rotation_error_deg(current_rotation, tool_rotation)
        return axial, lateral, orientation, current_tool, approach

    def plan_observation_pose(self, tcp_camera, found):
        """Plan a partial camera-centering move without changing TCP rotation."""
        cfg = self.config
        camera_point = np.asarray(found["camera_point"], dtype=float)
        if camera_point.shape != (3,) or not np.all(np.isfinite(camera_point)):
            raise RuntimeError("invalid camera point for observation pre-grasp")
        camera_depth = float(camera_point[2])
        available_depth = max(
            0.0,
            camera_depth - cfg.observation_min_camera_distance_m,
        )
        axial_advance = min(
            cfg.observation_max_advance_m,
            cfg.observation_axial_gain * available_depth,
        )
        if 0.0 < axial_advance < cfg.observation_min_advance_m:
            axial_advance = 0.0

        lateral = cfg.observation_centering_gain * camera_point[:2]
        lateral_norm = float(np.linalg.norm(lateral))
        if lateral_norm > cfg.observation_max_lateral_shift_m:
            lateral *= cfg.observation_max_lateral_shift_m / lateral_norm
        camera_translation = np.array(
            [lateral[0], lateral[1], axial_advance],
            dtype=float,
        )
        translation_norm = float(np.linalg.norm(camera_translation))
        if translation_norm > cfg.observation_max_translation_m:
            camera_translation *= cfg.observation_max_translation_m / translation_norm
            translation_norm = cfg.observation_max_translation_m
        if translation_norm < cfg.observation_min_advance_m:
            print(
                "[OBSERVE] target is already close to the camera observation centre; "
                "skipping observation motion but still re-detecting."
            )
            return None

        current = self.current_joint_position()
        with self.sdk_call():
            base_camera = get_base_camera_transform(
                self.robot,
                tcp_camera,
                current,
                tcp_in_joint6=cfg.tcp_in_joint6,
            )
        base_from_camera = np.asarray(base_camera[:3, :3], dtype=float)
        desired_camera = (
            np.asarray(base_camera[:3, 3], dtype=float)
            + base_from_camera @ camera_translation
        )
        tcp_rotation = np.asarray(tcp_camera[:3, :3], dtype=float)
        tcp_translation = np.asarray(tcp_camera[:3, 3], dtype=float)
        observation_rotation = base_from_camera @ tcp_rotation.T
        observation_tool = (
            desired_camera - observation_rotation @ tcp_translation
        )
        observation_joint6 = (
            observation_tool - observation_rotation @ cfg.tcp_in_joint6
        )
        planned = self.validate_candidate(
            observation_tool,
            observation_joint6,
            observation_rotation,
            (current,),
            cfg.observation_max_joint_step,
            label="camera observation",
            check_workspace=False,
        )
        if planned is None:
            print(
                "[OBSERVE] local observation IK is unavailable; keeping the "
                "current camera pose and continuing bounded multi-frame refinement.",
                flush=True,
            )
            return None
        print(
            "[OBSERVE] partial eye-in-hand correction: "
            f"camera_delta={np.round(camera_translation, 4)} m, "
            f"travel={translation_norm:.3f} m, tool={np.round(observation_tool, 3)} m"
        )
        return planned

    def plan_pre_grasp(self, found):
        """Plan a clamped half-gap standoff along the final approach axis."""
        cfg = self.config
        tool_target = np.asarray(found["tool_target"], dtype=float)
        tool_rotation = np.asarray(found["tool_rotation"], dtype=float)
        approach = np.asarray(tool_rotation[:, 0], dtype=float)
        approach_norm = float(np.linalg.norm(approach))
        if approach_norm < 1e-8:
            raise RuntimeError("pre-grasp approach direction has zero length")
        approach /= approach_norm

        current_joints = self.current_joint_position()
        axial, lateral, orientation, _current_tool, _approach = self.pre_grasp_alignment(
            found, current_joints
        )
        if axial <= 0.0:
            raise RuntimeError(
                f"tip is on or beyond the grasp plane: axial={axial:.3f} m"
            )
        if axial < cfg.pre_grasp_min_distance_m:
            raise RuntimeError(
                "tip is too close for a free-space orientation change: "
                f"axial={axial:.3f} m"
            )
        if (
            cfg.pre_grasp_min_distance_m <= axial <= cfg.pre_grasp_max_distance_m
            and lateral <= cfg.pre_grasp_lateral_tolerance_m
            and orientation <= cfg.pre_grasp_orientation_tolerance_deg
        ):
            found["approach_standoff_m"] = axial
            print(
                f"[PREGRASP] already inside approach corridor: axial={axial:.3f} m, "
                f"lateral={lateral:.3f} m, rotation={orientation:.2f} deg; "
                "skipping the free-space waypoint."
            )
            return None

        standoff = float(
            np.clip(
                cfg.pre_grasp_standoff_ratio * axial,
                cfg.pre_grasp_min_distance_m,
                cfg.pre_grasp_max_distance_m,
            )
        )
        pre_tool_target = tool_target - standoff * approach
        tcp_offset = tool_rotation @ cfg.tcp_in_joint6
        pre_joint6_target = pre_tool_target - tcp_offset
        seed = np.asarray(found.get("provisional_joints", current_joints), dtype=float)
        jump_limit = (
            cfg.graspnet_max_joint_jump if cfg.use_graspnet else cfg.max_home_to_grasp_step
        )
        planned = self.validate_candidate(
            pre_tool_target,
            pre_joint6_target,
            tool_rotation,
            (current_joints, seed, cfg.manual_grasp_ik_seed),
            jump_limit,
            label="pre-grasp",
        )
        if planned is None:
            raise RuntimeError("pre-grasp validation failed; direct fallback is unsafe")
        found["approach_standoff_m"] = standoff
        print(
            f"[PREGRASP] adaptive standoff={standoff:.3f} m "
            f"from current axial gap={axial:.3f} m: {np.round(pre_tool_target, 3)}"
        )
        return planned

    def plan_cartesian_approach(self, found):
        """Plan and validate the final converging TCP approach sample by sample."""
        cfg = self.config
        current = self.wait_until_stationary()
        if current is None:
            raise RuntimeError("arm is not stationary before Cartesian approach")
        axial, lateral, orientation, start_tool, approach = self.pre_grasp_alignment(
            found, current
        )
        if not (
            cfg.pre_grasp_min_distance_m - 0.003
            <= axial
            <= cfg.pre_grasp_max_distance_m + 0.005
            and lateral <= cfg.pre_grasp_lateral_tolerance_m
            and orientation <= cfg.pre_grasp_orientation_tolerance_deg
        ):
            raise RuntimeError(
                "final approach does not start inside the pre-grasp corridor: "
                f"axial={axial:.3f} m, lateral={lateral:.3f} m, "
                f"rotation={orientation:.2f} deg"
            )

        with self.sdk_call():
            fk = self.robot.forward_kinematics(current)
        if fk is None:
            raise RuntimeError("FK failed before Cartesian approach")
        start_pose = {
            "position": np.asarray(fk["position"], dtype=float),
            "rotation": np.asarray(fk["rotation"], dtype=float),
        }
        end_pose = {
            "position": np.asarray(found["joint6_target"], dtype=float),
            "rotation": np.asarray(found["tool_rotation"], dtype=float),
        }
        with self.sdk_call():
            planned, fraction = self.robot.compute_cartesian_path(
                [start_pose, end_pose], avoid_collisions=False
            )
        if planned is None or float(fraction) < 0.999999:
            raise RuntimeError(
                f"Cartesian approach planning incomplete: fraction={float(fraction):.3f}"
            )
        trajectory = [np.asarray(current, dtype=float)] + [
            np.asarray(joints, dtype=float) for joints in planned
        ]
        if len(trajectory) < 2:
            raise RuntimeError("Cartesian approach contains no motion samples")

        target_tool = np.asarray(found["tool_target"], dtype=float)
        path_vector = target_tool - start_tool
        path_length = float(np.linalg.norm(path_vector))
        if path_length < 1e-6:
            raise RuntimeError("Cartesian approach path is too short")
        path_direction = path_vector / path_length
        lateral_ceiling = min(
            cfg.pre_grasp_lateral_tolerance_m + 0.003,
            max(
                lateral + 0.003,
                cfg.approach_path_lateral_tolerance_m,
            ),
        )
        previous_progress = -1e-6
        previous_path_progress = -1e-6
        previous_remaining_lateral = lateral + 1e-6
        previous_joints = trajectory[0]
        jump_limit = float(getattr(self.robot, "jump_threshold", 1.5))
        max_path_cross_track = 0.0
        max_remaining_lateral = 0.0
        for index, joints in enumerate(trajectory):
            if (
                joints.shape != (6,)
                or not np.all(np.isfinite(joints))
                or np.any(joints < cfg.joint_lower)
                or np.any(joints > cfg.joint_upper)
            ):
                raise RuntimeError(f"Cartesian sample {index} violates joint limits")
            if index and float(np.max(np.abs(joints - previous_joints))) > jump_limit:
                raise RuntimeError(f"Cartesian sample {index} has a joint jump")
            with self.sdk_call():
                sample_fk = self.robot.forward_kinematics(joints)
            if sample_fk is None:
                raise RuntimeError(f"FK failed at Cartesian sample {index}")
            rotation = np.asarray(sample_fk["rotation"], dtype=float)
            joint6 = np.asarray(sample_fk["position"], dtype=float)
            tool = joint6 + rotation @ cfg.tcp_in_joint6
            if not workspace_ok(tool, joint6, cfg):
                raise RuntimeError(f"Cartesian sample {index} leaves the workspace")
            progress = float(np.dot(tool - start_tool, approach))
            path_progress = float(np.dot(tool - start_tool, path_direction))
            path_cross_track = float(
                np.linalg.norm(
                    (tool - start_tool) - path_progress * path_direction
                )
            )
            if progress + 1e-4 < previous_progress or progress > axial + 0.001:
                raise RuntimeError(f"Cartesian sample {index} is not axially monotonic")
            if (
                path_progress + 1e-4 < previous_path_progress
                or path_progress > path_length + 0.001
            ):
                raise RuntimeError(
                    f"Cartesian sample {index} does not advance along the planned segment"
                )
            if path_cross_track > cfg.approach_path_lateral_tolerance_m:
                raise RuntimeError(
                    f"Cartesian sample {index} leaves the commanded segment: "
                    f"cross-track={path_cross_track:.4f} m"
                )
            remaining = target_tool - tool
            remaining_axial = float(np.dot(remaining, approach))
            remaining_lateral = float(
                np.linalg.norm(remaining - remaining_axial * approach)
            )
            if remaining_lateral > lateral_ceiling:
                raise RuntimeError(
                    f"Cartesian sample {index} leaves the approach corridor: "
                    f"lateral={remaining_lateral:.4f} m"
                )
            if remaining_lateral > previous_remaining_lateral + 0.003:
                raise RuntimeError(
                    f"Cartesian sample {index} diverges laterally from the target"
                )
            if self.rotation_error_deg(rotation, end_pose["rotation"]) > (
                cfg.pre_grasp_orientation_tolerance_deg
            ):
                raise RuntimeError(f"Cartesian sample {index} rotates outside tolerance")
            previous_progress = progress
            previous_path_progress = path_progress
            previous_remaining_lateral = remaining_lateral
            previous_joints = joints
            max_path_cross_track = max(max_path_cross_track, path_cross_track)
            max_remaining_lateral = max(max_remaining_lateral, remaining_lateral)

        endpoint_tool, endpoint_rotation = self.current_tool_pose(trajectory[-1])
        endpoint_error = float(
            np.linalg.norm(endpoint_tool - np.asarray(found["tool_target"], dtype=float))
        )
        if endpoint_error > cfg.approach_endpoint_tolerance_m:
            raise RuntimeError(
                f"Cartesian endpoint error={endpoint_error:.4f} m exceeds tolerance"
            )
        if self.rotation_error_deg(endpoint_rotation, end_pose["rotation"]) > (
            cfg.pre_grasp_orientation_tolerance_deg
        ):
            raise RuntimeError("Cartesian endpoint orientation is inaccurate")
        print(
            "[PREGRASP] Cartesian approach verified: "
            f"delta_xyz={np.round(path_vector * 1000.0, 1)} mm, "
            f"travel={axial:.3f} m, "
            f"samples={len(trajectory)}, endpoint_error={endpoint_error:.4f} m, "
            f"segment_cross_track_max={max_path_cross_track:.4f} m, "
            f"approach_lateral_max={max_remaining_lateral:.4f} m"
        )
        return trajectory

    def plan_pre_grasp_realignment(self, found):
        """Correct residual lateral error before the strict axial approach.

        The free-space MoveJ waypoint can be accepted by the motor controller
        while the TCP is still a few centimetres from its Cartesian target.
        This bounded correction moves back to the already validated standoff;
        it never advances through the grasp plane.  The final approach then
        validates deviation from its commanded Cartesian segment separately
        from the lateral error that is converging toward the grasp axis.
        """
        cfg = self.config
        current = self.wait_until_stationary()
        if current is None:
            raise RuntimeError("arm is not stationary before pre-grasp realignment")
        axial, lateral, orientation, start_tool, approach = self.pre_grasp_alignment(
            found, current
        )
        if (
            cfg.pre_grasp_min_distance_m - 0.003
            <= axial
            <= cfg.pre_grasp_max_distance_m + 0.005
            and lateral <= cfg.pre_grasp_lateral_tolerance_m
            and orientation <= cfg.pre_grasp_orientation_tolerance_deg
        ):
            return None

        if axial < cfg.pre_grasp_min_distance_m - 0.003:
            raise RuntimeError(
                "pre-grasp realignment would start too close to the target: "
                f"axial={axial:.3f} m"
            )
        if orientation > cfg.pre_grasp_realign_max_orientation_deg:
            raise RuntimeError(
                "pre-grasp realignment orientation is too far from target: "
                f"rotation={orientation:.2f} deg"
            )

        tool_target = np.asarray(found["tool_target"], dtype=float)
        tool_rotation = np.asarray(found["tool_rotation"], dtype=float)
        standoff = float(
            np.clip(
                found.get("approach_standoff_m", cfg.pre_grasp_max_distance_m),
                cfg.pre_grasp_min_distance_m,
                cfg.pre_grasp_max_distance_m,
            )
        )
        desired_tool = tool_target - standoff * approach
        correction = desired_tool - start_tool
        correction_norm = float(np.linalg.norm(correction))
        if correction_norm > cfg.pre_grasp_realign_max_translation_m:
            raise RuntimeError(
                "pre-grasp residual is too large for bounded realignment: "
                f"translation={correction_norm:.3f} m"
            )

        with self.sdk_call():
            fk = self.robot.forward_kinematics(current)
        if fk is None:
            raise RuntimeError("FK failed before pre-grasp realignment")
        start_pose = {
            "position": np.asarray(fk["position"], dtype=float),
            "rotation": np.asarray(fk["rotation"], dtype=float),
        }
        desired_joint6 = desired_tool - tool_rotation @ cfg.tcp_in_joint6
        end_pose = {
            "position": desired_joint6,
            "rotation": tool_rotation,
        }
        with self.sdk_call():
            planned, fraction = self.robot.compute_cartesian_path(
                [start_pose, end_pose], avoid_collisions=False
            )
        if planned is None or float(fraction) < 0.999999:
            raise RuntimeError(
                "pre-grasp Cartesian realignment incomplete: "
                f"fraction={float(fraction):.3f}"
            )
        trajectory = [np.asarray(current, dtype=float)] + [
            np.asarray(joints, dtype=float) for joints in planned
        ]
        if len(trajectory) < 2:
            raise RuntimeError("pre-grasp realignment contains no motion samples")

        previous_distance = correction_norm + 1e-6
        previous_joints = trajectory[0]
        jump_limit = float(getattr(self.robot, "jump_threshold", 1.5))
        for index, joints in enumerate(trajectory):
            if (
                joints.shape != (6,)
                or not np.all(np.isfinite(joints))
                or np.any(joints < cfg.joint_lower)
                or np.any(joints > cfg.joint_upper)
            ):
                raise RuntimeError(
                    f"pre-grasp realignment sample {index} violates joint limits"
                )
            if index and float(np.max(np.abs(joints - previous_joints))) > jump_limit:
                raise RuntimeError(
                    f"pre-grasp realignment sample {index} has a joint jump"
                )
            sample_tool, sample_rotation = self.current_tool_pose(joints)
            sample_joint6 = sample_tool - sample_rotation @ cfg.tcp_in_joint6
            if not workspace_ok(sample_tool, sample_joint6, cfg):
                raise RuntimeError(
                    f"pre-grasp realignment sample {index} leaves the workspace"
                )
            sample_axial = float(np.dot(tool_target - sample_tool, approach))
            if not (
                cfg.pre_grasp_min_distance_m - 0.003
                <= sample_axial
                <= cfg.pre_grasp_max_distance_m + 0.010
            ):
                raise RuntimeError(
                    f"pre-grasp realignment sample {index} has unsafe "
                    f"axial gap={sample_axial:.4f} m"
                )
            distance = float(np.linalg.norm(sample_tool - desired_tool))
            if distance > previous_distance + 0.003:
                raise RuntimeError(
                    f"pre-grasp realignment sample {index} diverges from waypoint"
                )
            if self.rotation_error_deg(sample_rotation, tool_rotation) > (
                cfg.pre_grasp_realign_max_orientation_deg
            ):
                raise RuntimeError(
                    f"pre-grasp realignment sample {index} rotates outside tolerance"
                )
            previous_distance = distance
            previous_joints = joints

        endpoint_tool, endpoint_rotation = self.current_tool_pose(trajectory[-1])
        endpoint_error = float(np.linalg.norm(endpoint_tool - desired_tool))
        if endpoint_error > cfg.pre_grasp_realign_endpoint_tolerance_m:
            raise RuntimeError(
                "pre-grasp realignment endpoint error="
                f"{endpoint_error:.4f} m exceeds tolerance"
            )
        if self.rotation_error_deg(endpoint_rotation, tool_rotation) > (
            cfg.pre_grasp_orientation_tolerance_deg
        ):
            raise RuntimeError("pre-grasp realignment endpoint orientation is inaccurate")
        print(
            "[PREGRASP] bounded Cartesian realignment verified: "
            f"axial={axial:.3f} m, lateral={lateral:.3f} m, "
            f"correction_xyz={np.round(correction * 1000.0, 1)} mm, "
            f"correction={correction_norm:.3f} m, samples={len(trajectory)}"
        )
        return trajectory

    def validate_trajectory_start(self, trajectory, label):
        """Reject a stale plan if feedback moved after its first sample."""
        cfg = self.config
        expected = np.asarray(trajectory[0], dtype=float)
        actual = np.asarray(self.current_joint_position(), dtype=float)
        joint_gap = float(np.max(np.abs(actual - expected)))
        expected_tool, _ = self.current_tool_pose(expected)
        actual_tool, _ = self.current_tool_pose(actual)
        tcp_gap = float(np.linalg.norm(actual_tool - expected_tool))
        print(
            f"[{label}] start feedback: joint_gap={joint_gap:.4f} rad, "
            f"tcp_gap={tcp_gap * 1000.0:.1f} mm"
        )
        if (
            joint_gap > cfg.trajectory_start_joint_tolerance_rad
            or tcp_gap > cfg.trajectory_start_tcp_tolerance_m
        ):
            raise RuntimeError(
                f"{label} plan is stale before execution: "
                f"joint_gap={joint_gap:.4f} rad, tcp_gap={tcp_gap:.4f} m"
            )

    def execute_pre_grasp_realignment(self, trajectory):
        self.validate_trajectory_start(trajectory, "PRE GRASP CARTESIAN REALIGNMENT")
        with self.sdk_call():
            self.robot.execute_joint_trajectory_checked(
                trajectory,
                duration=self.config.pre_grasp_realign_duration,
                max_torque=self.config.max_torque,
                label="PRE GRASP CARTESIAN REALIGNMENT",
                control_period=self.config.trajectory_control_period_s,
            )
        self._remember_command(trajectory[-1])

    def execute_cartesian_approach(self, trajectory):
        self.validate_trajectory_start(trajectory, "FINAL CARTESIAN APPROACH")
        with self.sdk_call():
            self.robot.execute_joint_trajectory_checked(
                trajectory,
                duration=self.config.direct_grasp_duration,
                max_torque=self.config.max_torque,
                label="FINAL CARTESIAN APPROACH",
                control_period=self.config.trajectory_control_period_s,
            )
        self._remember_command(trajectory[-1])

    def grasp_and_close(
        self,
        final,
        approach_trajectory,
        streamer=None,
        found=None,
        gripper_preopened=False,
    ):
        cfg = self.config
        if not gripper_preopened:
            print("[GRASP] opening gripper before direct grasp ...")
            self.open_gripper()
            if self.interrupted.is_set():
                return False, float("nan"), float("nan")

        print("[GRASP] executing validated Cartesian final approach ...")
        self.execute_cartesian_approach(approach_trajectory)
        if not self.sleep_interruptible(cfg.direct_grasp_post_command_wait):
            return False, float("nan"), float("nan")

        actual = self.current_joint_position()
        print("[GRASP] final joint comparison")
        print(f"  planned: {np.round(final, 3)}")
        print(f"  actual : {np.round(actual, 3)}")
        print(f"  manual : {np.round(cfg.manual_grasp_ik_seed, 3)}")
        print(f"  delta to manual: {np.round(final - cfg.manual_grasp_ik_seed, 3)}")

        print("[GRASP] confirming grasp pose settled ...")
        settled = self.settle_at_grasp(
            final,
            duration=cfg.direct_grasp_settle_timeout,
            pos_tol=cfg.direct_grasp_joint_tolerance_rad,
            vel_tol=0.2,
        )
        if not settled:
            print(
                "[GRASP] final pose did not settle inside the bounded joint "
                "tolerance; gripper close aborted."
            )
            return False, float("nan"), float("nan")

        tcp_residual_norm = float("nan")
        if found is not None:
            settled_joints = self.current_joint_position()
            actual_tool, _actual_rotation = self.current_tool_pose(settled_joints)
            tool_target = np.asarray(found["tool_target"], dtype=float)
            residual_mm = (actual_tool - tool_target) * 1000.0
            tcp_residual_norm = float(np.linalg.norm(actual_tool - tool_target))
            print(
                "[ACCURACY] actual TCP - software target in Base XYZ: "
                f"{np.round(residual_mm, 1)} mm"
            )
            print(
                "[ACCURACY] actual TCP residual norm: "
                f"{tcp_residual_norm * 1000.0:.1f} mm"
            )
            if "base_point" in found:
                detected_object = np.asarray(found["base_point"], dtype=float)
                configured_offset_mm = (tool_target - detected_object) * 1000.0
                actual_object_delta_mm = (actual_tool - detected_object) * 1000.0
                print(
                    "[ACCURACY] software target - detected object in Base XYZ: "
                    f"{np.round(configured_offset_mm, 1)} mm"
                )
                print(
                    "[ACCURACY] actual TCP - detected object in Base XYZ: "
                    f"{np.round(actual_object_delta_mm, 1)} mm"
                )
                print(
                    "[ACCURACY] commanded approach-axis overtravel: "
                    f"{cfg.grasp_approach_overtravel_m * 1000.0:.1f} mm"
                )
        if (
            np.isfinite(tcp_residual_norm)
            and tcp_residual_norm > cfg.direct_grasp_tcp_tolerance_m
        ):
            print(
                "[GRASP] actual TCP is outside the final Cartesian tolerance: "
                f"{tcp_residual_norm:.4f} m > "
                f"{cfg.direct_grasp_tcp_tolerance_m:.4f} m; "
                "gripper close aborted."
            )
            return False, float("nan"), float("nan")

        print("[GRASP] closing gripper while holding arm ...")
        grasp_hold = self.current_joint_position()
        clamped = False
        gpos = float("nan")
        gtor = float("nan")

        def publish_force(position, torque):
            magnitude = abs(float(torque))
            force_text = f"{magnitude:.3f} Nm"
            print(
                f"[FORCE] gripper feedback: magnitude={force_text}, "
                f"raw_torque={float(torque):+.3f}, position={float(position):+.3f}",
                flush=True,
            )
            if streamer is not None:
                streamer.set_force_feedback(force_text)

        for attempt in range(1, cfg.gripper_close_attempts + 1):
            if attempt > 1:
                print(
                    f"[GRASP] close attempt {attempt}/{cfg.gripper_close_attempts}: "
                    "releasing and retrying ..."
                )
                self.open_gripper()
                grasp_hold = self.current_joint_position()
            clamped, gpos, gtor, reason = self.close_gripper(
                hold_joints=grasp_hold,
                force_callback=publish_force,
            )
            print(
                f"[GRASP] close attempt {attempt}/{cfg.gripper_close_attempts} -> "
                f"clamped={clamped} pos={gpos:+.3f} torque={gtor:+.3f} ({reason})"
            )
            if clamped:
                break
        if not clamped:
            print("[GRASP] clamp failed; releasing and aborting grasp")
            self.open_gripper()
            return False, gpos, gtor
        print(f"[GRASP] gripper clamped: pos={gpos:+.3f}, torque={gtor:+.3f}")
        self._say("已夹紧物体")
        if self.interrupted.is_set():
            self.open_gripper()
            return False, gpos, gtor
        return True, gpos, gtor

    def finish_place_sequence(self):
        cfg = self.config

        # Once force confirms a grasp, do not strand the object in the gripper
        # if a web-stop arrives during placement.  Finish the validated path to
        # PUT1 and release, then honour the stop before optional PUT2 motion.
        print("[GRASP] returning to HOME while holding the object ...")
        self.move_j(cfg.home, cfg.return_home_duration, "HOME")
        print(f"[PUT1] moving HOME -> PUT1 while holding: {np.round(cfg.put1, 3)}")
        self.move_j(cfg.put1, cfg.put1_duration, "PUT1")
        print("[PUT1] opening gripper fully to release the object ...")
        self.open_gripper(ignore_interrupt=True)
        if self.interrupted.is_set():
            print(
                "[PUT1] stop requested during placement; object released "
                "safely and PUT2 skipped."
            )
            return True
        self._say("物体已放置")
        print(f"[PUT2] moving to PUT2 with gripper open: {np.round(cfg.put2, 3)}")
        self.move_j(cfg.put2, cfg.put2_duration, "PUT2")
        return True

    def execute_grasp(self, final, approach_trajectory, streamer=None):
        clamped, _gpos, _gtor = self.grasp_and_close(
            final, approach_trajectory, streamer
        )
        if not clamped:
            return False
        return self.finish_place_sequence()

    def scan_joint1_values(self):
        cfg = self.config
        values = list(
            np.arange(cfg.scan_j1_start, cfg.scan_j1_end, -cfg.scan_j1_step, dtype=float)
        )
        if not values or not np.isclose(values[-1], cfg.scan_j1_end):
            values.append(cfg.scan_j1_end)
        return values

    def scan_pose(self, joint1):
        pose = self.config.home.copy()
        pose[0] = float(joint1)
        return pose

    def jog_joint1(self, direction):
        """Move J1 by one bounded web-jog step while the arm is at HOME shape."""
        cfg = self.config
        direction = str(direction).strip().lower()
        signs = {"left": 1.0, "right": -1.0}
        if direction not in signs:
            raise RuntimeError(f"invalid J1 jog direction: {direction!r}")
        current = self.wait_until_stationary()
        if current is None:
            raise RuntimeError("arm is not stationary; J1 jog rejected")
        non_j1_error = float(np.max(np.abs(current[1:] - cfg.home[1:])))
        if non_j1_error > cfg.joint1_jog_posture_tolerance_rad:
            raise RuntimeError(
                "arm is outside the safe J1-jog posture: "
                f"non_j1_error={non_j1_error:.3f} rad"
            )
        requested = float(current[0] + signs[direction] * cfg.joint1_jog_step_rad)
        target_j1 = float(
            np.clip(requested, cfg.joint_lower[0], cfg.joint_upper[0])
        )
        actual_step = target_j1 - float(current[0])
        if abs(actual_step) < 1e-3:
            side = "left" if direction == "left" else "right"
            raise RuntimeError(f"J1 is already at the {side} joint limit")
        target = current.copy()
        target[0] = target_j1
        side_label = "LEFT" if direction == "left" else "RIGHT"
        print(
            f"[WEB-JOG] J1 {side_label}: current={current[0]:+.3f}, "
            f"requested={requested:+.3f}, target={target_j1:+.3f} rad",
            flush=True,
        )
        self.move_j(target, cfg.joint1_jog_duration, f"WEB J1 {side_label}")
        settled = self.wait_until_stationary()
        if settled is None:
            raise RuntimeError("J1 jog completed but the arm did not settle")
        return float(settled[0])

    def _is_central_horizontal(self, detection):
        cfg = self.config
        ratio = float(np.clip(cfg.central_x_grasp_ratio, 0.0, 1.0))
        if ratio <= 0.0:
            return True
        x1, _y1, x2, _y2 = detection["bbox"]
        center_x = 0.5 * (float(x1) + float(x2))
        margin = (1.0 - ratio) / 2.0
        left = margin * cfg.width
        right = (1.0 - margin) * cfg.width
        return left <= center_x <= right

    def _select_graspnet_candidate(
        self,
        camera_feed,
        capture,
        intrinsic,
        base_camera,
        matches,
        scan_joint_position,
        joint1,
        label,
        requested_color=None,
        target_base_override=None,
    ):
        """Choose the highest-scoring GraspNet candidate that is reachable."""
        cfg = self.config
        if self.graspnet_provider is None:
            raise RuntimeError(
                "use_graspnet is enabled but no GraspNetCandidateProvider was passed"
            )

        current_joints = self.current_joint_position()
        for detection in matches:
            camera_point, measured_base_point = object_base_position(
                detection, intrinsic, base_camera
            )
            target_base_point = (
                np.asarray(target_base_override, dtype=float)
                if target_base_override is not None
                else measured_base_point
            )
            candidates = self.graspnet_provider.generate_candidates(
                capture["color_image"],
                capture["depth_image"],
                detection,
                intrinsic,
                camera_feed.depth_scale,
                base_camera,
                target_base_point=target_base_point,
            )
            for candidate in candidates:
                joint6_target = np.asarray(candidate["joint6_target"], dtype=float)
                tool_rotation = np.asarray(candidate["tool_rotation"], dtype=float)
                tool_target = np.asarray(candidate["tool_target"], dtype=float)

                provisional = self.validate_candidate(
                    tool_target,
                    joint6_target,
                    tool_rotation,
                    (current_joints, cfg.manual_grasp_ik_seed),
                    cfg.graspnet_max_joint_jump,
                    label="GraspNet grasp",
                )
                if provisional is None:
                    print(
                        f"[{label}] J1={joint1:+.2f}: "
                        f"GraspNet candidate score={candidate['score']:.3f} "
                        "is unreachable."
                    )
                    continue

                print("\n" + "=" * 68)
                print(f"[{label}] GraspNet candidate accepted at J1={joint1:+.2f} rad")
                print(f"Target          : {detection['class_name']} ({detection['color']})")
                print(f"Confidence      : {detection['confidence']:.3f}")
                print(f"Color vote      : {detection['color_confidence']:.1%}")
                print(
                    f"Depth           : {detection['depth_m']:.3f} m, "
                    f"samples={detection.get('depth_samples', 0)}, "
                    f"spread={detection.get('depth_spread_m', float('nan')) * 1000.0:.1f} mm"
                )
                print(f"Grasp score     : {candidate['score']:.3f}")
                print(f"Gripper width   : {candidate['gripper_width']:.3f} m")
                print(f"Tool target     : {np.round(tool_target, 3)} m")
                print(f"Joint6 target   : {np.round(joint6_target, 3)} m")
                print("=" * 68)
                self._say(f"发现{detection['color']}积木，准备抓取")
                return {
                    "camera_point": camera_point,
                    "joint6_target": joint6_target,
                    "tool_rotation": tool_rotation,
                    "tool_target": tool_target,
                    "base_point": target_base_point,
                    "detected_color": detection.get("color"),
                    "requested_color": requested_color,
                    "detection": detection,
                    "provisional_joints": provisional,
                    "scan_joint1": joint1,
                    "scan_joint_position": scan_joint_position,
                    "score": candidate["score"],
                    "gripper_width": candidate["gripper_width"],
                }

        print(f"[{label}] J1={joint1:+.2f}: no reachable GraspNet candidate.")
        return None

    @staticmethod
    def _color_matches_request(color, requested_color):
        if requested_color is None:
            return color != "unknown"
        return color == requested_color

    def _accumulate_candidate_color(
        self,
        camera_feed,
        detection,
        intrinsic,
        base_camera,
        requested_color,
        after_sequence,
        deadline=None,
    ):
        """Merge weak color evidence only while the arm is stationary."""
        cfg = self.config
        evidence_items = [detection["color_evidence"]]
        tracked = apply_accumulated_color(detection, evidence_items, 1, cfg)
        if (
            tracked["color"] != "unknown"
            and tracked["color_confidence"] >= cfg.color_single_frame_strong_ratio
        ):
            return tracked, after_sequence

        _, tracked_point = object_base_position(tracked, intrinsic, base_camera)
        matched_frames = 1
        marker = after_sequence
        if deadline is None:
            deadline = time.monotonic() + cfg.color_accumulation_timeout_s
        print(
            "[COLOR] weak single-frame evidence; accumulating within "
            f"{max(0.0, deadline - time.monotonic()):.1f}s budget.",
            flush=True,
        )
        for _frame_index in range(1, cfg.color_accumulation_max_frames):
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                break
            capture = camera_feed.wait_for_newer(
                marker,
                timeout=min(cfg.camera_detection_timeout, remaining),
            )
            if capture is None:
                break
            marker = capture["frame_seq"]

            nearest = None
            nearest_point = None
            nearest_distance = float("inf")
            for candidate in capture["detections"]:
                try:
                    _, candidate_point = object_base_position(
                        candidate,
                        intrinsic,
                        base_camera,
                    )
                except Exception:
                    continue
                distance = float(np.linalg.norm(candidate_point - tracked_point))
                if distance < nearest_distance:
                    nearest = candidate
                    nearest_point = candidate_point
                    nearest_distance = distance

            if (
                nearest is None
                or nearest_distance > cfg.color_track_position_tolerance_m
            ):
                continue

            tracked = dict(nearest)
            tracked_point = 0.7 * tracked_point + 0.3 * nearest_point
            evidence_items.append(tracked["color_evidence"])
            matched_frames += 1
            accumulated = apply_accumulated_color(
                tracked,
                evidence_items,
                matched_frames,
                cfg,
            )
            if (
                matched_frames >= cfg.color_accumulation_min_frames
                and accumulated["color_samples"] >= cfg.color_accumulation_min_samples
                and self._color_matches_request(
                    accumulated["color"],
                    requested_color,
                )
            ):
                return accumulated, marker

        return apply_accumulated_color(
            tracked,
            evidence_items,
            matched_frames,
            cfg,
        ), marker

    def _detect_at_pose(
        self,
        camera_feed,
        intrinsic,
        tcp_camera,
        selected_color,
        joint1,
        label,
        reference_base_point=None,
    ):
        cfg = self.config
        color_label = selected_color if selected_color is not None else "any colour"
        stable_joint_position = self.wait_until_stationary()
        if stable_joint_position is None:
            return None
        marker_method = getattr(camera_feed, "freshness_marker", None)
        after_timestamp = (
            marker_method() if callable(marker_method) else time.monotonic()
        )
        capture = None
        for warmup_index in range(cfg.scan_frame_warmup):
            capture = camera_feed.wait_for_newer(
                after_timestamp,
                timeout=cfg.camera_detection_timeout,
            )
            if capture is None:
                print(
                    f"[{label}] J1={joint1:+.2f}: camera frame "
                    f"{warmup_index + 1}/{cfg.scan_frame_warmup} timed out."
                )
                return None
            after_timestamp = capture.get(
                "frame_seq",
                capture["detections_timestamp"],
            )

        capture_intrinsic = capture.get("intrinsics")
        if capture_intrinsic is not None:
            intrinsic = capture_intrinsic

        scan_joint_position = self.current_joint_position()
        joint_drift = float(np.max(np.abs(scan_joint_position - stable_joint_position)))
        if joint_drift > cfg.detection_stationary_joint_tolerance:
            print(
                f"[{label}] arm drifted {joint_drift:.4f} rad during RGB-D inference; "
                "snapshot rejected."
            )
            return None
        print(
            f"[{label}] frame={capture['frame_seq']}, "
            f"infer={float(capture.get('inference_latency_s', 0.0)) * 1000.0:.0f} ms, "
            f"age={float(capture.get('snapshot_age_s', 0.0)) * 1000.0:.0f} ms, "
            f"joint_drift={joint_drift:.4f} rad"
        )
        with self.sdk_call():
            base_camera = get_base_camera_transform(
                self.robot,
                tcp_camera,
                scan_joint_position,
                tcp_in_joint6=cfg.tcp_in_joint6,
            )
        capture["robot_joint_position"] = scan_joint_position.copy()
        capture["base_camera"] = np.asarray(base_camera, dtype=float).copy()
        raw_detections = list(capture["detections"])
        if selected_color is None:
            known = [item for item in raw_detections if item.get("color") != "unknown"]
            relevant_detections = known if known else raw_detections
        else:
            exact = [
                item for item in raw_detections
                if item.get("color") == selected_color
            ]
            uncertain = [
                item for item in raw_detections
                if item.get("color") == "unknown"
                or float(item.get("color_confidence", 0.0))
                < cfg.color_single_frame_strong_ratio
            ]
            relevant_detections = exact if exact else uncertain
        relevant_detections.sort(
            key=lambda item: (
                float(item.get("color_confidence", 0.0)),
                float(item.get("confidence", 0.0)),
            ),
            reverse=True,
        )
        print(
            f"[COLOR] frame candidates={len(raw_detections)}, "
            f"relevant={len(relevant_detections)} for {color_label}.",
            flush=True,
        )
        detections = []
        color_marker = capture["frame_seq"]
        color_deadline = time.monotonic() + cfg.color_accumulation_timeout_s
        for candidate_index, detection in enumerate(relevant_detections, start=1):
            accumulated, color_marker = self._accumulate_candidate_color(
                camera_feed,
                detection,
                intrinsic,
                base_camera,
                selected_color,
                color_marker,
                deadline=color_deadline,
            )
            print(
                f"[COLOR] J1={joint1:+.2f} candidate {candidate_index}: "
                f"{accumulated['color']} "
                f"confidence={accumulated['color_confidence']:.1%}, "
                f"frames={accumulated['color_frames']}, "
                f"samples={accumulated['color_samples']}, "
                f"margin={accumulated['color_margin']:.1%}."
            )
            detections.append(accumulated)
        if selected_color is None:
            matches = [d for d in detections if d.get("color") != "unknown"]
        else:
            matches = [d for d in detections if d.get("color") == selected_color]
        if not matches:
            print(f"[{label}] J1={joint1:+.2f}: no {color_label} block.")
            return None

        central_matches = [
            detection for detection in matches
            if self._is_central_horizontal(detection)
        ]
        if not central_matches:
            for detection in matches:
                x1, _y1, x2, _y2 = detection["bbox"]
                print(
                    f"[{label}] J1={joint1:+.2f}: {color_label} block at "
                    f"x_center={0.5 * (x1 + x2):.1f} is outside central "
                    f"{self.config.central_x_grasp_ratio:.0%} region."
                )
            return None
        matches = central_matches

        if reference_base_point is not None:
            reference_base_point = np.asarray(reference_base_point, dtype=float)
            tracked = []
            for detection in matches:
                try:
                    _camera_point, candidate_base = object_base_position(
                        detection, intrinsic, base_camera
                    )
                except Exception:
                    continue
                tracked.append(
                    (
                        float(np.linalg.norm(candidate_base - reference_base_point)),
                        detection,
                    )
                )
            if not tracked:
                print(f"[{label}] original target could not be tracked.")
                return None
            target_shift, nearest_detection = min(tracked, key=lambda item: item[0])
            if target_shift > cfg.refine_target_match_radius_m:
                print(
                    f"[{label}] nearest same-colour target shifted {target_shift:.3f} m; "
                    "refusing to switch objects."
                )
                return None
            matches = [nearest_detection]

        if cfg.use_graspnet:
            return self._select_graspnet_candidate(
                camera_feed,
                capture,
                intrinsic,
                base_camera,
                matches,
                scan_joint_position,
                joint1,
                label,
                requested_color=selected_color,
            )

        for detection in matches:
            camera_point, base_point = object_base_position(
                detection, intrinsic, base_camera
            )
            try:
                (
                    tool_rotation,
                    approach_tilt_deg,
                    jaw_angle_deg,
                    jaw_image_angle_deg,
                    short_axis_projection_error_deg,
                ) = grasp_rotation_from_mask(
                    detection, camera_point, intrinsic, base_camera, cfg
                )
                tool_target, joint6_target = grasp_geometry(
                    base_point, tool_rotation, cfg
                )
            except Exception as exc:
                print(f"[{label}] J1={joint1:+.2f}: target geometry rejected: {exc!r}")
                continue
            provisional = self.validate_candidate(
                tool_target,
                joint6_target,
                tool_rotation,
                (cfg.manual_grasp_ik_seed, scan_joint_position),
                cfg.max_home_to_grasp_step,
                label="OBB grasp",
            )
            if provisional is None:
                print(f"[{label}] J1={joint1:+.2f}: target IK is invalid.")
                continue

            print("\n" + "=" * 68)
            print(f"[{label}] target found at J1={joint1:+.2f} rad")
            print(f"Requested color: {color_label}")
            print(f"Target: {detection['class_name']} ({detection['color']})")
            print(f"Confidence : {detection['confidence']:.3f}")
            print(f"Color vote : {detection['color_confidence']:.1%}")
            print(
                f"Pixel      : ({detection['pixel'][0]:.1f}, "
                f"{detection['pixel'][1]:.1f})"
            )
            print(
                f"Depth      : {detection['depth_m']:.3f} m, "
                f"samples={detection.get('depth_samples', 0)}, "
                f"spread={detection.get('depth_spread_m', float('nan')) * 1000.0:.1f} mm"
            )
            print(f"Camera xyz : {np.round(camera_point, 3)} m")
            print(f"Base xyz   : {np.round(base_point, 3)} m")
            print(f"Seeed-ray approach tilt      : {approach_tilt_deg:.1f} deg")
            print(f"Jaw axis projected to image  : {jaw_image_angle_deg:+.1f} deg")
            print(
                "OBB-to-jaw projection error  : "
                f"{short_axis_projection_error_deg:.1f} deg"
            )
            print(f"Jaw angle vs manual reference: {jaw_angle_deg:+.1f} deg")
            print(f"Tool target: {np.round(tool_target, 3)} m")
            print("=" * 68)
            self._say(f"发现{detection['color']}积木，准备抓取")
            return {
                "camera_point": camera_point,
                "joint6_target": joint6_target,
                "tool_rotation": tool_rotation,
                "tool_target": tool_target,
                "base_point": base_point,
                "detected_color": detection["color"],
                "requested_color": selected_color,
                "detection": detection,
                "provisional_joints": provisional,
                "scan_joint1": joint1,
                "scan_joint_position": scan_joint_position,
            }

        print(f"[{label}] J1={joint1:+.2f}: matching target is not graspable.")
        return None

    @staticmethod
    def _axis_error_deg(first, second, undirected=False):
        first = np.asarray(first, dtype=float)
        second = np.asarray(second, dtype=float)
        first /= float(np.linalg.norm(first))
        second /= float(np.linalg.norm(second))
        cosine = float(np.dot(first, second))
        if undirected:
            cosine = abs(cosine)
        return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))

    def refine_target_at_observation_pose(
        self,
        camera_feed,
        intrinsic,
        tcp_camera,
        selected_color,
        found,
        position_only=False,
    ):
        """Track one physical block over several coherent post-move snapshots."""
        cfg = self.config
        stage = "CLOSE-REFINE" if position_only else "REFINE"
        stable_joints = self.wait_until_stationary()
        if stable_joints is None:
            return None
        refine_started = time.monotonic()
        refine_deadline = refine_started + cfg.refine_total_timeout_s
        print(
            f"[{stage}] bounded frame budget={cfg.refine_total_timeout_s:.1f}s, "
            f"need={cfg.refine_required_observations} observation(s).",
            flush=True,
        )
        marker_method = getattr(camera_feed, "freshness_marker", None)
        marker = marker_method() if callable(marker_method) else time.monotonic()
        for warmup_index in range(cfg.refine_frame_warmup):
            remaining = refine_deadline - time.monotonic()
            if remaining <= 0.0:
                print(f"[{stage}] total frame budget expired during warmup.")
                return None
            capture = camera_feed.wait_for_newer(
                marker,
                timeout=min(cfg.refine_frame_timeout_s, remaining),
            )
            if capture is None:
                print(
                    f"[{stage}] warmup frame {warmup_index + 1}/"
                    f"{cfg.refine_frame_warmup} timed out."
                )
                return None
            marker = capture.get("frame_seq", capture["detections_timestamp"])

        current_joints = self.current_joint_position()
        if float(np.max(np.abs(current_joints - stable_joints))) > (
            cfg.detection_stationary_joint_tolerance
        ):
            print(f"[{stage}] arm drifted during warmup; observation rejected.")
            return None
        with self.sdk_call():
            base_camera = get_base_camera_transform(
                self.robot,
                tcp_camera,
                current_joints,
                tcp_in_joint6=cfg.tcp_in_joint6,
            )

        expected_base = np.asarray(found["base_point"], dtype=float)
        tracked_color = selected_color or found.get("detected_color")
        observations = []
        last_intrinsic = intrinsic
        quality_rejections = []
        for attempt in range(1, cfg.refine_max_frame_attempts + 1):
            remaining = refine_deadline - time.monotonic()
            if remaining <= 0.0:
                print(f"[{stage}] total frame budget expired.")
                break
            capture = camera_feed.wait_for_newer(
                marker,
                timeout=min(cfg.refine_frame_timeout_s, remaining),
            )
            if capture is None:
                print(
                    f"[{stage}] frame {attempt}/{cfg.refine_max_frame_attempts} "
                    "timed out."
                )
                continue
            marker = capture.get("frame_seq", capture["detections_timestamp"])
            capture_intrinsic = capture.get("intrinsics")
            if capture_intrinsic is not None:
                last_intrinsic = capture_intrinsic
            detections = list(capture.get("detections", []))
            if tracked_color is not None and not any(
                item.get("color") == tracked_color for item in detections
            ):
                fallback = detect_requested_color_regions(
                    capture["color_image"],
                    capture["depth_image"],
                    camera_feed.depth_scale,
                    tracked_color,
                    cfg,
                )
                if fallback:
                    print(
                        f"[{stage}] frame {attempt}: detector missed {tracked_color}; "
                        f"colour fallback found {len(fallback)} region(s)."
                    )
                    detections.extend(fallback)

            nearest = None
            nearest_camera = None
            nearest_base = None
            nearest_distance = float("inf")
            for detection in detections:
                if tracked_color is not None and detection.get("color") != tracked_color:
                    continue
                if position_only:
                    depth_m = float(detection.get("depth_m", float("nan")))
                    depth_spread_m = float(
                        detection.get("depth_spread_m", float("inf"))
                    )
                    bbox = np.asarray(detection.get("bbox", ()), dtype=float)
                    color_image = capture.get("color_image")
                    if color_image is None or bbox.shape != (4,):
                        quality_rejections.append("missing image/bbox")
                        continue
                    image_height, image_width = color_image.shape[:2]
                    x1, y1, x2, y2 = bbox
                    bbox_area_ratio = max(0.0, (x2 - x1) * (y2 - y1)) / float(
                        image_width * image_height
                    )
                    margin = cfg.close_refine_border_margin_px
                    if not np.isfinite(depth_m) or depth_m < cfg.close_refine_min_depth_m:
                        quality_rejections.append(f"depth={depth_m:.3f}m")
                        continue
                    if depth_spread_m > cfg.close_refine_max_depth_spread_m:
                        quality_rejections.append(
                            f"depth_spread={depth_spread_m * 1000.0:.1f}mm"
                        )
                        continue
                    if (
                        x1 <= margin
                        or y1 <= margin
                        or x2 >= image_width - margin
                        or y2 >= image_height - margin
                    ):
                        quality_rejections.append("bbox touches image border")
                        continue
                    if bbox_area_ratio > cfg.close_refine_max_bbox_area_ratio:
                        quality_rejections.append(
                            f"bbox_area={bbox_area_ratio:.1%}"
                        )
                        continue
                center = np.asarray(detection["pixel"], dtype=float)
                image_center = np.array(
                    [last_intrinsic.ppx, last_intrinsic.ppy],
                    dtype=float,
                )
                if float(np.linalg.norm(center - image_center)) > (
                    cfg.refine_max_center_distance_px
                ):
                    continue
                try:
                    camera_point, base_point = object_base_position(
                        detection,
                        last_intrinsic,
                        base_camera,
                    )
                except Exception:
                    continue
                distance = float(np.linalg.norm(base_point - expected_base))
                if distance < nearest_distance:
                    nearest = detection
                    nearest_camera = camera_point
                    nearest_base = base_point
                    nearest_distance = distance

            if nearest is None or nearest_distance > cfg.refine_target_match_radius_m:
                quality_detail = ""
                if position_only and quality_rejections:
                    quality_detail = f", quality={quality_rejections[-3:]}"
                print(
                    f"[{stage}] frame {attempt}/{cfg.refine_max_frame_attempts}: "
                    f"same target not found (detections={len(detections)}, "
                    f"nearest={nearest_distance:.3f} m{quality_detail})."
                )
                continue
            observations.append(
                {
                    "capture": capture,
                    "detection": nearest,
                    "camera_point": np.asarray(nearest_camera, dtype=float),
                    "base_point": np.asarray(nearest_base, dtype=float),
                    "distance": nearest_distance,
                }
            )
            print(
                f"[{stage}] observation {len(observations)}/"
                f"{cfg.refine_required_observations}: "
                f"base={np.round(nearest_base, 3)} m, match={nearest_distance:.3f} m, "
                f"pixel={np.round(nearest['pixel'], 1)}, "
                f"depth={float(nearest.get('depth_m', float('nan'))):.3f} m, "
                f"spread={float(nearest.get('depth_spread_m', float('nan'))) * 1000.0:.1f} mm"
            )
            if len(observations) >= cfg.refine_required_observations:
                break

        if len(observations) < cfg.refine_required_observations:
            print(
                f"[{stage}] not enough coherent observations after "
                f"{time.monotonic() - refine_started:.2f}s; returning to "
                "recognition pose."
            )
            return None
        points = np.asarray([item["base_point"] for item in observations], dtype=float)
        refined_base = np.median(points, axis=0)
        spread = float(np.max(np.linalg.norm(points - refined_base, axis=1)))
        raw_correction = refined_base - expected_base
        if position_only and cfg.close_refine_preserve_base_y:
            refined_base[1] = expected_base[1]
        correction = refined_base - expected_base
        correction_xy = float(np.linalg.norm(correction[:2]))
        correction_z = abs(float(correction[2]))
        correction_total = float(np.linalg.norm(correction))
        spread_limit = (
            cfg.close_refine_max_position_spread_m
            if position_only
            else cfg.refine_max_position_spread_m
        )
        xy_limit = (
            cfg.close_refine_max_xy_correction_m
            if position_only
            else cfg.refine_max_xy_correction_m
        )
        z_limit = (
            cfg.close_refine_max_z_correction_m
            if position_only
            else cfg.refine_max_z_correction_m
        )
        total_limit = (
            cfg.close_refine_max_total_correction_m
            if position_only
            else cfg.refine_max_total_correction_m
        )
        print(
            f"[{stage}] correction={np.round(correction, 4)} m, "
            f"spread={spread:.3f} m"
        )
        if position_only and cfg.close_refine_preserve_base_y:
            print(
                f"[{stage}] raw measured correction="
                f"{np.round(raw_correction * 1000.0, 1)} mm; "
                "Base-Y is preserved because the physical check found no "
                "repeatable Y bias."
            )
        if (
            spread > spread_limit
            or correction_xy > xy_limit
            or correction_z > z_limit
            or correction_total > total_limit
        ):
            print(f"[{stage}] correction/spread exceeds safety limits.")
            return None

        representative = min(
            observations,
            key=lambda item: float(np.linalg.norm(item["base_point"] - refined_base)),
        )
        if position_only:
            tool_rotation = np.asarray(found["tool_rotation"], dtype=float)
            try:
                tool_target, joint6_target = grasp_geometry(
                    refined_base,
                    tool_rotation,
                    cfg,
                )
            except Exception as exc:
                print(f"[{stage}] position-only geometry rejected: {exc!r}")
                return None
            provisional = self.validate_candidate(
                tool_target,
                joint6_target,
                tool_rotation,
                (
                    current_joints,
                    found.get("provisional_joints", current_joints),
                ),
                cfg.max_home_to_grasp_step,
                label="close-range position refinement",
            )
            if provisional is None:
                return None
            refreshed = dict(found)
            refreshed.update(
                {
                    "camera_point": representative["camera_point"],
                    "base_point": refined_base,
                    "tool_target": tool_target,
                    "joint6_target": joint6_target,
                    "provisional_joints": provisional,
                    "detection": representative["detection"],
                }
            )
            print(
                f"[{stage}] accepted position correction while preserving the "
                "far-field grasp orientation."
            )
            return refreshed
        if cfg.use_graspnet:
            refreshed = self._select_graspnet_candidate(
                camera_feed,
                representative["capture"],
                last_intrinsic,
                base_camera,
                [representative["detection"]],
                current_joints,
                float(current_joints[0]),
                "REFINE",
                requested_color=selected_color,
                target_base_override=refined_base,
            )
            if refreshed is not None:
                refreshed["scan_joint_position"] = found["scan_joint_position"]
                refreshed["scan_joint1"] = found["scan_joint1"]
            return refreshed
        try:
            (
                tool_rotation,
                approach_tilt_deg,
                _jaw_angle_deg,
                _jaw_image_angle_deg,
                _projection_error_deg,
            ) = grasp_rotation_from_mask(
                representative["detection"],
                representative["camera_point"],
                last_intrinsic,
                base_camera,
                cfg,
            )
            tool_target, joint6_target = grasp_geometry(
                refined_base,
                tool_rotation,
                cfg,
            )
        except Exception as exc:
            print(f"[REFINE] refined geometry rejected: {exc!r}")
            return None
        initial_rotation = np.asarray(found["tool_rotation"], dtype=float)
        open_axis_change = self._axis_error_deg(
            initial_rotation[:, 1],
            tool_rotation[:, 1],
            undirected=True,
        )
        if (
            approach_tilt_deg > cfg.refine_max_approach_tilt_deg
            or open_axis_change > cfg.refine_max_open_axis_change_deg
        ):
            print(
                "[REFINE] refined orientation rejected: "
                f"tilt={approach_tilt_deg:.1f} deg, "
                f"jaw_change={open_axis_change:.1f} deg"
            )
            return None
        provisional = self.validate_candidate(
            tool_target,
            joint6_target,
            tool_rotation,
            (
                current_joints,
                found.get("provisional_joints", current_joints),
                cfg.manual_grasp_ik_seed,
            ),
            cfg.max_home_to_grasp_step,
            label="refined OBB grasp",
        )
        if provisional is None:
            return None
        refreshed = dict(found)
        refreshed.update(
            {
                "camera_point": representative["camera_point"],
                "base_point": refined_base,
                "tool_target": tool_target,
                "joint6_target": joint6_target,
                "tool_rotation": tool_rotation,
                "provisional_joints": provisional,
                "detection": representative["detection"],
                "detected_color": representative["detection"]["color"],
            }
        )
        return refreshed

    def scan_for_target(
        self,
        camera_feed,
        intrinsic,
        tcp_camera,
        selected_color,
    ):
        cfg = self.config
        color_label = selected_color if selected_color is not None else "any colour"
        print(
            f"[SCAN] searching for {color_label} block: "
            f"J1 {cfg.scan_j1_start:+.2f} -> {cfg.scan_j1_end:+.2f} rad, "
            f"step {cfg.scan_j1_step:.2f} rad."
        )
        self._say(f"开始寻找{color_label}积木")

        current = self.current_joint_position()
        current_joint1 = float(current[0])
        print(
            f"[FAST] checking the current camera pose first at "
            f"J1={current_joint1:+.2f} rad ...",
            flush=True,
        )
        result = self._detect_at_pose(
            camera_feed,
            intrinsic,
            tcp_camera,
            selected_color,
            current_joint1,
            "FAST",
        )
        if result is not None:
            print("[FAST] target accepted without a J1 sweep.", flush=True)
            return result
        if self.interrupted.is_set():
            return None
        print("[FAST] target unavailable at current pose; starting J1 fallback sweep.")

        start_pose = self.scan_pose(cfg.scan_j1_start)
        print(f"[SCAN] moving HOME -> scan start J1={cfg.scan_j1_start:+.2f} rad ...")
        self.move_j(start_pose, cfg.scan_start_duration, "SCAN START")

        for index, joint1 in enumerate(self.scan_joint1_values()):
            if self.interrupted.is_set():
                return None
            if index:
                self.move_j(
                    self.scan_pose(joint1),
                    cfg.scan_step_duration,
                    f"SCAN J1={joint1:+.2f}",
                )
            if not self.sleep_interruptible(cfg.scan_camera_settle_time):
                return None

            result = self._detect_at_pose(
                camera_feed,
                intrinsic,
                tcp_camera,
                selected_color,
                joint1,
                "SCAN",
            )
            if result is not None:
                return result

        print(
            f"[SCAN] completed {cfg.scan_j1_start:+.2f} -> "
            f"{cfg.scan_j1_end:+.2f} rad; requested target was not found."
        )
        return None

    def detect_once_at_position(
        self,
        camera_feed,
        intrinsic,
        tcp_camera,
        selected_color,
        scan_joint_position,
    ):
        """Move back to a previously found scan pose and evaluate once."""
        cfg = self.config
        self.move_j(
            scan_joint_position,
            cfg.scan_step_duration,
            "RETRY SCAN POSE",
        )
        if not self.sleep_interruptible(cfg.scan_camera_settle_time):
            return None
        joint1 = float(scan_joint_position[0])
        return self._detect_at_pose(
            camera_feed,
            intrinsic,
            tcp_camera,
            selected_color,
            joint1,
            "RETRY",
        )

    def pre_grasp_and_redetect(
        self,
        camera_feed,
        intrinsic,
        tcp_camera,
        selected_color,
        found,
    ):
        """Partially centre the eye-in-hand camera, then refine over fresh frames."""
        cfg = self.config
        observation_joints = self.plan_observation_pose(tcp_camera, found)
        if observation_joints is not None:
            print("[OBSERVE] moving halfway toward a safer centred camera view ...")
            self.move_j(
                observation_joints,
                cfg.pre_grasp_duration,
                "CAMERA OBSERVATION",
            )
        with self.hold_current_pose("far-field RGB-D refinement"):
            if not self.sleep_interruptible(cfg.pre_grasp_camera_settle_time):
                return None
            print(
                "[REFINE] acquiring several coherent RGB-D observations before "
                "choosing the final grasp pose ..."
            )
            return self.refine_target_at_observation_pose(
                camera_feed,
                intrinsic,
                tcp_camera,
                selected_color,
                found,
            )

    def _return_to_recognition_pose(self, scan_joints, streamer, reason):
        """Recover from an empty grasp without terminating the operator loop."""
        scan_joints = np.asarray(scan_joints, dtype=float)
        print(f"[RECOVERY] {reason}; returning to the last recognition pose.")
        self.move_j(
            scan_joints,
            self.config.return_home_duration,
            "RECOGNITION POSE",
        )
        if streamer is not None:
            streamer.set_control_message(
                f"{reason}。机械臂已回到识别位置，请选择下一个目标。"
            )

    def run_grasp_loop(
        self,
        camera_feed,
        intrinsic,
        tcp_camera,
        streamer,
        select_target,
    ):
        """Run the interactive visual grasp loop with force-based retries."""
        cfg = self.config
        while not self.interrupted.is_set():
            selected_color = select_target()
            if selected_color is False:
                return False
            if streamer is not None:
                streamer.set_selected_color(selected_color)

            found = self.scan_for_target(
                camera_feed,
                intrinsic,
                tcp_camera,
                selected_color,
            )
            if self.interrupted.is_set():
                return False
            if found is None:
                print("[SCAN] no matching block found; returning HOME and asking again.")
                self.move_j(cfg.home, cfg.return_home_duration, "HOME")
                if streamer is not None:
                    streamer.set_control_message(
                        "没有识别到所选积木，已回到 HOME，请重新选择目标。"
                    )
                continue

            retry_scan_position = found["scan_joint_position"]
            task_complete = False
            for attempt in range(1, cfg.grasp_max_attempts + 1):
                if self.interrupted.is_set():
                    return False

                if attempt > 1:
                    print("[RETRY] returning to scan pose and re-detecting ...")
                    found = self.detect_once_at_position(
                        camera_feed,
                        intrinsic,
                        tcp_camera,
                        selected_color,
                        retry_scan_position,
                    )
                    if self.interrupted.is_set():
                        return False
                    if found is None:
                        self.open_gripper()
                        self._return_to_recognition_pose(
                            retry_scan_position,
                            streamer,
                            "重新识别失败",
                        )
                        break

                try:
                    if streamer is not None:
                        streamer.set_control_message(
                            "目标已锁定，正在计算观察位并进行多帧复检。"
                        )
                    found = self.pre_grasp_and_redetect(
                        camera_feed,
                        intrinsic,
                        tcp_camera,
                        selected_color,
                        found,
                    )
                except RuntimeError as exc:
                    print(f"[OBSERVE] planning/re-detection failed safely: {exc}")
                    found = None
                if self.interrupted.is_set():
                    return False
                if found is None:
                    self.open_gripper()
                    self._return_to_recognition_pose(
                        retry_scan_position,
                        streamer,
                        "预抓取复检失败",
                    )
                    break

                try:
                    if streamer is not None:
                        streamer.set_control_message(
                            "复检通过，正在规划自适应预抓取和直线接近。"
                        )
                    pre_grasp_joints = self.plan_pre_grasp(found)
                    if pre_grasp_joints is not None:
                        print(
                            "[PREGRASP] moving to the adaptive half-gap "
                            "approach waypoint ..."
                        )
                        self.move_j(
                            pre_grasp_joints,
                            cfg.pre_grasp_duration,
                            "ADAPTIVE PRE GRASP",
                            position_tolerance=cfg.pre_grasp_joint_tolerance_rad,
                        )
                        with self.hold_current_pose("pre-grasp settle"):
                            if not self.sleep_interruptible(
                                cfg.pre_grasp_camera_settle_time
                            ):
                                return False
                    realignment_trajectory = self.plan_pre_grasp_realignment(found)
                    if realignment_trajectory is not None:
                        print(
                            "[PREGRASP] correcting residual TCP error at the "
                            "safe standoff before the final approach ..."
                        )
                        self.execute_pre_grasp_realignment(realignment_trajectory)
                        with self.hold_current_pose("pre-grasp realignment settle"):
                            if not self.sleep_interruptible(
                                cfg.pre_grasp_camera_settle_time
                            ):
                                return False
                    if cfg.close_refine_enabled and not cfg.use_graspnet:
                        print(
                            "[CLOSE-REFINE] acquiring a bounded position-only "
                            "RGB-D correction at the safe standoff ..."
                        )
                        with self.hold_current_pose("close-range RGB-D refinement"):
                            close_refined = self.refine_target_at_observation_pose(
                                camera_feed,
                                intrinsic,
                                tcp_camera,
                                selected_color,
                                found,
                                position_only=True,
                            )
                        if close_refined is None:
                            print(
                                "[CLOSE-REFINE] close image is incomplete or "
                                "unstable; keeping the previously validated "
                                "far-field target."
                            )
                        else:
                            close_correction = (
                                np.asarray(close_refined["base_point"], dtype=float)
                                - np.asarray(found["base_point"], dtype=float)
                            )
                            found = close_refined
                            print(
                                "[CLOSE-REFINE] accepted Base correction: "
                                f"{np.round(close_correction * 1000.0, 1)} mm; "
                                "re-aligning the safe standoff."
                            )
                            close_realignment = self.plan_pre_grasp_realignment(found)
                            if close_realignment is not None:
                                self.execute_pre_grasp_realignment(close_realignment)
                                with self.hold_current_pose(
                                    "close-range realignment settle"
                                ):
                                    if not self.sleep_interruptible(
                                        cfg.pre_grasp_camera_settle_time
                                    ):
                                        return False
                    print(
                        "[GRASP] opening gripper while holding the validated "
                        "pre-grasp pose ..."
                    )
                    self.open_gripper()
                    if self.interrupted.is_set():
                        return False
                    print(
                        "[PREGRASP] waypoint aligned; validating the straight "
                        "final approach ..."
                    )
                    approach_trajectory = self.plan_cartesian_approach(found)
                except RuntimeError as exc:
                    print(f"[PREGRASP] final approach rejected safely: {exc}")
                    self.open_gripper()
                    self._return_to_recognition_pose(
                        retry_scan_position,
                        streamer,
                        "预抓取或直线接近规划失败",
                    )
                    break
                final = np.asarray(approach_trajectory[-1], dtype=float)
                print("[PLAN] full Cartesian approach verified; auto grasp starts now.")

                clamped, gpos, gtor = self.grasp_and_close(
                    final,
                    approach_trajectory,
                    streamer=streamer,
                    found=found,
                    gripper_preopened=True,
                )
                if not clamped:
                    print("[GRASP] clamp failed; releasing for another target.")
                    self.open_gripper()
                    self._return_to_recognition_pose(
                        retry_scan_position,
                        streamer,
                        "夹爪未抓到物体",
                    )
                    break

                force_magnitude = abs(float(gtor))
                print(
                    f"[GRASP] force after close: {force_magnitude:.3f} "
                    f"(threshold={cfg.grasp_min_force:.3f})"
                )
                if force_magnitude >= cfg.grasp_min_force:
                    if not self.finish_place_sequence():
                        raise RuntimeError("place sequence did not complete")
                    task_complete = True
                    break

                print(
                    f"[GRASP] force below threshold; releasing. "
                    f"attempt={attempt}/{cfg.grasp_max_attempts}"
                )
                self._say("抓取力不足，重新尝试")
                self.open_gripper()
                if attempt >= cfg.grasp_max_attempts:
                    print(
                        "[RETRY] second attempt still below threshold; "
                        "returning to recognition pose and asking again."
                    )
                    self._return_to_recognition_pose(
                        retry_scan_position,
                        streamer,
                        "夹爪力不足，未确认抓到物体",
                    )

            if task_complete:
                return True
        return False

    def pose_confirmed(self, target, label):
        cfg = self.config
        target = np.asarray(target, dtype=float)
        deadline = time.monotonic() + cfg.zero_verify_timeout
        stable = 0
        while time.monotonic() < deadline:
            with self.sdk_call():
                q = np.asarray(self.robot.get_current_pos(), dtype=float)
                dq = np.asarray(self.robot.get_current_vel(), dtype=float)
            if (
                q.shape != (6,)
                or dq.shape != (6,)
                or not np.all(np.isfinite(q))
                or not np.all(np.isfinite(dq))
            ):
                raise RuntimeError("invalid joint state during ZERO verification")
            if (
                np.max(np.abs(q - target)) <= cfg.zero_position_tolerance
                and np.max(np.abs(dq)) <= cfg.zero_velocity_tolerance
            ):
                stable += 1
                if stable >= cfg.zero_stable_samples:
                    print(
                        f"[SHUTDOWN] {label} confirmed: "
                        f"q={np.round(q, 4)}, dq={np.round(dq, 4)}"
                    )
                    return True
            else:
                stable = 0
            time.sleep(0.10)
        return False

    def zero_confirmed(self):
        return self.pose_confirmed(self.config.zero, "ZERO")

    def _fault_hold_and_stop(self, reason: Exception) -> bool:
        """Hold only a freshly measured pose, then stop; never replay stale HOME."""
        cfg = self.config
        self.lifecycle_state = RobotLifecycleState.FAULT_HOLD
        print(f"[SAFETY FAULT] controlled shutdown fallback: {reason!r}")
        try:
            hold = self.current_joint_position()
            if hold.shape != (6,) or not np.all(np.isfinite(hold)):
                raise RuntimeError("current joint feedback is invalid")
            with self.sdk_call():
                self.robot.Joint_Pos_Vel(
                    hold.tolist(),
                    [0.0] * 6,
                    cfg.max_torque,
                    iswait=False,
                )
            self._remember_command(hold)
            time.sleep(max(0.0, cfg.shutdown_fault_hold_time))
        except Exception as exc:
            print(
                "[SAFETY FAULT] no position hold was sent because fresh joint "
                f"feedback is unavailable: {exc!r}"
            )

        try:
            with self.sdk_call():
                self.robot.set_stop()
            self.lifecycle_state = RobotLifecycleState.STOPPED
            print("[SAFETY FAULT] motor stop command sent.")
        except Exception as stop_exc:
            print(f"[SAFETY FAULT] motor stop command failed: {stop_exc!r}")
        return False

    def safe_shutdown(
        self,
        pipeline=None,
        last_command=None,
        *,
        return_home=False,
        shutdown_target=None,
        shutdown_label=None,
    ):
        """Converge every exit path to a finite ZERO-or-fault-stop sequence.

        ``last_command`` remains in the signature for callers built against the
        previous API, but is deliberately ignored: replaying a stale HOME after
        joint-feedback failure is unsafe. ``shutdown_target`` lets the owning
        single-process entry return to a freshly captured pre-program pose;
        callers that do not provide it retain the ZERO/HOME compatibility path.
        """
        del last_command
        with self._shutdown_lock:
            if self._shutdown_complete:
                return self._shutdown_succeeded

            return self._safe_shutdown_once(
                pipeline,
                return_home=return_home,
                shutdown_target=shutdown_target,
                shutdown_label=shutdown_label,
            )

    def _safe_shutdown_once(
        self,
        pipeline,
        *,
        return_home=False,
        shutdown_target=None,
        shutdown_label=None,
    ) -> bool:
        cfg = self.config
        if shutdown_target is not None:
            target = np.asarray(shutdown_target, dtype=float)
            label = str(shutdown_label or "STARTUP POSE")
            duration = cfg.return_home_duration
        else:
            target = cfg.home if return_home else cfg.zero
            label = "HOME" if return_home else "ZERO"
            duration = cfg.return_home_duration if return_home else cfg.zero_duration
        timeout = max(cfg.zero_move_timeout, duration + 5.0)
        print(f"\n[SHUTDOWN] stopping camera and returning to {label} ...")
        self._say(f"任务结束，机械臂返回{label}")
        self.lifecycle_state = RobotLifecycleState.STOPPING
        if pipeline is not None:
            try:
                pipeline.stop()
            except Exception as exc:
                print(f"[SHUTDOWN] camera stop warning: {exc!r}")
        try:
            if (
                target.shape != (6,)
                or not np.all(np.isfinite(target))
                or np.any(target < cfg.joint_lower)
                or np.any(target > cfg.joint_upper)
            ):
                raise RuntimeError(f"invalid {label} shutdown target: {target!r}")
            with self.sdk_call():
                result = self.robot.moveJ(
                    target.tolist(),
                    duration=duration,
                    max_tqu=cfg.max_torque,
                    iswait=True,
                    timeout=timeout,
                )
            if result is False:
                raise RuntimeError(f"{label} move rejected or timed out")
            self._remember_command(target)
            time.sleep(cfg.zero_settle_time)
            if not self.pose_confirmed(target, label):
                raise RuntimeError(f"{label} was not confirmed")
            with self.sdk_call():
                self.robot.set_stop()
            self.lifecycle_state = RobotLifecycleState.STOPPED
            self._shutdown_complete = True
            self._shutdown_succeeded = True
            print("[SHUTDOWN] motors stopped.")
            return True
        except Exception as exc:
            print(f"[SHUTDOWN] failure: {exc!r}")
            result = self._fault_hold_and_stop(exc)
            self._shutdown_complete = True
            self._shutdown_succeeded = False
            return result
