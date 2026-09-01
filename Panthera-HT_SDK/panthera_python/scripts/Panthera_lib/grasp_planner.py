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
        with self.sdk_call():
            joints = self.robot.inverse_kinematics(
                target_joint6.tolist(),
                tool_rotation,
                seed.tolist(),
                multi_init=True,
                num_attempts=8,
            )
        if joints is None:
            print("[IK] solver returned no solution for this seed.")
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
        return joints

    def validate_candidate(
        self,
        tool_target,
        joint6_target,
        tool_rotation,
        seeds,
        jump_limit,
        label="candidate",
    ):
        """Apply the same workspace and IK policy to every grasp backend."""
        tool_target = np.asarray(tool_target, dtype=float)
        joint6_target = np.asarray(joint6_target, dtype=float)
        tool_rotation = np.asarray(tool_rotation, dtype=float)
        if not workspace_ok(tool_target, joint6_target, self.config):
            print(f"[PLAN] {label} rejected by workspace limits.")
            return None
        for seed in seeds:
            solution = self.solve_ik(
                joint6_target,
                tool_rotation,
                np.asarray(seed, dtype=float),
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

    def move_j(self, joints, duration, label, wait=True):
        joints = np.asarray(joints, dtype=float)
        with self.sdk_call():
            result = self.robot.move_j_checked(
                joints,
                duration=duration,
                max_torque=self.config.max_torque,
                label=label,
                wait=wait,
            )
        if result is False:
            raise RuntimeError(f"{label} move failed")
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

    def open_gripper(self):
        cfg = self.config
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
            if self.interrupted.is_set():
                return False
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
            with self.sdk_call():
                self.robot.Joint_Pos_Vel(
                    target_joints.tolist(),
                    [0.0] * 6,
                    self.config.max_torque,
                    iswait=False,
                )
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
        while time.monotonic() < deadline and not self.interrupted.is_set():
            with self.sdk_call():
                latest = np.asarray(self.robot.current_joint_position(), dtype=float)
                velocity = np.asarray(self.robot.get_current_vel(), dtype=float)
            if (
                latest.shape != (6,)
                or velocity.shape != (6,)
                or not np.all(np.isfinite(latest))
                or not np.all(np.isfinite(velocity))
            ):
                raise RuntimeError("invalid joint feedback while waiting for camera stability")
            if float(np.max(np.abs(velocity))) <= cfg.detection_stationary_velocity_tolerance:
                stable += 1
                if stable >= cfg.detection_stationary_stable_samples:
                    return latest
            else:
                stable = 0
            time.sleep(0.05)
        print("[VISION] arm did not become stationary before RGB-D capture.")
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

    def plan_pre_grasp(self, found):
        """Plan a waypoint 5 cm behind the compensated final grasp pose."""
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
        if (
            cfg.pre_grasp_min_distance_m <= axial <= cfg.pre_grasp_offset_m
            and lateral <= cfg.pre_grasp_lateral_tolerance_m
            and orientation <= cfg.pre_grasp_orientation_tolerance_deg
        ):
            print(
                f"[PREGRASP] already inside approach corridor: axial={axial:.3f} m, "
                f"lateral={lateral:.3f} m, rotation={orientation:.2f} deg; "
                "skipping the free-space waypoint."
            )
            return None
        if (
            axial < cfg.pre_grasp_min_distance_m
            and lateral <= cfg.pre_grasp_lateral_tolerance_m
        ):
            raise RuntimeError(
                f"tip is too close to or beyond the grasp plane: axial={axial:.3f} m"
            )

        pre_tool_target = tool_target - cfg.pre_grasp_offset_m * approach
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
        print(
            f"[PREGRASP] planned tip waypoint {cfg.pre_grasp_offset_m:.3f} m "
            f"before final grasp: {np.round(pre_tool_target, 3)}"
        )
        return planned

    def plan_cartesian_approach(self, found):
        """Plan and validate the final straight TCP approach sample by sample."""
        cfg = self.config
        current = self.wait_until_stationary()
        if current is None:
            raise RuntimeError("arm is not stationary before Cartesian approach")
        axial, lateral, orientation, start_tool, approach = self.pre_grasp_alignment(
            found, current
        )
        if not (
            cfg.pre_grasp_min_distance_m <= axial <= cfg.pre_grasp_offset_m + 0.005
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

        previous_progress = -1e-6
        previous_joints = trajectory[0]
        jump_limit = float(getattr(self.robot, "jump_threshold", 1.5))
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
            cross_track = float(
                np.linalg.norm((tool - start_tool) - progress * approach)
            )
            if progress + 1e-4 < previous_progress or progress > axial + 0.001:
                raise RuntimeError(f"Cartesian sample {index} is not axially monotonic")
            if cross_track > cfg.approach_path_lateral_tolerance_m:
                raise RuntimeError(
                    f"Cartesian sample {index} cross-track error={cross_track:.4f} m"
                )
            if self.rotation_error_deg(rotation, end_pose["rotation"]) > (
                cfg.pre_grasp_orientation_tolerance_deg
            ):
                raise RuntimeError(f"Cartesian sample {index} rotates outside tolerance")
            previous_progress = progress
            previous_joints = joints

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
            f"[PREGRASP] Cartesian approach verified: travel={axial:.3f} m, "
            f"samples={len(trajectory)}, endpoint_error={endpoint_error:.4f} m"
        )
        return trajectory

    def execute_cartesian_approach(self, trajectory):
        with self.sdk_call():
            self.robot.execute_joint_trajectory_checked(
                trajectory,
                duration=self.config.direct_grasp_duration,
                max_torque=self.config.max_torque,
                label="FINAL CARTESIAN APPROACH",
            )
        self._remember_command(trajectory[-1])

    def grasp_and_close(self, final, approach_trajectory, streamer=None):
        cfg = self.config
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
            pos_tol=0.015,
            vel_tol=0.2,
        )
        if not settled:
            raise RuntimeError("final grasp pose did not settle; gripper close aborted")

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

        print("[GRASP] returning to HOME while holding the object ...")
        self.move_j(cfg.home, cfg.return_home_duration, "HOME")
        if self.interrupted.is_set():
            return False
        print(f"[PUT1] moving HOME -> PUT1 while holding: {np.round(cfg.put1, 3)}")
        self.move_j(cfg.put1, cfg.put1_duration, "PUT1")
        if self.interrupted.is_set():
            return False
        print("[PUT1] opening gripper fully to release the object ...")
        self.open_gripper()
        if self.interrupted.is_set():
            return False
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
    ):
        """Choose the highest-scoring GraspNet candidate that is reachable."""
        cfg = self.config
        if self.graspnet_provider is None:
            raise RuntimeError(
                "use_graspnet is enabled but no GraspNetCandidateProvider was passed"
            )

        current_joints = self.current_joint_position()
        for detection in matches:
            _camera_point, target_base_point = object_base_position(
                detection, intrinsic, base_camera
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
                    "joint6_target": joint6_target,
                    "tool_rotation": tool_rotation,
                    "tool_target": tool_target,
                    "base_point": target_base_point,
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
        for _frame_index in range(1, cfg.color_accumulation_max_frames):
            capture = camera_feed.wait_for_newer(
                marker,
                timeout=cfg.camera_detection_timeout,
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
            )
        capture["robot_joint_position"] = scan_joint_position.copy()
        capture["base_camera"] = np.asarray(base_camera, dtype=float).copy()
        detections = []
        color_marker = capture["frame_seq"]
        for candidate_index, detection in enumerate(capture["detections"], start=1):
            accumulated, color_marker = self._accumulate_candidate_color(
                camera_feed,
                detection,
                intrinsic,
                base_camera,
                selected_color,
                color_marker,
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
            if target_shift > cfg.pre_grasp_abort_shift_m:
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
                "joint6_target": joint6_target,
                "tool_rotation": tool_rotation,
                "tool_target": tool_target,
                "base_point": base_point,
                "provisional_joints": provisional,
                "scan_joint1": joint1,
                "scan_joint_position": scan_joint_position,
            }

        print(f"[{label}] J1={joint1:+.2f}: matching target is not graspable.")
        return None

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
        """Move to pre-grasp, re-detect the same object, then realign once."""
        cfg = self.config
        original_base_point = np.asarray(found["base_point"], dtype=float)
        pre_grasp_joints = self.plan_pre_grasp(found)
        if pre_grasp_joints is not None:
            print("[PREGRASP] moving to the 5 cm waypoint ...")
            self.move_j(
                pre_grasp_joints,
                cfg.pre_grasp_duration,
                "PRE GRASP",
            )
        if not self.sleep_interruptible(cfg.pre_grasp_camera_settle_time):
            return None

        current = self.current_joint_position()
        print("[PREGRASP] acquiring a fresh RGB-D detection before final grasp ...")
        refreshed = self._detect_at_pose(
            camera_feed,
            intrinsic,
            tcp_camera,
            selected_color,
            float(current[0]),
            "PREGRASP",
            reference_base_point=original_base_point,
        )
        if refreshed is None:
            raise RuntimeError(
                "target was not confirmed from the pre-grasp pose; grasp aborted"
            )
        target_shift = float(
            np.linalg.norm(np.asarray(refreshed["base_point"]) - original_base_point)
        )
        print(f"[PREGRASP] refreshed target shift={target_shift:.3f} m")

        correction = self.plan_pre_grasp(refreshed)
        if correction is not None:
            if target_shift > cfg.pre_grasp_abort_shift_m:
                raise RuntimeError(
                    f"refreshed target shifted {target_shift:.3f} m; grasp aborted"
                )
            print("[PREGRASP] realigning to the refreshed 5 cm waypoint ...")
            self.move_j(
                correction,
                cfg.pre_grasp_duration,
                "REFRESHED PRE GRASP",
            )
            if not self.sleep_interruptible(cfg.pre_grasp_camera_settle_time):
                return None
        return refreshed

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

                found = self.pre_grasp_and_redetect(
                    camera_feed,
                    intrinsic,
                    tcp_camera,
                    selected_color,
                    found,
                )
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

                print("[PREGRASP] refreshed target found; validating straight approach ...")
                approach_trajectory = self.plan_cartesian_approach(found)
                final = np.asarray(approach_trajectory[-1], dtype=float)
                print("[PLAN] full Cartesian approach verified; auto grasp starts now.")

                clamped, gpos, gtor = self.grasp_and_close(
                    final,
                    approach_trajectory,
                    streamer=streamer,
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
