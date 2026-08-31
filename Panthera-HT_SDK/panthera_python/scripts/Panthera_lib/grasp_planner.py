"""High-level grasp planning, execution and safe shutdown."""

from __future__ import annotations

import contextlib
import signal
import threading
import time

import numpy as np

from .grasp_config import GraspConfig
from .vision_pipeline import (
    get_base_camera_transform,
    grasp_geometry,
    grasp_rotation_from_mask,
    object_base_position,
    workspace_ok,
)


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
            return None
        joints = np.asarray(joints, dtype=float)
        if (
            joints.shape != (6,)
            or not np.all(np.isfinite(joints))
            or np.any(joints < self.config.joint_lower)
            or np.any(joints > self.config.joint_upper)
            or np.max(np.abs(joints - seed)) > jump_limit
        ):
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
        with self.sdk_call():
            self.robot.move_j_checked(
                joints,
                duration=duration,
                max_torque=self.config.max_torque,
                label=label,
                wait=wait,
            )

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

    def plan_graspnet_pre_grasp(self, found):
        """Plan an approach waypoint just before the GraspNet grasp pose."""
        cfg = self.config
        tool_target = np.asarray(found["tool_target"], dtype=float)
        tool_rotation = np.asarray(found["tool_rotation"], dtype=float)
        approach = tool_rotation[:, 0]
        pre_tool_target = tool_target - cfg.graspnet_pre_grasp_offset * approach
        tcp_offset = tool_rotation @ cfg.tcp_in_joint6
        pre_joint6_target = pre_tool_target - tcp_offset
        seed = found.get("provisional_joints", cfg.manual_grasp_ik_seed)
        return self.solve_ik(
            pre_joint6_target,
            tool_rotation,
            np.asarray(seed, dtype=float),
            cfg.graspnet_max_joint_jump,
        )

    def grasp_and_close(self, final, streamer=None, pre_grasp_joints=None):
        cfg = self.config
        print("[GRASP] opening gripper before direct grasp ...")
        self.open_gripper()
        if self.interrupted.is_set():
            return False, float("nan"), float("nan")

        if pre_grasp_joints is not None:
            pre_grasp_joints = np.asarray(pre_grasp_joints, dtype=float)
            print("[GRASP] moving to pre-grasp waypoint ...")
            self.move_j(
                pre_grasp_joints,
                cfg.direct_grasp_duration,
                "PRE GRASP",
                wait=False,
            )
            if not self.sleep_interruptible(
                cfg.direct_grasp_duration + cfg.direct_grasp_post_command_wait
            ):
                return False, float("nan"), float("nan")
            self.settle_at_grasp(
                pre_grasp_joints,
                duration=cfg.direct_grasp_settle_timeout,
                pos_tol=0.015,
                vel_tol=0.2,
            )

        print("[GRASP] moving directly HOME -> grasp pose ...")
        self.move_j(final, cfg.direct_grasp_duration, "DIRECT GRASP", wait=False)
        print("[GRASP] direct command sent; waiting for motion ...")
        if not self.sleep_interruptible(
            cfg.direct_grasp_duration + cfg.direct_grasp_post_command_wait
        ):
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
            print("[GRASP] strict settle timed out; holding final command before closing.")
            self.hold_arm(final, 0.5)

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

    def execute_grasp(self, final, streamer=None):
        clamped, _gpos, _gtor = self.grasp_and_close(final, streamer)
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

                provisional = None
                for seed in (current_joints, cfg.manual_grasp_ik_seed):
                    provisional = self.solve_ik(
                        joint6_target,
                        tool_rotation,
                        np.asarray(seed, dtype=float),
                        cfg.graspnet_max_joint_jump,
                    )
                    if provisional is not None:
                        break
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
                print(f"Depth           : {detection['depth_m']:.3f} m")
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
                    "provisional_joints": provisional,
                    "scan_joint1": joint1,
                    "scan_joint_position": scan_joint_position,
                    "score": candidate["score"],
                    "gripper_width": candidate["gripper_width"],
                }

        print(f"[{label}] J1={joint1:+.2f}: no reachable GraspNet candidate.")
        return None

    def _detect_at_pose(
        self,
        camera_feed,
        intrinsic,
        tcp_camera,
        selected_color,
        joint1,
        label,
    ):
        cfg = self.config
        color_label = selected_color if selected_color is not None else "any colour"
        sample_time = time.monotonic()
        capture = camera_feed.wait_for_newer(
            sample_time, timeout=self.config.camera_detection_timeout
        )
        if capture is None:
            print(f"[{label}] J1={joint1:+.2f}: invalid camera frame.")
            return None

        with self.sdk_call():
            base_camera = get_base_camera_transform(self.robot, tcp_camera)
        detections = capture["detections"]
        if selected_color is None:
            matches = detections
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

        scan_joint_position = self.current_joint_position()
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
            if not workspace_ok(tool_target, joint6_target, cfg):
                continue

            provisional = self.solve_ik(
                joint6_target,
                tool_rotation,
                cfg.manual_grasp_ik_seed,
                cfg.max_home_to_grasp_step,
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
            print(f"Depth      : {detection['depth_m']:.3f} m")
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

        print("[SCAN] completed +2.30 -> -2.30 rad; requested target was not found.")
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
            if found is None or self.interrupted.is_set():
                return False

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
                    if found is None or self.interrupted.is_set():
                        return False

                print("[SCAN] valid target found; checking direct scan-pose IK ...")
                if cfg.use_graspnet and "provisional_joints" in found:
                    final = np.asarray(found["provisional_joints"], dtype=float)
                    print("[PLAN] reusing the GraspNet-candidate IK solution.")
                else:
                    final = self.plan_grasp(
                        found["joint6_target"], found["tool_rotation"]
                    )
                print("[PLAN] direct scan-pose grasp IK verified; auto grasp starts now.")

                pre_grasp_joints = None
                if cfg.use_graspnet and "tool_target" in found:
                    pre_grasp_joints = self.plan_graspnet_pre_grasp(found)
                    if pre_grasp_joints is None:
                        print("[GRASP] pre-grasp IK failed; using direct grasp only.")

                clamped, gpos, gtor = self.grasp_and_close(
                    final,
                    streamer=streamer,
                    pre_grasp_joints=pre_grasp_joints,
                )
                if not clamped:
                    print("[GRASP] clamp failed; releasing and returning HOME.")
                    self.open_gripper()
                    self.move_j(
                        cfg.home,
                        cfg.return_home_duration,
                        "HOME",
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
                        "returning HOME and asking again."
                    )
                    self.move_j(
                        cfg.home,
                        cfg.return_home_duration,
                        "HOME",
                    )

            if task_complete:
                return True
        return False

    def zero_confirmed(self):
        cfg = self.config
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
                np.max(np.abs(q - cfg.zero)) <= cfg.zero_position_tolerance
                and np.max(np.abs(dq)) <= cfg.zero_velocity_tolerance
            ):
                stable += 1
                if stable >= cfg.zero_stable_samples:
                    print(f"[SHUTDOWN] ZERO confirmed: q={np.round(q, 4)}, dq={np.round(dq, 4)}")
                    return True
            else:
                stable = 0
            time.sleep(0.10)
        return False

    def hold_position_forever(self, last_command):
        print("[SAFETY FAULT] ZERO not confirmed; set_stop is forbidden. Holding with power.")
        try:
            hold = self.current_joint_position()
        except Exception:
            hold = np.asarray(last_command, dtype=float)
        while True:
            try:
                with self.sdk_call():
                    self.robot.Joint_Pos_Vel(
                        hold.tolist(),
                        [0.10] * 6,
                        self.config.max_torque,
                        iswait=False,
                    )
            except Exception as exc:
                print(f"[SAFETY FAULT] hold command failed: {exc!r}")
            time.sleep(0.10)

    def safe_shutdown(self, pipeline, last_command):
        cfg = self.config
        print("\n[SHUTDOWN] stopping camera and returning to ZERO ...")
        self._say("任务结束，机械臂回零")
        if pipeline is not None:
            try:
                pipeline.stop()
            except Exception as exc:
                print(f"[SHUTDOWN] camera stop warning: {exc!r}")
        try:
            with self.sdk_call():
                result = self.robot.moveJ(
                    cfg.zero.tolist(),
                    duration=cfg.zero_duration,
                    max_tqu=cfg.max_torque,
                    iswait=True,
                )
            if result is False:
                raise RuntimeError("ZERO move rejected or timed out")
            time.sleep(cfg.zero_settle_time)
            if not self.zero_confirmed():
                raise RuntimeError("ZERO was not confirmed")
            with self.sdk_call():
                self.robot.set_stop()
            print("[SHUTDOWN] motors stopped.")
        except Exception as exc:
            print(f"[SHUTDOWN] failure: {exc!r}")
            self.hold_position_forever(last_command)
