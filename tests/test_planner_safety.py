from __future__ import annotations

import threading
import unittest
from unittest.mock import patch

import numpy as np

from Panthera_lib.grasp_config import GraspConfig
from Panthera_lib.grasp_planner import (
    GraspPlanner,
    RobotLifecycleState,
)
from Panthera_lib.vision_pipeline import grasp_geometry


class NoopRobot:
    pass


class PlannerSafetyTests(unittest.TestCase):
    def test_planner_ik_uses_one_bounded_explicit_seed(self):
        class Robot:
            def __init__(self):
                self.kwargs = None

            def inverse_kinematics(self, *_args, **kwargs):
                self.kwargs = kwargs
                return np.zeros(6)

            def forward_kinematics(self, _joints):
                return {"position": np.zeros(3), "rotation": np.eye(3)}

        robot = Robot()
        config = GraspConfig()
        planner = GraspPlanner(robot, config, threading.Event())

        result = planner.solve_ik(
            np.zeros(3),
            np.eye(3),
            np.zeros(6),
            jump_limit=1.0,
        )

        self.assertIsNotNone(result)
        self.assertFalse(robot.kwargs["multi_init"])
        self.assertEqual(
            robot.kwargs["max_iter"],
            config.ik_single_seed_max_iterations,
        )

    def test_candidate_validation_deduplicates_and_caps_seed_attempts(self):
        config = GraspConfig()
        config.ik_max_seed_attempts = 2
        planner = GraspPlanner(NoopRobot(), config, threading.Event())
        attempts = []
        planner.solve_ik = lambda _target, _rotation, seed, _jump: (
            attempts.append(np.asarray(seed).copy()) or None
        )

        result = planner.validate_candidate(
            np.array([0.30, 0.0, 0.10]),
            np.array([0.20, 0.0, 0.10]),
            np.eye(3),
            (
                np.zeros(6),
                np.zeros(6),
                np.ones(6),
                np.full(6, 2.0),
            ),
            3.0,
            label="bounded test",
        )

        self.assertIsNone(result)
        self.assertEqual(len(attempts), 2)
        self.assertTrue(np.allclose(attempts[0], np.zeros(6)))
        self.assertTrue(np.allclose(attempts[1], np.ones(6)))

    def test_move_wait_uses_duration_bounded_feedback_timeout(self):
        class Robot:
            def __init__(self):
                self.kwargs = None

            def move_j_checked(self, *_args, **kwargs):
                self.kwargs = kwargs
                return True

        robot = Robot()
        config = GraspConfig()
        planner = GraspPlanner(robot, config, threading.Event())

        planner.move_j(np.zeros(6), 3.0, "TEST MOVE")

        self.assertEqual(robot.kwargs["timeout"], 7.0)

    def test_move_can_request_tighter_pre_grasp_joint_tolerance(self):
        class Robot:
            def __init__(self):
                self.kwargs = None

            def move_j_checked(self, *_args, **kwargs):
                self.kwargs = kwargs
                return True

        robot = Robot()
        planner = GraspPlanner(robot, GraspConfig(), threading.Event())

        planner.move_j(
            np.zeros(6),
            3.0,
            "PREGRASP",
            position_tolerance=0.02,
        )

        self.assertEqual(robot.kwargs["tolerance"], 0.02)

    def test_edge_visible_block_is_inside_expanded_image_region(self):
        planner = GraspPlanner(NoopRobot(), GraspConfig(), threading.Event())
        self.assertTrue(
            planner._is_central_horizontal({"bbox": (70, 100, 130, 180)})
        )
        self.assertFalse(
            planner._is_central_horizontal({"bbox": (0, 100, 40, 180)})
        )

    def test_web_joint1_jog_moves_only_joint1_by_half_radian(self):
        class Robot:
            def __init__(self):
                self.command = None

            def move_j_checked(self, joints, **_kwargs):
                self.command = np.asarray(joints, dtype=float)
                return True

        robot = Robot()
        config = GraspConfig()
        planner = GraspPlanner(robot, config, threading.Event())
        initial = config.home.copy()
        settled = initial.copy()
        settled[0] = 0.5
        samples = iter((initial, settled))
        planner.wait_until_stationary = lambda: next(samples)

        final_j1 = planner.jog_joint1("left")

        self.assertAlmostEqual(final_j1, 0.5)
        self.assertTrue(np.allclose(robot.command[1:], initial[1:]))
        self.assertAlmostEqual(float(robot.command[0]), 0.5)

    def test_grasp_overtravel_follows_pose_specific_approach_axis(self):
        config = GraspConfig()
        config.grasp_approach_overtravel_m = 0.005
        base_point = np.array([0.10, -0.02, 0.12])
        tool_rotation = np.array(
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        tool_target, joint6_target = grasp_geometry(
            base_point,
            tool_rotation,
            config,
        )

        nominal = base_point + config.grasp_offset_base
        self.assertTrue(
            np.allclose(
                tool_target - nominal,
                [0.0, config.grasp_approach_overtravel_m, 0.0],
            )
        )
        self.assertTrue(
            np.allclose(
                joint6_target,
                tool_target - tool_rotation @ config.tcp_in_joint6,
            )
        )

    def test_grasp_settle_is_passive_and_does_not_resend_zero_velocity(self):
        class Robot:
            @staticmethod
            def get_current_vel():
                return np.zeros(6)

            def Joint_Pos_Vel(self, *_args, **_kwargs):
                self.fail("settle verification must not resend a zero-velocity move")

        planner = GraspPlanner(Robot(), GraspConfig(), threading.Event())
        samples = iter((np.full(6, 0.02), np.zeros(6)))
        planner.current_joint_position = lambda: next(samples)

        self.assertTrue(
            planner.settle_at_grasp(
                np.zeros(6),
                duration=0.2,
                pos_tol=0.01,
                vel_tol=0.1,
            )
        )

    def test_web_joint1_jog_rejects_non_home_arm_shape(self):
        config = GraspConfig()
        planner = GraspPlanner(NoopRobot(), config, threading.Event())
        unsafe = config.home.copy()
        unsafe[1] += config.joint1_jog_posture_tolerance_rad + 0.1
        planner.wait_until_stationary = lambda: unsafe

        with self.assertRaises(RuntimeError):
            planner.jog_joint1("right")

    def test_stationary_check_refreshes_motor_feedback_before_sampling(self):
        class Robot:
            def __init__(self):
                self.refresh_count = 0

            def refresh_motor_state(self):
                self.refresh_count += 1

            def current_joint_position(self):
                return np.zeros(6)

            def get_current_vel(self):
                return np.zeros(6)

        config = GraspConfig()
        config.detection_stationary_stable_samples = 2
        config.detection_stationary_timeout = 0.3
        robot = Robot()
        planner = GraspPlanner(robot, config, threading.Event())

        stable = planner.wait_until_stationary()

        self.assertTrue(np.allclose(stable, np.zeros(6)))
        self.assertGreaterEqual(robot.refresh_count, 3)

    def test_pre_grasp_uses_half_current_axial_gap(self):
        class Robot:
            def current_joint_position(self):
                return np.zeros(6)

            def forward_kinematics(self, joints):
                return {"position": [0.10, 0.0, 0.10], "rotation": np.eye(3)}

        config = GraspConfig()
        planner = GraspPlanner(Robot(), config, threading.Event())
        captured = {}

        def validate(tool, joint6, rotation, seeds, jump_limit, label):
            captured["tool"] = tool
            captured["joint6"] = joint6
            captured["label"] = label
            return np.ones(6)

        planner.validate_candidate = validate
        found = {
            "tool_target": np.array([0.315, 0.0, 0.10]),
            "tool_rotation": np.eye(3),
            "provisional_joints": np.zeros(6),
        }

        result = planner.plan_pre_grasp(found)

        self.assertTrue(np.all(result == 1.0))
        self.assertTrue(np.allclose(captured["tool"], [0.290, 0.0, 0.10]))
        self.assertTrue(np.allclose(captured["joint6"], [0.125, 0.0, 0.10]))
        self.assertAlmostEqual(found["approach_standoff_m"], 0.025)
        self.assertEqual(captured["label"], "pre-grasp")

    def test_pre_grasp_is_skipped_inside_adaptive_approach_corridor(self):
        class Robot:
            def current_joint_position(self):
                return np.zeros(6)

            def forward_kinematics(self, joints):
                return {"position": [0.205, 0.0, 0.10], "rotation": np.eye(3)}

        planner = GraspPlanner(Robot(), GraspConfig(), threading.Event())
        planner.validate_candidate = lambda *args, **kwargs: self.fail(
            "IK must not run when pre-grasp is skipped"
        )

        result = planner.plan_pre_grasp(
            {
                "tool_target": np.array([0.40, 0.0, 0.10]),
                "tool_rotation": np.eye(3),
            }
        )

        self.assertIsNone(result)

    def test_pre_grasp_half_gap_is_clamped_to_maximum(self):
        class Robot:
            def current_joint_position(self):
                return np.zeros(6)

            def forward_kinematics(self, joints):
                return {"position": [0.10, 0.0, 0.10], "rotation": np.eye(3)}

        planner = GraspPlanner(Robot(), GraspConfig(), threading.Event())
        captured = {}

        def validate(tool, *_args, **_kwargs):
            captured["tool"] = np.asarray(tool)
            return np.ones(6)

        planner.validate_candidate = validate
        found = {
            "tool_target": np.array([0.45, 0.0, 0.10]),
            "tool_rotation": np.eye(3),
        }

        planner.plan_pre_grasp(found)

        self.assertAlmostEqual(found["approach_standoff_m"], 0.040)
        self.assertTrue(np.allclose(captured["tool"], [0.410, 0.0, 0.10]))

    def test_observation_move_halves_image_error_and_preserves_camera_pose(self):
        class Robot:
            def current_joint_position(self):
                return np.zeros(6)

        config = GraspConfig()
        planner = GraspPlanner(Robot(), config, threading.Event())
        captured = {}

        def validate(tool, joint6, rotation, *_args, **_kwargs):
            captured["tool"] = np.asarray(tool)
            captured["joint6"] = np.asarray(joint6)
            captured["rotation"] = np.asarray(rotation)
            return np.ones(6)

        planner.validate_candidate = validate
        base_camera = np.eye(4)
        base_camera[:3, 3] = [0.20, 0.00, 0.20]
        with patch(
            "Panthera_lib.grasp_planner.get_base_camera_transform",
            return_value=base_camera,
        ):
            result = planner.plan_observation_pose(
                np.eye(4),
                {"camera_point": np.array([0.04, -0.02, 0.30])},
            )

        self.assertTrue(np.all(result == 1.0))
        self.assertTrue(np.allclose(captured["rotation"], np.eye(3)))
        self.assertTrue(np.allclose(captured["tool"], [0.22, -0.01, 0.225]))
        self.assertTrue(
            np.allclose(captured["joint6"], [0.055, -0.01, 0.225])
        )

    def test_unreachable_observation_pose_falls_back_to_stationary_refinement(self):
        class Robot:
            def current_joint_position(self):
                return np.zeros(6)

        planner = GraspPlanner(Robot(), GraspConfig(), threading.Event())
        planner.validate_candidate = lambda *args, **kwargs: None
        with patch(
            "Panthera_lib.grasp_planner.get_base_camera_transform",
            return_value=np.eye(4),
        ):
            result = planner.plan_observation_pose(
                np.eye(4),
                {"camera_point": np.array([0.04, 0.00, 0.30])},
            )

        self.assertIsNone(result)

    def test_scan_accepts_current_pose_before_starting_joint_sweep(self):
        planner = GraspPlanner(NoopRobot(), GraspConfig(), threading.Event())
        current = np.array([0.4, 0.2, 1.0, -1.2, 0.0, 0.0])
        planner.current_joint_position = lambda: current.copy()
        labels = []
        expected = {"scan_joint_position": current.copy()}

        def detect(*_args, **_kwargs):
            labels.append(_args[5])
            return expected

        planner._detect_at_pose = detect
        planner.move_j = lambda *_args, **_kwargs: self.fail(
            "J1 sweep must not start when current-pose detection succeeds"
        )

        result = planner.scan_for_target(
            camera_feed=object(),
            intrinsic=object(),
            tcp_camera=np.eye(4),
            selected_color="blue",
        )

        self.assertIs(result, expected)
        self.assertEqual(labels, ["FAST"])

    def test_pre_grasp_does_not_skip_a_laterally_offset_pose(self):
        class Robot:
            def current_joint_position(self):
                return np.zeros(6)

            def forward_kinematics(self, joints):
                return {"position": [0.205, 0.020, 0.10], "rotation": np.eye(3)}

        planner = GraspPlanner(Robot(), GraspConfig(), threading.Event())
        planner.validate_candidate = lambda *args, **kwargs: np.ones(6)

        result = planner.plan_pre_grasp(
            {
                "tool_target": np.array([0.40, 0.0, 0.10]),
                "tool_rotation": np.eye(3),
            }
        )

        self.assertIsNotNone(result)

    def test_pre_grasp_rejects_tip_beyond_grasp_plane(self):
        class Robot:
            def current_joint_position(self):
                return np.zeros(6)

            def forward_kinematics(self, joints):
                return {"position": [0.245, 0.0, 0.10], "rotation": np.eye(3)}

        planner = GraspPlanner(Robot(), GraspConfig(), threading.Event())

        with self.assertRaises(RuntimeError):
            planner.plan_pre_grasp(
                {
                    "tool_target": np.array([0.40, 0.0, 0.10]),
                    "tool_rotation": np.eye(3),
                }
            )

    def test_pre_grasp_rejects_tip_too_close_for_orientation_change(self):
        class Robot:
            def current_joint_position(self):
                return np.zeros(6)

            def forward_kinematics(self, joints):
                return {"position": [0.225, 0.0, 0.10], "rotation": np.eye(3)}

        planner = GraspPlanner(Robot(), GraspConfig(), threading.Event())

        with self.assertRaises(RuntimeError):
            planner.plan_pre_grasp(
                {
                    "tool_target": np.array([0.40, 0.0, 0.10]),
                    "tool_rotation": np.eye(3),
                }
            )

    def test_cartesian_approach_is_sampled_and_monotonic(self):
        class Robot:
            jump_threshold = 1.5

            def forward_kinematics(self, joints):
                return {
                    "position": [0.195 + float(joints[0]), 0.0, 0.10],
                    "rotation": np.eye(3),
                }

            def compute_cartesian_path(self, waypoints, avoid_collisions=False):
                return [
                    np.array([step, 0, 0, 0, 0, 0], dtype=float)
                    for step in np.linspace(0.008, 0.04, 5)
                ], 1.0

        planner = GraspPlanner(Robot(), GraspConfig(), threading.Event())
        planner.wait_until_stationary = lambda: np.zeros(6)

        trajectory = planner.plan_cartesian_approach(
            {
                "tool_target": np.array([0.40, 0.0, 0.10]),
                "joint6_target": np.array([0.235, 0.0, 0.10]),
                "tool_rotation": np.eye(3),
            }
        )

        self.assertEqual(len(trajectory), 6)
        self.assertAlmostEqual(float(trajectory[-1][0]), 0.04)

    def test_cartesian_approach_accepts_lateral_convergence_to_target(self):
        class Robot:
            jump_threshold = 1.5

            def __init__(self):
                self.start_joint6 = np.array([0.195, 0.008, 0.10])
                self.end_joint6 = np.array([0.235, 0.000, 0.10])

            def forward_kinematics(self, joints):
                progress = float(np.asarray(joints)[0])
                position = self.start_joint6 + progress * (
                    self.end_joint6 - self.start_joint6
                )
                return {"position": position, "rotation": np.eye(3)}

            def compute_cartesian_path(self, waypoints, avoid_collisions=False):
                return [
                    np.array([step, 0, 0, 0, 0, 0], dtype=float)
                    for step in np.linspace(0.2, 1.0, 5)
                ], 1.0

        planner = GraspPlanner(Robot(), GraspConfig(), threading.Event())
        planner.wait_until_stationary = lambda: np.zeros(6)

        trajectory = planner.plan_cartesian_approach(
            {
                "tool_target": np.array([0.40, 0.0, 0.10]),
                "joint6_target": np.array([0.235, 0.0, 0.10]),
                "tool_rotation": np.eye(3),
            }
        )

        self.assertEqual(len(trajectory), 6)
        endpoint_tool, _ = planner.current_tool_pose(trajectory[-1])
        self.assertTrue(np.allclose(endpoint_tool, [0.40, 0.0, 0.10]))

    def test_cartesian_approach_rejects_true_departure_from_commanded_segment(self):
        class Robot:
            jump_threshold = 1.5

            def forward_kinematics(self, joints):
                joints = np.asarray(joints, dtype=float)
                return {
                    "position": [0.195 + 0.040 * joints[0], joints[1], 0.10],
                    "rotation": np.eye(3),
                }

            def compute_cartesian_path(self, waypoints, avoid_collisions=False):
                return [
                    np.array([0.5, 0.008, 0, 0, 0, 0], dtype=float),
                    np.array([1.0, 0.000, 0, 0, 0, 0], dtype=float),
                ], 1.0

        planner = GraspPlanner(Robot(), GraspConfig(), threading.Event())
        planner.wait_until_stationary = lambda: np.zeros(6)

        with self.assertRaisesRegex(RuntimeError, "leaves the commanded segment"):
            planner.plan_cartesian_approach(
                {
                    "tool_target": np.array([0.40, 0.0, 0.10]),
                    "joint6_target": np.array([0.235, 0.0, 0.10]),
                    "tool_rotation": np.eye(3),
                }
            )

    def test_pre_grasp_realignment_retreats_and_removes_lateral_error(self):
        class Robot:
            jump_threshold = 1.5

            def __init__(self):
                self.start_joint6 = np.array([0.205, 0.025, 0.10])
                self.end_joint6 = np.array([0.195, 0.0, 0.10])

            def forward_kinematics(self, joints):
                progress = float(np.asarray(joints)[0])
                position = self.start_joint6 + progress * (
                    self.end_joint6 - self.start_joint6
                )
                return {"position": position, "rotation": np.eye(3)}

            def compute_cartesian_path(self, waypoints, avoid_collisions=False):
                return [
                    np.array([0.5, 0, 0, 0, 0, 0], dtype=float),
                    np.array([1.0, 0, 0, 0, 0, 0], dtype=float),
                ], 1.0

        planner = GraspPlanner(Robot(), GraspConfig(), threading.Event())
        planner.wait_until_stationary = lambda: np.zeros(6)
        found = {
            "tool_target": np.array([0.40, 0.0, 0.10]),
            "tool_rotation": np.eye(3),
            "approach_standoff_m": 0.040,
        }

        trajectory = planner.plan_pre_grasp_realignment(found)

        self.assertEqual(len(trajectory), 3)
        endpoint_tool, _ = planner.current_tool_pose(trajectory[-1])
        self.assertTrue(np.allclose(endpoint_tool, [0.36, 0.0, 0.10]))

    def test_pre_grasp_realignment_is_skipped_inside_strict_corridor(self):
        class Robot:
            def forward_kinematics(self, joints):
                return {"position": [0.205, 0.0, 0.10], "rotation": np.eye(3)}

        planner = GraspPlanner(Robot(), GraspConfig(), threading.Event())
        planner.wait_until_stationary = lambda: np.zeros(6)

        result = planner.plan_pre_grasp_realignment(
            {
                "tool_target": np.array([0.40, 0.0, 0.10]),
                "tool_rotation": np.eye(3),
                "approach_standoff_m": 0.030,
            }
        )

        self.assertIsNone(result)

    def test_partial_cartesian_approach_is_rejected_before_execution(self):
        class Robot:
            jump_threshold = 1.5

            def forward_kinematics(self, joints):
                return {
                    "position": [0.195 + float(joints[0]), 0.0, 0.10],
                    "rotation": np.eye(3),
                }

            def compute_cartesian_path(self, waypoints, avoid_collisions=False):
                return [np.array([0.01, 0, 0, 0, 0, 0], dtype=float)], 0.2

        planner = GraspPlanner(Robot(), GraspConfig(), threading.Event())
        planner.wait_until_stationary = lambda: np.zeros(6)

        with self.assertRaises(RuntimeError):
            planner.plan_cartesian_approach(
                {
                    "tool_target": np.array([0.40, 0.0, 0.10]),
                    "joint6_target": np.array([0.235, 0.0, 0.10]),
                    "tool_rotation": np.eye(3),
                }
            )

    def test_observation_always_refines_even_when_move_is_skipped(self):
        planner = GraspPlanner(NoopRobot(), GraspConfig(), threading.Event())
        planner.plan_observation_pose = lambda tcp_camera, found: None
        planner.sleep_interruptible = lambda seconds: True
        planner.refine_target_at_observation_pose = lambda *args, **kwargs: {
            "refreshed": True,
            "base_point": np.zeros(3),
        }

        result = planner.pre_grasp_and_redetect(
            camera_feed=object(),
            intrinsic=object(),
            tcp_camera=np.eye(4),
            selected_color="green",
            found={"camera_point": np.zeros(3), "base_point": np.zeros(3)},
        )

        self.assertTrue(result["refreshed"])

    def test_refinement_uses_three_coherent_observations_not_one_frame(self):
        config = GraspConfig()
        config.refine_frame_warmup = 1
        config.refine_required_observations = 3
        config.refine_max_frame_attempts = 3
        planner = GraspPlanner(NoopRobot(), config, threading.Event())
        planner.wait_until_stationary = lambda: np.zeros(6)
        planner.current_joint_position = lambda: np.zeros(6)
        planner.validate_candidate = lambda *args, **kwargs: np.ones(6)

        class Intrinsic:
            ppx = 320.0
            ppy = 240.0

        def capture(sequence, point):
            detection = {
                "pixel": (320.0, 240.0),
                "color": "red",
                "point": np.asarray(point, dtype=float),
            }
            return {
                "frame_seq": sequence,
                "detections_timestamp": float(sequence),
                "detections": [detection],
                "intrinsics": Intrinsic(),
                "color_image": np.zeros((2, 2, 3), dtype=np.uint8),
                "depth_image": np.ones((2, 2), dtype=np.uint16),
            }

        frames = iter(
            [
                capture(1, [0.300, 0.000, 0.100]),
                capture(2, [0.302, 0.001, 0.101]),
                capture(3, [0.299, -0.001, 0.100]),
                capture(4, [0.301, 0.000, 0.099]),
            ]
        )

        class Feed:
            depth_scale = 0.001

            @staticmethod
            def freshness_marker():
                return 0

            @staticmethod
            def wait_for_newer(_marker, timeout):
                return next(frames)

        found = {
            "base_point": np.array([0.300, 0.000, 0.100]),
            "tool_rotation": np.eye(3),
            "provisional_joints": np.zeros(6),
            "scan_joint_position": np.zeros(6),
            "scan_joint1": 0.0,
            "detected_color": "red",
        }
        with (
            patch(
                "Panthera_lib.grasp_planner.get_base_camera_transform",
                return_value=np.eye(4),
            ),
            patch(
                "Panthera_lib.grasp_planner.object_base_position",
                side_effect=lambda detection, *_args: (
                    detection["point"],
                    detection["point"],
                ),
            ),
            patch(
                "Panthera_lib.grasp_planner.grasp_rotation_from_mask",
                return_value=(np.eye(3), 0.0, 0.0, 0.0, 0.0),
            ),
            patch(
                "Panthera_lib.grasp_planner.grasp_geometry",
                side_effect=lambda point, _rotation, _config: (
                    np.asarray(point),
                    np.asarray(point),
                ),
            ),
        ):
            result = planner.refine_target_at_observation_pose(
                Feed(),
                Intrinsic(),
                np.eye(4),
                "red",
                found,
            )

        self.assertIsNotNone(result)
        self.assertTrue(
            np.allclose(result["base_point"], [0.301, 0.000, 0.100])
        )

    def test_close_refinement_updates_position_but_preserves_far_orientation(self):
        config = GraspConfig()
        config.refine_frame_warmup = 1
        config.refine_required_observations = 3
        config.refine_max_frame_attempts = 3
        planner = GraspPlanner(NoopRobot(), config, threading.Event())
        planner.wait_until_stationary = lambda: np.zeros(6)
        planner.current_joint_position = lambda: np.zeros(6)
        planner.validate_candidate = lambda *args, **kwargs: np.ones(6)

        class Intrinsic:
            ppx = 320.0
            ppy = 240.0

        def capture(sequence, point):
            detection = {
                "pixel": (320.0, 240.0),
                "bbox": (280, 200, 360, 280),
                "color": "blue",
                "depth_m": 0.100,
                "depth_spread_m": 0.002,
                "point": np.asarray(point, dtype=float),
            }
            return {
                "frame_seq": sequence,
                "detections_timestamp": float(sequence),
                "detections": [detection],
                "intrinsics": Intrinsic(),
                "color_image": np.zeros((480, 640, 3), dtype=np.uint8),
                "depth_image": np.ones((480, 640), dtype=np.uint16),
            }

        frames = iter(
            [
                capture(1, [0.300, 0.000, 0.100]),
                capture(2, [0.328, 0.012, 0.109]),
                capture(3, [0.331, 0.011, 0.110]),
                capture(4, [0.329, 0.013, 0.111]),
            ]
        )

        class Feed:
            depth_scale = 0.001

            @staticmethod
            def freshness_marker():
                return 0

            @staticmethod
            def wait_for_newer(_marker, timeout):
                return next(frames)

        far_rotation = np.array(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        found = {
            "base_point": np.array([0.300, 0.000, 0.100]),
            "tool_rotation": far_rotation,
            "provisional_joints": np.zeros(6),
            "detected_color": "blue",
        }
        with (
            patch(
                "Panthera_lib.grasp_planner.get_base_camera_transform",
                return_value=np.eye(4),
            ),
            patch(
                "Panthera_lib.grasp_planner.object_base_position",
                side_effect=lambda detection, *_args: (
                    detection["point"],
                    detection["point"],
                ),
            ),
            patch(
                "Panthera_lib.grasp_planner.grasp_rotation_from_mask",
                side_effect=AssertionError(
                    "close refinement must preserve far-field orientation"
                ),
            ),
        ):
            result = planner.refine_target_at_observation_pose(
                Feed(),
                Intrinsic(),
                np.eye(4),
                "blue",
                found,
                position_only=True,
            )

        self.assertIsNotNone(result)
        self.assertTrue(np.allclose(result["tool_rotation"], far_rotation))
        self.assertTrue(
            np.allclose(result["base_point"], [0.329, 0.000, 0.110])
        )

    def test_weak_color_evidence_is_confirmed_across_matched_frames(self):
        config = GraspConfig()
        planner = GraspPlanner(NoopRobot(), config, threading.Event())

        def detection(red, blue):
            return {
                "color": "unknown",
                "color_confidence": 0.0,
                "color_evidence": {
                    "core_pixels": red + blue,
                    "dark": 0,
                    "white": 0,
                    "chromatic_pixels": red + blue,
                    "votes": {
                        "red": red,
                        "yellow": 0,
                        "green": 0,
                        "blue": blue,
                    },
                },
            }

        first = detection(30, 30)
        second = detection(55, 5)

        class Feed:
            def wait_for_newer(self, marker, timeout):
                self.last_request = (marker, timeout)
                return {"frame_seq": 2, "detections": [second]}

        with patch(
            "Panthera_lib.grasp_planner.object_base_position",
            return_value=(np.zeros(3), np.array([0.3, 0.0, 0.1])),
        ):
            accumulated, marker = planner._accumulate_candidate_color(
                Feed(),
                first,
                intrinsic=None,
                base_camera=np.eye(4),
                requested_color="red",
                after_sequence=1,
            )

        self.assertEqual(marker, 2)
        self.assertEqual(accumulated["color"], "red")
        self.assertEqual(accumulated["color_frames"], 2)

    def test_scan_does_not_accumulate_irrelevant_strong_colours(self):
        planner = GraspPlanner(NoopRobot(), GraspConfig(), threading.Event())
        planner.wait_until_stationary = lambda: np.zeros(6)
        planner.current_joint_position = lambda: np.zeros(6)
        planner._accumulate_candidate_color = lambda *args, **kwargs: self.fail(
            "irrelevant strong colours must not start multi-frame accumulation"
        )

        class Feed:
            @staticmethod
            def freshness_marker():
                return 1

            @staticmethod
            def wait_for_newer(_marker, timeout):
                return {
                    "frame_seq": 2,
                    "detections_timestamp": 2.0,
                    "inference_latency_s": 0.01,
                    "snapshot_age_s": 0.01,
                    "detections": [
                        {"color": "red", "color_confidence": 1.0},
                        {"color": "yellow", "color_confidence": 1.0},
                    ],
                }

        with patch(
            "Panthera_lib.grasp_planner.get_base_camera_transform",
            return_value=np.eye(4),
        ):
            result = planner._detect_at_pose(
                Feed(),
                intrinsic=object(),
                tcp_camera=np.eye(4),
                selected_color="green",
                joint1=0.0,
                label="TEST",
            )

        self.assertIsNone(result)

    def test_open_gripper_uses_current_positive_calibration(self):
        class GripperRobot:
            def __init__(self):
                self.command = None

            def gripper_control(self, position, velocity, torque):
                self.command = (position, velocity, torque)
                return True

            def gripper_state(self):
                return self.command[0], 0.0

        config = GraspConfig()
        robot = GripperRobot()
        planner = GraspPlanner(robot, config, threading.Event())

        self.assertTrue(planner.open_gripper())
        self.assertEqual(robot.command[0], 1.8)

    def test_stationary_camera_hold_refreshes_arm_commands(self):
        class Robot:
            def __init__(self):
                self.commands = 0

            @staticmethod
            def current_joint_position():
                return np.zeros(6)

            def Joint_Pos_Vel(self, *_args, **_kwargs):
                self.commands += 1
                return True

        robot = Robot()
        config = GraspConfig()
        config.stationary_hold_period_s = 0.02
        planner = GraspPlanner(robot, config, threading.Event())

        with planner.hold_current_pose("test camera wait"):
            threading.Event().wait(0.07)

        self.assertGreaterEqual(robot.commands, 3)

    def test_cartesian_hold_stays_in_mit_mode(self):
        class Robot:
            def __init__(self):
                self.mit_commands = 0

            @staticmethod
            def current_joint_position():
                return np.zeros(6)

            def hold_joints_mit_once(self, joints, _max_torque):
                self.mit_commands += 1
                self.asserted_target = np.asarray(joints, dtype=float)
                return True

            @staticmethod
            def Joint_Pos_Vel(*_args, **_kwargs):
                raise AssertionError("MIT trajectory hold must not switch controllers")

        robot = Robot()
        config = GraspConfig()
        config.stationary_hold_period_s = 0.02
        planner = GraspPlanner(robot, config, threading.Event())
        planner._control_mode = "mit"
        planner._remember_command(np.zeros(6))

        with planner.hold_current_pose("test MIT continuity"):
            threading.Event().wait(0.07)

        self.assertGreaterEqual(robot.mit_commands, 3)
        self.assertTrue(np.allclose(robot.asserted_target, 0.0))

    def test_retry_retreat_reverses_only_outward_cartesian_samples(self):
        class Robot:
            jump_threshold = 1.5

            @staticmethod
            def current_joint_position():
                return np.array([0.294, 0.0, 0.0, 0.0, 0.0, 0.0])

            @staticmethod
            def forward_kinematics(joints):
                return {
                    "position": np.array([float(joints[0]) - 0.165, 0.0, 0.10]),
                    "rotation": np.eye(3),
                }

        planner = GraspPlanner(Robot(), GraspConfig(), threading.Event())
        original = []
        for x in np.linspace(0.260, 0.300, 5):
            joints = np.zeros(6)
            joints[0] = x
            original.append(joints)
        found = {
            "tool_target": np.array([0.300, 0.0, 0.10]),
            "tool_rotation": np.eye(3),
            "approach_standoff_m": 0.040,
        }

        retreat = planner.plan_retry_retreat(original, found)
        axial_gaps = [0.300 - float(joints[0]) for joints in retreat]

        self.assertTrue(np.allclose(retreat[0], Robot.current_joint_position()))
        self.assertTrue(np.all(np.diff(axial_gaps) >= -1e-9))
        self.assertAlmostEqual(axial_gaps[-1], 0.040, places=3)

    def test_stale_cartesian_start_is_rejected_before_motor_execution(self):
        class Robot:
            @staticmethod
            def current_joint_position():
                return np.array([0.10, 0.0, 0.0, 0.0, 0.0, 0.0])

            @staticmethod
            def forward_kinematics(joints):
                return {
                    "position": np.array([float(joints[0]), 0.0, 0.0]),
                    "rotation": np.eye(3),
                }

        planner = GraspPlanner(Robot(), GraspConfig(), threading.Event())
        with self.assertRaisesRegex(RuntimeError, "plan is stale"):
            planner.validate_trajectory_start(
                [np.zeros(6), np.full(6, 0.01)],
                "TEST TRAJECTORY",
            )

    def test_successful_place_follows_put2_put1_release_put2_home_order(self):
        planner = GraspPlanner(NoopRobot(), GraspConfig(), threading.Event())
        moves = []
        opens = []
        planner.move_j = lambda _joints, _duration, label, **_kwargs: moves.append(
            label
        )
        planner.open_gripper = lambda ignore_interrupt=False: opens.append(
            ignore_interrupt
        ) or True

        self.assertTrue(planner.finish_place_sequence())
        self.assertEqual(moves, ["PUT2", "PUT1", "PUT2", "READY HOME"])
        self.assertEqual(opens, [True])

    def test_stop_during_put1_still_releases_before_skipping_ready_home(self):
        interrupted = threading.Event()
        planner = GraspPlanner(NoopRobot(), GraspConfig(), interrupted)
        moves = []
        opens = []

        def move_j(_joints, _duration, label, **_kwargs):
            moves.append(label)
            if label == "PUT1":
                interrupted.set()

        planner.move_j = move_j
        planner.open_gripper = lambda ignore_interrupt=False: opens.append(
            ignore_interrupt
        ) or True

        self.assertTrue(planner.finish_place_sequence())
        self.assertEqual(moves, ["PUT2", "PUT1"])
        self.assertEqual(opens, [True])

    def test_successful_cycle_returns_home_and_waits_for_another_target(self):
        config = GraspConfig()
        config.close_refine_enabled = False
        planner = GraspPlanner(NoopRobot(), config, threading.Event())
        found = {"scan_joint_position": np.zeros(6)}
        selected = iter(["red", False])
        finished = []

        class Streamer:
            def __init__(self):
                self.selected = []
                self.cleared = []

            def set_selected_color(self, color):
                self.selected.append(color)

            def clear_selected_target(self, message):
                self.cleared.append(message)

            def set_control_message(self, _message):
                pass

        streamer = Streamer()
        planner.scan_for_target = lambda *_args: found
        planner.pre_grasp_and_redetect = lambda *_args: found
        planner.plan_pre_grasp = lambda _found: None
        planner.plan_pre_grasp_realignment = lambda _found: None
        planner.open_gripper = lambda *args, **kwargs: True
        planner.plan_cartesian_approach = lambda _found: [
            np.zeros(6),
            np.full(6, 0.01),
        ]
        planner.grasp_and_close = lambda *args, **kwargs: (
            True,
            0.3,
            config.grasp_min_force,
        )
        planner.finish_place_sequence = lambda: finished.append(True) or True

        result = planner.run_grasp_loop(
            None,
            None,
            None,
            streamer,
            lambda: next(selected),
        )

        self.assertFalse(result)
        self.assertEqual(finished, [True])
        self.assertEqual(streamer.selected, ["red"])
        self.assertEqual(len(streamer.cleared), 1)

    def test_exhausted_pregrasp_retry_returns_home_before_next_target(self):
        config = GraspConfig()
        config.close_refine_enabled = False
        planner = GraspPlanner(NoopRobot(), config, threading.Event())
        found = {"scan_joint_position": np.zeros(6)}
        selected = iter(["red", False])
        moves = []
        redetections = []
        grasp_attempts = []

        class Streamer:
            def __init__(self):
                self.cleared = []

            def set_selected_color(self, _color):
                pass

            def set_control_message(self, _message):
                pass

            def clear_selected_target(self, message):
                self.cleared.append(message)

        streamer = Streamer()
        planner.scan_for_target = lambda *_args: found
        planner.pre_grasp_and_redetect = lambda *_args: found
        planner.redetect_at_pre_grasp = lambda *_args: (
            redetections.append(True) or found
        )
        planner.plan_pre_grasp = lambda _found: None
        planner.plan_pre_grasp_realignment = lambda _found: None
        planner.open_gripper = lambda *args, **kwargs: True
        planner.plan_cartesian_approach = lambda _found: [
            np.zeros(6),
            np.full(6, 0.01),
        ]
        planner.grasp_and_close = lambda *args, **kwargs: (
            grasp_attempts.append(True) or (False, float("nan"), float("nan"))
        )
        planner._prepare_pregrasp_retry = lambda *_args: True
        planner.move_j = lambda _joints, _duration, label, **_kwargs: moves.append(
            label
        )

        result = planner.run_grasp_loop(
            None,
            None,
            None,
            streamer,
            lambda: next(selected),
        )

        self.assertFalse(result)
        self.assertEqual(len(grasp_attempts), 2)
        self.assertEqual(len(redetections), 1)
        self.assertEqual(moves, ["READY HOME"])
        self.assertEqual(len(streamer.cleared), 1)

    def test_obb_and_graspnet_share_workspace_validation(self):
        planner = GraspPlanner(NoopRobot(), GraspConfig(), threading.Event())
        calls = []
        planner.solve_ik = lambda *args: calls.append(args) or np.zeros(6)
        tool = np.array([0.30, 0.0, 0.10])
        joint6 = np.array([0.20, 0.0, 0.0])
        rotation = np.eye(3)
        seeds = (np.zeros(6),)

        for backend in ("OBB grasp", "GraspNet grasp"):
            result = planner.validate_candidate(
                tool,
                joint6,
                rotation,
                seeds,
                2.6,
                label=backend,
            )
            self.assertIsNotNone(result)

        before = len(calls)
        rejected = planner.validate_candidate(
            np.array([0.90, 0.0, 0.10]),
            joint6,
            rotation,
            seeds,
            2.6,
            label="GraspNet grasp",
        )
        self.assertIsNone(rejected)
        self.assertEqual(len(calls), before)


    def test_failed_grasp_recovery_returns_to_recognition_pose(self):
        planner = GraspPlanner(NoopRobot(), GraspConfig(), threading.Event())
        moves = []
        messages = []
        planner.move_j = lambda joints, duration, label: moves.append(
            (np.asarray(joints), duration, label)
        )

        class Streamer:
            def set_control_message(self, message):
                messages.append(message)

        recognition_pose = np.array([0.4, 0.8, 0.7, -1.0, 0.0, 0.0])
        planner._return_to_recognition_pose(
            recognition_pose,
            Streamer(),
            "夹爪未抓到物体",
        )

        self.assertTrue(np.allclose(moves[0][0], recognition_pose))
        self.assertEqual(moves[0][2], "RECOGNITION POSE")
        self.assertIn("下一个目标", messages[0])

    def test_shutdown_returns_captured_startup_pose_before_stop(self):
        class ShutdownRobot:
            def __init__(self, target):
                self.target = np.asarray(target)
                self.move_targets = []
                self.stop_calls = 0

            def moveJ(self, target, **_kwargs):
                self.move_targets.append(np.asarray(target))
                return True

            def get_current_pos(self):
                return self.target.copy()

            def get_current_vel(self):
                return np.zeros(6)

            def set_stop(self):
                self.stop_calls += 1

        config = GraspConfig()
        config.zero_settle_time = 0.0
        config.zero_stable_samples = 1
        config.zero_verify_timeout = 0.2
        startup = np.array([0.1, 0.3, 1.0, -1.2, 0.1, -0.1])
        robot = ShutdownRobot(startup)
        planner = GraspPlanner(robot, config, threading.Event())

        self.assertTrue(
            planner.safe_shutdown(
                None,
                shutdown_target=startup,
                shutdown_label="STARTUP POSE",
            )
        )
        self.assertTrue(np.allclose(robot.move_targets[0], startup))
        self.assertEqual(robot.stop_calls, 1)

    def test_shutdown_feedback_failure_never_replays_stale_home(self):
        class FaultRobot:
            def __init__(self):
                self.hold_commands = []
                self.stop_calls = 0

            def moveJ(self, *args, **kwargs):
                return False

            def current_joint_position(self):
                raise RuntimeError("joint feedback lost")

            def Joint_Pos_Vel(self, *args, **kwargs):
                self.hold_commands.append(args[0])

            def set_stop(self):
                self.stop_calls += 1

        config = GraspConfig()
        config.zero_settle_time = 0.0
        config.shutdown_fault_hold_time = 0.0
        robot = FaultRobot()
        planner = GraspPlanner(robot, config, threading.Event())
        planner._remember_command(config.home)

        self.assertFalse(planner.safe_shutdown(None, config.home))
        self.assertEqual(robot.hold_commands, [])
        self.assertEqual(robot.stop_calls, 1)
        self.assertEqual(planner.lifecycle_state, RobotLifecycleState.STOPPED)
        self.assertFalse(planner.safe_shutdown(None))
        self.assertEqual(robot.stop_calls, 1)
