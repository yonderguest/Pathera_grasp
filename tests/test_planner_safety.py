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


class NoopRobot:
    pass


class PlannerSafetyTests(unittest.TestCase):
    def test_edge_visible_block_is_inside_expanded_image_region(self):
        planner = GraspPlanner(NoopRobot(), GraspConfig(), threading.Event())
        self.assertTrue(
            planner._is_central_horizontal({"bbox": (70, 100, 130, 180)})
        )
        self.assertFalse(
            planner._is_central_horizontal({"bbox": (0, 100, 40, 180)})
        )

    def test_pre_grasp_retreats_five_centimetres_along_approach(self):
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
            "tool_target": np.array([0.40, 0.0, 0.10]),
            "tool_rotation": np.eye(3),
            "provisional_joints": np.zeros(6),
        }

        result = planner.plan_pre_grasp(found)

        self.assertTrue(np.all(result == 1.0))
        self.assertTrue(np.allclose(captured["tool"], [0.35, 0.0, 0.10]))
        self.assertTrue(np.allclose(captured["joint6"], [0.185, 0.0, 0.10]))
        self.assertEqual(captured["label"], "pre-grasp")

    def test_pre_grasp_is_skipped_when_tip_is_already_within_five_centimetres(self):
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

    def test_cartesian_approach_is_sampled_and_monotonic(self):
        class Robot:
            jump_threshold = 1.5

            def forward_kinematics(self, joints):
                return {
                    "position": [0.185 + float(joints[0]), 0.0, 0.10],
                    "rotation": np.eye(3),
                }

            def compute_cartesian_path(self, waypoints, avoid_collisions=False):
                return [
                    np.array([step, 0, 0, 0, 0, 0], dtype=float)
                    for step in np.linspace(0.01, 0.05, 5)
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
        self.assertAlmostEqual(float(trajectory[-1][0]), 0.05)

    def test_partial_cartesian_approach_is_rejected_before_execution(self):
        class Robot:
            jump_threshold = 1.5

            def forward_kinematics(self, joints):
                return {
                    "position": [0.185 + float(joints[0]), 0.0, 0.10],
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

    def test_pre_grasp_always_redetects_even_when_move_is_skipped(self):
        planner = GraspPlanner(NoopRobot(), GraspConfig(), threading.Event())
        planner.plan_pre_grasp = lambda found: None
        planner.sleep_interruptible = lambda seconds: True
        planner.current_joint_position = lambda: np.array([0.3, 0, 0, 0, 0, 0])
        planner._detect_at_pose = lambda *args, **kwargs: {
            "refreshed": True,
            "base_point": np.zeros(3),
        }

        result = planner.pre_grasp_and_redetect(
            camera_feed=object(),
            intrinsic=object(),
            tcp_camera=np.eye(4),
            selected_color="green",
            found={"tool_target": np.zeros(3), "base_point": np.zeros(3)},
        )

        self.assertTrue(result["refreshed"])

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
