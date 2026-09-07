from __future__ import annotations

import threading
import time
import unittest

import numpy as np

from Panthera_lib.hand_follow import (
    CpuYoloHandDetector,
    HandDetection,
    HandFollowController,
    HandFollowSettings,
    HandFollowState,
    HandTrackGate,
    SemanticStepLimitError,
    hand_candidate_from_mask,
    plan_bounded_follow_step,
)


class _Tensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def cpu(self):
        return self

    def numpy(self):
        return self.value.copy()


class _Boxes:
    def __init__(self, confidences, classes, boxes):
        self.conf = _Tensor(confidences)
        self.cls = _Tensor(classes)
        self.xyxy = _Tensor(boxes)

    def __len__(self):
        return len(self.conf.value)


class _Masks:
    def __init__(self, masks):
        self.data = _Tensor(masks)


class _Model:
    def __init__(self, result):
        self.result = result
        self.kwargs = None

    def predict(self, _image, **kwargs):
        self.kwargs = kwargs
        return [self.result]


class _Result:
    def __init__(self, masks, names=None):
        self.boxes = _Boxes([0.8] * len(masks), [0] * len(masks), [[20, 20, 80, 90]] * len(masks))
        self.masks = _Masks(masks)
        self.names = names or {0: "hand"}


def _candidate(point=(0.05, 0.0, 0.30)):
    return HandDetection(
        confidence=0.8,
        pixel=np.array([50.0, 50.0]),
        camera_point=np.asarray(point, dtype=float),
        depth_m=float(point[2]),
        depth_spread_m=0.0,
        depth_samples=100,
        bbox=(20, 20, 80, 90),
        mask=np.ones((100, 100), dtype=np.uint8),
    )


class _Config:
    def __init__(self):
        self.home = np.zeros(6)
        self.tcp_in_joint6 = np.zeros(3)
        self.joint_lower = np.full(6, -1.0)
        self.joint_upper = np.full(6, 1.0)
        self.return_home_duration = 1.0
        self.tool_x_range = (0.0, 1.0)
        self.tool_y_range = (-1.0, 1.0)
        self.tool_z_range = (0.0, 1.0)
        self.radial_range = (0.0, 1.0)
        self.wrist_z_range = (0.0, 1.0)


class _FakePlanner:
    def __init__(self):
        self.config = _Config()
        self.interrupted = threading.Event()
        self.q = np.zeros(6)
        self.origin = np.array([0.30, 0.0, 0.10])
        self.moves = []
        self.holds = []
        self.hold_times = []
        self.robot_threads = []
        self.pose_queries = 0

    def wait_until_stationary(self):
        self.robot_threads.append(threading.get_ident())
        return self.q.copy()

    def current_joint_position(self):
        self.robot_threads.append(threading.get_ident())
        return self.q.copy()

    def current_tool_pose(self, joints):
        self.robot_threads.append(threading.get_ident())
        self.pose_queries += 1
        joints = np.asarray(joints, dtype=float)
        return self.origin + joints[:3], np.eye(3)

    def validate_candidate(
        self,
        _tool_target,
        joint6_target,
        _rotation,
        seeds,
        jump_limit,
        label,
    ):
        self.robot_threads.append(threading.get_ident())
        del label
        seed = np.asarray(tuple(seeds)[0], dtype=float)
        result = seed.copy()
        result[:3] = np.asarray(joint6_target, dtype=float) - self.origin
        return result if np.max(np.abs(result - seed)) <= jump_limit else None

    def refresh_arm_hold(self, joints):
        self.robot_threads.append(threading.get_ident())
        self.holds.append(np.asarray(joints, dtype=float).copy())
        self.hold_times.append(time.monotonic())
        return True

    def move_j(self, joints, duration, label, **kwargs):
        self.robot_threads.append(threading.get_ident())
        self.q = np.asarray(joints, dtype=float).copy()
        self.moves.append((self.q.copy(), float(duration), label, dict(kwargs)))


class _BulgingPlanner(_FakePlanner):
    def current_tool_pose(self, joints):
        self.robot_threads.append(threading.get_ident())
        self.pose_queries += 1
        joints = np.asarray(joints, dtype=float)
        position = self.origin + joints[:3]
        # The endpoint is on target, but the interpolated MoveJ path bows more
        # than the approved 20 mm envelope.
        position[0] += 0.020 * np.sin(np.pi * joints[0] / 0.020)
        return position, np.eye(3)


class _WorkspaceBulgingPlanner(_FakePlanner):
    def __init__(self):
        super().__init__()
        self.config.tool_y_range = (-0.003, 0.003)

    def current_tool_pose(self, joints):
        self.robot_threads.append(threading.get_ident())
        self.pose_queries += 1
        joints = np.asarray(joints, dtype=float)
        position = self.origin + joints[:3]
        position[1] += 0.004 * np.sin(np.pi * joints[0] / 0.005)
        return position, np.eye(3)


class _Feed:
    def __init__(self):
        self.sequence = 0
        self.depth_scale = 0.001
        self.old_wait_calls = 0

    def freshness_marker(self):
        return self.sequence

    def wait_for_newer(self, _marker, timeout):
        del timeout
        self.old_wait_calls += 1
        self.sequence += 1
        return {
            "frame_seq": self.sequence,
            "timestamp": time.monotonic(),
            "color_image": np.zeros((100, 100, 3), dtype=np.uint8),
            "depth_image": np.full((100, 100), 300, dtype=np.uint16),
        }


class _CaptureFeed(_Feed):
    def __init__(self):
        super().__init__()
        self.capture_wait_calls = 0

    def capture_freshness_marker(self):
        return self.sequence

    def wait_for_new_capture(self, _marker, timeout):
        del timeout
        self.capture_wait_calls += 1
        self.sequence += 1
        return {
            "frame_seq": self.sequence,
            "timestamp": time.monotonic(),
            "capture_timestamp_ns": time.monotonic_ns(),
            "color_image": np.zeros((100, 100, 3), dtype=np.uint8),
            "depth_image": np.full((100, 100), 300, dtype=np.uint16),
        }


class _Detector:
    def __init__(self, detection, settings):
        self.detection = detection
        self.settings = settings

    def detect(self, *_args):
        return [self.detection]


class _SlowDetector(_Detector):
    def __init__(self, detection, settings, delay_s=0.12):
        super().__init__(detection, settings)
        self.delay_s = delay_s
        self.thread_ids = []

    def detect(self, *_args):
        self.thread_ids.append(threading.get_ident())
        time.sleep(self.delay_s)
        return [self.detection]


class HandFollowTests(unittest.TestCase):
    def test_hard_limits_cannot_be_relaxed(self):
        defaults = HandFollowSettings()
        self.assertEqual(defaults.prompt, "hand")
        self.assertEqual(defaults.confidence_threshold, 0.10)
        with self.assertRaises(ValueError):
            HandFollowSettings(max_tcp_step_m=0.021)
        with self.assertRaises(ValueError):
            HandFollowSettings(max_joint_speed_rad_s=0.121)
        with self.assertRaises(ValueError):
            HandFollowSettings(desired_camera_distance_m=0.29)
        with self.assertRaises(ValueError):
            HandFollowSettings(min_tcp_hand_distance_m=0.19)
        with self.assertRaises(ValueError):
            HandFollowSettings(joint_path_samples=4)
        with self.assertRaises(ValueError):
            HandFollowSettings(inference_hold_period_s=0.051)

    def test_mask_core_produces_metric_hand_point(self):
        settings = HandFollowSettings(min_mask_area_px=100)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:90, 20:80] = 1
        depth = np.full((100, 100), 300, dtype=np.uint16)
        hand = hand_candidate_from_mask(
            mask,
            (20, 20, 80, 90),
            0.8,
            depth,
            0.001,
            {"fx": 100.0, "fy": 100.0, "ppx": 50.0, "ppy": 50.0},
            settings,
        )

        self.assertIsNotNone(hand)
        self.assertAlmostEqual(hand.depth_m, 0.30)
        self.assertLess(abs(hand.camera_point[0]), 0.01)
        self.assertLess(abs(hand.camera_point[1]), 0.02)

    def test_border_hand_and_noisy_depth_are_rejected(self):
        settings = HandFollowSettings(min_mask_area_px=100)
        depth = np.full((100, 100), 300, dtype=np.uint16)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[2:60, 2:60] = 1
        diagnostics = {}
        self.assertIsNone(
            hand_candidate_from_mask(
                mask,
                (2, 2, 60, 60),
                0.8,
                depth,
                0.001,
                {"fx": 100.0, "fy": 100.0, "ppx": 50.0, "ppy": 50.0},
                settings,
                diagnostics=diagnostics,
            )
        )
        self.assertIn("touches", diagnostics["rejection_reason"])

        mask[:] = 0
        mask[20:90, 20:80] = 1
        noisy = depth.copy()
        noisy[:, 50:] = 350
        self.assertIsNone(
            hand_candidate_from_mask(
                mask,
                (20, 20, 80, 90),
                0.8,
                noisy,
                0.001,
                {"fx": 100.0, "fy": 100.0, "ppx": 50.0, "ppy": 50.0},
                settings,
            )
        )

    def test_cpu_detector_uses_approved_low_resolution_settings(self):
        settings = HandFollowSettings(min_mask_area_px=100)
        mask = np.zeros((100, 100), dtype=np.float32)
        mask[20:90, 20:80] = 1.0
        model = _Model(_Result([mask]))
        detector = CpuYoloHandDetector(model, settings)
        hands = detector.detect(
            np.zeros((100, 100, 3), dtype=np.uint8),
            np.full((100, 100), 300, dtype=np.uint16),
            0.001,
            {"fx": 100.0, "fy": 100.0, "ppx": 50.0, "ppy": 50.0},
        )

        self.assertEqual(len(hands), 1)
        self.assertEqual(model.kwargs["imgsz"], 320)
        self.assertEqual(model.kwargs["device"], "cpu")
        self.assertEqual(detector.last_diagnostics["raw_count"], 1)
        self.assertEqual(detector.last_diagnostics["matching_count"], 1)
        self.assertEqual(detector.last_diagnostics["accepted_count"], 1)
        self.assertEqual(detector.last_diagnostics["geometry_rejected"], 0)
        self.assertEqual(model.kwargs["max_det"], settings.max_detections)

    def test_unique_hand_needs_three_frames_and_two_to_reacquire(self):
        gate = HandTrackGate()
        gate.reset(now=0.0)
        hand = _candidate()
        self.assertIsNone(gate.update([hand], now=0.1))
        self.assertIsNone(gate.update([hand], now=0.2))
        self.assertIs(gate.update([hand], now=0.3), hand)
        self.assertEqual(gate.state, HandFollowState.TRACKING)

        gate.miss("lost", now=0.4)
        self.assertEqual(gate.state, HandFollowState.HOLD_LOST)
        self.assertIsNone(gate.update([hand], now=0.5))
        self.assertIs(gate.update([hand], now=0.6), hand)
        self.assertEqual(gate.state, HandFollowState.TRACKING)

    def test_multiple_hands_and_long_loss_pause(self):
        gate = HandTrackGate()
        gate.reset(now=0.0)
        gate.update([_candidate(), _candidate((0.06, 0.0, 0.30))], now=0.1)
        self.assertEqual(gate.state, HandFollowState.ARMING)
        # A failed initial observation restarts the full three-frame gate.
        hand = _candidate()
        self.assertIsNone(gate.update([hand], now=0.2))
        self.assertIsNone(gate.update([hand], now=0.3))
        self.assertIs(gate.update([hand], now=0.4), hand)
        gate.miss("lost", now=0.5)
        self.assertEqual(gate.state, HandFollowState.HOLD_LOST)
        gate.miss("still lost", now=1.3)
        self.assertEqual(gate.state, HandFollowState.HOLD_LOST)
        gate.miss("still lost", now=1.7)
        self.assertEqual(gate.state, HandFollowState.PAUSED)

    def test_initial_no_hand_does_not_require_manual_reenable(self):
        gate = HandTrackGate()
        gate.reset(now=0.0)

        self.assertIsNone(gate.update([], now=2.0, observed_count=0))

        self.assertEqual(gate.state, HandFollowState.ARMING)
        self.assertEqual(gate.reason, "no hand")

    def test_raw_second_hand_blocks_even_if_its_geometry_is_rejected(self):
        gate = HandTrackGate()
        gate.reset(now=0.0)
        accepted = _candidate()

        self.assertIsNone(
            gate.update([accepted], now=0.1, observed_count=2)
        )

        self.assertEqual(gate.state, HandFollowState.ARMING)
        self.assertEqual(gate.reason, "multiple hands are ambiguous")

    def test_planned_step_is_at_most_twenty_mm_and_speed_limited(self):
        planner = _FakePlanner()
        settings = HandFollowSettings()
        motion = plan_bounded_follow_step(
            planner,
            planner.q,
            np.eye(3),
            _candidate((0.10, 0.0, 0.30)),
            np.eye(4),
            settings,
        )

        self.assertIsNotNone(motion)
        self.assertLessEqual(np.linalg.norm(motion.camera_step), 0.020 + 1e-12)
        self.assertAlmostEqual(np.linalg.norm(motion.camera_step), 0.020)
        joint_delta = np.max(np.abs(motion.target_joints - planner.q))
        self.assertLessEqual(joint_delta / motion.duration_s, 0.12 + 1e-12)
        self.assertGreaterEqual(motion.tcp_hand_clearance_m, 0.20)
        self.assertGreaterEqual(planner.pose_queries, settings.joint_path_samples + 1)

    def test_semantic_step_over_10cm_holds_before_ik(self):
        planner = _FakePlanner()
        with self.assertRaises(SemanticStepLimitError):
            plan_bounded_follow_step(
                planner,
                planner.q,
                np.eye(3),
                _candidate((0.0, 0.0, 0.55)),
                np.eye(4),
            )
        self.assertEqual(planner.pose_queries, 0)

    def test_semantic_limit_does_not_change_physical_20mm_step(self):
        planner = _FakePlanner()
        settings = HandFollowSettings(max_semantic_step_m=0.10)
        motion = plan_bounded_follow_step(
            planner,
            planner.q,
            np.eye(3),
            _candidate((0.18, 0.0, 0.30)),
            np.eye(4),
            settings,
        )
        self.assertIsNotNone(motion)
        self.assertLessEqual(np.linalg.norm(motion.camera_step), 0.020 + 1e-12)
        self.assertAlmostEqual(np.linalg.norm(motion.camera_step), 0.020)

    def test_movej_interpolation_rejects_nonlinear_tcp_excursion(self):
        planner = _BulgingPlanner()
        with self.assertRaisesRegex(RuntimeError, "20.5 mm envelope"):
            plan_bounded_follow_step(
                planner,
                planner.q,
                np.eye(3),
                _candidate((0.10, 0.0, 0.30)),
                np.eye(4),
            )

    def test_movej_interpolation_rejects_intermediate_workspace_exit(self):
        planner = _WorkspaceBulgingPlanner()
        with self.assertRaisesRegex(RuntimeError, "leaves the workspace"):
            plan_bounded_follow_step(
                planner,
                planner.q,
                np.eye(3),
                _candidate((0.10, 0.0, 0.30)),
                np.eye(4),
            )

    def test_deadband_holds_without_motion(self):
        planner = _FakePlanner()
        motion = plan_bounded_follow_step(
            planner,
            planner.q,
            np.eye(3),
            _candidate((0.005, 0.005, 0.31)),
            np.eye(4),
        )
        self.assertIsNone(motion)

    def test_clearance_violation_rejects_motion(self):
        planner = _FakePlanner()
        with self.assertRaisesRegex(RuntimeError, "clearance"):
            plan_bounded_follow_step(
                planner,
                planner.q,
                np.eye(3),
                _candidate((0.0, 0.0, 0.19)),
                np.eye(4),
            )

    def test_controller_runs_on_main_thread_and_returns_home(self):
        settings = HandFollowSettings(max_snapshot_age_s=1.0)
        planner = _FakePlanner()
        feed = _Feed()
        detector = _Detector(_candidate((0.10, 0.0, 0.30)), settings)
        states = []
        controller = HandFollowController(
            planner,
            feed,
            {"fx": 100.0, "fy": 100.0, "ppx": 50.0, "ppy": 50.0},
            np.eye(4),
            detector,
            settings,
            status_callback=states.append,
        )

        def enabled():
            return not any(move[2] == "HAND FOLLOW 20MM STEP" for move in planner.moves)

        result = controller.run(enabled)

        labels = [move[2] for move in planner.moves]
        self.assertIn("HAND FOLLOW 20MM STEP", labels)
        self.assertEqual(labels[-1], "HAND FOLLOW HOME")
        self.assertTrue(np.allclose(planner.q, planner.config.home))
        self.assertEqual(result.state, HandFollowState.IDLE)
        self.assertEqual(result.completed_steps, 1)
        self.assertTrue(states)

    def test_raw_capture_preview_and_worker_keep_robot_on_main_thread(self):
        settings = HandFollowSettings(max_snapshot_age_s=1.0)
        planner = _FakePlanner()
        feed = _CaptureFeed()
        detector = _SlowDetector(_candidate((0.10, 0.0, 0.30)), settings)
        previews = []
        controller = HandFollowController(
            planner,
            feed,
            {"fx": 100.0, "fy": 100.0, "ppx": 50.0, "ppy": 50.0},
            np.eye(4),
            detector,
            settings,
            preview_callback=lambda snapshot, hands: previews.append(
                (snapshot["frame_seq"], list(hands))
            ),
        )

        def enabled():
            return not any(move[2] == "HAND FOLLOW 20MM STEP" for move in planner.moves)

        result = controller.run(enabled)

        self.assertEqual(result.completed_steps, 1)
        self.assertEqual(feed.old_wait_calls, 0)
        self.assertEqual(feed.capture_wait_calls, 3)
        self.assertEqual(len(previews), 3)
        self.assertTrue(all(len(hands) == 1 for _, hands in previews))
        self.assertTrue(detector.thread_ids)
        self.assertNotIn(threading.get_ident(), detector.thread_ids)
        self.assertTrue(planner.robot_threads)
        self.assertEqual(set(planner.robot_threads), {threading.get_ident()})
        self.assertGreaterEqual(len(planner.hold_times), 9)

    def test_disable_during_cpu_inference_does_not_execute_follow_step(self):
        settings = HandFollowSettings(max_snapshot_age_s=1.0)
        planner = _FakePlanner()
        feed = _CaptureFeed()
        detector = _SlowDetector(
            _candidate((0.10, 0.0, 0.30)),
            settings,
            delay_s=0.12,
        )
        controller = HandFollowController(
            planner,
            feed,
            {"fx": 100.0, "fy": 100.0, "ppx": 50.0, "ppy": 50.0},
            np.eye(4),
            detector,
            settings,
        )
        deadline = time.monotonic() + 0.05

        result = controller.run(lambda: time.monotonic() < deadline)

        labels = [move[2] for move in planner.moves]
        self.assertNotIn("HAND FOLLOW 20MM STEP", labels)
        self.assertEqual(result.completed_steps, 0)
        self.assertTrue(detector.thread_ids)
        self.assertGreaterEqual(len(planner.hold_times), 3)


if __name__ == "__main__":
    unittest.main()
