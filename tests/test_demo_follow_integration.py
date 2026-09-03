from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

import grasp_demo
from Panthera_lib.hand_follow import HandFollowResult, HandFollowState


class DemoFollowIntegrationTests(unittest.TestCase):
    def setUp(self):
        grasp_demo.shutdown_requested.clear()

    def tearDown(self):
        grasp_demo.shutdown_requested.clear()

    def test_process_lock_rejects_a_second_hardware_owner(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pathera.lock"
            first = grasp_demo.acquire_process_lock(path)
            try:
                with self.assertRaisesRegex(RuntimeError, "owns the hardware"):
                    grasp_demo.acquire_process_lock(path)
            finally:
                first.close()

            second = grasp_demo.acquire_process_lock(path)
            second.close()

    def test_follow_session_enters_from_home_and_restores_object_inference(self):
        events = []

        class Planner:
            def home(self):
                events.append("home")

            def open_gripper(self):
                events.append("open")

        class Streamer:
            is_closed = False

            def activate_follow_mode(self, message):
                events.append("activate")

            def poll_follow_command(self):
                return None

            def update_follow_feedback(self, *args):
                events.append("feedback")

            def finish_follow_mode(self, message):
                events.append("finish")

        class Feed:
            def set_object_inference_enabled(self, enabled):
                events.append(f"objects:{enabled}")

        class Controller:
            planner = Planner()

            def run(self, mode_enabled):
                events.append("run")
                self.assert_enabled = mode_enabled()
                return HandFollowResult(HandFollowState.IDLE, "done", 2)

        controller = Controller()
        result = grasp_demo.run_hand_follow_from_prompt(
            Streamer(), controller, Feed()
        )

        self.assertTrue(controller.assert_enabled)
        self.assertEqual(result.completed_steps, 2)
        self.assertEqual(
            events,
            [
                "home",
                "open",
                "activate",
                "objects:False",
                "run",
                "objects:True",
                "finish",
            ],
        )

    def test_follow_callbacks_publish_mask_preview_and_preserve_confidence(self):
        class Streamer:
            def __init__(self):
                self.confidence = 0.0
                self.feedback = []
                self.published = []

            def control_status(self):
                return {"follow_hand_confidence": self.confidence}

            def update_follow_feedback(self, visible, confidence=0.0, message=None):
                self.confidence = float(confidence)
                self.feedback.append((bool(visible), float(confidence), message))

            def publish(self, color, detections, depth, depth_scale):
                self.published.append((color, detections, depth, depth_scale))

        class Detection:
            confidence = 0.83

            @staticmethod
            def as_stream_detection():
                return {"class_name": "human hand", "confidence": 0.83}

        streamer = Streamer()
        feed = SimpleNamespace(depth_scale=0.001)
        status, preview = grasp_demo.build_hand_follow_callbacks(streamer, feed)
        snapshot = {
            "color_image": np.zeros((4, 5, 3), dtype=np.uint8),
            "depth_image": np.ones((4, 5), dtype=np.uint16),
        }

        preview(snapshot, [Detection()])
        status({"hand_seen": True, "message": "tracking"})

        self.assertEqual(len(streamer.published), 1)
        self.assertEqual(streamer.published[0][1][0]["class_name"], "human hand")
        self.assertEqual(streamer.feedback[-1], (True, 0.83, "tracking"))

    def test_global_stop_during_follow_arming_skips_controller_loop(self):
        class Planner:
            def home(self):
                pass

            def open_gripper(self):
                pass

        class Streamer:
            is_closed = False

            def poll_follow_command(self):
                return None

            def activate_follow_mode(self, _message):
                grasp_demo.shutdown_requested.set()

        class Controller:
            planner = Planner()

            def run(self, _mode_enabled):
                raise AssertionError("controller must not start after a global stop")

        result = grasp_demo.run_hand_follow_from_prompt(
            Streamer(), Controller(), SimpleNamespace()
        )

        self.assertEqual(result.state, HandFollowState.STOPPING)
        self.assertEqual(result.completed_steps, 0)

    def test_global_stop_inside_follow_does_not_restart_object_inference(self):
        class Planner:
            def home(self):
                pass

            def open_gripper(self):
                pass

        class Streamer:
            is_closed = False

            def activate_follow_mode(self, _message):
                pass

            def poll_follow_command(self):
                return None

        class Feed:
            def __init__(self):
                self.transitions = []

            def set_object_inference_enabled(self, enabled):
                self.transitions.append(bool(enabled))

        class Controller:
            planner = Planner()

            def run(self, _mode_enabled):
                grasp_demo.shutdown_requested.set()
                return HandFollowResult(
                    HandFollowState.STOPPING,
                    "global shutdown requested",
                    0,
                )

        feed = Feed()
        result = grasp_demo.run_hand_follow_from_prompt(
            Streamer(), Controller(), feed
        )

        self.assertEqual(result.state, HandFollowState.STOPPING)
        self.assertEqual(feed.transitions, [False])

    def test_cancel_during_lazy_load_is_checked_before_any_robot_motion(self):
        events = []

        class Planner:
            def home(self):
                events.append("home")

            def open_gripper(self):
                events.append("open")

        class Streamer:
            is_closed = False

            def poll_follow_command(self):
                return False

            def finish_follow_mode(self, _message):
                events.append("finish")

        controller = SimpleNamespace(planner=Planner())
        result = grasp_demo.run_hand_follow_from_prompt(
            Streamer(), controller, SimpleNamespace()
        )

        self.assertEqual(result.state, HandFollowState.IDLE)
        self.assertEqual(events, ["finish"])

    def test_hand_model_is_loaded_only_on_first_follow_request_and_reused(self):
        messages = []

        class Streamer:
            is_closed = False

            def set_control_message(self, message):
                messages.append(message)

        streamer = Streamer()
        config = SimpleNamespace(
            model_path=Path("model.pt"),
            text_encoder_path=Path("encoder.ts"),
            follow_max_semantic_step_m=0.10,
            follow_max_tcp_step_m=0.020,
        )
        planner = Mock()
        feed = SimpleNamespace(depth_scale=0.001)
        model = object()
        detector = object()
        controller = object()
        result = HandFollowResult(HandFollowState.IDLE, "done", 0)

        with patch.object(
            grasp_demo, "load_cpu_hand_model", return_value=model
        ) as load, patch.object(
            grasp_demo,
            "CpuYoloHandDetector",
            return_value=detector,
        ) as detector_type, patch.object(
            grasp_demo,
            "HandFollowController",
            return_value=controller,
        ) as controller_type, patch.object(
            grasp_demo,
            "run_hand_follow_from_prompt",
            return_value=result,
        ) as run_session:
            runner = grasp_demo.build_lazy_hand_follow_runner(
                streamer,
                planner,
                feed,
                object(),
                np.eye(4),
                config,
            )
            load.assert_not_called()
            controller_type.assert_not_called()

            self.assertIs(runner(), result)
            self.assertIs(runner(), result)

        load.assert_called_once()
        detector_type.assert_called_once()
        controller_type.assert_called_once()
        settings = controller_type.call_args.kwargs["settings"]
        self.assertEqual(settings.max_semantic_step_m, 0.10)
        self.assertEqual(settings.max_tcp_step_m, 0.020)
        self.assertEqual(run_session.call_count, 2)
        self.assertEqual(len(messages), 1)


if __name__ == "__main__":
    unittest.main()
