from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import patch

import numpy as np
from rclpy.qos import DurabilityPolicy

import panthera_grasp_brain.grasp_brain_node as brain_module
from panthera_grasp_brain.grasp_brain_node import (
    PantheraGraspBrainNode,
    SynchronizedFrameBuffer,
    camera_info_qos,
)


class RosTransportTests(unittest.TestCase):
    def test_ros_entrypoint_patch_uses_abi_compatible_conda_python(self):
        patch_script = (
            Path(__file__).resolve().parents[1]
            / "ros2_ws"
            / "patch_shebangs.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("sys.version_info[:2] != (3, 10)", patch_script)
        self.assertIn('"rclpy"', patch_script)
        self.assertIn("PANTHERA_ROS_PYTHON", patch_script)

    def test_camera_info_is_durable_for_late_joiners(self):
        self.assertEqual(
            camera_info_qos().durability,
            DurabilityPolicy.TRANSIENT_LOCAL,
        )

    def test_ros_frame_buffer_rejects_cross_frame_mix_and_times_out(self):
        frames = SynchronizedFrameBuffer()
        color_10 = np.full((1, 1, 3), 10, dtype=np.uint8)
        color_11 = np.full((1, 1, 3), 11, dtype=np.uint8)
        depth_10 = np.full((1, 1), 100, dtype=np.uint16)

        frames.put_depth(10_000, depth_10)
        frames.put_color(11_000, color_11)
        frames.put_detections(10, 10_000, [{"frame": 10}])
        self.assertIsNone(frames.wait_for_newer(-1, timeout=0.01))

        frames.put_color(10_000, color_10)
        snapshot = frames.wait_for_newer(-1, timeout=0.01)
        self.assertEqual(snapshot["frame_seq"], 10)
        self.assertTrue(np.all(snapshot["color_image"] == 10))
        self.assertTrue(np.all(snapshot["depth_image"] == 100))
        self.assertEqual(snapshot["detections"], [{"frame": 10}])
        self.assertIsNone(frames.wait_for_newer(10, timeout=0.01))


    def test_use_voice_false_never_requests_voice(self):
        class MustNotBeCalled:
            def wait_for_command(self, timeout):
                raise AssertionError("voice request was made while disabled")

        node = PantheraGraspBrainNode.__new__(PantheraGraspBrainNode)
        node.config = SimpleNamespace(use_voice=False, voice_prompt_duration=3.5)
        node.voice = MustNotBeCalled()
        node.announcer = None
        node._shutdown_event = threading.Event()
        node._parse_target_command = lambda text: (None, False)
        with patch.object(
            brain_module.sys,
            "stdin",
            SimpleNamespace(isatty=lambda: False),
        ):
            self.assertIs(node._select_target(), False)

    def test_launch_passes_use_voice_to_voice_node(self):
        launch_path = (
            Path(__file__).resolve().parents[1]
            / "ros2_ws"
            / "src"
            / "grasp_bringup"
            / "launch"
            / "grasp_system.launch.py"
        )
        tree = ast.parse(launch_path.read_text(encoding="utf-8"))
        helper = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "voice_node_parameters"
        )
        returned = next(
            node.value
            for node in helper.body
            if isinstance(node, ast.Return)
        )
        self.assertIsInstance(returned, ast.Dict)
        parameters = {
            key.value: value.id
            for key, value in zip(returned.keys, returned.values)
        }
        self.assertEqual(
            parameters,
            {
                "voice_enabled": "use_voice",
                "voice_prompt_duration": "voice_prompt_duration",
            },
        )
