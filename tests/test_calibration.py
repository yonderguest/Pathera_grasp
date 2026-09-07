from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np

from Panthera_lib.grasp_config import GraspConfig
from Panthera_lib.vision_pipeline import get_base_camera_transform, load_hand_eye


class CalibrationTests(unittest.TestCase):
    def test_current_project_calibration_is_the_831_result(self):
        root = Path(__file__).resolve().parents[1]
        payload = json.loads(
            (root / "hand_eye_calibration.json").read_text(encoding="utf-8")
        )
        self.assertEqual(payload["timestamp"], "2026-08-31 07:18:17")
        self.assertEqual(payload["num_samples"], 20)

        config = GraspConfig(calibration_file=root / "hand_eye_calibration.json")
        transform = load_hand_eye(config)
        self.assertEqual(transform.shape, (4, 4))
        self.assertTrue(np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0]))
        self.assertTrue(np.isclose(np.linalg.det(transform[:3, :3]), 1.0, atol=1e-5))

    def test_camera_transform_bridges_joint6_to_calibrated_tcp(self):
        class Robot:
            @staticmethod
            def forward_kinematics(_joints):
                return {
                    "position": [0.10, 0.20, 0.30],
                    "rotation": np.array(
                        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
                    ),
                }

        tcp_camera = np.eye(4)
        tcp_camera[:3, 3] = [0.01, 0.02, 0.03]
        result = get_base_camera_transform(
            Robot(),
            tcp_camera,
            np.zeros(6),
            tcp_in_joint6=np.array([0.165, 0.0, 0.0]),
        )

        # joint6->TCP and TCP->Camera translations both rotate with the wrist.
        self.assertTrue(np.allclose(result[:3, 3], [0.08, 0.375, 0.33]))
        self.assertTrue(
            np.allclose(
                result[:3, :3],
                [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            )
        )
