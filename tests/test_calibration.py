from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np

from Panthera_lib.grasp_config import GraspConfig
from Panthera_lib.vision_pipeline import load_hand_eye


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
