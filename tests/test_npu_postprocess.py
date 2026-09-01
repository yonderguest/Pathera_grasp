from __future__ import annotations

import unittest

import numpy as np

from Panthera_lib.npu_inference import NpuYoloDetector


class NpuPostprocessTests(unittest.TestCase):
    def test_nms_keeps_separate_blocks_and_suppresses_duplicate(self):
        boxes = np.array(
            [
                [0, 0, 100, 100],
                [5, 5, 98, 98],
                [200, 0, 260, 60],
            ],
            dtype=np.float32,
        )
        scores = np.array([0.95, 0.90, 0.80], dtype=np.float32)

        kept = NpuYoloDetector._nms_indices(boxes, scores, 0.45, 50, 20)

        self.assertEqual(kept, [0, 2])


if __name__ == "__main__":
    unittest.main()
