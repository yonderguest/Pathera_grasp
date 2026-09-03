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

    def test_nms_keeps_overlapping_different_object_classes(self):
        boxes = np.array([[0, 0, 100, 100], [2, 2, 98, 98]], dtype=np.float32)
        scores = np.array([0.95, 0.90], dtype=np.float32)

        kept = NpuYoloDetector._nms_indices(
            boxes,
            scores,
            0.45,
            50,
            20,
            labels=np.array([0, 1]),
        )

        self.assertEqual(kept, [0, 1])

    def test_decode_rejects_invalid_labels_and_off_image_boxes(self):
        detector = NpuYoloDetector.__new__(NpuYoloDetector)
        detector.names = ("a", "b", "c", "d")
        detector.confidence = 0.15
        detector.iou_threshold = 0.45
        detector.pre_nms_top_k = 50
        detector.max_detections = 20
        detector.input_size = 640

        pred = np.zeros((300, 38), dtype=np.float32)
        pred[:, 5] = 99.0
        pred[0, :6] = [100, 100, 200, 200, 0.8, 1]
        pred[1, :6] = [100, 100, 200, 200, 0.9, 9]
        pred[2, :6] = [-100, 50, -20, 100, 0.9, 2]
        pred[3, :6] = [300, 100, 400, 200, 0.9, 1.5]
        proto = np.zeros((32, 160, 160), dtype=np.float32)

        result = detector._decode(
            [pred.tobytes(), proto.tobytes()],
            height=480,
            width=640,
            ratio=1.0,
            dx=0,
            dy=80,
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["cls"], 1)
        self.assertGreaterEqual(detector.last_decode_stats["invalid_label"], 2)
        self.assertGreaterEqual(detector.last_decode_stats["invalid_box"], 1)


if __name__ == "__main__":
    unittest.main()
