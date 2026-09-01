from __future__ import annotations

import numpy as np
import unittest
import cv2

from Panthera_lib.grasp_config import GraspConfig
from Panthera_lib.vision_pipeline import (
    CameraFeed,
    classify_color,
    classify_color_evidence,
    detect_requested_color_regions,
    robust_surface_point,
)


def _capture(frame_seq: int, value: int) -> dict:
    return {
        "color_image": np.full((2, 2, 3), value, dtype=np.uint8),
        "depth_image": np.full((2, 2), value * 10, dtype=np.uint16),
        "timestamp": float(frame_seq),
        "capture_timestamp_ns": frame_seq * 1_000,
        "frame_seq": frame_seq,
    }


class PerceptionSnapshotTests(unittest.TestCase):
    def test_camera_feed_stop_releases_pipeline_before_join(self):
        class Pipeline:
            def __init__(self):
                self.stopped = 0

            def stop(self):
                self.stopped += 1

        pipeline = Pipeline()
        feed = CameraFeed(
            pipeline,
            align=None,
            config=GraspConfig(),
            depth_scale=0.001,
        )

        feed.stop()
        feed.stop()

        self.assertEqual(pipeline.stopped, 1)

    def test_depth_uses_one_coherent_near_surface_for_pixel_and_z(self):
        config = GraspConfig()
        depth = np.full((20, 20), 550, dtype=np.uint16)
        depth[:8, :] = 450
        depth[0, 0] = 100
        mask = np.ones((20, 20), dtype=np.uint8)

        surface = robust_surface_point(depth, mask, 0.001, config)

        self.assertIsNotNone(surface)
        self.assertAlmostEqual(surface["depth_m"], 0.45, places=3)
        self.assertLess(surface["pixel"][1], 8.0)
        self.assertGreaterEqual(surface["depth_samples"], config.depth_min_surface_pixels)

    def test_screenshot_calibrated_yellow_green_boundary(self):
        config = GraspConfig()
        mask = np.ones((20, 20), dtype=np.uint8)

        def classify_hue(hue):
            hsv = np.full((20, 20, 3), [hue, 220, 180], dtype=np.uint8)
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            return classify_color(bgr, mask, config)[0]

        self.assertEqual(classify_hue(27), "yellow")
        self.assertEqual(classify_hue(42), "green")

    def test_adaptive_color_core_rejects_mask_border_background(self):
        config = GraspConfig()
        image = np.zeros((30, 30, 3), dtype=np.uint8)
        image[5:25, 5:25] = [255, 0, 0]
        image[8:22, 8:22] = [0, 0, 255]
        mask = np.zeros((30, 30), dtype=np.uint8)
        mask[5:25, 5:25] = 1

        color, confidence = classify_color(image, mask, config)

        self.assertEqual(color, "red")
        self.assertGreaterEqual(confidence, config.color_dominant_ratio)

    def test_color_vote_margin_rejects_ambiguous_hues(self):
        config = GraspConfig()
        evidence = {
            "core_pixels": 100,
            "dark": 0,
            "white": 0,
            "chromatic_pixels": 100,
            "votes": {"red": 50, "yellow": 3, "green": 2, "blue": 45},
        }

        color, ratio, samples, margin = classify_color_evidence(evidence, config)

        self.assertEqual(color, "unknown")
        self.assertEqual(ratio, 0.5)
        self.assertEqual(samples, 100)
        self.assertLess(margin, config.color_min_margin)

    def test_refinement_color_fallback_recovers_known_green_block(self):
        config = GraspConfig()
        image = np.zeros((120, 160, 3), dtype=np.uint8)
        image[35:85, 55:105] = [0, 220, 0]
        depth = np.full((120, 160), 500, dtype=np.uint16)

        detections = detect_requested_color_regions(
            image,
            depth,
            0.001,
            "green",
            config,
        )

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["color"], "green")
        self.assertAlmostEqual(detections[0]["depth_m"], 0.5, places=3)

    def test_snapshot_keeps_rgb_depth_and_detections_on_same_frame(self):
        feed = CameraFeed(None, None, GraspConfig(), 0.001, intrinsic="K")
        first = _capture(7, 7)
        feed._store_inference_result(first, [{"id": 7}])
        feed._latest_capture = _capture(8, 8)

        snapshot = feed.latest()
        self.assertEqual(snapshot["frame_seq"], 7)
        self.assertEqual(snapshot["capture_timestamp_ns"], 7_000)
        self.assertEqual(snapshot["detections"], [{"id": 7}])
        self.assertTrue(np.all(snapshot["color_image"] == 7))
        self.assertTrue(np.all(snapshot["depth_image"] == 70))
        self.assertEqual(snapshot["intrinsics"], "K")


    def test_wait_for_newer_times_out_instead_of_returning_stale_frame(self):
        feed = CameraFeed(None, None, GraspConfig(), 0.001)
        feed._store_inference_result(_capture(3, 3), [])
        marker = feed.freshness_marker()

        self.assertEqual(marker, 3)
        self.assertIsNone(feed.wait_for_newer(marker, timeout=0.02))

        feed._store_inference_result(_capture(4, 4), [])
        self.assertEqual(feed.wait_for_newer(marker, timeout=0.02)["frame_seq"], 4)
