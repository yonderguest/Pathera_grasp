from __future__ import annotations

import numpy as np
import unittest
import cv2

from Panthera_lib.grasp_config import GraspConfig
from Panthera_lib.vision_pipeline import (
    _detection_from_result,
    CameraFeed,
    classify_color,
    classify_color_evidence,
    detect_requested_color_regions,
    robust_surface_point,
)
from Panthera_lib.vision_streamer import VisionStreamer


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

    def test_grasp_ray_uses_obb_centre_instead_of_nearest_tail_pixels(self):
        config = GraspConfig()
        image = np.zeros((80, 120, 3), dtype=np.uint8)
        image[20:60, 20:100] = [0, 0, 255]
        mask = np.zeros((80, 120), dtype=np.uint8)
        mask[20:60, 20:100] = 1
        depth = np.full((80, 120), 700, dtype=np.uint16)
        depth[20:60, 20:60] = 450
        depth[20:60, 60:100] = 550

        detection = _detection_from_result(
            "Lego brick",
            0.9,
            (20, 20, 100, 60),
            mask,
            image,
            depth,
            0.001,
            config,
        )

        self.assertIsNotNone(detection)
        self.assertTrue(np.allclose(detection["pixel"], [59.5, 39.5], atol=1.0))
        self.assertLess(detection["depth_pixel"][0], detection["pixel"][0])
        self.assertGreater(detection["grasp_center_shift_px"], 8.0)
        self.assertAlmostEqual(detection["depth_m"], 0.45, places=3)

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

    def test_raw_capture_api_does_not_wait_for_object_inference(self):
        feed = CameraFeed(None, None, GraspConfig(), 0.001)
        feed._latest_capture = _capture(11, 11)

        self.assertEqual(feed.capture_freshness_marker(), 11)
        capture = feed.latest_capture()
        self.assertEqual(capture["frame_seq"], 11)
        self.assertTrue(np.all(capture["color_image"] == 11))
        self.assertIsNone(feed.wait_for_new_capture(11, timeout=0.01))

        feed._latest_capture = _capture(12, 12)
        self.assertEqual(feed.wait_for_new_capture(11, timeout=0.01)["frame_seq"], 12)

    def test_hand_mode_pause_clears_stale_object_snapshot(self):
        feed = CameraFeed(None, None, GraspConfig(), 0.001)
        feed._store_inference_result(_capture(5, 5), [{"id": 5}])
        self.assertIsNotNone(feed.latest())

        feed.set_object_inference_enabled(False)

        self.assertFalse(feed._object_inference_enabled)
        self.assertIsNone(feed.latest())
        self.assertEqual(feed.freshness_marker(), -1)
        feed._store_inference_result(_capture(6, 6), [{"id": 6}])
        self.assertIsNone(feed.latest())

        feed.set_object_inference_enabled(True)
        self.assertTrue(feed._object_inference_enabled)
        self.assertIsNone(feed.latest())

    def test_paused_object_result_cannot_overwrite_hand_preview(self):
        class Streamer:
            def __init__(self):
                self.calls = []

            def publish(self, *args, **kwargs):
                self.calls.append((args, kwargs))

        streamer = Streamer()
        feed = CameraFeed(
            None,
            None,
            GraspConfig(),
            0.001,
            streamer=streamer,
        )
        snapshot = feed._store_inference_result(_capture(7, 7), [], 0.01)
        self.assertIsNotNone(snapshot)

        feed.set_object_inference_enabled(False)

        self.assertFalse(feed._publish_object_snapshot_if_current(snapshot))
        self.assertEqual(streamer.calls, [])

    def test_streamer_keeps_raw_and_analysis_generations_separate(self):
        streamer = VisionStreamer(preview_fps=1000.0)
        raw = np.full((2, 2, 3), 8, dtype=np.uint8)
        analysis = np.full((2, 2, 3), 9, dtype=np.uint8)
        streamer.publish_capture(raw, np.ones((2, 2), dtype=np.uint16), 0.001)
        raw_snapshot = streamer._wait_for_next_frame(-1, kind="raw")
        self.assertEqual(raw_snapshot[0][0, 0, 0], 8)
        streamer.publish(analysis, [{"bbox": (0, 0, 1, 1)}])
        yolo_snapshot = streamer._wait_for_next_frame(-1, kind="yolo")
        self.assertEqual(yolo_snapshot[0][0, 0, 0], 9)
