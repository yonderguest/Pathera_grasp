from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from Panthera_lib.grasp_config import (
    GraspConfig,
    canonical_object_name,
)
from Panthera_lib.lab_color import classify_lab_features, load_color_calibration


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class RecognitionProfileTests(unittest.TestCase):
    def test_object3_asset_and_manifest_are_consistent(self):
        model = (
            PROJECT_ROOT
            / "third_party"
            / "qnn"
            / "yoloe-26s-seg_640_iq9075_qnn_object3.bin"
        )
        self.assertEqual(model.stat().st_size, 22_294_528)
        digest = hashlib.sha256(model.read_bytes()).hexdigest()
        self.assertEqual(
            digest,
            "25dbff5010f0dee5f15c9c39cb0eeb03a384a7d66401281ced98ddf35d02add6",
        )
        manifest = json.loads(
            (PROJECT_ROOT / "config" / "recognition_profiles.json").read_text(
                encoding="utf-8"
            )
        )
        profile = manifest["profiles"]["object3"]
        self.assertEqual(profile["sha256"], digest)
        self.assertEqual(
            profile["classes"], ["bottle", "box", "toy building block"]
        )

    def test_calibrated_centres_and_multiframe_lab_aggregation(self):
        model = load_color_calibration(
            PROJECT_ROOT / "config" / "color_calibration.json"
        )
        for color, item in model["classes"].items():
            center = item["center_lab"]
            feature = {
                "core_pixels": 100,
                "valid_pixels": 90,
                "white_reference_pixels": 200,
                "central_lightness": center[0],
                "central_chroma": (
                    (center[1] - 128.0) ** 2 + (center[2] - 128.0) ** 2
                ) ** 0.5,
                "lab": center,
            }
            classified = classify_lab_features([feature, dict(feature)], model)
            self.assertEqual(classified[0], color)
            self.assertGreater(classified[1], 0.5)
            self.assertEqual(classified[2], 180)

    def test_profiles_and_canonical_object_names_remain_coupled(self):
        config = GraspConfig()
        config.validate()
        self.assertEqual(
            tuple(canonical_object_name(name) for name in config.npu_class_names),
            ("bottle", "box", "toy building block"),
        )
        config.apply_recognition_profile("block4")
        config.validate()
        self.assertEqual(
            set(canonical_object_name(name) for name in config.npu_class_names),
            {"toy building block"},
        )

    def test_exposure_tolerance_keeps_margin_gate(self):
        model = load_color_calibration(
            PROJECT_ROOT / "config" / "color_calibration.json"
        )
        shifted_red = {
            "core_pixels": 100,
            "valid_pixels": 90,
            "white_reference_pixels": 200,
            "central_lightness": 90.0,
            "central_chroma": 50.0,
            "lab": [90.0, 179.0, 168.0],
        }
        ambiguous_yellow_green = {
            "core_pixels": 100,
            "valid_pixels": 90,
            "white_reference_pixels": 200,
            "central_lightness": 130.0,
            "central_chroma": 50.0,
            "lab": [128.0, 116.0, 179.0],
        }
        brown_background = {
            "core_pixels": 100,
            "valid_pixels": 90,
            "white_reference_pixels": 200,
            "central_lightness": 90.0,
            "central_chroma": 20.0,
            "lab": [90.0, 133.0, 145.0],
        }
        self.assertEqual(
            classify_lab_features([shifted_red], model, distance_scale=2.5)[0],
            "red",
        )
        self.assertEqual(
            classify_lab_features(
                [ambiguous_yellow_green], model, distance_scale=2.5
            )[0],
            "unknown",
        )
        self.assertEqual(
            classify_lab_features(
                [brown_background], model, distance_scale=2.5
            )[0],
            "unknown",
        )


if __name__ == "__main__":
    unittest.main()
