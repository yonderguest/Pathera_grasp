#!/usr/bin/env python3
"""Camera/NPU recognition smoke test that never initialises the robot."""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SDK_SCRIPTS = PROJECT_ROOT / "Panthera-HT_SDK" / "panthera_python" / "scripts"
for path in (PROJECT_ROOT, SDK_SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from grasp_demo import build_config  # noqa: E402
from Panthera_lib.npu_inference import NpuYoloDetector  # noqa: E402
from Panthera_lib.lab_color import classify_lab_features  # noqa: E402
from Panthera_lib.vision_pipeline import detect_targets_npu, init_camera  # noqa: E402
from Panthera_lib.vision_streamer import draw_detections  # noqa: E402


def main() -> int:
    config = build_config()
    if not config.use_npu:
        raise RuntimeError("set YOLO_NPU=1 for the NPU smoke test")

    pipeline = None
    detector = None
    latencies = []
    class_counts: dict[str, int] = {}
    try:
        pipeline, align, _intrinsic, depth_scale = init_camera(config)
        detector = NpuYoloDetector(
            config,
            confidence=config.npu_confidence_threshold,
        )
        for frame_index in range(10):
            frames = align.process(pipeline.wait_for_frames())
            color = frames.get_color_frame()
            depth = frames.get_depth_frame()
            if not color or not depth:
                continue
            import numpy as np

            image = np.asanyarray(color.get_data())
            started = time.perf_counter()
            detections = detect_targets_npu(
                detector,
                image,
                depth,
                depth_scale,
                config,
            )
            latency = time.perf_counter() - started
            latencies.append(latency)
            if frame_index == 0:
                import cv2

                cv2.imwrite(
                    "/tmp/pathera_recognition_smoke.jpg",
                    draw_detections(image, detections),
                )
            labels = []
            for detection in detections:
                name = str(detection["class_name"])
                class_counts[name] = class_counts.get(name, 0) + 1
                labels.append(
                    f"{name}/{detection['color']}@{detection['confidence']:.2f}"
                )
                if frame_index == 0 and detection.get("color_evidence", {}).get(
                    "backend"
                ) == "lab":
                    result = classify_lab_features(
                        detection["color_evidence"]["lab_features"],
                        config.color_calibration_model,
                        distance_scale=config.color_lab_distance_scale,
                    )
                    diagnostics = result[4]
                    print(
                        "  lab="
                        f"{diagnostics.get('lab')} reason={diagnostics.get('reason', 'ok')} "
                        f"distance={diagnostics.get('best_distance')} "
                        f"limit={diagnostics.get('max_distance')}",
                        flush=True,
                    )
            print(
                f"frame={frame_index + 1:02d} latency={latency * 1000.0:.1f}ms "
                f"detections={labels}",
                flush=True,
            )
    finally:
        if detector is not None:
            detector.close()
        if pipeline is not None:
            pipeline.stop()

    if not latencies:
        raise RuntimeError("no aligned RGB-D frames were processed")
    median = statistics.median(latencies)
    print(
        f"summary: profile={config.recognition_profile}, classes={config.npu_class_names}, "
        f"median={median * 1000.0:.1f}ms, effective_fps={1.0 / median:.2f}, "
        f"class_counts={class_counts}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
