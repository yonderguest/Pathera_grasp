#!/usr/bin/env python3
"""Run one still image through one or more project-compatible QNN contexts."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SDK_SCRIPTS = PROJECT_ROOT / "Panthera-HT_SDK" / "panthera_python" / "scripts"
sys.path.insert(0, str(SDK_SCRIPTS))

from Panthera_lib.grasp_config import GraspConfig  # noqa: E402
from Panthera_lib.npu_inference import NpuYoloDetector  # noqa: E402


def run_context(image, context: Path, confidence: float) -> dict:
    config = GraspConfig(project_root=PROJECT_ROOT)
    config.npu_context_path = str(context)
    detector = None
    started = time.monotonic()
    try:
        detector = NpuYoloDetector(config, confidence=confidence)
        load_seconds = time.monotonic() - started
        infer_started = time.monotonic()
        detections = detector.infer(image)
        infer_seconds = time.monotonic() - infer_started
        return {
            "context": str(context),
            "count": len(detections),
            "load_seconds": round(load_seconds, 3),
            "infer_seconds": round(infer_seconds, 3),
            "detections": [
                {
                    "box": item["box"],
                    "class_index": item["cls"],
                    "confidence": round(item["conf"], 4),
                }
                for item in detections
            ],
        }
    finally:
        if detector is not None:
            detector.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("contexts", nargs="+", type=Path)
    parser.add_argument("--confidence", type=float, default=0.30)
    args = parser.parse_args()

    image = cv2.imread(str(args.image))
    if image is None:
        raise SystemExit(f"cannot read image: {args.image}")
    results = [run_context(image, path, args.confidence) for path in args.contexts]
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
