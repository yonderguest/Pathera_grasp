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


def run_context(
    image,
    context: Path,
    confidences: list[float],
    class_names: list[str] | None,
) -> dict:
    config = GraspConfig(project_root=PROJECT_ROOT)
    config.npu_context_path = str(context)
    if class_names is not None:
        config.npu_class_names = tuple(class_names)
    detector = None
    started = time.monotonic()
    try:
        detector = NpuYoloDetector(config, confidence=confidences[0])
        load_seconds = time.monotonic() - started
        sweeps = []
        for confidence in confidences:
            detector.confidence = float(confidence)
            infer_started = time.monotonic()
            detections = detector.infer(image)
            infer_seconds = time.monotonic() - infer_started
            sweeps.append(
                {
                    "confidence_threshold": confidence,
                    "count": len(detections),
                    "infer_seconds": round(infer_seconds, 3),
                    "decode_stats": dict(detector.last_decode_stats),
                    "detections": [
                        {
                            "box": item["box"],
                            "class_index": item["cls"],
                            "class_name": detector.names[item["cls"]],
                            "confidence": round(item["conf"], 4),
                        }
                        for item in detections
                    ],
                }
            )
        return {
            "context": str(context),
            "context_bytes": context.stat().st_size,
            "decoder_class_names": list(detector.names),
            "class_names_source": (
                "--class-names"
                if class_names is not None
                else "GraspConfig.npu_class_names"
            ),
            "load_seconds": round(load_seconds, 3),
            "threshold_sweep": sweeps,
        }
    finally:
        if detector is not None:
            detector.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("contexts", nargs="+", type=Path)
    parser.add_argument(
        "--confidences",
        type=float,
        nargs="+",
        default=[0.10, 0.15, 0.20, 0.30],
        help="post-process thresholds to compare without changing the QNN context",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=None,
        help="backward-compatible single-threshold alias",
    )
    parser.add_argument(
        "--class-names",
        nargs="+",
        default=None,
        help=(
            "exact class order used when the context was compiled; compare "
            "contexts with different label sets in separate invocations"
        ),
    )
    args = parser.parse_args()

    image = cv2.imread(str(args.image))
    if image is None:
        raise SystemExit(f"cannot read image: {args.image}")
    confidences = (
        [args.confidence] if args.confidence is not None else args.confidences
    )
    if any(not 0.0 < value < 1.0 for value in confidences):
        raise SystemExit("all --confidences values must be in (0, 1)")
    results = [
        run_context(image, path, confidences, args.class_names)
        for path in args.contexts
    ]
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
