#!/usr/bin/env python3
"""Camera/CPU-YOLOE hand smoke test that never initialises the robot."""

from __future__ import annotations

import argparse
from dataclasses import replace
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SDK_SCRIPTS = PROJECT_ROOT / "Panthera-HT_SDK" / "panthera_python" / "scripts"
for path in (PROJECT_ROOT, SDK_SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from grasp_demo import build_config  # noqa: E402
from Panthera_lib.hand_follow import (  # noqa: E402
    CpuYoloHandDetector,
    HandFollowSettings,
    HandTrackGate,
    load_cpu_hand_model,
)
from Panthera_lib.vision_pipeline import init_camera  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="CPU YOLOE hand camera smoke test; never opens the robot"
    )
    parser.add_argument("--prompt", default="hand")
    parser.add_argument("--confidence", type=float, default=0.10)
    parser.add_argument("--frames", type=int, default=15)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/pathera_hand_smoke"),
    )
    args = parser.parse_args()
    if not 0.0 < args.confidence < 1.0:
        parser.error("--confidence must be between 0 and 1")
    if not 1 <= args.frames <= 60:
        parser.error("--frames must be between 1 and 60")
    return args


def _draw_diagnostics(image, diagnostics, accepted):
    annotated = np.asarray(image, dtype=np.uint8).copy()
    for item in diagnostics.get("raw_candidates", []):
        x1, y1, x2, y2 = (int(round(value)) for value in item["bbox"])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 165, 255), 2)
        cv2.putText(
            annotated,
            f"raw {item['name']} {item['confidence']:.2f}",
            (max(0, x1), max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 165, 255),
            2,
            cv2.LINE_AA,
        )
    for candidate in accepted:
        x1, y1, x2, y2 = candidate.bbox
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 90), 3)
        cv2.putText(
            annotated,
            f"accepted depth={candidate.depth_m:.3f}m",
            (max(0, x1), min(annotated.shape[0] - 8, y2 + 22)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 90),
            2,
            cv2.LINE_AA,
        )
    return annotated


def main() -> int:
    args = parse_args()
    config = build_config()
    settings = replace(
        HandFollowSettings(),
        prompt=args.prompt.strip(),
        confidence_threshold=float(args.confidence),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = load_cpu_hand_model(
        config.model_path,
        config.text_encoder_path,
        settings.prompt,
    )
    detector = CpuYoloHandDetector(model, settings)
    gate = HandTrackGate(settings)

    pipeline = None
    latencies = []
    candidate_counts = []
    try:
        pipeline, align, intrinsic, depth_scale = init_camera(config)
        # Camera/model startup time is not a hand-loss interval. Start the
        # arming clock only once fresh RGB-D observations are about to begin.
        gate.reset()
        for frame_index in range(args.frames):
            frames = align.process(pipeline.wait_for_frames())
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue
            color = np.asanyarray(color_frame.get_data())
            depth = np.asanyarray(depth_frame.get_data())
            started = time.perf_counter()
            candidates = detector.detect(
                color,
                depth,
                depth_scale,
                intrinsic,
            )
            latency = time.perf_counter() - started
            diagnostics = dict(detector.last_diagnostics)
            matching_count = int(
                diagnostics.get("matching_count", len(candidates))
            )
            authorized_hand = gate.update(
                candidates,
                now=time.monotonic(),
                observed_count=matching_count,
            )
            latencies.append(latency)
            candidate_counts.append(len(candidates))
            raw_summary = [
                (
                    f"{item['name']}@{item['confidence']:.2f}"
                    + (
                        f" rejected={item['rejection_reason']}"
                        if item.get("rejection_reason")
                        else ""
                    )
                )
                for item in diagnostics.get("raw_candidates", [])
            ]
            summary = [
                (
                    f"confidence={candidate.confidence:.2f}, "
                    f"depth={candidate.depth_m:.3f}m, "
                    f"spread={candidate.depth_spread_m * 1000.0:.1f}mm"
                )
                for candidate in candidates
            ]
            print(
                f"frame={frame_index + 1:02d} latency={latency * 1000.0:.1f}ms "
                f"raw={raw_summary} geometry_rejected="
                f"{diagnostics.get('geometry_rejected', 0)} "
                f"matching={matching_count} hands={summary} "
                f"gate={gate.state.value!r} "
                f"motion_authorized={authorized_hand is not None} "
                f"reason={gate.reason!r}",
                flush=True,
            )
            annotated = _draw_diagnostics(image=color, diagnostics=diagnostics, accepted=candidates)
            cv2.imwrite(
                str(args.output_dir / f"frame_{frame_index + 1:02d}.jpg"),
                annotated,
            )
    finally:
        if pipeline is not None:
            pipeline.stop()

    if not latencies:
        raise RuntimeError("no aligned RGB-D frames were processed")
    median = statistics.median(latencies)
    print(
        f"summary: backend=CPU YOLOE, prompt={settings.prompt!r}, "
        f"median={median * 1000.0:.1f}ms, effective_fps={1.0 / median:.2f}, "
        f"candidate_counts={candidate_counts}, output_dir={args.output_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
