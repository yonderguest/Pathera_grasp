#!/usr/bin/env python3
"""Serve a synthetic RGB-D frame for browser UI checks; no hardware imports."""

from __future__ import annotations

import signal
import sys
import time
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SDK_SCRIPTS = PROJECT_ROOT / "Panthera-HT_SDK" / "panthera_python" / "scripts"
for path in (PROJECT_ROOT, SDK_SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Panthera_lib.vision_streamer import VisionStreamer  # noqa: E402


def main() -> int:
    streamer = VisionStreamer("0.0.0.0", 8091, 92)
    if not streamer.start():
        return 2

    height, width = 480, 640
    y_axis = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    x_axis = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    color = np.zeros((height, width, 3), dtype=np.uint8)
    color[:, :, 0] = (35.0 + 70.0 * x_axis).astype(np.uint8)
    color[:, :, 1] = (25.0 + 65.0 * y_axis).astype(np.uint8)
    color[:, :, 2] = (25.0 + 30.0 * (1.0 - x_axis)).astype(np.uint8)
    depth = (650.0 + 350.0 * (x_axis + y_axis)).astype(np.uint16)
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[155:325, 235:405] = 1
    color[155:325, 235:405] = [155, 95, 55]
    streamer.publish(
        color,
        [
            {
                "class_name": "human hand",
                "confidence": 0.83,
                "color": "unknown",
                "color_confidence": 0.0,
                "color_frames": 1,
                "depth_m": 0.30,
                "depth_spread_m": 0.006,
                "bbox": (235, 155, 405, 325),
                "mask": mask,
            }
        ],
        depth,
        0.001,
    )
    streamer.set_accepting_targets(True)
    print(f"CONTROL_URL={streamer.control_url}", flush=True)

    stopping = False

    def request_stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    try:
        while not stopping:
            time.sleep(0.2)
    finally:
        streamer.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
