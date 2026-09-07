"""Panthera-HT 的 WSL2 MuJoCo 专用入口。

本模块只在 ``main()`` 内延迟导入 ``sim`` 包，不导入或启动 Panthera SDK、
CAN、RealSense、QNN/NPU、ROS2 或语音硬件。真机入口始终是 ``grasp_demo.py``。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("recognition", "grasp", "sort"), default="sort"
    )
    parser.add_argument("--detector", choices=("yolo", "colour"), default="colour")
    parser.add_argument("--target-color", choices=("red", "green", "blue"), default="green")
    parser.add_argument(
        "--scene-seed",
        type=int,
        default=None,
        help="Optional fixed seed. Omit it to create a new tabletop layout each run.",
    )
    parser.add_argument(
        "--sort-colours",
        choices=("red", "green", "blue"),
        nargs="+",
        default=("green", "red", "blue"),
        help="Ordered colours for --mode sort; all must be placed in PUT1.",
    )
    parser.add_argument(
        "--weights", default=str(PROJECT_ROOT / "sim" / "models" / "block_detector.pt")
    )
    parser.add_argument(
        "--save-dir", default=str(PROJECT_ROOT / "reports" / "mujoco_sim")
    )
    parser.add_argument(
        "--record-video",
        default=None,
        help="Optional MP4 path for an external-camera recording of the simulation.",
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Open a real-time interactive MuJoCo window (requires WSLg).",
    )
    parser.add_argument(
        "--camera-view",
        action="store_true",
        help="With --viewer, open a live virtual RealSense RGB window.",
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Serve the local MuJoCo browser dashboard instead of auto-starting a task.",
    )
    parser.add_argument("--web-host", default="127.0.0.1")
    parser.add_argument("--web-port", type=int, default=8765)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    # Import only the simulation package.  None of the legacy Panthera SDK,
    # RealSense, QNN/NPU or audio modules are imported on this path.
    from sim.scripts import run_sim_demo

    original_argv = sys.argv
    try:
        sys.argv = [
            "run_sim_demo",
            "--mode",
            args.mode,
            "--save-dir",
            args.save_dir,
            "--detector",
            args.detector,
            "--target-color",
            args.target_color,
            *(
                ["--scene-seed", str(args.scene_seed)]
                if args.scene_seed is not None
                else []
            ),
            "--sort-colours",
            *args.sort_colours,
            "--weights",
            args.weights,
            *(
                ["--record-video", args.record_video]
                if args.record_video is not None
                else []
            ),
            *(["--viewer"] if args.viewer else []),
            *(["--camera-view"] if args.camera_view else []),
            *(["--web"] if args.web else []),
            "--web-host",
            args.web_host,
            "--web-port",
            str(args.web_port),
        ]
        return run_sim_demo.main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    raise SystemExit(main())
