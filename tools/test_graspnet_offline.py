#!/usr/bin/env python3
"""Offline GraspNet candidate smoke test.

This script does **not** connect to the robot.  It loads a saved RGB-D frame,
an optional object mask, and a camera intrinsic file, then prints the GraspNet
candidates in the camera frame.  Use it to inspect candidate scores and pose
axes before enabling ``use_graspnet`` on the real robot.

Example:

    python tools/test_graspnet_offline.py \
        --data-dir /path/to/saved_frame \
        --checkpoint third_party/graspnet-baseline/checkpoint-rs.tar

Expected files in ``--data-dir``:

    color.png        BGR colour image
    depth.png        16-bit aligned depth image
    intrinsic.json   {"width":640,"height":480,"fx":..., "fy":...,
                      "ppx":..., "ppy":..., "depth_scale":0.001}
    mask.png         optional 8-bit binary object mask
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SDK_ROOT = PROJECT_ROOT / "Panthera-HT_SDK" / "panthera_python"
SDK_SCRIPTS = SDK_ROOT / "scripts"
for path in (SDK_ROOT, SDK_SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Panthera_lib.grasp_config import GraspConfig  # noqa: E402
from Panthera_lib.graspnet_pipeline import GraspNetCandidateProvider  # noqa: E402


def load_intrinsics(data_dir: Path):
    payload = json.loads((data_dir / "intrinsic.json").read_text(encoding="utf-8"))
    intrinsic = rs.intrinsics()
    intrinsic.width = int(payload["width"])
    intrinsic.height = int(payload["height"])
    intrinsic.ppx = float(payload["ppx"])
    intrinsic.ppy = float(payload["ppy"])
    intrinsic.fx = float(payload["fx"])
    intrinsic.fy = float(payload["fy"])
    intrinsic.model = rs.distortion.inverse_brown_conrady
    intrinsic.coeffs = [0.0, 0.0, 0.0, 0.0, 0.0]
    return intrinsic, float(payload["depth_scale"])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument(
        "--checkpoint",
        default="third_party/graspnet-baseline/checkpoint-rs.tar",
    )
    parser.add_argument("--max-candidates", type=int, default=10)
    parser.add_argument("--score-threshold", type=float, default=0.0)
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    color_path = data_dir / "color.png"
    depth_path = data_dir / "depth.png"
    mask_path = data_dir / "mask.png"
    if not color_path.is_file() or not depth_path.is_file():
        raise SystemExit(f"missing colour/depth images in {data_dir}")

    color = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if color is None or depth is None:
        raise SystemExit("failed to read colour/depth images")

    intrinsic, depth_scale = load_intrinsics(data_dir)
    if mask_path.is_file():
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.uint8)
    else:
        mask = (depth > 0).astype(np.uint8)
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        raise SystemExit("object mask contains no foreground pixels")
    bbox = [
        int(xs.min()),
        int(ys.min()),
        int(xs.max()) + 1,
        int(ys.max()) + 1,
    ]

    config = GraspConfig()
    config.project_root = PROJECT_ROOT
    config.use_graspnet = True
    config.graspnet_checkpoint_path = args.checkpoint
    config.graspnet_max_candidates = args.max_candidates
    config.graspnet_score_threshold = args.score_threshold
    config.validate()

    provider = GraspNetCandidateProvider(config)
    provider.load()
    base_camera = np.eye(4, dtype=float)

    candidates = provider.generate_candidates(
        color,
        depth,
        {"mask": mask, "bbox": bbox},
        intrinsic,
        depth_scale,
        base_camera,
    )
    print(f"[OFFLINE] generated {len(candidates)} candidate(s)")
    for index, candidate in enumerate(candidates, start=1):
        print(
            f"[{index:02d}] score={candidate['score']:.4f} "
            f"width={candidate['gripper_width']:.4f} "
            f"tool_target={np.round(candidate['tool_target'], 4)}"
        )
        print(f"      rotation={np.round(candidate['tool_rotation'], 4).tolist()}")


if __name__ == "__main__":
    main()
