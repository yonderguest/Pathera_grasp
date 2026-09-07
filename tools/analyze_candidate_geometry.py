#!/usr/bin/env python3
"""Offline Base-frame reconstruction for screenshot/terminal RGB-D candidates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SDK_ROOT = PROJECT_ROOT / "Panthera-HT_SDK" / "panthera_python"
SDK_SCRIPTS = SDK_ROOT / "scripts"
sys.path.insert(0, str(SDK_ROOT))
sys.path.insert(0, str(SDK_SCRIPTS))

from Panthera_lib.Panthera import Panthera  # noqa: E402
from Panthera_lib.grasp_config import GraspConfig  # noqa: E402
from Panthera_lib.vision_pipeline import get_base_camera_transform  # noqa: E402


def offline_robot(config_path: Path) -> Panthera:
    """Load only YAML + Pinocchio; never construct CAN/serial hardware."""
    robot = Panthera.__new__(Panthera)
    with config_path.open("r", encoding="utf-8") as handle:
        robot.config = yaml.safe_load(handle)
    robot.config_dir = str(config_path.parent)
    robot.model = None
    robot.data = None
    robot.joint_ids = []
    robot.joint_names = []
    limits = robot.config["robot"]["joint_limits"]
    robot.joint_limits = {
        "lower": np.asarray(limits["lower"], dtype=float),
        "upper": np.asarray(limits["upper"], dtype=float),
    }
    robot._load_urdf_model()
    if robot.model is None:
        raise RuntimeError("offline URDF load failed")
    return robot


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--u", type=float, required=True)
    parser.add_argument("--v", type=float, required=True)
    parser.add_argument("--depth", type=float, required=True)
    parser.add_argument("--joint", type=float, nargs=6, required=True)
    parser.add_argument("--offset", type=float, nargs=3)
    parser.add_argument("--fx", type=float, default=393.88287353515625)
    parser.add_argument("--fy", type=float, default=393.34136962890625)
    parser.add_argument("--cx", type=float, default=318.4827880859375)
    parser.add_argument("--cy", type=float, default=234.91738891601562)
    args = parser.parse_args()

    config = GraspConfig(project_root=PROJECT_ROOT)
    if args.offset is not None:
        config.grasp_offset_base = np.asarray(args.offset, dtype=float)
    robot = offline_robot(
        SDK_ROOT / "robot_param" / "Leader.yaml"
    )
    fk = robot.forward_kinematics(np.asarray(args.joint, dtype=float))
    rotation = np.asarray(fk["rotation"], dtype=float)
    joint6_to_tcp_base = rotation @ config.tcp_in_joint6
    calibration = json.loads(
        (PROJECT_ROOT / "hand_eye_calibration.json").read_text(encoding="utf-8")
    )
    tcp_camera = np.asarray(calibration["T_tcp_camera"], dtype=float)
    camera_tcp = np.linalg.inv(tcp_camera)
    tcp_to_camera_base = rotation @ tcp_camera[:3, 3]
    joint6_to_camera_base = joint6_to_tcp_base + tcp_to_camera_base
    base_camera = get_base_camera_transform(
        robot,
        tcp_camera,
        np.asarray(args.joint, dtype=float),
        tcp_in_joint6=config.tcp_in_joint6,
    )
    camera_point = np.array(
        [
            (args.u - args.cx) / args.fx * args.depth,
            (args.v - args.cy) / args.fy * args.depth,
            args.depth,
        ],
        dtype=float,
    )
    base_point = (base_camera @ np.append(camera_point, 1.0))[:3]
    tool_target = base_point + config.grasp_offset_base
    radial = float(np.hypot(tool_target[0], tool_target[1]))
    in_tool_workspace = bool(
        config.tool_x_range[0] <= tool_target[0] <= config.tool_x_range[1]
        and config.tool_y_range[0] <= tool_target[1] <= config.tool_y_range[1]
        and config.tool_z_range[0] <= tool_target[2] <= config.tool_z_range[1]
        and config.radial_range[0] <= radial <= config.radial_range[1]
    )
    print(
        json.dumps(
            {
                "camera_point_m": camera_point.tolist(),
                "tcp_to_camera_tcp_m": tcp_camera[:3, 3].tolist(),
                "tcp_origin_in_camera_m": camera_tcp[:3, 3].tolist(),
                "tcp_to_camera_base_m": tcp_to_camera_base.tolist(),
                "joint6_to_tcp_base_m": joint6_to_tcp_base.tolist(),
                "joint6_to_camera_base_m": joint6_to_camera_base.tolist(),
                "base_point_m": base_point.tolist(),
                "grasp_offset_base_m": config.grasp_offset_base.tolist(),
                "tool_target_m": tool_target.tolist(),
                "tool_workspace_ok_without_wrist_check": in_tool_workspace,
                "radial_m": radial,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
