#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Panthera-HT coloured building-block visual grasp.

Workflow:
    select colour -> scan J1 -> detect -> plan -> direct grasp -> place -> ZERO
"""

import os
import signal
import sys
import threading
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
SDK_ROOT = PROJECT_ROOT / "Panthera-HT_SDK" / "panthera_python"
SDK_SCRIPTS = SDK_ROOT / "scripts"
ROBOT_CONFIG = (
    SDK_ROOT / "robot_param" / "Leader.yaml"
)
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "yoloe-26s-seg.pt"
MODEL_PATH = Path(os.environ.get("YOLOE_MODEL_PATH", str(DEFAULT_MODEL_PATH)))
DEFAULT_TEXT_ENCODER_PATH = PROJECT_ROOT / "mobileclip2_b.ts"
TEXT_ENCODER_PATH = Path(
    os.environ.get("YOLOE_TEXT_ENCODER_PATH", str(DEFAULT_TEXT_ENCODER_PATH))
)
CALIBRATION_FILE = PROJECT_ROOT / "hand_eye_calibration.json"

if str(SDK_ROOT) not in sys.path:
    sys.path.insert(0, str(SDK_ROOT))
if str(SDK_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SDK_SCRIPTS))

from Panthera_lib.grasp_config import GraspConfig, parse_target_command  # noqa: E402
from Panthera_lib.grasp_planner import GraspPlanner  # noqa: E402
from Panthera_lib.vision_pipeline import (  # noqa: E402
    CameraFeed,
    init_camera,
    load_hand_eye,
    load_target_model,
)
from Panthera_lib.vision_streamer import VisionStreamer  # noqa: E402
from Panthera_lib.Panthera import Panthera  # noqa: E402
from Panthera_lib.graspnet_pipeline import (  # noqa: E402
    GraspNetCandidateProvider,
)


shutdown_requested = threading.Event()


def safe_print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    try:
        print(*args, **kwargs)
    except Exception:
        pass


def on_signal(signum, _frame):
    shutdown_requested.set()
    safe_print(f"\n[SIGNAL] {signum} received; returning to ZERO.")


signal.signal(signal.SIGINT, on_signal)
signal.signal(signal.SIGTERM, on_signal)


def choose_target_at_start():
    safe_print(
        "\n您想要抓取什么？（例如：红色积木 / 黄色积木 / 蓝色积木，q 退出）"
    )
    while not shutdown_requested.is_set():
        try:
            command = input("> ")
        except EOFError:
            return False
        command = command.strip()
        if command.lower() == "q":
            return False
        color, accepted = parse_target_command(command)
        if accepted:
            selected = color if color is not None else "任意颜色"
            safe_print(
                f"[TARGET] 已选择 {selected} 积木；检测到可抓取目标后自动执行一次抓取。"
            )
            return color
        safe_print("未识别颜色，请输入红/黄/蓝/绿/白/黑积木，或 q 退出。")
    return False


def build_config():
    config = GraspConfig()
    config.sdk_scripts = str(SDK_SCRIPTS)
    config.robot_config = str(ROBOT_CONFIG)
    config.model_path = str(MODEL_PATH)
    config.project_root = PROJECT_ROOT
    config.text_encoder_path = TEXT_ENCODER_PATH
    config.calibration_file = CALIBRATION_FILE
    config.use_graspnet = os.environ.get("GRASPNET_USE", "0") == "1"
    config.graspnet_checkpoint_path = os.environ.get(
        "GRASPNET_CHECKPOINT_PATH", config.graspnet_checkpoint_path
    )
    config.stream_host = os.environ.get("VISION_STREAM_HOST", "0.0.0.0")
    config.stream_port = int(os.environ.get("VISION_STREAM_PORT", "8080"))
    config.stream_jpeg_quality = int(
        os.environ.get("VISION_STREAM_JPEG_QUALITY", "85")
    )
    config.validate()
    return config


def main():
    robot = None
    pipeline = None
    streamer = None
    camera_feed = None
    config = build_config()
    last_command = config.zero.copy()
    try:
        safe_print("=" * 68)
        safe_print("Panthera-HT coloured building-block visual grasp")
        safe_print("=" * 68)

        streamer = VisionStreamer(
            config.stream_host,
            config.stream_port,
            config.stream_jpeg_quality,
        )
        if not streamer.start():
            streamer = None
        if streamer is not None:
            safe_print(f"[STREAM] 网页推流已启动：{streamer.url}")

        pipeline, align, intrinsic, depth_scale = init_camera(config)
        camera_feed = CameraFeed(
            pipeline,
            align,
            config,
            depth_scale,
            streamer=streamer,
            model=None,
        )
        camera_feed.start()
        safe_print("[VISION] camera preview started.")

        robot = Panthera(config.robot_config)
        graspnet_provider = None
        if config.use_graspnet:
            safe_print("[GRASPNET] loading GraspNet candidate provider ...")
            graspnet_provider = GraspNetCandidateProvider(config)
            graspnet_provider.load()
            safe_print("[GRASPNET] GraspNet candidate provider loaded.")
        planner = GraspPlanner(
            robot,
            config,
            shutdown_requested,
            graspnet_provider=graspnet_provider,
        )

        planner.home()
        last_command = config.home.copy()
        safe_print("[INIT] opening gripper at HOME ...")
        planner.open_gripper()

        tcp_camera = load_hand_eye(config)
        model = load_target_model(config)
        camera_feed.set_model(model)
        safe_print("[VISION] YOLOE model loaded.")

        task_complete = planner.run_grasp_loop(
            camera_feed,
            intrinsic,
            tcp_camera,
            streamer,
            choose_target_at_start,
        )
        if task_complete:
            safe_print("[GRASP] scan, grasp and placement completed; returning to ZERO.")
    except Exception as exc:
        safe_print(f"[MAIN] exception: {exc!r}")
    finally:
        if camera_feed is not None:
            camera_feed.stop()
        if streamer is not None:
            safe_print("[STREAM] closing web preview ...")
            streamer.stop()
            safe_print("[STREAM] web preview closed.")
        if "planner" in locals():
            planner.safe_shutdown(pipeline, last_command)
        elif pipeline is not None:
            try:
                pipeline.stop()
            except Exception:
                pass


if __name__ == "__main__":
    main()
