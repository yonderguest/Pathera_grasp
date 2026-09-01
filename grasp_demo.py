#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Panthera-HT coloured building-block visual grasp.

Workflow:
    select colour -> scan J1 -> detect -> 5 cm pre-grasp -> re-detect
    -> final grasp -> place -> ZERO
"""

import os
import select
import signal
import sys
import threading
import time
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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from voice_controller import VoiceInterface  # noqa: E402
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
from Panthera_lib.npu_inference import NpuYoloDetector  # noqa: E402


shutdown_requested = threading.Event()


def safe_print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    try:
        print(*args, **kwargs)
    except Exception:
        pass


def on_signal(signum, _frame):
    shutdown_requested.set()
    safe_print(f"\n[SIGNAL] {signum} received; returning to startup pose.")


def on_web_stop_requested():
    """Request a graceful stop whose final destination is the startup pose."""
    shutdown_requested.set()
    safe_print("\n[WEB] end requested; returning to startup pose before motor stop.")


signal.signal(signal.SIGINT, on_signal)
signal.signal(signal.SIGTERM, on_signal)


def read_terminal_command(streamer=None) -> str | None:
    """Read one terminal or browser command while remaining interruptible."""
    safe_print("> ", end="")
    terminal_available = bool(sys.stdin.isatty())
    while not shutdown_requested.is_set():
        if streamer is not None:
            command = streamer.poll_target_command()
            if command is not None:
                safe_print(f"\n[WEB] 收到网页目标：{command}")
                return command
            if streamer.is_closed and not terminal_available:
                return None
        if not terminal_available:
            time.sleep(0.2)
            continue
        try:
            readable, _, _ = select.select([sys.stdin], [], [], 0.2)
        except (OSError, ValueError):
            return None
        if readable:
            line = sys.stdin.readline()
            return line if line else None
    return None


def choose_target_at_start(voice=None, streamer=None):
    def finish(value):
        if streamer is not None:
            streamer.set_accepting_targets(False)
        return value

    safe_print(
        "\n您想要抓取什么？（例如：红色积木 / 黄色积木 / 蓝色积木，q 退出）"
    )
    if streamer is not None:
        streamer.set_accepting_targets(True)
        streamer.set_control_message(
            "请选择目标颜色并确认现场安全；提交后机械臂会自动开始扫描。"
        )
    if voice is not None and voice.available:
        voice.say("请说出要抓取的颜色，例如红色积木")
        command = voice.listen_for_command()
        if command:
            command = command.strip()
            if command.lower() == "q":
                return finish(False)
            color, accepted = parse_target_command(command)
            if accepted:
                selected = color if color is not None else "任意颜色"
                safe_print(
                    f"[TARGET] 语音识别：已选择 {selected} 积木；检测到可抓取目标后自动执行一次抓取。"
                )
                voice.say(f"已选择{selected}积木")
                return finish(color)
            safe_print(
                f"[VOICE] 未能从语音识别颜色：{command!r}，请继续用终端输入。"
            )
    while not shutdown_requested.is_set():
        try:
            command = read_terminal_command(streamer)
        except (EOFError, KeyboardInterrupt):
            return finish(False)
        if command is None:
            return finish(False)
        command = command.strip()
        if command.lower() == "q":
            if streamer is not None:
                streamer.set_control_message("收到安全退出命令。")
            return finish(False)
        color, accepted = parse_target_command(command)
        if accepted:
            selected = color if color is not None else "任意颜色"
            safe_print(
                f"[TARGET] 已选择 {selected} 积木；检测到可抓取目标后自动执行一次抓取。"
            )
            if voice is not None:
                voice.say(f"已选择{selected}积木")
            if streamer is not None:
                streamer.set_control_message(f"目标有效：{selected}积木。")
            return finish(color)
        if streamer is not None:
            streamer.set_control_message(
                f"无法识别“{command}”，请输入红/黄/蓝/绿/白/黑或任意颜色。"
            )
        safe_print("未识别颜色，请输入红/黄/蓝/绿/白/黑积木，或 q 退出。")
    return finish(False)


def build_config():
    config = GraspConfig()
    config.sdk_scripts = str(SDK_SCRIPTS)
    config.robot_config = str(ROBOT_CONFIG)
    config.model_path = str(MODEL_PATH)
    config.project_root = PROJECT_ROOT
    config.text_encoder_path = TEXT_ENCODER_PATH
    config.calibration_file = CALIBRATION_FILE
    # OBB / Seeed is the default grasp backend. GraspNet is opt-in only
    # when the operator explicitly sets GRASPNET_USE=1.
    config.use_graspnet = False
    if os.environ.get("GRASPNET_USE", "0") == "1":
        config.use_graspnet = True
    config.use_npu = os.environ.get("YOLO_NPU", "1") == "1"
    config.graspnet_checkpoint_path = os.environ.get(
        "GRASPNET_CHECKPOINT_PATH", config.graspnet_checkpoint_path
    )
    config.stream_host = os.environ.get("VISION_STREAM_HOST", "0.0.0.0")
    config.stream_port = int(os.environ.get("VISION_STREAM_PORT", "8080"))
    config.stream_jpeg_quality = int(
        os.environ.get("VISION_STREAM_JPEG_QUALITY", "92")
    )
    config.camera_serial = os.environ.get("REALSENSE_SERIAL", "").strip()
    voice_input_flag = os.environ.get("VOICE_INPUT", "0").strip().lower()
    config.use_voice = voice_input_flag not in {"0", "false", "no", "off"}
    config.voice_asr_model_dir = os.environ.get(
        "IQ9075_ASR_MODEL_DIR",
        str(PROJECT_ROOT / "models" / "sensevoice"),
    )
    config.voice_tts_model_dir = os.environ.get(
        "IQ9075_SHERPA_TTS_MODEL_DIR",
        str(PROJECT_ROOT / "models" / "sherpa_tts" / "vits-melo-tts-zh_en"),
    )
    config.voice_prompt_duration = float(
        os.environ.get("VOICE_PROMPT_DURATION", "3.5")
    )
    config.validate()
    return config


def main():
    exit_code = 0
    robot = None
    planner = None
    pipeline = None
    streamer = None
    camera_feed = None
    npu_detector = None
    voice = None
    startup_joint_position = None
    config = build_config()
    try:
        safe_print("=" * 68)
        safe_print("Panthera-HT coloured building-block visual grasp")
        safe_print("=" * 68)

        streamer = VisionStreamer(
            config.stream_host,
            config.stream_port,
            config.stream_jpeg_quality,
            stop_callback=on_web_stop_requested,
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
            intrinsic=intrinsic,
        )
        camera_feed.start()
        safe_print("[VISION] camera preview started.")

        # Load the selected vision backend before taking ownership of or
        # moving the arm.  Backend startup failures therefore remain motion-free.
        tcp_camera = load_hand_eye(config)
        if config.use_npu:
            safe_print("[VISION] selected backend: Qualcomm QNN HTP NPU")
            npu_detector = NpuYoloDetector(
                config, confidence=config.npu_confidence_threshold
            )
            camera_feed.set_npu_detector(npu_detector)
            safe_print("[VISION] YOLOE NPU detector ready.")
        else:
            safe_print("[VISION] selected backend: CPU YOLOE (slow fallback)")
            model = load_target_model(config)
            camera_feed.set_model(model)
            safe_print("[VISION] YOLOE CPU model loaded.")

        robot = Panthera(config.robot_config)
        startup_joint_position = np.asarray(
            robot.current_joint_position(),
            dtype=float,
        )
        if (
            startup_joint_position.shape != (6,)
            or not np.all(np.isfinite(startup_joint_position))
            or np.any(startup_joint_position < config.joint_lower)
            or np.any(startup_joint_position > config.joint_upper)
        ):
            raise RuntimeError(
                f"invalid startup joint position: {startup_joint_position!r}"
            )
        safe_print(
            "[INIT] captured pre-program startup pose: "
            f"{np.round(startup_joint_position, 4)}"
        )
        if config.use_voice:
            voice = VoiceInterface(
                config.project_root,
                model_dir=config.voice_asr_model_dir,
                tts_model_dir=config.voice_tts_model_dir,
                prompt_duration=config.voice_prompt_duration,
                enabled=True,
            )
            if voice.available:
                safe_print("[VOICE] voice interface ready (offline ASR + TTS).")
            else:
                safe_print("[VOICE] voice interface unavailable; using terminal input.")
        planner = GraspPlanner(
            robot,
            config,
            shutdown_requested,
            graspnet_provider=None,
            voice=voice,
        )
        graspnet_provider = None
        if config.use_graspnet:
            safe_print("[GRASPNET] loading GraspNet candidate provider ...")
            graspnet_provider = GraspNetCandidateProvider(config)
            graspnet_provider.load()
            planner.graspnet_provider = graspnet_provider
            safe_print("[GRASPNET] GraspNet candidate provider loaded.")

        planner.home()
        safe_print("[INIT] opening gripper at HOME ...")
        planner.open_gripper()

        task_complete = planner.run_grasp_loop(
            camera_feed,
            intrinsic,
            tcp_camera,
            streamer,
            lambda: choose_target_at_start(voice, streamer),
        )
        if task_complete:
            safe_print("[GRASP] scan, grasp and placement completed; returning to ZERO.")
    except Exception as exc:
        safe_print(f"[MAIN] exception: {exc!r}")
        exit_code = 1
    finally:
        if camera_feed is not None:
            camera_feed.stop()
        if npu_detector is not None:
            npu_detector.close()
        if streamer is not None:
            safe_print("[STREAM] closing web preview ...")
            streamer.stop()
            safe_print("[STREAM] web preview closed.")
        if planner is not None:
            if not planner.safe_shutdown(
                pipeline,
                shutdown_target=startup_joint_position,
                shutdown_label="STARTUP POSE",
            ):
                exit_code = max(exit_code, 2)
        elif pipeline is not None:
            try:
                pipeline.stop()
            except Exception:
                pass
        if voice is not None:
            voice.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
