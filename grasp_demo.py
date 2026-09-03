#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Panthera-HT visual grasp and conservative hand-follow demo.

Workflow:
    select object/colour -> scan J1 -> detect -> pre-grasp -> re-detect
    -> final grasp -> place -> HOME, or HOME -> CPU hand follow -> HOME.
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
from Panthera_lib.grasp_config import (  # noqa: E402
    GraspConfig,
    parse_target_request,
)
from Panthera_lib.grasp_planner import GraspPlanner  # noqa: E402
from Panthera_lib.vision_pipeline import (  # noqa: E402
    CameraFeed,
    init_camera,
    load_color_model,
    load_hand_eye,
    load_target_model,
)
from Panthera_lib.vision_streamer import VisionStreamer  # noqa: E402
from Panthera_lib.Panthera import Panthera  # noqa: E402
from Panthera_lib.graspnet_pipeline import (  # noqa: E402
    GraspNetCandidateProvider,
)
from Panthera_lib.npu_inference import NpuYoloDetector  # noqa: E402
from Panthera_lib.hand_follow import (  # noqa: E402
    CpuYoloHandDetector,
    HandFollowController,
    HandFollowResult,
    HandFollowSettings,
    HandFollowState,
    load_cpu_hand_model,
)


shutdown_requested = threading.Event()


def acquire_process_lock(path="/tmp/pathera_grasp.lock"):
    """Prevent two processes from opening the camera/CAN hardware together."""
    import fcntl

    handle = open(path, "a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.seek(0)
        owner = handle.read().strip() or "unknown"
        handle.close()
        raise RuntimeError(
            f"another pathera_grasp process owns the hardware (pid={owner})"
        ) from exc
    handle.seek(0)
    handle.truncate()
    handle.write(str(os.getpid()))
    handle.flush()
    return handle


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


def run_hand_follow_from_prompt(streamer, controller, camera_feed):
    """Run one web-requested follow session in the robot-owning main thread."""
    if streamer is None or controller is None:
        raise RuntimeError("hand-follow controller is unavailable")

    # The web request may be cancelled while the lazily loaded CPU model is
    # starting.  Check both the global stop latch and the queued off request
    # before issuing HOME or any other robot command.
    if shutdown_requested.is_set() or streamer.is_closed:
        return HandFollowResult(
            HandFollowState.STOPPING,
            "global shutdown requested before hand following",
            0,
        )
    if streamer.poll_follow_command() is False:
        message = "随动启动已取消；机械臂保持当前位置，继续等待抓取目标。"
        streamer.finish_follow_mode(message)
        return HandFollowResult(HandFollowState.IDLE, message, 0)

    # A prior J1 viewing jog is allowed while idle, but follow mode itself must
    # always arm from the calibrated HOME posture with the gripper open.
    controller.planner.home()
    controller.planner.open_gripper()
    streamer.activate_follow_mode("随动模式正在确认唯一且稳定的人手。")
    if shutdown_requested.is_set() or streamer.is_closed:
        return HandFollowResult(
            HandFollowState.STOPPING,
            "global shutdown requested before hand following",
            0,
        )
    active = True

    def mode_enabled():
        nonlocal active
        if shutdown_requested.is_set() or streamer.is_closed:
            active = False
            return False
        transition = streamer.poll_follow_command()
        if transition is False:
            active = False
        return active

    camera_feed.set_object_inference_enabled(False)
    resume_object_inference = False
    try:
        if shutdown_requested.is_set() or streamer.is_closed:
            return HandFollowResult(
                HandFollowState.STOPPING,
                "global shutdown requested before hand inference",
                0,
            )
        result = controller.run(mode_enabled)
        resume_object_inference = (
            not shutdown_requested.is_set() and not streamer.is_closed
        )
    except Exception as exc:
        streamer.update_follow_feedback(
            False,
            0.0,
            f"随动故障，机械臂保持并进入安全停机：{exc}",
        )
        raise
    finally:
        # Resume only after a normal follow exit.  A global stop or fault is
        # already entering camera teardown and must not start another NPU call.
        if resume_object_inference:
            camera_feed.set_object_inference_enabled(True)

    if not shutdown_requested.is_set():
        streamer.finish_follow_mode(
            f"随动模式已停止，机械臂已回到 HOME；共执行 {result.completed_steps} 个小步。"
        )
    return result


def build_lazy_hand_follow_runner(
    streamer,
    planner,
    camera_feed,
    intrinsic,
    tcp_camera,
    config,
):
    """Create a click-triggered hand-follow runner without loading CPU YOLOE.

    The default process remains in NPU object-grasp mode.  The independent CPU
    hand model and controller are constructed only after the main thread has
    consumed a web ``随动模式`` request, and are reused by later sessions.
    """
    cached_controller = None

    def run():
        nonlocal cached_controller
        if shutdown_requested.is_set() or streamer.is_closed:
            return HandFollowResult(
                HandFollowState.STOPPING,
                "global shutdown requested before hand model loading",
                0,
            )

        if cached_controller is None:
            settings = HandFollowSettings(
                max_semantic_step_m=getattr(
                    config, "follow_max_semantic_step_m", 0.10
                ),
                max_tcp_step_m=getattr(config, "follow_max_tcp_step_m", 0.020),
            )
            message = "随动请求已接收，正在按需加载 CPU 手部模型；加载期间机械臂不运动。"
            streamer.set_control_message(message)
            safe_print("[FOLLOW] request received; loading CPU YOLOE hand model ...")
            try:
                model = load_cpu_hand_model(
                    config.model_path,
                    config.text_encoder_path,
                    settings.prompt,
                )
                detector = CpuYoloHandDetector(model, settings)
                status_callback, preview_callback = build_hand_follow_callbacks(
                    streamer,
                    camera_feed,
                )
                cached_controller = HandFollowController(
                    planner,
                    camera_feed,
                    intrinsic,
                    tcp_camera,
                    detector,
                    settings=settings,
                    status_callback=status_callback,
                    preview_callback=preview_callback,
                )
                safe_print("[FOLLOW] CPU YOLOE hand model ready.")
            except Exception as exc:
                message = f"随动模型加载失败，已返回抓取待机：{exc}"
                safe_print(f"[FOLLOW] lazy model load failed: {exc!r}")
                streamer.finish_follow_mode(message)
                return HandFollowResult(HandFollowState.PAUSED, message, 0)

        return run_hand_follow_from_prompt(
            streamer,
            cached_controller,
            camera_feed,
        )

    return run


def build_hand_follow_callbacks(streamer, camera_feed):
    """Bridge controller telemetry to the web page without commanding motion."""

    def publish_status(payload):
        visible = bool(payload.get("hand_seen", False))
        confidence = 0.0
        if visible:
            # The preview callback owns detector confidence.  Preserve its most
            # recent value when a later controller status message has no score.
            confidence = float(
                streamer.control_status().get("follow_hand_confidence", 0.0)
            )
        streamer.update_follow_feedback(
            visible,
            confidence,
            str(payload.get("message", "随动模式状态已更新。")),
        )

    def publish_preview(snapshot, candidates):
        confidence = max(
            (float(candidate.confidence) for candidate in candidates),
            default=0.0,
        )
        streamer.update_follow_feedback(bool(candidates), confidence)
        streamer.publish(
            snapshot["color_image"],
            [candidate.as_stream_detection() for candidate in candidates],
            snapshot["depth_image"],
            camera_feed.depth_scale,
        )

    return publish_status, publish_preview


def read_terminal_command(
    streamer=None,
    joint1_jog_callback=None,
    follow_callback=None,
) -> str | None:
    """Read one terminal or browser command while remaining interruptible."""
    safe_print("> ", end="")
    terminal_available = bool(sys.stdin.isatty())
    while not shutdown_requested.is_set():
        if streamer is not None:
            poll_follow = getattr(streamer, "poll_follow_command", None)
            follow_transition = poll_follow() if callable(poll_follow) else None
            if follow_transition is True:
                if follow_callback is None:
                    raise RuntimeError("hand-follow controller is unavailable")
                result = follow_callback()
                safe_print(
                    "\n[FOLLOW] session ended: "
                    f"state={result.state.value}, steps={result.completed_steps}, "
                    f"reason={result.reason}"
                )
                safe_print("> ", end="")
                continue
            if follow_transition is False:
                # A disable request is normally consumed by the active follow
                # loop. Recover the web state if it arrives after that loop.
                finish_follow = getattr(streamer, "finish_follow_mode", None)
                if callable(finish_follow):
                    finish_follow("随动模式已经停止，机械臂位于 HOME。")
                safe_print("> ", end="")
                continue
            poll_jog = getattr(streamer, "poll_joint1_jog", None)
            jog_direction = poll_jog() if callable(poll_jog) else None
            if jog_direction is not None:
                try:
                    if joint1_jog_callback is None:
                        raise RuntimeError("J1 jog controller is unavailable")
                    final_j1 = float(joint1_jog_callback(jog_direction))
                    label = "左" if jog_direction == "left" else "右"
                    message = f"一号关节已向{label}转动；当前 J1={final_j1:+.2f} rad。"
                    safe_print(f"\n[WEB-JOG] {message}")
                    finish_jog = getattr(streamer, "finish_joint1_jog", None)
                    if callable(finish_jog):
                        finish_jog(message)
                except Exception as exc:
                    message = f"一号关节转动被拒绝：{exc}"
                    safe_print(f"\n[WEB-JOG] {message}")
                    finish_jog = getattr(streamer, "finish_joint1_jog", None)
                    if callable(finish_jog):
                        finish_jog(message)
                safe_print("> ", end="")
                continue
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


def choose_target_at_start(
    voice=None,
    streamer=None,
    joint1_jog_callback=None,
    follow_callback=None,
):
    def finish(value):
        if streamer is not None:
            streamer.set_accepting_targets(False)
        return value

    safe_print(
        "\n您想要抓取什么？（例如：红色积木；瓶子/盒子当前仅识别展示；q 退出）"
    )
    if streamer is not None:
        streamer.set_accepting_targets(True)
        streamer.set_control_message(
            "请输入目标；积木可抓取，瓶子和盒子当前仅识别展示。"
        )
    if voice is not None and voice.available:
        voice.say("请说出要抓取的颜色，例如红色积木")
        command = voice.listen_for_command()
        if command:
            command = command.strip()
            if command.lower() == "q":
                return finish(False)
            request, accepted = parse_target_request(command)
            if accepted:
                selected = request.label
                safe_print(
                    f"[TARGET] 语音识别：已选择 {selected}；检测到可抓取目标后自动执行一次抓取。"
                )
                voice.say(f"已选择{selected}")
                return finish(request)
            safe_print(
                f"[VOICE] 未能从语音识别颜色：{command!r}，请继续用终端输入。"
            )
    while not shutdown_requested.is_set():
        try:
            command = read_terminal_command(
                streamer,
                joint1_jog_callback,
                follow_callback,
            )
        except (EOFError, KeyboardInterrupt):
            return finish(False)
        if command is None:
            return finish(False)
        command = command.strip()
        if command.lower() == "q":
            if streamer is not None:
                streamer.set_control_message("收到安全退出命令。")
            return finish(False)
        request, accepted = parse_target_request(command)
        if accepted:
            selected = request.label
            safe_print(
                f"[TARGET] 已选择 {selected}；检测到可抓取目标后自动执行一次抓取。"
            )
            if voice is not None:
                voice.say(f"已选择{selected}")
            if streamer is not None:
                streamer.set_control_message(f"目标有效：{selected}。")
            return finish(request)
        if streamer is not None:
            reject_target = getattr(streamer, "reject_target_command", None)
            message = f"无法识别“{command}”；请一次只输入一个对象和一种颜色。"
            if callable(reject_target):
                reject_target(message)
            else:
                streamer.set_control_message(message)
        safe_print("未识别目标，请输入对象和颜色，例如红色积木，或 q 退出。")
    return finish(False)


def build_config():
    config = GraspConfig()
    config.sdk_scripts = str(SDK_SCRIPTS)
    config.robot_config = str(ROBOT_CONFIG)
    config.model_path = str(MODEL_PATH)
    config.project_root = PROJECT_ROOT
    config.text_encoder_path = TEXT_ENCODER_PATH
    config.calibration_file = CALIBRATION_FILE
    config.color_calibration_file = PROJECT_ROOT / "config" / "color_calibration.json"
    config.apply_recognition_profile(
        os.environ.get("VISION_MODEL_PROFILE", "object3")
    )
    if "NPU_CONFIDENCE_THRESHOLD" in os.environ:
        config.npu_confidence_threshold = float(
            os.environ["NPU_CONFIDENCE_THRESHOLD"]
        )
    config.color_classifier_backend = os.environ.get(
        "COLOR_CLASSIFIER", "lab"
    ).strip().lower()
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
    config.stream_preview_fps = float(os.environ.get("VISION_STREAM_PREVIEW_FPS", "15"))
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
    load_color_model(config)
    config.validate()
    return config


def main():
    exit_code = 0
    process_lock = None
    robot = None
    planner = None
    pipeline = None
    streamer = None
    camera_feed = None
    npu_detector = None
    hand_follow_runner = None
    voice = None
    startup_joint_position = None
    config = build_config()
    try:
        process_lock = acquire_process_lock()
        safe_print("=" * 68)
        safe_print("Panthera-HT coloured building-block visual grasp")
        safe_print("=" * 68)

        streamer = VisionStreamer(
            config.stream_host,
            config.stream_port,
            config.stream_jpeg_quality,
            stop_callback=on_web_stop_requested,
            preview_fps=config.stream_preview_fps,
        )
        if not streamer.start():
            streamer = None
        if streamer is not None:
            safe_print(f"[STREAM] 安全控制页面已启动：{streamer.control_url}")

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

        safe_print(
            "[FOLLOW] disabled by default; the CPU hand model will load only "
            "after a web follow request."
        )

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
        if streamer is not None:
            # Factory creation is motion-free and model-free.  CPU YOLOE is
            # loaded only when the page's follow button is actually consumed.
            hand_follow_runner = build_lazy_hand_follow_runner(
                streamer,
                planner,
                camera_feed,
                intrinsic,
                tcp_camera,
                config,
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

        planner.run_grasp_loop(
            camera_feed,
            intrinsic,
            tcp_camera,
            streamer,
            lambda: choose_target_at_start(
                voice,
                streamer,
                planner.jog_joint1,
                hand_follow_runner,
            ),
        )
    except Exception as exc:
        safe_print(f"[MAIN] exception: {exc!r}")
        exit_code = 1
    finally:
        pipeline_for_shutdown = pipeline
        if camera_feed is not None:
            camera_feed.stop()
            pipeline_for_shutdown = None
        if npu_detector is not None:
            npu_detector.close()
        if streamer is not None:
            safe_print("[STREAM] closing web preview ...")
            streamer.stop()
            safe_print("[STREAM] web preview closed.")
        if planner is not None:
            if not planner.safe_shutdown(
                pipeline_for_shutdown,
                shutdown_target=startup_joint_position,
                shutdown_label="STARTUP POSE",
            ):
                exit_code = max(exit_code, 2)
        elif pipeline_for_shutdown is not None:
            try:
                pipeline_for_shutdown.stop()
            except Exception:
                pass
        if voice is not None:
            voice.close()
        if process_lock is not None:
            process_lock.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
