"""Run the hardware-isolated pathera_grasp MuJoCo demo.

Usage from the project root:
    MUJOCO_GL=egl python -m sim.scripts.run_sim_demo --mode recognition
    MUJOCO_GL=egl python -m sim.scripts.run_sim_demo --mode sort
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from sim.detection import detect_coloured_block, detect_with_ultralytics
from sim.live_viewer import LiveMujocoViewer, ViewerClosedError, WristCameraView
from sim.mujoco_backend import MujocoRobot
from sim.web_dashboard import SimWebDashboard


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
        help="Optional fixed seed. Omit it to scatter a new tabletop scene each run.",
    )
    parser.add_argument(
        "--sort-colours",
        choices=("red", "green", "blue"),
        nargs="+",
        default=("green", "red", "blue"),
        help="Ordered task colours for --mode sort; the task exits only after all succeed.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("sim/models/block_detector.pt"),
        help="Ultralytics weights used when --detector yolo is selected.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("reports") / "mujoco_sim",
        help="Directory for reproducible rendered RGB-D arrays.",
    )
    parser.add_argument(
        "--record-video",
        type=Path,
        default=None,
        help="Optional MP4 path for an external-camera recording of the full simulation.",
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Open the interactive WSLg MuJoCo window and run controls in real time.",
    )
    parser.add_argument(
        "--camera-view",
        action="store_true",
        help="With --viewer, open the live virtual RealSense RGB view.",
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Serve the local browser dashboard; the browser starts simulation tasks.",
    )
    parser.add_argument("--web-host", default="127.0.0.1")
    parser.add_argument("--web-port", type=int, default=8765)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.viewer and os.environ.get("MUJOCO_GL", "").lower() == "egl":
        raise ValueError(
            "--viewer needs a windowed OpenGL backend. Run `unset MUJOCO_GL` "
            "(or `export MUJOCO_GL=glfw`) instead of MUJOCO_GL=egl."
        )
    if args.camera_view and not args.viewer:
        raise ValueError("--camera-view requires --viewer")
    if args.web and args.mode != "sort":
        raise ValueError("--web currently supports --mode sort only")
    if args.web and args.viewer:
        raise ValueError("--web and --viewer are separate interactive frontends")
    if args.mode == "sort" and args.detector == "yolo":
        raise ValueError(
            "--mode sort requires --detector colour because the bundled "
            "YOLO weights do not distinguish block colours."
        )
    args.save_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "sort" and len(set(args.sort_colours)) != len(args.sort_colours):
        raise ValueError("--sort-colours cannot contain duplicates")
    initial_colour = args.sort_colours[0] if args.mode == "sort" else args.target_color
    active_colours = args.sort_colours if args.mode == "sort" else (args.target_color,)
    robot = MujocoRobot(
        target_colour=initial_colour,
        active_colours=active_colours,
        scene_seed=args.scene_seed,
    )
    recorder = None
    live_viewer = None
    camera_view = None
    dashboard = None
    try:
        print(f"SCENE_SEED={robot.scene_seed}", flush=True)
        if args.viewer:
            live_viewer = LiveMujocoViewer(
                robot.model,
                robot.data,
                control_period_s=robot.control_period_s,
            )
            robot.add_control_step_callback(live_viewer.on_control_step)
            print("LIVE_VIEWER=started controls=mouse,space-pause", flush=True)
        if args.camera_view:
            camera_view = WristCameraView(robot.render_wrist_rgbd)
            robot.add_control_step_callback(camera_view.on_control_step)
            print("VIRTUAL_REALSENSE_VIEW=started", flush=True)
        if args.web:
            dashboard = SimWebDashboard(
                robot, host=args.web_host, port=args.web_port
            )
            dashboard.start()
            robot.add_control_step_callback(dashboard.on_control_step)
            print(f"SIM_WEB_URL={dashboard.url}", flush=True)
        if args.record_video is not None:
            from sim.recording import ExternalVideoRecorder

            recorder = ExternalVideoRecorder(robot.model, args.record_video)
            robot.add_control_step_callback(recorder.on_control_step)
            recorder.capture(robot.data)
        if args.mode == "recognition":
            frame = robot.render_wrist_rgbd()
            detections = (
                detect_with_ultralytics(frame, str(args.weights))
                if args.detector == "yolo"
                else detect_coloured_block(frame, colour=args.target_color)
            )
            if not detections:
                raise RuntimeError("no target found in rendered RGB-D frame")
            detection = detections[0]
            np.save(args.save_dir / "wrist_rgb.npy", frame.rgb)
            np.save(args.save_dir / "wrist_depth_m.npy", frame.depth_m)
            print(f"DETECTION_COUNT={len(detections)} source={detection.source}")
            print(
                "TARGET_WORLD="
                + ",".join(f"{value:.4f}" for value in detection.position_world)
            )
            return 0

        # Keep Pinocchio optional for recognition-only simulation while making
        # transfer modes explicitly depend on the conda-forge package.
        from sim.grasp_pipeline import SimGraspPipeline
        from sim.ik import PinocchioPositionIK

        pipeline = SimGraspPipeline(
            robot,
            PinocchioPositionIK(),
            detector=args.detector,
            weights=str(args.weights),
            target_colour=args.target_color,
        )
        if args.web:
            def run_web_task() -> dict[str, object]:
                if args.mode == "sort":
                    result = pipeline.sort_to_put1(tuple(args.sort_colours))
                    print(
                        "TASK_COMPLETE=" + ",".join(result.completed_colours),
                        flush=True,
                    )
                    return {"completed_colours": list(result.completed_colours)}
                run = pipeline.plan_and_approach()
                return {
                    "completed_colours": [args.target_color]
                    if run.bilateral_contact
                    else []
                }

            try:
                dashboard.run(run_web_task)
            except KeyboardInterrupt:
                print("SIM_WEB=stopped_by_terminal", flush=True)
            return 0
        if args.mode == "sort":
            result = pipeline.sort_to_put1(tuple(args.sort_colours))
            for transfer in result.transfers:
                print(
                    f"PUT1_TRANSFER={transfer.colour} "
                    f"grasped={transfer.grasp.bilateral_contact} "
                    f"placed={transfer.placed_in_put1} "
                    "final_world="
                    + ",".join(f"{value:.4f}" for value in transfer.final_block_world),
                    flush=True,
                )
            print("TASK_COMPLETE=" + ",".join(result.completed_colours), flush=True)
            return 0
        run = pipeline.plan_and_approach()
        print(f"DETECTION_COUNT=1 source={run.detection.source}")
        print(
            f"PREGRASP_IK=reached:{run.pregrasp_ik.reached} "
            f"error_m:{run.pregrasp_ik.position_error_m:.4f}"
        )
        print(
            f"GRASP_IK=reached:{run.grasp_ik.reached} "
            f"error_m:{run.grasp_ik.position_error_m:.4f} "
            f"orientation_error_deg:{run.grasp_ik.orientation_error_deg:.2f}"
        )
        print(
            f"BILATERAL_CONTACT={run.bilateral_contact} "
            f"force_n={run.contact_force_n[0]:.3f},{run.contact_force_n[1]:.3f}"
        )
        print(f"LIFTED_HEIGHT_M={run.lifted_height_m:.4f}")
        print(
            "FINAL_TCP_WORLD="
            + ",".join(f"{value:.4f}" for value in run.final_tcp_position)
        )
        return 0
    except ViewerClosedError:
        # 关闭窗口是操作者主动取消，不应输出 Python 回溯误导为仿真故障。
        print("LIVE_VIEWER=closed_by_operator")
        return 130
    finally:
        if recorder is not None:
            recorder.close()
            print(f"SIMULATION_VIDEO={args.record_video} frames={recorder.frame_count}")
        if live_viewer is not None:
            live_viewer.close()
        if camera_view is not None:
            camera_view.close()
        if dashboard is not None:
            dashboard.close()
        robot.close()


if __name__ == "__main__":
    raise SystemExit(main())
