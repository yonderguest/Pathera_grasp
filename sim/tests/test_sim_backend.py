"""Standard-library offline checks; no physical hardware imports or access."""

from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from sim.mujoco_backend import HOME, MujocoRobot


SIM_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SIM_ROOT.parent
SDK_SCRIPTS = PROJECT_ROOT / "Panthera-HT_SDK" / "panthera_python" / "scripts"
if str(SDK_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SDK_SCRIPTS))

from Panthera_lib.grasp_config import GraspConfig


YOLO_WEIGHTS = SIM_ROOT / "models" / "block_detector.pt"


def _xyz(element: ET.Element, attribute: str) -> np.ndarray:
    return np.fromstring(element.attrib[attribute], sep=" ", dtype=float)


def _rotation_from_wxyz(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternion
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


class MujocoBackendTest(unittest.TestCase):
    def test_mujoco_entrypoint_import_is_hardware_isolated(self) -> None:
        """Importing the PC entry point must not load any real-hardware stack."""

        module = importlib.import_module("run_mujoco")
        self.assertTrue(callable(module.main))
        forbidden = {"hightorque_robot", "pyrealsense2", "rclpy"}
        self.assertFalse(forbidden.intersection(sys.modules))

    def test_sim_geometry_matches_current_project_configuration(self) -> None:
        """Fail when a later real-arm calibration leaves the simulator stale."""

        config = GraspConfig()
        np.testing.assert_allclose(HOME, config.home, atol=1e-12)

        scene = ET.parse(SIM_ROOT / "models" / "panthera_grasp_scene.xml")
        tool = scene.find(".//body[@name='tool_link']")
        camera = scene.find(".//body[@name='d405_visual_mount']")
        self.assertIsNotNone(tool)
        self.assertIsNotNone(camera)
        np.testing.assert_allclose(_xyz(tool, "pos"), config.tcp_in_joint6, atol=1e-12)

        calibration = json.loads(
            (PROJECT_ROOT / "hand_eye_calibration.json").read_text(encoding="utf-8")
        )
        tcp_to_camera = np.asarray(calibration["T_tcp_camera"], dtype=float)
        np.testing.assert_allclose(_xyz(camera, "pos"), tcp_to_camera[:3, 3], atol=1e-8)
        # 标定矩阵使用 OpenCV 光学坐标（x 右、y 下、z 前）；MuJoCo 相机
        # 使用 x 右、y 上、-z 前，因此需要在相机局部坐标中翻转 y/z。
        optical_to_mujoco = np.diag([1.0, -1.0, -1.0])
        np.testing.assert_allclose(
            _rotation_from_wxyz(_xyz(camera, "quat")),
            tcp_to_camera[:3, :3] @ optical_to_mujoco,
            atol=1e-8,
        )

    def test_scene_and_rgbd(self) -> None:
        from sim.detection import detect_green_block

        robot = MujocoRobot(width=320, height=240)
        try:
            self.assertEqual(robot.current_joint_position().shape, (6,))
            frame = robot.render_wrist_rgbd()
            self.assertEqual(frame.rgb.shape, (240, 320, 3))
            self.assertEqual(frame.depth_m.shape, (240, 320))
            self.assertTrue(np.isfinite(frame.depth_m).any())
            detections = detect_green_block(frame)
            self.assertGreaterEqual(len(detections), 1)
            self.assertLess(
                np.linalg.norm(detections[0].position_world - robot.block_position()),
                0.012,
            )
        finally:
            robot.close()

    def test_three_colours_are_independently_localized(self) -> None:
        from sim.detection import detect_coloured_block

        for colour in ("red", "green", "blue"):
            robot = MujocoRobot(
                width=320,
                height=240,
                target_colour=colour,
                scene_seed=20260903,
            )
            try:
                detections = detect_coloured_block(
                    robot.render_wrist_rgbd(), colour=colour
                )
                self.assertGreaterEqual(len(detections), 1, colour)
                self.assertLess(
                    np.linalg.norm(detections[0].position_world - robot.block_position()),
                    0.025,
                    colour,
                )
            finally:
                robot.close()

    @unittest.skipUnless(
        importlib.util.find_spec("ultralytics") and YOLO_WEIGHTS.exists(),
        "Ultralytics weights missing",
    )
    def test_ultralytics_detects_rendered_block(self) -> None:
        from sim.detection import detect_with_ultralytics

        robot = MujocoRobot(width=640, height=480, scene_seed=20260903)
        try:
            detections = detect_with_ultralytics(
                robot.render_wrist_rgbd(), str(YOLO_WEIGHTS)
            )
            self.assertGreaterEqual(len(detections), 1)
            self.assertEqual(detections[0].source, "ultralytics_yolo")
        finally:
            robot.close()

    @unittest.skipUnless(importlib.util.find_spec("pinocchio"), "pinocchio missing")
    def test_recognize_grasp_contact_and_lift(self) -> None:
        from sim.grasp_pipeline import SimGraspPipeline
        from sim.ik import PinocchioPositionIK

        robot = MujocoRobot(width=320, height=240)
        try:
            run = SimGraspPipeline(robot, PinocchioPositionIK()).plan_and_approach()
            self.assertTrue(run.pregrasp_ik.reached)
            self.assertTrue(run.grasp_ik.reached)
            self.assertTrue(run.bilateral_contact)
            self.assertGreater(min(run.contact_force_n), 0.05)
            self.assertGreater(run.lifted_height_m, 0.030)
        finally:
            robot.close()

    @unittest.skipUnless(importlib.util.find_spec("pinocchio"), "pinocchio missing")
    def test_full_three_colour_sort_reaches_put1_before_exit(self) -> None:
        from sim.grasp_pipeline import SimGraspPipeline
        from sim.ik import PinocchioPositionIK

        robot = MujocoRobot(
            width=320,
            height=240,
            target_colour="green",
            active_colours=("green", "red", "blue"),
            scene_seed=20260904,
        )
        try:
            result = SimGraspPipeline(robot, PinocchioPositionIK()).sort_to_put1()
            self.assertTrue(result.complete)
            self.assertEqual(result.completed_colours, ("green", "red", "blue"))
            self.assertTrue(robot.block_is_in_put1("green"))
            self.assertTrue(robot.block_is_in_put1("red"))
            self.assertTrue(robot.block_is_in_put1("blue"))
        finally:
            robot.close()

    def test_scene_seed_repositions_active_red_and_green_blocks(self) -> None:
        first = MujocoRobot(
            width=160,
            height=120,
            target_colour="green",
            active_colours=("green", "red"),
            scene_seed=1001,
        )
        second = MujocoRobot(
            width=160,
            height=120,
            target_colour="green",
            active_colours=("green", "red"),
            scene_seed=1002,
        )
        try:
            self.assertFalse(
                np.allclose(first.block_position("green"), second.block_position("green"))
            )
            self.assertFalse(
                np.allclose(first.block_position("red"), second.block_position("red"))
            )
        finally:
            first.close()
            second.close()

    @unittest.skipUnless(importlib.util.find_spec("pinocchio"), "pinocchio missing")
    def test_pinocchio_matches_home_frame(self) -> None:
        from sim.ik import PinocchioPositionIK

        robot = MujocoRobot(width=160, height=120)
        try:
            ik = PinocchioPositionIK()
            rng = np.random.default_rng(5005)
            samples = [HOME]
            for _ in range(10):
                samples.append(rng.uniform(robot.joint_lower, robot.joint_upper))
            for joints in samples:
                robot.data.qpos[robot._joint_qpos] = joints
                mujoco.mj_forward(robot.model, robot.data)
                self.assertLess(
                    np.linalg.norm(ik.forward_tcp(joints) - robot.tcp_position()),
                    1e-6,
                )
        finally:
            robot.close()


if __name__ == "__main__":
    unittest.main()
