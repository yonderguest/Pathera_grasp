"""Closed-loop simulated recognition, localization, IK and grasp approach."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .detection import Detection, detect_coloured_block, detect_with_ultralytics
from .ik import IkResult, PinocchioPositionIK
from .mujoco_backend import HOME, MujocoRobot


@dataclass(frozen=True)
class GraspRun:
    detection: Detection
    pregrasp_ik: IkResult
    grasp_ik: IkResult
    final_tcp_position: np.ndarray
    bilateral_contact: bool
    contact_force_n: tuple[float, float]
    lifted_height_m: float


@dataclass(frozen=True)
class PickPlaceRun:
    colour: str
    grasp: GraspRun
    preplace_ik: IkResult
    place_ik: IkResult
    put1_target_world: np.ndarray
    final_block_world: np.ndarray
    placed_in_put1: bool


@dataclass(frozen=True)
class SortingRun:
    transfers: tuple[PickPlaceRun, ...]

    @property
    def completed_colours(self) -> tuple[str, ...]:
        return tuple(run.colour for run in self.transfers if run.placed_in_put1)

    @property
    def complete(self) -> bool:
        return bool(self.transfers) and all(run.placed_in_put1 for run in self.transfers)


class SimGraspPipeline:
    """A hardware-isolated counterpart of the pathera_grasp scan/approach path."""

    def __init__(
        self,
        robot: MujocoRobot,
        ik: PinocchioPositionIK,
        *,
        detector: str = "colour",
        weights: str | None = None,
        target_colour: str = "green",
    ) -> None:
        self.robot = robot
        self.ik = ik
        if detector not in {"colour", "yolo"}:
            raise ValueError("detector must be 'colour' or 'yolo'")
        self.detector = detector
        self.weights = weights
        self.target_colour = target_colour

    @staticmethod
    def _grasp_pose() -> tuple[np.ndarray, np.ndarray]:
        rotation = np.array(
            [[0.2102, -0.0070, 0.9776], [0.1167, 0.9930, -0.0180], [-0.9707, 0.1178, 0.2096]],
            dtype=float,
        )
        u, _singular, vt = np.linalg.svd(rotation)
        rotation = u @ vt
        return rotation, rotation[:, 0]

    def _select_colour(self, colour: str) -> None:
        self.robot.set_target_colour(colour)
        self.target_colour = colour

    def recognize(self) -> Detection:
        frame = self.robot.render_wrist_rgbd()
        if self.detector == "yolo":
            if self.target_colour != "green":
                raise RuntimeError(
                    "the bundled YOLO weight is trained for green blocks; "
                    "use --detector colour for red/blue targets"
                )
            if not self.weights:
                raise RuntimeError("YOLO detector requires --weights")
            detections = detect_with_ultralytics(frame, self.weights)
        else:
            detections = detect_coloured_block(frame, colour=self.target_colour)
        if not detections:
            raise RuntimeError("no target found in rendered RGB-D frame")
        return detections[0]

    def plan_and_approach(self) -> GraspRun:
        self.robot.open_gripper()
        self.robot.move_j(HOME, duration_s=1.0)
        detection = self.recognize()
        grasp_rotation, approach_axis = self._grasp_pose()
        grasp_position = detection.position_world
        pregrasp_position = grasp_position - 0.080 * approach_axis
        seed = self.robot.current_joint_position()
        pregrasp_ik = self.ik.solve(
            pregrasp_position, seed, target_rotation=grasp_rotation
        )
        if not pregrasp_ik.reached:
            raise RuntimeError(
                f"pre-grasp IK failed: error={pregrasp_ik.position_error_m:.4f} m"
            )
        self.robot.move_j(pregrasp_ik.joints, duration_s=1.2)
        pregrasp_error = float(
            np.linalg.norm(self.robot.tcp_position() - pregrasp_position)
        )
        if pregrasp_error > 0.020:
            raise RuntimeError(
                f"pre-grasp TCP tracking failed: error={pregrasp_error:.4f} m"
            )
        grasp_ik = self.ik.solve(
            grasp_position,
            pregrasp_ik.joints,
            target_rotation=grasp_rotation,
        )
        if not grasp_ik.reached:
            raise RuntimeError(
                f"grasp IK failed: error={grasp_ik.position_error_m:.4f} m"
            )
        self.robot.move_j(grasp_ik.joints, duration_s=0.8)
        grasp_tcp_error = float(np.linalg.norm(self.robot.tcp_position() - grasp_position))
        if grasp_tcp_error > 0.020:
            raise RuntimeError(
                f"grasp TCP tracking failed: error={grasp_tcp_error:.4f} m"
            )
        initial_block_z = float(self.robot.block_position()[2])
        self.robot.close_gripper()
        contacts = self.robot.gripper_block_contacts()
        contact_forces = self.robot.gripper_block_contact_forces()
        bilateral = bool(all(contacts) and min(contact_forces) >= 0.05)
        if not bilateral:
            raise RuntimeError(
                "grasp contact validation failed: "
                f"contacts={contacts}, force_n={contact_forces}"
            )
        self.robot.attach_target_block()
        lift_position = grasp_position - 0.080 * approach_axis
        lift_ik = self.ik.solve(
            lift_position,
            grasp_ik.joints,
            target_rotation=grasp_rotation,
        )
        if not lift_ik.reached:
            raise RuntimeError(
                f"lift IK failed: error={lift_ik.position_error_m:.4f} m"
            )
        self.robot.move_j(lift_ik.joints, duration_s=1.0)
        lifted_height = float(self.robot.block_position()[2] - initial_block_z)
        if lifted_height < 0.030:
            raise RuntimeError(
                f"lift validation failed: block_delta_z={lifted_height:.4f} m"
            )
        return GraspRun(
            detection=detection,
            pregrasp_ik=pregrasp_ik,
            grasp_ik=grasp_ik,
            final_tcp_position=self.robot.tcp_position(),
            bilateral_contact=bilateral,
            contact_force_n=contact_forces,
            lifted_height_m=lifted_height,
        )

    def pick_and_place(self, colour: str) -> PickPlaceRun:
        """Pick one identified block, release it in PUT1, then verify it."""
        self._select_colour(colour)
        grasp = self.plan_and_approach()
        grasp_rotation, approach_axis = self._grasp_pose()
        put1_target = self.robot.put1_position(colour)
        preplace_position = put1_target - 0.080 * approach_axis
        preplace_ik = self.ik.solve(
            preplace_position,
            self.robot.current_joint_position(),
            target_rotation=grasp_rotation,
        )
        if not preplace_ik.reached:
            raise RuntimeError(f"PUT1 pre-place IK failed for {colour}")
        self.robot.move_j(preplace_ik.joints, duration_s=1.2)
        if float(np.linalg.norm(self.robot.tcp_position() - preplace_position)) > 0.020:
            raise RuntimeError(f"PUT1 pre-place TCP tracking failed for {colour}")
        place_ik = self.ik.solve(
            put1_target, preplace_ik.joints, target_rotation=grasp_rotation
        )
        if not place_ik.reached:
            raise RuntimeError(f"PUT1 place IK failed for {colour}")
        self.robot.move_j(place_ik.joints, duration_s=0.8)
        if float(np.linalg.norm(self.robot.tcp_position() - put1_target)) > 0.020:
            raise RuntimeError(f"PUT1 TCP tracking failed for {colour}")
        self.robot.release_target_at_put1(colour)
        self.robot.open_gripper()
        for _ in range(25):
            self.robot.step_control_period()
        placed_in_put1 = self.robot.block_is_in_put1(colour)
        final_block = self.robot.block_position(colour)
        if not placed_in_put1:
            raise RuntimeError(f"PUT1 placement validation failed for {colour}")
        self.robot.lock_block_in_put1(colour)
        self.robot.move_j(preplace_ik.joints, duration_s=0.8)
        self.robot.move_j(HOME, duration_s=1.0)
        return PickPlaceRun(
            colour=colour,
            grasp=grasp,
            preplace_ik=preplace_ik,
            place_ik=place_ik,
            put1_target_world=put1_target,
            final_block_world=final_block,
            placed_in_put1=placed_in_put1,
        )

    def sort_to_put1(
        self, colours: tuple[str, ...] = ("green", "red", "blue")
    ) -> SortingRun:
        """Stop only after every requested colour is still verified in PUT1."""
        if not colours or len(set(colours)) != len(colours):
            raise ValueError("colours must be a non-empty sequence without duplicates")
        result = SortingRun(tuple(self.pick_and_place(colour) for colour in colours))
        if not result.complete or not all(
            self.robot.block_is_in_put1(colour) for colour in colours
        ):
            raise RuntimeError(f"sorting incomplete: completed={result.completed_colours}")
        return result
