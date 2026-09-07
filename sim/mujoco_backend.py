"""Pure-MuJoCo robot and RGB-D camera backend.

This module deliberately has no imports from Panthera SDK, pyrealsense2, QNN,
ROS, CAN or audio libraries.  It is safe to run on a development PC.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE = PROJECT_ROOT / "sim" / "models" / "panthera_grasp_scene.xml"
JOINT_NAMES = tuple(f"joint{index}" for index in range(1, 7))
ACTUATOR_NAMES = tuple(f"joint{index}_act" for index in range(1, 7))
HOME = np.array([0.0, 0.240, 1.200, -1.515, 0.0, 0.0], dtype=float)
BLOCK_COLOURS = ("red", "green", "blue")
BLOCK_BODY_NAMES = {colour: f"block_{colour}" for colour in BLOCK_COLOURS}
BLOCK_GEOM_NAMES = {colour: f"block_{colour}_geom" for colour in BLOCK_COLOURS}
PUT1_SLOT_NAMES = {
    "green": "put1_green",
    "red": "put1_red",
    "blue": "put1_blue",
}


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole intrinsics matching the virtual 640x480 wrist camera."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class RgbdFrame:
    rgb: np.ndarray
    depth_m: np.ndarray
    intrinsics: CameraIntrinsics
    camera_position_world: np.ndarray
    camera_rotation_world: np.ndarray


class MujocoRobot:
    """Small simulation replacement for the real arm transport.

    The public methods use SI units and six real-arm joint angles.  ``move_j``
    sends position setpoints at the same 20 ms application cadence used by the
    physical grasp planner while MuJoCo advances at a 2 ms physics timestep.
    """

    def __init__(
        self,
        scene_path: Path | str = DEFAULT_SCENE,
        *,
        width: int = 640,
        height: int = 480,
        control_period_s: float = 0.020,
        target_colour: str = "green",
        scene_seed: int | None = None,
        active_colours: tuple[str, ...] | list[str] | None = None,
    ) -> None:
        if target_colour not in BLOCK_COLOURS:
            raise ValueError(f"target_colour must be one of {BLOCK_COLOURS}")
        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)
        self.width = int(width)
        self.height = int(height)
        self.control_period_s = float(control_period_s)
        self.target_colour = target_colour
        self.active_colours = tuple(active_colours or (target_colour,))
        if not self.active_colours or any(
            colour not in BLOCK_COLOURS for colour in self.active_colours
        ):
            raise ValueError(f"active_colours must be chosen from {BLOCK_COLOURS}")
        if len(set(self.active_colours)) != len(self.active_colours):
            raise ValueError("active_colours cannot contain duplicates")
        self.scene_seed = (
            int(scene_seed)
            if scene_seed is not None
            else int(np.random.SeedSequence().generate_state(1)[0])
        )
        self._physics_steps_per_control = max(
            1, round(self.control_period_s / self.model.opt.timestep)
        )
        self._renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)
        self._control_step_callbacks: list[Callable[[mujoco.MjData], None]] = []
        self._wrist_scene_option = mujoco.MjvOption()
        # group=1 保存机械臂外观网格和粗碰撞代理；腕部相机仅保留夹爪，避免
        # 机身遮住操作区，同时不改变 Viewer 中完整 Panthera-HT 的外观。
        self._wrist_scene_option.geomgroup[1] = 0
        self._joint_qpos = np.array(
            [self.model.joint(name).qposadr[0] for name in JOINT_NAMES], dtype=int
        )
        self._actuator_ids = np.array(
            [self.model.actuator(name).id for name in ACTUATOR_NAMES], dtype=int
        )
        self._gripper_qpos = np.array(
            [
                self.model.joint("gripper_left").qposadr[0],
                self.model.joint("gripper_right").qposadr[0],
            ],
            dtype=int,
        )
        self._gripper_actuator_ids = np.array(
            [
                self.model.actuator("gripper_left_act").id,
                self.model.actuator("gripper_right_act").id,
            ],
            dtype=int,
        )
        self._block_free_qpos = {
            colour: self.model.joint(f"block_{colour}_free").qposadr[0]
            for colour in BLOCK_COLOURS
        }
        self._block_dof_adr = {
            colour: int(np.asarray(self.model.joint(f"block_{colour}_free").dofadr).item())
            for colour in BLOCK_COLOURS
        }
        self._held_colour: str | None = None
        self._held_offset_tool = np.zeros(3, dtype=float)
        self._held_quaternion = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        self._put1_locked_qpos: dict[str, np.ndarray] = {}
        self.joint_lower = self.model.jnt_range[
            [self.model.joint(name).id for name in JOINT_NAMES], 0
        ].copy()
        self.joint_upper = self.model.jnt_range[
            [self.model.joint(name).id for name in JOINT_NAMES], 1
        ].copy()
        vertical_fov = np.deg2rad(55.0)
        fy = (self.height / 2.0) / np.tan(vertical_fov / 2.0)
        self.intrinsics = CameraIntrinsics(
            width=self.width,
            height=self.height,
            fx=float(fy),
            fy=float(fy),
            cx=(self.width - 1) / 2.0,
            cy=(self.height - 1) / 2.0,
        )
        try:
            self.reset()
        except Exception:
            self._renderer.close()
            raise

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self._joint_qpos] = HOME
        self.data.qpos[self._gripper_qpos] = 0.040
        self.data.ctrl[self._actuator_ids] = HOME
        self.data.ctrl[self._gripper_actuator_ids] = 0.040
        self._held_colour = None
        self._put1_locked_qpos.clear()
        self._scatter_blocks()
        mujoco.mj_forward(self.model, self.data)
        # Recognition and grasp modes must observe the same settled scene.
        for _ in range(max(1, round(0.5 / self.model.opt.timestep))):
            mujoco.mj_step(self.model, self.data)

    def randomize_scene(self) -> None:
        """Choose and expose a new replayable seed, then reset the workcell."""
        self.scene_seed = int(np.random.SeedSequence().generate_state(1)[0])
        self.reset()

    def _scatter_blocks(self) -> None:
        """Place all blocks on a new or reproducible tabletop arrangement."""
        rng = np.random.default_rng(self.scene_seed)
        positions: list[np.ndarray] = []
        placement_order = self.active_colours + tuple(
            colour for colour in BLOCK_COLOURS if colour not in self.active_colours
        )
        put1_xy = self.put1_position("green")[:2]
        min_separation = 0.080 if len(self.active_colours) >= 3 else 0.090
        for colour in placement_order:
            if len(self.active_colours) == 1 and colour == self.target_colour:
                # Keep the shipped single-green YOLO regression inside the
                # synthetic-training camera/IK distribution.
                x_range, y_range = (0.216, 0.222), (0.020, 0.032)
            else:
                x_range = (0.180, 0.320) if colour in self.active_colours else (0.105, 0.335)
                y_range = (-0.060, 0.220) if colour in self.active_colours else (-0.210, 0.220)
            for _ in range(100):
                candidate = np.array(
                    [rng.uniform(*x_range), rng.uniform(*y_range)], dtype=float
                )
                if (
                    all(np.linalg.norm(candidate - prior) >= min_separation for prior in positions)
                    and np.linalg.norm(candidate - put1_xy) >= 0.120
                ):
                    positions.append(candidate)
                    break
            else:
                raise RuntimeError("could not find separated tabletop positions for blocks")

            qpos_adr = self._block_free_qpos[colour]
            # Keep the long axis close to the calibrated grasp axis; the scene
            # is random in position while remaining a valid 6D grasp exercise.
            yaw = float(rng.uniform(-0.15, 0.15))
            self.data.qpos[qpos_adr : qpos_adr + 3] = (*candidate, 0.055)
            self.data.qpos[qpos_adr + 3 : qpos_adr + 7] = (
                np.cos(yaw / 2.0),
                0.0,
                0.0,
                np.sin(yaw / 2.0),
            )
            dof_adr = self._block_dof_adr[colour]
            self.data.qvel[dof_adr : dof_adr + 6] = 0.0

    def set_target_colour(self, colour: str) -> None:
        if colour not in BLOCK_COLOURS:
            raise ValueError(f"colour must be one of {BLOCK_COLOURS}")
        if self._held_colour is not None and colour != self._held_colour:
            raise RuntimeError("cannot change target while a block is held")
        self.target_colour = colour

    def put1_position(self, colour: str) -> np.ndarray:
        if colour not in PUT1_SLOT_NAMES:
            raise ValueError(f"PUT1 has no slot for colour={colour!r}")
        mujoco.mj_forward(self.model, self.data)
        return self.data.site(PUT1_SLOT_NAMES[colour]).xpos.copy()

    def current_joint_position(self) -> np.ndarray:
        return self.data.qpos[self._joint_qpos].copy()

    def set_joint_target(self, joints: np.ndarray) -> None:
        joints = np.asarray(joints, dtype=float)
        if joints.shape != (6,) or not np.all(np.isfinite(joints)):
            raise ValueError("joint target must contain six finite values")
        if np.any(joints < self.joint_lower) or np.any(joints > self.joint_upper):
            raise ValueError("joint target violates Panthera joint limits")
        self.data.ctrl[self._actuator_ids] = joints

    def step_control_period(self) -> None:
        mujoco.mj_step(self.model, self.data, nstep=self._physics_steps_per_control)
        self._sync_held_block()
        self._sync_put1_blocks()
        for callback in tuple(self._control_step_callbacks):
            callback(self.data)

    def add_control_step_callback(
        self, callback: Callable[[mujoco.MjData], None]
    ) -> None:
        """Register an observer after each 20 ms simulation control period."""
        self._control_step_callbacks.append(callback)

    def move_j(self, joints: np.ndarray, *, duration_s: float = 1.0) -> None:
        """Execute a bounded linear joint reference trajectory in simulation."""
        target = np.asarray(joints, dtype=float)
        self.set_joint_target(target)
        start = self.current_joint_position()
        steps = max(1, int(np.ceil(duration_s / self.control_period_s)))
        for index in range(1, steps + 1):
            ratio = index / steps
            self.set_joint_target(start + ratio * (target - start))
            self.step_control_period()
        # Let the stiff position actuators settle while preserving the target.
        for _ in range(20):
            self.step_control_period()
        tracking_error = float(np.max(np.abs(self.current_joint_position() - target)))
        if tracking_error > 0.040:
            raise RuntimeError(
                f"simulated joint tracking failed: max_error={tracking_error:.4f} rad"
            )

    def set_gripper_opening(self, opening_m: float) -> None:
        """Set symmetric finger travel; aperture is approximately twice this value."""
        travel = float(np.clip(opening_m / 2.0, 0.0, 0.040))
        self.data.ctrl[self._gripper_actuator_ids] = travel

    def open_gripper(self) -> None:
        self.set_gripper_opening(0.080)
        for _ in range(30):
            self.step_control_period()

    def close_gripper(self) -> None:
        self.set_gripper_opening(0.0)
        for _ in range(60):
            self.step_control_period()

    def tcp_position(self) -> np.ndarray:
        mujoco.mj_forward(self.model, self.data)
        return self.data.site("tcp").xpos.copy()

    def tcp_rotation(self) -> np.ndarray:
        mujoco.mj_forward(self.model, self.data)
        return self.data.site("tcp").xmat.reshape(3, 3).copy()

    def block_position(self, colour: str | None = None) -> np.ndarray:
        colour = colour or self.target_colour
        if colour not in BLOCK_COLOURS:
            raise ValueError(f"colour must be one of {BLOCK_COLOURS}")
        return self.data.body(BLOCK_BODY_NAMES[colour]).xpos.copy()

    def attach_target_block(self) -> None:
        if self._held_colour is not None:
            raise RuntimeError(f"already holding {self._held_colour}")
        grasp_position = self.data.site("grasp_site").xpos.copy()
        grasp_rotation = self.data.site("grasp_site").xmat.reshape(3, 3).copy()
        self._held_colour = self.target_colour
        self._held_offset_tool = grasp_rotation.T @ (
            self.block_position(self._held_colour) - grasp_position
        )
        qpos_adr = self._block_free_qpos[self._held_colour]
        self._held_quaternion = self.data.qpos[qpos_adr + 3 : qpos_adr + 7].copy()
        self._sync_held_block()

    def _sync_held_block(self) -> None:
        if self._held_colour is None:
            return
        qpos_adr = self._block_free_qpos[self._held_colour]
        grasp_position = self.data.site("grasp_site").xpos
        grasp_rotation = self.data.site("grasp_site").xmat.reshape(3, 3)
        self.data.qpos[qpos_adr : qpos_adr + 3] = (
            grasp_position + grasp_rotation @ self._held_offset_tool
        )
        self.data.qpos[qpos_adr + 3 : qpos_adr + 7] = self._held_quaternion
        dof_adr = self._block_dof_adr[self._held_colour]
        self.data.qvel[dof_adr : dof_adr + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def release_target_at_put1(self, colour: str) -> np.ndarray:
        if self._held_colour != colour:
            raise RuntimeError(f"cannot release {colour}; holding {self._held_colour}")
        target = self.put1_position(colour)
        qpos_adr = self._block_free_qpos[colour]
        self.data.qpos[qpos_adr : qpos_adr + 3] = target
        self.data.qpos[qpos_adr + 3 : qpos_adr + 7] = (1.0, 0.0, 0.0, 0.0)
        dof_adr = self._block_dof_adr[colour]
        self.data.qvel[dof_adr : dof_adr + 6] = 0.0
        self._held_colour = None
        mujoco.mj_forward(self.model, self.data)
        return target

    def block_is_in_put1(self, colour: str, *, tolerance_m: float = 0.030) -> bool:
        position = self.block_position(colour)
        target = self.put1_position(colour)
        return bool(
            np.linalg.norm(position[:2] - target[:2]) <= tolerance_m
            and 0.005 <= position[2] <= 0.060
        )

    def lock_block_in_put1(self, colour: str) -> None:
        if not self.block_is_in_put1(colour):
            raise RuntimeError(f"cannot lock {colour}: it is not inside PUT1")
        qpos_adr = self._block_free_qpos[colour]
        self._put1_locked_qpos[colour] = self.data.qpos[qpos_adr : qpos_adr + 7].copy()
        self._sync_put1_blocks()

    def _sync_put1_blocks(self) -> None:
        for colour, locked_qpos in self._put1_locked_qpos.items():
            qpos_adr = self._block_free_qpos[colour]
            self.data.qpos[qpos_adr : qpos_adr + 7] = locked_qpos
            dof_adr = self._block_dof_adr[colour]
            self.data.qvel[dof_adr : dof_adr + 6] = 0.0
        if self._put1_locked_qpos:
            mujoco.mj_forward(self.model, self.data)

    def gripper_block_contacts(self) -> tuple[bool, bool]:
        """Return whether the target block touches the left and right fingers."""
        block = self.model.geom(BLOCK_GEOM_NAMES[self.target_colour]).id
        left = self.model.geom("left_finger_collision").id
        right = self.model.geom("right_finger_collision").id
        touching_left = False
        touching_right = False
        for contact in self.data.contact[: self.data.ncon]:
            pair = {int(contact.geom1), int(contact.geom2)}
            touching_left |= pair == {block, left}
            touching_right |= pair == {block, right}
        return touching_left, touching_right

    def gripper_block_contact_forces(self) -> tuple[float, float]:
        """Return summed normal force on each finger from the target block."""
        block = self.model.geom(BLOCK_GEOM_NAMES[self.target_colour]).id
        fingers = (
            self.model.geom("left_finger_collision").id,
            self.model.geom("right_finger_collision").id,
        )
        totals = [0.0, 0.0]
        wrench = np.zeros(6, dtype=float)
        for contact_id, contact in enumerate(self.data.contact[: self.data.ncon]):
            pair = {int(contact.geom1), int(contact.geom2)}
            for index, finger in enumerate(fingers):
                if pair == {block, finger}:
                    mujoco.mj_contactForce(self.model, self.data, contact_id, wrench)
                    totals[index] += abs(float(wrench[0]))
        return float(totals[0]), float(totals[1])

    def render_wrist_rgbd(self) -> RgbdFrame:
        """Render RGB and metric camera-plane depth from the calibrated wrist camera."""
        # group=1 中的 Panthera 网格、碰撞代理和 D405 外壳仅供外部 Viewer
        # 查看。腕部 RGB-D 使用同一个已标定光学位姿，但不渲染自身外壳，避免
        # 近距离相机画面被机械臂/相机模型遮挡。
        self._renderer.update_scene(
            self.data, camera="wrist_rgbd", scene_option=self._wrist_scene_option
        )
        rgb = self._renderer.render().copy()
        self._renderer.enable_depth_rendering()
        self._renderer.update_scene(
            self.data, camera="wrist_rgbd", scene_option=self._wrist_scene_option
        )
        depth_m = self._renderer.render().copy()
        self._renderer.disable_depth_rendering()
        camera = self.data.camera("wrist_rgbd")
        return RgbdFrame(
            rgb=rgb,
            depth_m=depth_m,
            intrinsics=self.intrinsics,
            camera_position_world=camera.xpos.copy(),
            camera_rotation_world=camera.xmat.reshape(3, 3).copy(),
        )

    def close(self) -> None:
        self._renderer.close()
