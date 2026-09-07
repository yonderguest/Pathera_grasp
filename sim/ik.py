"""Pinocchio pose IK used by the hardware-isolated MuJoCo grasp demo."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pinocchio as pin


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URDF = PROJECT_ROOT / "sim" / "assets" / "panthera_follower.urdf"


@dataclass(frozen=True)
class IkResult:
    joints: np.ndarray
    reached: bool
    position_error_m: float
    orientation_error_deg: float
    iterations: int


class PinocchioPositionIK:
    """Numerical 6D IK over the same joint limits used by the MuJoCo scene."""

    # Keep this identical to GraspConfig.tcp_in_joint6 in the real planner.
    TOOL_OFFSET_LINK6 = np.array([0.165, 0.0, 0.0], dtype=float)

    def __init__(self, urdf_path: Path | str = DEFAULT_URDF) -> None:
        self.model = pin.buildModelFromUrdf(str(urdf_path))
        self.data = self.model.createData()
        self.frame_id = self.model.getFrameId("link6")
        if self.frame_id >= len(self.model.frames):
            raise RuntimeError("link6 frame is missing from Panthera URDF")
        self.lower = self.model.lowerPositionLimit.copy()
        self.upper = self.model.upperPositionLimit.copy()
        if self.model.nq != 6:
            raise RuntimeError(f"expected six Panthera joints, got {self.model.nq}")

    def forward_tcp(self, joints: np.ndarray) -> np.ndarray:
        return self.forward_pose(joints)[0]

    def forward_pose(self, joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        joints = np.asarray(joints, dtype=float)
        pin.forwardKinematics(self.model, self.data, joints)
        pin.updateFramePlacements(self.model, self.data)
        placement = self.data.oMf[self.frame_id]
        position = placement.translation + placement.rotation @ self.TOOL_OFFSET_LINK6
        return position.copy(), placement.rotation.copy()

    def solve(
        self,
        target_position: np.ndarray,
        seed: np.ndarray,
        *,
        target_rotation: np.ndarray | None = None,
        max_iterations: int = 160,
        tolerance_m: float = 0.008,
        orientation_tolerance_deg: float = 4.0,
    ) -> IkResult:
        target = np.asarray(target_position, dtype=float)
        desired_rotation = None
        if target_rotation is not None:
            desired_rotation = np.asarray(target_rotation, dtype=float)
            if desired_rotation.shape != (3, 3):
                raise ValueError("target_rotation must be 3x3")
        joints = np.clip(np.asarray(seed, dtype=float), self.lower, self.upper)
        epsilon = 1e-5

        def errors(q: np.ndarray) -> tuple[np.ndarray, float, float]:
            current_position, current_rotation = self.forward_pose(q)
            position_error = target - current_position
            if desired_rotation is None:
                rotation_error = np.zeros(3, dtype=float)
            else:
                rotation_error = pin.log3(current_rotation.T @ desired_rotation)
            residual = np.concatenate((position_error, 0.12 * rotation_error))
            return (
                residual,
                float(np.linalg.norm(position_error)),
                float(np.degrees(np.linalg.norm(rotation_error))),
            )

        for iteration in range(1, max_iterations + 1):
            residual, position_error, orientation_error = errors(joints)
            if (
                position_error <= tolerance_m
                and orientation_error <= orientation_tolerance_deg
            ):
                return IkResult(
                    joints, True, position_error, orientation_error, iteration
                )
            # Central finite-difference Jacobian of the full pose residual.
            jacobian = np.empty((6, 6), dtype=float)
            for axis in range(6):
                plus = joints.copy()
                minus = joints.copy()
                plus[axis] = min(self.upper[axis], plus[axis] + epsilon)
                minus[axis] = max(self.lower[axis], minus[axis] - epsilon)
                jacobian[:, axis] = (errors(plus)[0] - errors(minus)[0]) / max(
                    plus[axis] - minus[axis], epsilon
                )
            damping = 5e-4
            delta = np.linalg.solve(
                jacobian.T @ jacobian + damping * np.eye(6),
                -jacobian.T @ residual,
            )
            joints = np.clip(joints + 0.45 * delta, self.lower, self.upper)
        _residual, position_error, orientation_error = errors(joints)
        reached = (
            position_error <= tolerance_m
            and orientation_error <= orientation_tolerance_deg
        )
        return IkResult(
            joints, reached, position_error, orientation_error, max_iterations
        )
