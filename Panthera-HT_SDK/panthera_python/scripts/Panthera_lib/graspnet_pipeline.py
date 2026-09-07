"""GraspNet candidate generation and Panthera base-frame conversion.

This module is intentionally optional.  The default grasp workflow uses the
OBB/Seeed geometric planner.  Set ``GraspConfig.use_graspnet = True`` and
provide the required repositories/checkpoint to enable the deep candidate
generator.

It owns only visual/geometric logic.  Motion planning, workspace and IK checks
remain in ``grasp_planner.py``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pyrealsense2 as rs

from .grasp_config import GraspConfig


def graspnet_dependencies_available(config: GraspConfig) -> bool:
    """Return True when both GraspNet repositories look present."""
    project_root = Path(config.project_root)
    repo_path = project_root / config.graspnet_repo_path
    api_path = project_root / config.graspnet_api_path
    return repo_path.is_dir() and api_path.is_dir()


def _downsample(points, colors, count, seed: int = 0):
    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if colors.shape != points.shape:
        raise ValueError("colors must have the same shape as points")
    if points.shape[0] == 0:
        return points, colors

    rng = np.random.default_rng(seed)
    if points.shape[0] >= count:
        indices = rng.choice(points.shape[0], count, replace=False)
    else:
        indices = np.concatenate(
            [
                np.arange(points.shape[0]),
                rng.choice(points.shape[0], count - points.shape[0], replace=True),
            ]
        )
    return points[indices], colors[indices]


def masked_point_cloud(
    color_image,
    depth_image,
    mask,
    intrinsic,
    depth_scale: float,
    config: GraspConfig,
    num_points: int | None = None,
):
    """Build a camera-frame RGB point cloud from a binary object mask."""
    depth = np.asanyarray(depth_image, dtype=np.float32)
    color = np.asanyarray(color_image)
    mask_bool = np.asarray(mask, dtype=bool)
    if mask_bool.shape != depth.shape:
        raise ValueError(
            f"mask shape {mask_bool.shape} does not match depth shape {depth.shape}"
        )

    ys, xs = np.where(mask_bool)
    if xs.size == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)

    depth_values = depth[ys, xs]
    valid_depth = (depth_values > 0.0) & (
        depth_values * depth_scale >= config.min_depth_m
    ) & (depth_values * depth_scale <= config.max_depth_m)
    if not np.any(valid_depth):
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)

    xs = xs[valid_depth]
    ys = ys[valid_depth]
    depth_values = depth_values[valid_depth]

    points = np.empty((xs.size, 3), dtype=np.float32)
    for index, (u, v) in enumerate(zip(xs, ys)):
        point = rs.rs2_deproject_pixel_to_point(
            intrinsic, [float(u), float(v)], float(depth_values[index] * depth_scale)
        )
        points[index] = point

    colors = color[ys, xs].astype(np.float32)
    if colors.shape[1:] == (3,):
        colors = colors[:, ::-1].copy() / 255.0

    if num_points is None:
        num_points = int(config.graspnet_num_point)
    return _downsample(points, colors, num_points)


def _scene_mask_from_detection(depth_image, detection, config: GraspConfig):
    """Build a local scene mask around a detection bbox.

    GraspNet needs context (for example the table) around the object to
    predict stable approach directions.  Feeding only the object mask makes the
    model emit mostly side grasps.
    """
    depth = np.asanyarray(depth_image)
    x1, y1, x2, y2 = [int(value) for value in detection["bbox"]]
    expand = int(config.graspnet_scene_expand_px)
    x1 = max(0, x1 - expand)
    y1 = max(0, y1 - expand)
    x2 = min(depth.shape[1], x2 + expand)
    y2 = min(depth.shape[0], y2 + expand)

    mask = np.zeros(depth.shape[:2], dtype=bool)
    mask[y1:y2, x1:x2] = depth[y1:y2, x1:x2] > 0
    return mask


class GraspNetCandidateProvider:
    """Lazy loader around ``graspnet-baseline`` and ``graspnetAPI``."""

    def __init__(self, config: GraspConfig) -> None:
        self.config = config
        self._net = None
        self._device = None
        self._GraspGroup = None
        self._ModelFreeCollisionDetector = None
        self._pred_decode = None

    def _import_dependencies(self):
        project_root = Path(self.config.project_root)
        repo_path = project_root / self.config.graspnet_repo_path
        api_path = project_root / self.config.graspnet_api_path
        if not repo_path.is_dir():
            raise FileNotFoundError(
                "GraspNet baseline not found. Clone graspnet-baseline into "
                f"{repo_path} first."
            )
        if not api_path.is_dir():
            raise FileNotFoundError(
                "GraspNetAPI not found. Clone graspnetAPI into "
                f"{api_path} first."
            )

        for path in (
            api_path,
            repo_path / "models",
            repo_path / "utils",
        ):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))

        try:
            import torch  # noqa: F401
            from graspnetAPI import GraspGroup
            from graspnet import GraspNet, pred_decode
            from collision_detector import ModelFreeCollisionDetector
        except Exception as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "Unable to import graspnet-baseline or graspnetAPI. "
                "Compile pointnet2/knn and install the API before enabling "
                "use_graspnet."
            ) from exc

        self._GraspGroup = GraspGroup
        self._pred_decode = pred_decode
        self._ModelFreeCollisionDetector = ModelFreeCollisionDetector
        return GraspNet

    def load(self) -> None:
        if self._net is not None:
            return
        GraspNet = self._import_dependencies()
        import torch

        checkpoint_path = Path(self.config.project_root) / self.config.graspnet_checkpoint_path
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"GraspNet checkpoint not found: {checkpoint_path}"
            )

        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net = GraspNet(
            input_feature_dim=0,
            num_view=300,
            num_angle=12,
            num_depth=4,
            cylinder_radius=0.05,
            hmin=-0.02,
            hmax_list=[0.01, 0.02, 0.03, 0.04],
            is_training=False,
        )
        net.to(self._device)
        checkpoint = torch.load(checkpoint_path, map_location=self._device)
        net.load_state_dict(checkpoint["model_state_dict"])
        net.eval()
        self._net = net

    @property
    def loaded(self) -> bool:
        return self._net is not None

    def generate_candidates(
        self,
        color_image,
        depth_image,
        detection,
        intrinsic,
        depth_scale: float,
        base_camera,
        target_base_point=None,
    ) -> list[dict]:
        """Generate camera-frame candidates and convert them to base frame."""
        if not self.loaded:
            self.load()

        import torch

        scene_mask = _scene_mask_from_detection(
            depth_image, detection, self.config
        )
        points, colors = masked_point_cloud(
            color_image,
            depth_image,
            scene_mask,
            intrinsic,
            depth_scale,
            self.config,
        )
        if points.shape[0] == 0:
            return []

        cloud_sampled = torch.from_numpy(points[np.newaxis].astype(np.float32))
        cloud_colors = torch.from_numpy(colors.astype(np.float32))
        cloud_sampled = cloud_sampled.to(self._device)
        cloud_colors = cloud_colors.to(self._device)

        with torch.no_grad():
            end_points = {
                "point_clouds": cloud_sampled,
                "cloud_colors": cloud_colors,
            }
            end_points = self._net(end_points)
            grasp_preds = self._pred_decode(end_points)

        gg_array = grasp_preds[0].detach().cpu().numpy()
        gg = self._GraspGroup(gg_array)

        if self._ModelFreeCollisionDetector is not None and points.shape[0] > 0:
            try:
                collision_detector = self._ModelFreeCollisionDetector(
                    points,
                    voxel_size=self.config.graspnet_collision_voxel_size,
                )
                collision_mask = collision_detector.detect(
                    gg,
                    approach_dist=self.config.graspnet_collision_approach_dist,
                    collision_thresh=self.config.graspnet_collision_thresh,
                )
                gg = gg[~collision_mask]
            except Exception as exc:
                print(f"[GRASPNET] collision filter failed closed: {exc!r}")
                return []

        gg = gg.nms(
            translation_thresh=self.config.graspnet_nms_translation,
            rotation_thresh=np.deg2rad(self.config.graspnet_nms_rotation),
        )
        if gg is None:
            raise RuntimeError("GraspNet NMS returned no candidate group")
        gg.sort_by_score()

        candidates = []
        fix_rotation = np.asarray(
            self.config.graspnet_gripper_fix_rotation, dtype=float
        )
        base_from_camera = np.asarray(base_camera, dtype=float)

        for grasp in gg:
            if len(candidates) >= self.config.graspnet_max_candidates:
                break
            if float(grasp.score) < self.config.graspnet_score_threshold:
                continue

            tool_rotation_camera = np.asarray(grasp.rotation_matrix, dtype=float)
            tool_rotation = base_from_camera[:3, :3] @ tool_rotation_camera @ fix_rotation
            tool_target_camera = np.asarray(grasp.translation, dtype=float)
            tool_target = (base_from_camera @ np.append(tool_target_camera, 1.0))[:3]
            tool_target = tool_target + np.array(
                [0.0, 0.0, self.config.graspnet_z_offset_m],
                dtype=float,
            )

            if target_base_point is not None:
                distance = float(
                    np.linalg.norm(
                        (tool_target - np.asarray(target_base_point))[:2]
                    )
                )
                if distance > self.config.graspnet_target_radius_m:
                    continue

            approach = tool_rotation[:, 0]
            manual_approach = self.config.manual_grasp_rotation[:, 0]
            approach = approach / max(float(np.linalg.norm(approach)), 1e-12)
            manual_approach = manual_approach / max(
                float(np.linalg.norm(manual_approach)), 1e-12
            )
            approach_angle = float(
                np.degrees(
                    np.arccos(
                        np.clip(
                            float(np.dot(approach, manual_approach)),
                            -1.0,
                            1.0,
                        )
                    )
                )
            )
            if approach_angle > self.config.graspnet_approach_max_angle_deg:
                continue

            tcp_offset = tool_rotation @ self.config.tcp_in_joint6
            joint6_target = tool_target - tcp_offset

            candidates.append(
                {
                    "score": float(grasp.score),
                    "gripper_width": float(grasp.width),
                    "tool_target": tool_target,
                    "tool_rotation": tool_rotation,
                    "joint6_target": joint6_target,
                }
            )

        return candidates
