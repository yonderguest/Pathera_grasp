"""Headless external-camera recording for reproducible MuJoCo demonstrations."""

from __future__ import annotations

from pathlib import Path

import cv2
import mujoco


class ExternalVideoRecorder:
    """Record the complete workcell without changing wrist RGB-D perception."""

    def __init__(
        self,
        model: mujoco.MjModel,
        output_path: Path | str,
        *,
        width: int = 640,
        height: int = 480,
        fps: float = 25.0,
        capture_every_control_steps: int = 2,
    ) -> None:
        if capture_every_control_steps < 1:
            raise ValueError("capture_every_control_steps must be positive")
        self.output_path = Path(output_path)
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.capture_every_control_steps = int(capture_every_control_steps)
        self._renderer = mujoco.Renderer(model, height=self.height, width=self.width)
        self._camera = mujoco.MjvCamera()
        self._camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self._camera.lookat[:] = (0.04, 0.0, 0.08)
        self._camera.distance = 0.86
        self._camera.azimuth = 128.0
        self._camera.elevation = -22.0
        self._writer: cv2.VideoWriter | None = None
        self._control_steps = 0
        self.frame_count = 0

    def capture(self, data: mujoco.MjData) -> None:
        """Append one external RGB frame at the current physical state."""
        self._renderer.update_scene(data, camera=self._camera)
        rgb = self._renderer.render()
        if self._writer is None:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            codec = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(
                str(self.output_path), codec, self.fps, (self.width, self.height)
            )
            if not self._writer.isOpened():
                raise RuntimeError(f"cannot create simulation video: {self.output_path}")
        self._writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        self.frame_count += 1

    def on_control_step(self, data: mujoco.MjData) -> None:
        """Hook for the robot's 20 ms control loop; records at a stable FPS."""
        self._control_steps += 1
        if self._control_steps % self.capture_every_control_steps == 0:
            self.capture(data)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        self._renderer.close()
