"""MuJoCo 的 WSLg 实时交互查看器。

该查看器与抓取脚本共用同一份 ``MjModel`` / ``MjData``，因此窗口展示
的是控制器正在执行的真实仿真状态，而不是录制完成后再播放的视频。
"""

from __future__ import annotations

import time
from collections.abc import Callable
import os

# The conda OpenCV Qt plugin omits bundled fonts.  WSL Ubuntu provides DejaVu;
# point Qt to it before the first HighGUI window is created.
os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype/dejavu")

import cv2
import mujoco
import mujoco.viewer
import numpy as np


class ViewerClosedError(RuntimeError):
    """操作者关闭实时窗口时，用于正常结束仿真流程。"""


class WristCameraView:
    """WSLg 中实时显示虚拟 D405 的 RGB 输出，独立于外部 Viewer 视角。"""

    window_name = "Virtual RealSense D405 RGB"

    def __init__(self, render_rgb: Callable[[], object], *, every_steps: int = 2) -> None:
        self._render_rgb = render_rgb
        self._every_steps = every_steps
        self._steps = 0
        self._closed = False
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 640, 480)

    def on_control_step(self, _data: mujoco.MjData) -> None:
        if self._closed:
            return
        self._steps += 1
        if self._steps % self._every_steps:
            return
        frame = self._render_rgb()
        # RgbdFrame.rgb is intentionally duck-typed here to avoid a backend
        # import cycle; it remains the same image used by recognition.
        rgb = np.asarray(frame.rgb)
        cv2.imshow(self.window_name, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)
        if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
            self._closed = True

    def close(self) -> None:
        if not self._closed:
            cv2.destroyWindow(self.window_name)
            cv2.waitKey(1)
            self._closed = True


class LiveMujocoViewer:
    """把机械臂控制周期同步到 MuJoCo WSLg 窗口，并按真实时间节流。

    鼠标交互由 MuJoCo Viewer 原生提供。空格键在本类中额外实现暂停/继续，
    因为被动查看器不会替应用程序自动暂停控制循环。
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        control_period_s: float,
    ) -> None:
        self._control_period_s = control_period_s
        self._paused = False
        self._viewer = mujoco.viewer.launch_passive(
            model,
            data,
            key_callback=self._on_key,
            show_left_ui=True,
            show_right_ui=True,
        )

    def _on_key(self, keycode: int) -> None:
        # GLFW_KEY_SPACE 的值是 32；避免引入额外的 glfw Python 依赖。
        if keycode == 32:
            self._paused = not self._paused
            print(f"LIVE_VIEWER_PAUSED={self._paused}", flush=True)

    def on_control_step(self, _data: mujoco.MjData) -> None:
        """由 ``MujocoRobot`` 在每个控制周期的物理步进后调用。"""
        while self._paused:
            self._ensure_open()
            self._viewer.sync()
            time.sleep(0.02)

        self._ensure_open()
        self._viewer.sync()
        # 现有控制轨迹按秒定义；节流使它以接近真实的速度可视化。
        time.sleep(self._control_period_s)

    def _ensure_open(self) -> None:
        if not self._viewer.is_running():
            raise ViewerClosedError("MuJoCo live viewer was closed by the operator")

    def close(self) -> None:
        """关闭窗口；该操作在异常和正常完成两条路径上都可安全调用。"""
        # WSLg 用户可能刚好在抓取完成时手动关窗；MuJoCo 的 Handle 在该
        # 情况下已经自行释放，再次 close 不应把成功任务变成非零退出。
        if self._viewer.is_running():
            self._viewer.close()
