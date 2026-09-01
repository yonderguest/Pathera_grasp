"""Offline test path setup; never imports or starts a hardware entry point."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SDK_ROOT = PROJECT_ROOT / "Panthera-HT_SDK" / "panthera_python"
SDK_SCRIPTS = SDK_ROOT / "scripts"
ROS_SRC = PROJECT_ROOT / "ros2_ws" / "src"

for path in (
    PROJECT_ROOT,
    SDK_ROOT,
    SDK_SCRIPTS,
    ROS_SRC / "panthera_grasp_brain",
    ROS_SRC / "panthera_vision",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
