#!/usr/bin/env python3
"""Run TASK-003 offline unit tests without pytest or hardware access."""

from __future__ import annotations

import sys
import unittest
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


if __name__ == "__main__":
    python_roots = (
        PROJECT_ROOT / "grasp_demo.py",
        PROJECT_ROOT / "voice_controller.py",
        PROJECT_ROOT / "tools",
        PROJECT_ROOT / "tests",
        SDK_SCRIPTS / "Panthera_lib",
        ROS_SRC,
    )
    python_files = []
    for root in python_roots:
        python_files.extend(root.rglob("*.py") if root.is_dir() else [root])
    for path in python_files:
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
    print(f"[STATIC] compiled {len(python_files)} Python files in memory")

    suite = unittest.defaultTestLoader.discover(str(PROJECT_ROOT / "tests"))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
