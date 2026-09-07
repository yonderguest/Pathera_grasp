#!/bin/bash
# ROS Humble on this device is Python 3.10, while /usr/bin/python3 is 3.12.
# Force generated Python entry points to the dependency-complete 3.10 conda env.
set -eo pipefail

WORKSPACE="$(cd "$(dirname "$0")" && pwd)"
EXPECTED_ENV="/home/ubuntu/miniconda3/envs/pathera_grasp"
ACTIVE_ENV="${CONDA_PREFIX:-$EXPECTED_ENV}"
PY_BIN="${PANTHERA_ROS_PYTHON:-$ACTIVE_ENV/bin/python}"
ROS_SETUP="${PANTHERA_ROS_SETUP:-/opt/ros/humble/setup.bash}"

if [ ! -x "$PY_BIN" ]; then
  echo "ROS Python interpreter does not exist: $PY_BIN" >&2
  exit 1
fi
if [ ! -f "$ROS_SETUP" ]; then
  echo "ROS setup file does not exist: $ROS_SETUP" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$ROS_SETUP"
"$PY_BIN" - <<'PY'
import sys

if sys.version_info[:2] != (3, 10):
    raise SystemExit(
        f"ROS Humble requires Python 3.10 on this device; got {sys.version}"
    )

for module in ("rclpy", "numpy", "cv2", "pyrealsense2", "torch", "ultralytics"):
    __import__(module)
print(f"validated ROS interpreter: {sys.executable} ({sys.version.split()[0]})")
PY

patched=0
for script in "$WORKSPACE"/install/*/lib/*/*; do
  if [ -x "$script" ] && [ -f "$script" ] && [ "$(head -c 2 "$script")" = "#!" ]; then
    sed -i "1s|^#!.*|#!${PY_BIN}|" "$script"
    patched=$((patched + 1))
  fi
done

if [ "$patched" -eq 0 ]; then
  echo "no generated Python entry scripts found under $WORKSPACE/install" >&2
  exit 1
fi

echo "patched $patched ROS entry scripts to use $PY_BIN"
