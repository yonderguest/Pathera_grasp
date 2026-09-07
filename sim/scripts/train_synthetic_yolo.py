"""Train the small offline block detector from MuJoCo-rendered RGB frames."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil

import cv2
import mujoco
import numpy as np
from ultralytics import YOLO

from sim.mujoco_backend import MujocoRobot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=80)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("sim/models/block_detector.pt"))
    parser.add_argument("--work-dir", type=Path, default=Path("reports/mujoco_sim/yolo_train"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = args.output.resolve()
    dataset = args.work_dir / "dataset"
    allowed_root = (Path("reports") / "mujoco_sim").resolve()
    if allowed_root not in dataset.resolve().parents:
        raise ValueError("--work-dir must remain under reports/mujoco_sim")
    if dataset.exists():
        shutil.rmtree(dataset)
    for split in ("train", "val"):
        (dataset / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset / "labels" / split).mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(5005)
    robot = MujocoRobot(width=320, height=320)
    free_adr = robot.model.joint("block_green_free").qposadr[0]
    written = {"train": 0, "val": 0}
    try:
        for index in range(args.samples):
            split = "val" if index % 5 == 0 else "train"
            robot.data.qpos[free_adr : free_adr + 3] = [
                rng.uniform(0.175, 0.215),
                rng.uniform(-0.035, 0.035),
                0.025,
            ]
            robot.data.qpos[free_adr + 3 : free_adr + 7] = [1.0, 0.0, 0.0, 0.0]
            robot.data.qvel[:] = 0.0
            mujoco.mj_forward(robot.model, robot.data)
            frame = robot.render_wrist_rgbd()
            rgb = frame.rgb
            mask = (
                (rgb[..., 0] < 55)
                & (rgb[..., 1] > 80)
                & (rgb[..., 2] < 70)
            ).astype(np.uint8)
            component_count, labels, stats, _centres = cv2.connectedComponentsWithStats(
                mask
            )
            if component_count <= 1:
                continue
            target_component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            rows, cols = np.nonzero(labels == target_component)
            if rows.size < 20:
                continue
            x1, x2 = int(cols.min()), int(cols.max())
            y1, y2 = int(rows.min()), int(rows.max())
            height, width = rgb.shape[:2]
            xc = (x1 + x2 + 1) / (2.0 * width)
            yc = (y1 + y2 + 1) / (2.0 * height)
            bw = (x2 - x1 + 1) / width
            bh = (y2 - y1 + 1) / height
            stem = f"block_{index:04d}"
            cv2.imwrite(
                str(dataset / "images" / split / f"{stem}.png"),
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            )
            (dataset / "labels" / split / f"{stem}.txt").write_text(
                f"0 {xc:.8f} {yc:.8f} {bw:.8f} {bh:.8f}\n",
                encoding="utf-8",
            )
            written[split] += 1
    finally:
        robot.close()

    yaml_path = dataset / "dataset.yaml"
    yaml_path.write_text(
        f"path: {dataset.resolve()}\ntrain: images/train\nval: images/val\n"
        "names:\n  0: toy_building_block\n",
        encoding="utf-8",
    )
    if min(written.values()) == 0:
        raise RuntimeError(f"dataset generation failed: {written}")
    # Fine-tuning official pretrained CPU weights converges reliably on the
    # small synthetic dataset; a random-from-YAML model does not.
    original_cwd = Path.cwd()
    args.work_dir.resolve().mkdir(parents=True, exist_ok=True)
    try:
        os.chdir(args.work_dir.resolve())
        model = YOLO("yolo11n.pt")
    finally:
        os.chdir(original_cwd)
    result = model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=320,
        batch=8,
        device="cpu",
        workers=0,
        project=str(args.work_dir.resolve()),
        name="run",
        exist_ok=True,
        seed=5005,
        verbose=False,
    )
    best = Path(result.save_dir) / "weights" / "best.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, output_path)
    print(f"DATASET_COUNTS={written}")
    print(f"YOLO_WEIGHTS={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
