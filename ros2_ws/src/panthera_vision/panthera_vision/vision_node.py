#!/usr/bin/env python3
"""ROS2 vision node.

Publishes:
  /vision/image_raw       sensor_msgs/Image  (BGR8)
  /vision/depth_image     sensor_msgs/Image  (16UC1 millimetres)
  /vision/annotated       sensor_msgs/Image  (BGR8 with detections drawn)
  /vision/detections      std_msgs/String    (JSON detection list)
  /vision/camera_info     std_msgs/String    (JSON intrinsics + depth scale)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String


def _project_root() -> Path:
    # <project>/ros2_ws/src/panthera_vision/panthera_vision/vision_node.py
    return Path(__file__).resolve().parents[4]


def _add_paths(root: Path) -> None:
    for path in (
        root,
        root / "Panthera-HT_SDK" / "panthera_python",
        root / "Panthera-HT_SDK" / "panthera_python" / "scripts",
    ):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def _set_image_header(msg: Image, timestamp_ns: int, frame_id: str) -> None:
    msg.header.stamp.sec = int(timestamp_ns // 1_000_000_000)
    msg.header.stamp.nanosec = int(timestamp_ns % 1_000_000_000)
    msg.header.frame_id = frame_id


def bgr_to_image(
    image: np.ndarray,
    timestamp_ns: int = 0,
    frame_id: str = "camera_color_optical_frame",
) -> Image:
    msg = Image()
    msg.height = int(image.shape[0])
    msg.width = int(image.shape[1])
    msg.encoding = "bgr8"
    msg.is_bigendian = 0
    msg.step = int(image.shape[1] * image.shape[2])
    msg.data = np.asarray(image, dtype=np.uint8).tobytes()
    _set_image_header(msg, timestamp_ns, frame_id)
    return msg


def depth_to_image(
    depth: np.ndarray,
    timestamp_ns: int = 0,
    frame_id: str = "camera_color_optical_frame",
) -> Image:
    msg = Image()
    msg.height = int(depth.shape[0])
    msg.width = int(depth.shape[1])
    msg.encoding = "16UC1"
    msg.is_bigendian = 0
    msg.step = int(depth.shape[1] * 2)
    msg.data = np.asarray(depth, dtype=np.uint16).tobytes()
    _set_image_header(msg, timestamp_ns, frame_id)
    return msg


def _detections_json(detections) -> str:
    payload = []
    for detection in detections:
        pixel = np.asarray(detection.get("pixel", [0.0, 0.0]), dtype=float)
        short_axis = np.asarray(
            detection.get("short_axis_uv", [1.0, 0.0]), dtype=float
        )
        payload.append(
            {
                "class_name": str(detection.get("class_name") or "object"),
                "color": str(detection.get("color") or "unknown"),
                "confidence": float(detection.get("confidence") or 0.0),
                "color_confidence": float(detection.get("color_confidence") or 0.0),
                "depth_m": float(detection.get("depth_m") or 0.0),
                "bbox": [float(v) for v in detection["bbox"]],
                "pixel": pixel.tolist(),
                "short_axis_uv": short_axis.tolist(),
                "obb_angle_deg": float(detection.get("obb_angle_deg") or 0.0),
            }
        )
    return json.dumps(payload, ensure_ascii=False)


def _detections_stamped_json(
    detections,
    frame_seq: int,
    capture_timestamp_ns: int,
) -> str:
    """Compatibility extension; the legacy detection-list topic is unchanged."""
    return json.dumps(
        {
            "frame_seq": int(frame_seq),
            "capture_timestamp_ns": int(capture_timestamp_ns),
            "detections": json.loads(_detections_json(detections)),
        },
        ensure_ascii=False,
    )


def camera_info_qos() -> QoSProfile:
    """Durable CameraInfo-like JSON for late-joining grasp clients."""
    return QoSProfile(
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )


class PantheraVisionNode(Node):
    def __init__(self) -> None:
        super().__init__("panthera_vision")
        default_root = _project_root()
        self.declare_parameter("project_root", str(default_root))
        root = Path(str(self.get_parameter("project_root").value)).expanduser().resolve()
        _add_paths(root)

        from Panthera_lib.grasp_config import GraspConfig
        from Panthera_lib.npu_inference import NpuYoloDetector
        from Panthera_lib.vision_pipeline import (
            CameraFeed,
            init_camera,
            load_target_model,
        )
        from Panthera_lib.vision_streamer import draw_detections

        self._draw_detections = draw_detections

        self.declare_parameter("model_path", str(root / "models" / "yoloe-26s-seg.pt"))
        self.declare_parameter("text_encoder_path", str(root / "mobileclip2_b.ts"))
        self.declare_parameter(
            "calibration_file", str(root / "hand_eye_calibration.json")
        )
        self.declare_parameter("use_npu", False)
        self.declare_parameter("publish_hz", 30.0)

        config = GraspConfig()
        config.project_root = root
        config.model_path = self.get_parameter("model_path").value
        config.text_encoder_path = Path(self.get_parameter("text_encoder_path").value)
        config.calibration_file = Path(self.get_parameter("calibration_file").value)
        config.use_npu = bool(self.get_parameter("use_npu").value)
        config.validate()

        self._pub_raw = self.create_publisher(Image, "vision/image_raw", 10)
        self._pub_depth = self.create_publisher(Image, "vision/depth_image", 10)
        self._pub_annotated = self.create_publisher(Image, "vision/annotated", 10)
        self._pub_detections = self.create_publisher(String, "vision/detections", 10)
        self._pub_detections_stamped = self.create_publisher(
            String,
            "vision/detections_stamped",
            10,
        )
        self._pub_camera_info = self.create_publisher(
            String,
            "vision/camera_info",
            camera_info_qos(),
        )

        self.pipeline, self.align, self.intrinsic, self.depth_scale = init_camera(config)
        self.camera_feed = CameraFeed(
            self.pipeline,
            self.align,
            config,
            self.depth_scale,
            streamer=None,
            model=None,
            intrinsic=self.intrinsic,
        )
        if config.use_npu:
            self.npu = NpuYoloDetector(config, confidence=config.npu_confidence_threshold)
            self.camera_feed.set_npu_detector(self.npu)
        else:
            self.model = load_target_model(config)
            self.camera_feed.set_model(self.model)
            self.npu = None
        self.camera_feed.start()

        info = {
            "width": self.intrinsic.width,
            "height": self.intrinsic.height,
            "fx": self.intrinsic.fx,
            "fy": self.intrinsic.fy,
            "ppx": self.intrinsic.ppx,
            "ppy": self.intrinsic.ppy,
            "depth_scale": self.depth_scale,
            "frame_id": "camera_color_optical_frame",
        }
        self._camera_info_json = json.dumps(info)
        self._publish_camera_info()
        self._camera_info_timer = self.create_timer(5.0, self._publish_camera_info)

        hz = max(1.0, float(self.get_parameter("publish_hz").value))
        self._last_published_seq = -1
        self._timer = self.create_timer(1.0 / hz, self._publish)
        self.get_logger().info("panthera_vision started")

    def _publish_camera_info(self) -> None:
        msg = String()
        msg.data = self._camera_info_json
        self._pub_camera_info.publish(msg)

    def _publish(self) -> None:
        capture = self.camera_feed.latest()
        if capture is None:
            return
        frame_seq = int(capture["frame_seq"])
        if frame_seq == self._last_published_seq:
            return
        self._last_published_seq = frame_seq
        capture_timestamp_ns = int(capture["capture_timestamp_ns"])
        color = capture["color_image"]
        depth = capture["depth_image"]
        detections = capture["detections"]

        self._pub_raw.publish(bgr_to_image(color, capture_timestamp_ns))
        self._pub_depth.publish(depth_to_image(depth, capture_timestamp_ns))
        annotated = self._draw_detections(color, detections)
        self._pub_annotated.publish(bgr_to_image(annotated, capture_timestamp_ns))

        msg = String()
        msg.data = _detections_json(detections)
        self._pub_detections.publish(msg)

        stamped = String()
        stamped.data = _detections_stamped_json(
            detections,
            frame_seq,
            capture_timestamp_ns,
        )
        self._pub_detections_stamped.publish(stamped)

    def destroy_node(self) -> None:
        if hasattr(self, "camera_feed"):
            self.camera_feed.stop()
        if hasattr(self, "npu") and self.npu is not None:
            self.npu.close()
        if hasattr(self, "pipeline"):
            try:
                self.pipeline.stop()
            except Exception:
                pass
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PantheraVisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
