#!/usr/bin/env python3
"""ROS2 voice node.

Topics:
  /voice/say           std_msgs/String   -> speak this text
  /voice/listen_request std_msgs/Bool    -> request one microphone recognition
  /voice/command       std_msgs/String   -> recognized text (published by this node)
  /voice/status        std_msgs/String   -> availability/status
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String


def _project_root() -> Path:
    # <project>/ros2_ws/src/panthera_voice/panthera_voice/voice_node.py
    return Path(__file__).resolve().parents[4]


class PantheraVoiceNode(Node):
    def __init__(self) -> None:
        super().__init__("panthera_voice")
        root = _project_root()
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        from voice_controller import VoiceInterface

        self.declare_parameter("voice_enabled", True)
        self.declare_parameter("voice_prompt_duration", 3.5)
        self.declare_parameter("listen_debounce", 0.2)
        enabled = bool(self.get_parameter("voice_enabled").value)
        duration = float(self.get_parameter("voice_prompt_duration").value)
        self._prompt_duration = duration
        self._listen_debounce = max(
            0.0,
            float(self.get_parameter("listen_debounce").value),
        )

        self.voice = VoiceInterface(
            root,
            prompt_duration=duration,
            enabled=enabled,
        )

        self._pub_command = self.create_publisher(String, "voice/command", 10)
        self._pub_status = self.create_publisher(String, "voice/status", 10)
        self._sub_say = self.create_subscription(String, "voice/say", self._cb_say, 10)
        self._sub_listen = self.create_subscription(
            Bool, "voice/listen_request", self._cb_listen_request, 10
        )

        self._listen_lock = threading.Lock()
        self._closing = threading.Event()
        self._listen_thread: threading.Thread | None = None
        self._status(f"available={self.voice.available}")

    def _status(self, text: str) -> None:
        msg = String()
        msg.data = text
        self._pub_status.publish(msg)
        self.get_logger().info(f"[voice] {text}")

    def _cb_say(self, msg: String) -> None:
        self.voice.say(msg.data)

    def _cb_listen_request(self, msg: Bool) -> None:
        if not msg.data or self._closing.is_set():
            return
        if not self.voice.available:
            self._status("voice unavailable")
            return
        if not self._listen_lock.acquire(blocking=False):
            self._status("listener busy")
            return

        def _worker() -> None:
            try:
                # Say/listen use different compatible topics. Give an earlier
                # prompt callback time to enqueue before VoiceInterface drains
                # TTS and opens the microphone.
                time.sleep(self._listen_debounce)
                text = self.voice.listen_for_command()
                out = String()
                out.data = text or ""
                self._pub_command.publish(out)
                self._status(f"command={out.data!r}")
            finally:
                self._listen_lock.release()

        self._listen_thread = threading.Thread(
            target=_worker,
            name="voice-listener",
            daemon=False,
        )
        self._listen_thread.start()

    def destroy_node(self) -> None:
        self._closing.set()
        if self._listen_thread is not None:
            self._listen_thread.join(timeout=self._prompt_duration + 2.0)
            if self._listen_thread.is_alive():
                self.get_logger().error("voice listener did not stop before shutdown")
        self.voice.close()
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PantheraVoiceNode()
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
