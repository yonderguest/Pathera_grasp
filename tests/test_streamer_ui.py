from __future__ import annotations

import json
import unittest
import urllib.request

import numpy as np

from Panthera_lib.vision_streamer import VisionStreamer, _INDEX_HTML


class StreamerUiTests(unittest.TestCase):
    def test_browser_target_queue_keeps_latest_command(self):
        streamer = VisionStreamer()
        streamer.set_accepting_targets(True)
        streamer.submit_target_command("red")
        streamer.submit_target_command("green")

        self.assertEqual(streamer.poll_target_command(), "green")
        self.assertIsNone(streamer.poll_target_command())

    def test_browser_target_is_rejected_outside_operator_prompt(self):
        streamer = VisionStreamer()

        with self.assertRaises(RuntimeError):
            streamer.submit_target_command("red")

        self.assertFalse(streamer.control_status()["accepting_targets"])

    def test_control_page_has_idle_joint1_jog_without_checkbox(self):
        self.assertIn("/api/target", _INDEX_HTML)
        self.assertIn("/api/stop", _INDEX_HTML)
        self.assertIn("/api/joint1", _INDEX_HTML)
        self.assertIn("结束程序并回启动姿态", _INDEX_HTML)
        self.assertIn("Panthera 抓取", _INDEX_HTML)
        self.assertIn("jog-left", _INDEX_HTML)
        self.assertIn("jog-right", _INDEX_HTML)
        self.assertIn("J1 左转 0.5 rad", _INDEX_HTML)
        self.assertIn("J1 右转 0.5 rad", _INDEX_HTML)
        self.assertNotIn("safety-confirm", _INDEX_HTML)
        self.assertNotIn("已确认机械臂路径内无人、无障碍物", _INDEX_HTML)
        self.assertIn("confirmed:true", _INDEX_HTML)

    def test_joint1_jog_queue_is_idle_gated_and_serialized(self):
        streamer = VisionStreamer()
        with self.assertRaises(RuntimeError):
            streamer.submit_joint1_jog("left")

        streamer.set_accepting_targets(True)
        streamer.submit_joint1_jog("left")
        self.assertEqual(streamer.poll_joint1_jog(), "left")
        self.assertTrue(streamer.control_status()["jog_active"])
        self.assertFalse(streamer.control_status()["accepting_jog"])
        with self.assertRaises(RuntimeError):
            streamer.submit_target_command("red")

        streamer.finish_joint1_jog("done")
        self.assertFalse(streamer.control_status()["jog_active"])
        self.assertTrue(streamer.control_status()["accepting_jog"])

    def test_web_stop_is_latched_and_callback_runs_once(self):
        callbacks = []
        streamer = VisionStreamer(stop_callback=lambda: callbacks.append("stop"))
        streamer.set_accepting_targets(True)

        self.assertTrue(streamer.request_stop())
        self.assertFalse(streamer.request_stop())
        self.assertEqual(callbacks, ["stop"])
        self.assertTrue(streamer.control_status()["stop_requested"])
        self.assertFalse(streamer.control_status()["accepting_targets"])
        with self.assertRaises(RuntimeError):
            streamer.submit_target_command("red")

    def test_http_stop_endpoint_and_clean_page(self):
        callbacks = []
        streamer = VisionStreamer(
            host="127.0.0.1",
            port=0,
            stop_callback=lambda: callbacks.append("stop"),
        )
        self.assertTrue(streamer.start())
        try:
            html = urllib.request.urlopen(streamer.url, timeout=2.0).read().decode()
            request = urllib.request.Request(
                streamer.url + "api/stop",
                data=json.dumps({"confirmed": True}).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            response = json.loads(
                urllib.request.urlopen(request, timeout=2.0).read().decode()
            )
            self.assertIn("启动姿态", response["message"])
            self.assertIn("结束程序并回启动姿态", html)
            self.assertIn("Panthera 抓取", html)
            self.assertIn("jog-left", html)
            self.assertIn("jog-right", html)
            self.assertNotIn("safety-confirm", html)
            self.assertLess(html.index("/stream/depth"), html.index("/stream/yolo"))
            self.assertIn("深度画面", html)
            self.assertIn("YOLO 识别", html)
            self.assertEqual(callbacks, ["stop"])
        finally:
            streamer.stop()

    def test_http_joint1_endpoint_only_queues_during_idle_prompt(self):
        streamer = VisionStreamer(host="127.0.0.1", port=0)
        streamer.set_accepting_targets(True)
        self.assertTrue(streamer.start())
        try:
            request = urllib.request.Request(
                streamer.url + "api/joint1",
                data=json.dumps({"direction": "right"}).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            response = json.loads(
                urllib.request.urlopen(request, timeout=2.0).read().decode()
            )
            self.assertIn("向右", response["message"])
            self.assertEqual(streamer.poll_joint1_jog(), "right")
        finally:
            streamer.stop()

    def test_depth_frame_is_rendered_as_colour_information(self):
        streamer = VisionStreamer(jpeg_quality=92)
        depth = np.full((8, 12), 1700, dtype=np.uint16)
        rendered = streamer._render_depth(depth, 0.0001, (8, 12, 3))

        self.assertEqual(rendered.shape, (8, 12, 3))
        self.assertEqual(rendered.dtype, np.uint8)
        self.assertGreater(int(np.max(rendered)), 0)


if __name__ == "__main__":
    unittest.main()
