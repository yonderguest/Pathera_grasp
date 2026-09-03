from __future__ import annotations

import json
import unittest
import urllib.error
import urllib.request

import numpy as np

import Panthera_lib.vision_streamer as vision_streamer_module
from Panthera_lib.grasp_config import TargetRequest
from Panthera_lib.vision_streamer import ControlMode, VisionStreamer, _INDEX_HTML


class StreamerUiTests(unittest.TestCase):
    @staticmethod
    def _post(
        streamer,
        path,
        payload,
        *,
        token="test-control-token",
        origin=None,
        content_type="application/json",
    ):
        headers = {}
        if token is not None:
            headers["X-Panthera-Control-Token"] = token
        if origin is not None:
            headers["Origin"] = origin
        if content_type is not None:
            headers["Content-Type"] = content_type
        request = urllib.request.Request(
            streamer.url + path.lstrip("/"),
            data=json.dumps(payload).encode(),
            headers=headers,
            method="POST",
        )
        try:
            response = urllib.request.urlopen(request, timeout=2.0)
            return response.status, json.loads(response.read().decode()), response.headers
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read().decode()), exc.headers

    def test_browser_target_queue_rejects_overwrite(self):
        streamer = VisionStreamer()
        streamer.set_accepting_targets(True)
        streamer.submit_target_command("red")
        with self.assertRaises(RuntimeError):
            streamer.submit_target_command("green")

        self.assertEqual(streamer.control_status()["mode"], ControlMode.GRASPING.value)
        self.assertEqual(streamer.poll_target_command(), "red")
        self.assertIsNone(streamer.poll_target_command())

        streamer.reject_target_command("invalid command")
        self.assertEqual(streamer.control_status()["mode"], ControlMode.IDLE.value)
        self.assertTrue(streamer.control_status()["accepting_targets"])

    def test_browser_target_is_rejected_outside_operator_prompt(self):
        streamer = VisionStreamer()

        with self.assertRaises(RuntimeError):
            streamer.submit_target_command("red")

        self.assertFalse(streamer.control_status()["accepting_targets"])

    def test_completed_target_is_cleared_before_next_prompt(self):
        streamer = VisionStreamer()
        streamer.set_selected_color("red")
        self.assertEqual(streamer.control_status()["selected_color"], "red")

        streamer.clear_selected_target("ready for next target")

        status = streamer.control_status()
        self.assertIsNone(status["selected_color"])
        self.assertEqual(status["message"], "ready for next target")

    def test_selected_target_carries_object_and_colour(self):
        streamer = VisionStreamer()
        streamer.set_selected_target(TargetRequest("toy building block", "yellow"))

        status = streamer.control_status()
        self.assertEqual(status["selected_object"], "toy building block")
        self.assertEqual(status["selected_color"], "yellow")

        streamer.clear_selected_target()
        self.assertIsNone(streamer.control_status()["selected_object"])

    def test_control_page_has_idle_joint1_jog_without_checkbox(self):
        self.assertFalse(hasattr(vision_streamer_module, "_LEGACY_INDEX_HTML"))
        self.assertIn("/api/target", _INDEX_HTML)
        self.assertIn("/api/stop", _INDEX_HTML)
        self.assertIn("/api/joint1", _INDEX_HTML)
        self.assertIn("/api/follow", _INDEX_HTML)
        self.assertIn("/api/auth", _INDEX_HTML)
        self.assertIn('idle:"抓取待机"', _INDEX_HTML)
        self.assertIn("authValid", _INDEX_HTML)
        self.assertIn("结束程序并回启动姿态", _INDEX_HTML)
        self.assertIn("Panthera 抓取", _INDEX_HTML)
        self.assertIn("随动模式", _INDEX_HTML)
        self.assertIn("瓶子", _INDEX_HTML)
        self.assertIn("盒子", _INDEX_HTML)
        self.assertNotIn("X-Panthera-Control-Token", _INDEX_HTML)
        self.assertIn("credentials:\"same-origin\"", _INDEX_HTML)
        self.assertIn("preview-kind", _INDEX_HTML)
        self.assertIn('grid-template-areas:"videos videos" "left right"', _INDEX_HTML)
        self.assertIn(
            'canStopFollow=mode==="follow_arming"||mode==="following"',
            _INDEX_HTML,
        )
        self.assertIn('mode==="returning"?"正在返回 HOME"', _INDEX_HTML)
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

    def test_follow_mode_is_mutually_exclusive_and_acknowledged_by_main_thread(self):
        streamer = VisionStreamer()
        streamer.set_accepting_targets(True)

        request_id = streamer.submit_follow_command(True)
        self.assertGreater(request_id, 0)
        self.assertEqual(streamer.control_status()["mode"], "follow_arming")
        self.assertTrue(streamer.control_status()["follow_active"])
        with self.assertRaises(RuntimeError):
            streamer.submit_target_command("green block")
        with self.assertRaises(RuntimeError):
            streamer.submit_joint1_jog("left")

        self.assertIs(streamer.poll_follow_command(), True)
        streamer.activate_follow_mode("following")
        self.assertEqual(streamer.control_status()["mode"], "following")
        streamer.update_follow_feedback(True, 0.87, "hand locked")
        status = streamer.control_status()
        self.assertTrue(status["follow_hand_visible"])
        self.assertAlmostEqual(status["follow_hand_confidence"], 0.87)

        streamer.submit_follow_command(False)
        self.assertIs(streamer.poll_follow_command(), False)
        self.assertEqual(streamer.control_status()["mode"], "returning")
        streamer.finish_follow_mode("HOME ready")
        self.assertEqual(streamer.control_status()["mode"], "idle")
        self.assertTrue(streamer.control_status()["accepting_targets"])

    def test_global_stop_overrides_pending_follow(self):
        callbacks = []
        streamer = VisionStreamer(stop_callback=lambda: callbacks.append("stop"))
        streamer.set_accepting_targets(True)
        streamer.submit_follow_command(True)

        self.assertTrue(streamer.request_stop())
        status = streamer.control_status()
        self.assertEqual(status["mode"], "stopping")
        self.assertFalse(status["pending"])
        self.assertFalse(status["accepting_follow"])
        self.assertEqual(callbacks, ["stop"])

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
            control_token="test-control-token",
        )
        self.assertTrue(streamer.start())
        try:
            page_response = urllib.request.urlopen(streamer.url, timeout=2.0)
            html = page_response.read().decode()
            page_headers = page_response.headers
            self.assertIn("panthera_control=", page_headers.get("Set-Cookie", ""))
            request = urllib.request.Request(
                streamer.url + "api/stop",
                data=json.dumps({"confirmed": True}).encode(),
                headers={
                    "Content-Type": "application/json",
                    "X-Panthera-Control-Token": "test-control-token",
                },
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
            self.assertIn("/stream/raw", html)
            self.assertIn("/stream/yolo", html)
            self.assertIn("preview-kind", html)
            self.assertIn("YOLO 识别", html)
            self.assertIn("Content-Security-Policy", page_headers)
            self.assertEqual(callbacks, ["stop"])
        finally:
            streamer.stop()

    def test_http_joint1_endpoint_only_queues_during_idle_prompt(self):
        streamer = VisionStreamer(
            host="127.0.0.1",
            port=0,
            control_token="test-control-token",
        )
        streamer.set_accepting_targets(True)
        self.assertTrue(streamer.start())
        try:
            request = urllib.request.Request(
                streamer.url + "api/joint1",
                data=json.dumps({"direction": "right"}).encode(),
                headers={
                    "Content-Type": "application/json",
                    "X-Panthera-Control-Token": "test-control-token",
                },
                method="POST",
            )
            response = json.loads(
                urllib.request.urlopen(request, timeout=2.0).read().decode()
            )
            self.assertIn("向右", response["message"])
            self.assertEqual(streamer.poll_joint1_jog(), "right")
        finally:
            streamer.stop()

    def test_control_http_rejects_unauthenticated_cross_origin_and_wrong_media(self):
        streamer = VisionStreamer(
            host="127.0.0.1",
            port=0,
            control_token="test-control-token",
        )
        streamer.set_accepting_targets(True)
        self.assertTrue(streamer.start())
        try:
            status, _, _ = self._post(
                streamer,
                "/api/target",
                {"command": "green block", "confirmed": True},
                token=None,
            )
            self.assertEqual(status, 403)

            status, _, _ = self._post(
                streamer,
                "/api/target",
                {"command": "green block", "confirmed": True},
                origin="http://evil.invalid",
            )
            self.assertEqual(status, 403)

            status, _, _ = self._post(
                streamer,
                "/api/target",
                {"command": "green block", "confirmed": True},
                content_type="text/plain",
            )
            self.assertEqual(status, 415)
            self.assertEqual(streamer.control_status()["mode"], "idle")
        finally:
            streamer.stop()

    def test_control_http_rejects_non_object_json_and_pending_overwrite(self):
        streamer = VisionStreamer(
            host="127.0.0.1",
            port=0,
            control_token="test-control-token",
        )
        streamer.set_accepting_targets(True)
        self.assertTrue(streamer.start())
        try:
            status, _, headers = self._post(streamer, "/api/target", [])
            self.assertEqual(status, 400)
            self.assertEqual(headers.get("X-Content-Type-Options"), "nosniff")

            status, first, _ = self._post(
                streamer,
                "/api/target",
                {"command": "green block", "confirmed": True},
            )
            self.assertEqual(status, 202)
            self.assertGreater(first["request_id"], 0)
            status, _, _ = self._post(
                streamer,
                "/api/target",
                {"command": "red block", "confirmed": True},
            )
            self.assertEqual(status, 409)
            self.assertEqual(streamer.poll_target_command(), "green block")
        finally:
            streamer.stop()

    def test_http_follow_only_queues_intent(self):
        streamer = VisionStreamer(
            host="127.0.0.1",
            port=0,
            control_token="test-control-token",
        )
        streamer.set_accepting_targets(True)
        self.assertTrue(streamer.start())
        try:
            status, payload, _ = self._post(
                streamer,
                "/api/follow",
                {"enabled": True, "confirmed": True},
            )
            self.assertEqual(status, 202)
            self.assertIn("request_id", payload)
            self.assertIs(streamer.poll_follow_command(), True)
            self.assertEqual(streamer.control_status()["mode"], "follow_arming")
        finally:
            streamer.stop()

    def test_http_auth_probe_accepts_current_token_and_rejects_stale_token(self):
        streamer = VisionStreamer(
            host="127.0.0.1",
            port=0,
            control_token="test-control-token",
        )
        self.assertTrue(streamer.start())
        try:
            status, payload, _ = self._post(streamer, "/api/auth", {})
            self.assertEqual(status, 200)
            self.assertIn("有效", payload["message"])

            status, payload, _ = self._post(
                streamer,
                "/api/auth",
                {},
                token="stale-control-token",
            )
            self.assertEqual(status, 403)
            self.assertIn("无效", payload["message"])
        finally:
            streamer.stop()

    def test_control_url_keeps_secret_out_of_http_request(self):
        streamer = VisionStreamer(
            host="127.0.0.1",
            port=8765,
            control_token="secret token",
        )
        self.assertEqual(streamer.url, "http://127.0.0.1:8765/")
        self.assertEqual(streamer.control_url, streamer.url)

    def test_browser_cookie_auth_works_without_url_token(self):
        streamer = VisionStreamer(host="127.0.0.1", port=0)
        self.assertTrue(streamer.start())
        try:
            cookie_jar = urllib.request.HTTPCookieProcessor()
            opener = urllib.request.build_opener(cookie_jar)
            opener.open(streamer.url, timeout=2.0).read()
            request = urllib.request.Request(
                streamer.url + "api/auth",
                data=b"{}",
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            response = opener.open(request, timeout=2.0)
            self.assertEqual(response.status, 200)
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
