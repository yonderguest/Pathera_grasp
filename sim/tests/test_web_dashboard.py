"""HTTP smoke tests for the hardware-isolated simulation dashboard."""

from __future__ import annotations

import json
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from sim.mujoco_backend import MujocoRobot
from sim.web_dashboard import SimWebDashboard


class SimWebDashboardTest(unittest.TestCase):
    def test_status_start_and_wrist_stream(self) -> None:
        robot = MujocoRobot(
            width=320,
            height=240,
            scene_seed=20260904,
            active_colours=("green", "red", "blue"),
        )
        dashboard = SimWebDashboard(robot, port=0)
        try:
            dashboard.start()
            with urlopen(dashboard.url + "api/status", timeout=3) as response:
                status = json.load(response)
            with urlopen(
                Request(dashboard.url + "api/start", method="POST"), timeout=3
            ) as response:
                queued = json.load(response)
            with self.assertRaises(HTTPError) as duplicate:
                urlopen(
                    Request(dashboard.url + "api/start", method="POST"), timeout=3
                )
            self.assertEqual(duplicate.exception.code, 409)
            with urlopen(dashboard.url + "stream/wrist.mjpg", timeout=3) as response:
                preview = response.read(160)
            self.assertEqual(status["phase"], "ready")
            self.assertTrue(queued["ok"])
            self.assertIn(b"image/jpeg", preview)
        finally:
            dashboard.close()
            robot.close()

    def test_command_state_guards_and_duplicate_active_colours(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicates"):
            MujocoRobot(active_colours=("green", "green"))

        robot = MujocoRobot(width=160, height=120, scene_seed=20260904)
        dashboard = SimWebDashboard(robot, port=0)
        try:
            dashboard.start()
            paused, _message = dashboard.request_pause()
            first_start, _message = dashboard.request_start()
            duplicate_start, _message = dashboard.request_start()
            reset_while_queued, _message = dashboard.request_reset()
            self.assertFalse(paused)
            self.assertTrue(first_start)
            self.assertFalse(duplicate_start)
            self.assertFalse(reset_while_queued)
        finally:
            dashboard.close()
            robot.close()


if __name__ == "__main__":
    unittest.main()
