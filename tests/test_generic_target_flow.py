from __future__ import annotations

import threading
import unittest

from Panthera_lib.grasp_config import GraspConfig, TargetRequest
from Panthera_lib.grasp_planner import GraspPlanner


class GenericTargetFlowTests(unittest.TestCase):
    def test_plan_only_object_clears_browser_state_without_robot_motion(self):
        class Robot:
            pass

        class Streamer:
            def __init__(self):
                self.cleared = []

            def clear_selected_target(self, message):
                self.cleared.append(message)

        planner = GraspPlanner(Robot(), GraspConfig(), threading.Event())
        streamer = Streamer()
        requests = iter((TargetRequest("bottle", None), False))
        planner.scan_for_target = lambda *_args: self.fail(
            "plan-only targets must never enter the motion pipeline"
        )

        result = planner.run_grasp_loop(
            None,
            None,
            None,
            streamer,
            lambda: next(requests),
        )

        self.assertFalse(result)
        self.assertEqual(len(streamer.cleared), 1)
        self.assertIn("不会运动", streamer.cleared[0])


if __name__ == "__main__":
    unittest.main()
