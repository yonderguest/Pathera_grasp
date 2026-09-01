from __future__ import annotations

import unittest

import numpy as np


try:
    # The hardware SDK is imported but no robot object is initialized.  Hosts
    # without the vendor/scientific runtime can still run the remaining tests.
    from Panthera_lib.Panthera import Panthera
except (ImportError, SystemExit):  # pragma: no cover - host dependency boundary
    Panthera = None


@unittest.skipIf(Panthera is None, "Panthera scientific/vendor runtime unavailable")
class CartesianExecutionTests(unittest.TestCase):
    def test_checked_executor_densifies_without_joint_overshoot(self):
        robot = Panthera.__new__(Panthera)
        robot.motor_count = 6
        robot.joint_limits = {
            "lower": np.full(6, -1.0),
            "upper": np.full(6, 1.0),
        }
        robot.velocity_limits = np.full(6, 1.0)
        robot.acceleration_limits = np.full(6, 2.0)
        captured = {}

        def execute(trajectory, timestamps, velocities, _max_torque):
            captured["trajectory"] = np.asarray(trajectory, dtype=float)
            captured["timestamps"] = np.asarray(timestamps, dtype=float)
            captured["velocities"] = np.asarray(velocities, dtype=float)
            return True

        robot._execute_trajectory = execute
        sparse = np.zeros((4, 6), dtype=float)
        sparse[:, 0] = [0.00, 0.02, 0.05, 0.08]
        sparse[:, 1] = [0.00, -0.01, -0.015, -0.02]

        self.assertTrue(
            robot.execute_joint_trajectory_checked(
                sparse,
                duration=2.0,
                max_torque=[1.0] * 6,
                label="TEST CARTESIAN",
                control_period=0.02,
            )
        )
        dense = captured["trajectory"]
        velocities = captured["velocities"]
        self.assertGreaterEqual(len(dense), 101)
        self.assertTrue(np.allclose(dense[0], sparse[0]))
        self.assertTrue(np.allclose(dense[-1], sparse[-1]))
        self.assertTrue(np.allclose(velocities[[0, -1]], 0.0))
        self.assertTrue(np.all(np.diff(dense[:, 0]) >= -1e-12))
        self.assertTrue(np.all(np.diff(dense[:, 1]) <= 1e-12))
        self.assertGreaterEqual(float(dense[:, 0].min()), float(sparse[:, 0].min()))
        self.assertLessEqual(float(dense[:, 0].max()), float(sparse[:, 0].max()))
        self.assertAlmostEqual(captured["timestamps"][-1], 2.0)


if __name__ == "__main__":
    unittest.main()
