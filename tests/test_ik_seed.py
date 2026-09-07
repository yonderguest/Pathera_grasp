from __future__ import annotations

import numpy as np
import unittest

from Panthera_lib.Panthera import Panthera


def _solver_with_recorder(recorder):
    robot = Panthera.__new__(Panthera)
    robot.motor_count = 6
    robot.joint_limits = {
        "lower": np.full(6, -2.0),
        "upper": np.full(6, 2.0),
    }
    robot.get_current_pos = lambda: np.full(6, 0.25)

    def single(**kwargs):
        recorder.append(np.asarray(kwargs["init_q"], dtype=float).copy())
        return None

    robot._inverse_kinematics_dls_single_impl = single
    return robot


def _run_multi(robot, seed):
    return robot._inverse_kinematics_dls_multi_init_impl(
        target_position=np.zeros(3),
        target_rotation=np.eye(3),
        init_q=seed,
        num_attempts=8,
        max_iter=1,
        eps=1e-3,
        damping=1e-2,
        adaptive_damping=True,
    )


class IkSeedTests(unittest.TestCase):
    def test_multi_init_uses_explicit_seed_first_and_is_reproducible(self):
        seed = np.linspace(-0.5, 0.5, 6)
        first, second = [], []
        _run_multi(_solver_with_recorder(first), seed)
        _run_multi(_solver_with_recorder(second), seed)

        self.assertTrue(np.allclose(first[0], seed))
        self.assertEqual(len(first), 8)
        self.assertEqual(len(second), 8)
        self.assertTrue(all(np.allclose(a, b) for a, b in zip(first, second)))
