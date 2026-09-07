from __future__ import annotations

import os
import unittest

from Panthera_lib.npu_inference import NpuYoloDetector


class NpuTimeoutTests(unittest.TestCase):
    def test_npu_read_has_bounded_timeout(self):
        class Process:
            @staticmethod
            def poll():
                return None

        read_fd, write_fd = os.pipe()
        detector = NpuYoloDetector.__new__(NpuYoloDetector)
        detector.proc = Process()
        detector.response_timeout = 0.02
        try:
            with self.assertRaises(TimeoutError):
                detector._read_exact(read_fd, 4)
        finally:
            os.close(read_fd)
            os.close(write_fd)
