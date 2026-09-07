from __future__ import annotations

import os
from pathlib import Path
import threading
import time
import unittest

from iq9075_tts.speaker import Speaker


class TtsSafetyTests(unittest.TestCase):
    def test_speaker_uses_private_temporary_directory_and_cleans_it(self):
        played = threading.Event()
        output_paths = []

        class CompletedProcess:
            def wait(self, timeout=None):
                played.set()
                return 0

            def kill(self):
                return None

        class FakeBackend:
            output_suffix = ".wav"

            def synthesize(self, _text, out_path):
                # OfflineWavBackend relies on a non-existent output path to
                # select its prerecorded source, so the secure parent may
                # exist but the final file must not be pre-created.
                self.assertFalse(os.path.exists(out_path))
                Path(out_path).write_bytes(b"RIFF")
                output_paths.append(out_path)
                return True

            def play(self, audio_path):
                self.assertEqual(Path(audio_path).read_bytes(), b"RIFF")
                return CompletedProcess()

            def __init__(self, case):
                self.assertFalse = case.assertFalse
                self.assertEqual = case.assertEqual

        speaker = Speaker(backend=FakeBackend(self))
        try:
            speaker.say("archive smoke")
            self.assertTrue(played.wait(2.0))
            deadline = time.monotonic() + 2.0
            while (
                time.monotonic() < deadline
                and output_paths
                and Path(output_paths[0]).parent.exists()
            ):
                time.sleep(0.01)
        finally:
            speaker.close()

        self.assertEqual(len(output_paths), 1)
        self.assertFalse(Path(output_paths[0]).exists())
        self.assertFalse(Path(output_paths[0]).parent.exists())


if __name__ == "__main__":
    unittest.main()
