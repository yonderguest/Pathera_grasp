from __future__ import annotations

import threading
import unittest
from unittest.mock import patch

from Panthera_lib.grasp_config import GraspConfig, parse_target_command
from grasp_demo import choose_target_at_start
from voice_controller import VoiceInterface


class ConfigAndVoiceTests(unittest.TestCase):
    def test_browser_target_can_replace_terminal_input(self):
        class Streamer:
            is_closed = False

            def __init__(self):
                self.command = "green"
                self.messages = []
                self.accepting = False

            def poll_target_command(self):
                command, self.command = self.command, None
                return command

            def set_control_message(self, message):
                self.messages.append(message)

            def set_accepting_targets(self, accepting):
                self.accepting = bool(accepting)

        class HeadlessInput:
            @staticmethod
            def isatty():
                return False

        streamer = Streamer()
        with patch("grasp_demo.sys.stdin", HeadlessInput()):
            self.assertEqual(choose_target_at_start(None, streamer), "green")
        self.assertTrue(any("green" in message for message in streamer.messages))
        self.assertFalse(streamer.accepting)

    def test_positions_restore_previous_home_and_keep_validated_puts(self):
        config = GraspConfig()
        self.assertEqual(config.gripper_limit_lower, 0.0)
        self.assertEqual(config.gripper_limit_upper, 2.0)
        self.assertEqual(config.gripper_open_position, 1.8)
        self.assertEqual(config.gripper_close_position, 0.0)
        self.assertEqual(config.gripper_clamped_position, 0.22)
        self.assertGreater(config.gripper_open_position, config.gripper_close_position)
        self.assertEqual(config.home.tolist(), [0.0, 0.24, 1.2, -1.515, 0.0, 0.0])
        self.assertEqual(config.put1.tolist(), [1.6, 1.3, 0.55, -0.3, 0.0, 0.0])
        self.assertEqual(config.put2.tolist(), [1.5, 0.5, 0.56, -0.075, 0.0, 0.0])
        self.assertEqual(config.grasp_offset_base.tolist(), [0.0, 0.0, 0.0])
        self.assertEqual(config.grasp_approach_overtravel_m, 0.0)
        self.assertEqual(config.observation_centering_gain, 0.50)
        self.assertEqual(config.observation_axial_gain, 0.50)
        self.assertEqual(config.observation_max_advance_m, 0.025)
        self.assertEqual(config.observation_max_joint_step, 0.75)
        self.assertEqual(config.pre_grasp_standoff_ratio, 0.50)
        self.assertEqual(config.pre_grasp_min_distance_m, 0.015)
        self.assertEqual(config.pre_grasp_max_distance_m, 0.040)
        self.assertEqual(config.ik_single_seed_max_iterations, 600)
        self.assertEqual(config.ik_max_seed_attempts, 3)
        self.assertEqual(config.move_wait_min_timeout_s, 6.0)
        self.assertEqual(config.color_accumulation_timeout_s, 1.0)
        self.assertEqual(config.refine_total_timeout_s, 4.0)
        self.assertTrue(config.close_refine_enabled)
        self.assertTrue(config.close_refine_preserve_base_y)
        self.assertEqual(config.close_refine_min_depth_m, 0.075)
        self.assertEqual(config.close_refine_max_xy_correction_m, 0.035)
        self.assertEqual(config.close_refine_max_z_correction_m, 0.018)
        self.assertEqual(config.close_refine_max_total_correction_m, 0.035)
        self.assertEqual(config.trajectory_control_period_s, 0.020)
        self.assertEqual(config.direct_grasp_duration, 3.5)
        self.assertEqual(config.grasp_retry_retreat_duration, 2.0)
        self.assertEqual(config.direct_grasp_joint_tolerance_rad, 0.050)
        self.assertEqual(config.direct_grasp_tcp_tolerance_m, 0.012)
        self.assertEqual(config.pre_grasp_lateral_tolerance_m, 0.015)
        self.assertEqual(config.pre_grasp_orientation_tolerance_deg, 8.0)
        self.assertEqual(config.pre_grasp_joint_tolerance_rad, 0.040)
        self.assertEqual(config.pre_grasp_realign_max_translation_m, 0.045)
        self.assertEqual(config.pre_grasp_realign_endpoint_tolerance_m, 0.010)
        self.assertEqual(config.ik_rotation_tolerance_deg, 8.0)
        self.assertEqual(config.approach_endpoint_tolerance_m, 0.015)
        self.assertEqual(config.approach_path_lateral_tolerance_m, 0.007)
        self.assertEqual(config.joint1_jog_step_rad, 0.5)
        self.assertEqual(config.central_x_grasp_ratio, 0.80)
        self.assertEqual(config.stream_jpeg_quality, 92)
        self.assertEqual(config.color_yellow_green_boundary, 38)
        self.assertTrue(config.use_npu)
        self.assertEqual(config.npu_confidence_threshold, 0.15)
        self.assertTrue(config.npu_context_path.endswith("_block4.bin"))
        self.assertEqual(config.npu_class_names, config.target_prompts)
        self.assertEqual(
            (config.scan_j1_start, config.scan_j1_end, config.scan_j1_step),
            (1.8, -1.8, 0.3),
        )
        config.validate()

    def test_colour_parser_understands_negation(self):
        cases = [
            ("红色积木", ("red", True)),
            ("不要红色", (None, False)),
            ("不要红色，要蓝色", ("blue", True)),
            ("不是黄色，抓绿色", ("green", True)),
            ("not red, blue", ("blue", True)),
            ("任意颜色", (None, True)),
        ]
        for command, expected in cases:
            with self.subTest(command=command):
                self.assertEqual(parse_target_command(command), expected)

    def test_voice_listen_waits_for_tts_to_drain(self):
        events = []

        class Speaker:
            def __init__(self):
                self.busy = [True, False]

            def is_busy(self):
                events.append("busy")
                return self.busy.pop(0)

            def stop(self):
                events.append("stop")

        class Recognizer:
            def record_and_transcribe(self, duration):
                events.append(("record", duration))
                return "蓝色积木"

        voice = VoiceInterface.__new__(VoiceInterface)
        voice.available = True
        voice._speaker = Speaker()
        voice._recognizer = Recognizer()
        voice._audio_lock = threading.Lock()
        voice._playback_settle_seconds = 0.0
        voice.prompt_duration = 3.5

        with patch("voice_controller.time.sleep", lambda _seconds: None):
            self.assertEqual(voice.listen_for_command(), "蓝色积木")
        self.assertEqual(events[-1], ("record", 3.5))
        self.assertEqual(events[:2], ["busy", "busy"])
