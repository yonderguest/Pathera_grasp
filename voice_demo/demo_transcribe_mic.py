# -*- coding: utf-8 -*-
"""可选 demo：arecord 录音 3 秒 -> 识别（需要系统装有 arecord / ALSA）。

用法:
    python demo_transcribe_mic.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iq9075_speech import SpeechRecognizer, AsrConfig


def main():
    asr = SpeechRecognizer(AsrConfig())
    asr.load()
    print("请对着麦克风说话（录音 3 秒）...")
    text = asr.record_and_transcribe(duration=3.0)
    print("识别结果:", text)
    asr.close()


if __name__ == "__main__":
    main()
