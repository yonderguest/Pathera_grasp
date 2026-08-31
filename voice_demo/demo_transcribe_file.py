# -*- coding: utf-8 -*-
"""最小 demo：在全新空白项目中，导入语音识别组件，识别一个 WAV 文件。

用法:
    python demo_transcribe_file.py [音频.wav]
    或设置环境变量 IQ9075_ASR_MODEL_DIR 指向模型目录（默认取下方相对路径）。
"""
import os
import sys

# 使组件包可被直接导入（放在与 iq9075_speech 平级时无需此行）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iq9075_speech import SpeechRecognizer, AsrConfig


def main():
    wav_path = sys.argv[1] if len(sys.argv) > 1 else "sample_zh.wav"
    # 1) 构造识别器（模型目录 + 线程数）
    config = AsrConfig(num_threads=2)
    asr = SpeechRecognizer(config)

    # 2) 加载模型
    asr.load()
    print(f"模型已加载: {config.model_dir}")

    # 3) 识别音频文件 -> 文本
    text = asr.transcribe_file(wav_path)
    print("识别结果:", text)

    # 4) 释放
    asr.close()


if __name__ == "__main__":
    main()
