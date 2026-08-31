#!/usr/bin/env python3
"""最小 demo：语音播报。

三种后端：
  - 默认：edge-tts 在线合成
  - --offline <wav>：直接播放预录音频
  - --sherpa：sherpa-onnx 离线合成（VITS，断网可用）
"""
from __future__ import annotations

import argparse
import os
import sys
import time

# 允许从项目任意目录运行：把 pathera_grasp 根目录加入导入路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iq9075_tts import (
    EdgeTtsBackend,
    OfflineWavBackend,
    SherpaTtsBackend,
    SherpaTtsConfig,
    Speaker,
    TtsConfig,
)


def main() -> int:
    ap = argparse.ArgumentParser(description="iq9075_tts 语音播报 demo")
    ap.add_argument("text", nargs="?", default="小八在，检测到红色方块")
    ap.add_argument("--offline", default=None, help="离线兜底：直接播放该 wav 文件")
    ap.add_argument("--sherpa", action="store_true", help="离线合成：sherpa-onnx VITS")
    ap.add_argument("--edge", action="store_true", help="在线合成：edge-tts")
    args = ap.parse_args()

    if args.sherpa:
        speaker = Speaker(backend=SherpaTtsBackend(SherpaTtsConfig()))
        speaker.say(args.text)
    elif args.offline:
        config = TtsConfig()
        speaker = Speaker(backend=OfflineWavBackend(args.offline, config), config=config)
        speaker.say("离线播报")
    elif args.edge:
        config = TtsConfig()
        speaker = Speaker(backend=EdgeTtsBackend(config), config=config)
        speaker.say(args.text)
    else:
        # 默认离线 VITS，保证换机器后无需在线服务即可播报
        speaker = Speaker()
        speaker.say(args.text)

    time.sleep(8)
    speaker.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
