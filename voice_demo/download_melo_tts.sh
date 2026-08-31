#!/bin/bash
# 下载 sherpa-onnx 中文离线 TTS 模型 vits-melo-tts-zh_en（中英混读，163MB）。
# 默认放到 /home/ubuntu/work/sherpa_tts_models/vits-melo-tts-zh_en，
# 与 SherpaTtsConfig 默认 model_dir 保持一致（也可用环境变量覆盖）。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DEST="${IQ9075_SHERPA_TTS_MODEL_DIR:-$PROJECT_ROOT/models/sherpa_tts/vits-melo-tts-zh_en}"
mkdir -p "$DEST"

TARBALL="vits-melo-tts-zh_en.tar.bz2"
BASE="https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/$TARBALL"
# 依次尝试：国内镜像 -> 备用镜像 -> 官方直连
URLS=(
  "https://gh-proxy.com/$BASE"
  "https://ghfast.top/$BASE"
  "$BASE"
)

if [ -f "$DEST/model.onnx" ] && [ -f "$DEST/lexicon.txt" ] && [ -f "$DEST/tokens.txt" ]; then
  echo "模型已存在: $DEST"
  exit 0
fi

cd "$(dirname "$DEST")"
ok=0
for url in "${URLS[@]}"; do
  echo "下载: $url"
  if wget -O "$TARBALL" "$url"; then
    ok=1
    break
  fi
  echo "下载失败，尝试下一个镜像..."
done

if [ "$ok" -ne 1 ] || [ ! -f "$TARBALL" ]; then
  echo "所有镜像均下载失败" >&2
  exit 1
fi

tar xvf "$TARBALL"
rm -f "$TARBALL"

echo "done: $DEST"
echo "验证:"
ls -lh "$DEST/model.onnx" "$DEST/lexicon.txt" "$DEST/tokens.txt"
echo "用法:"
echo "  python3 demo/demo_say.py --sherpa"
