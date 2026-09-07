#!/bin/bash
# Download the offline sherpa-onnx SenseVoice ASR model.
# Default target: <project_root>/models/sensevoice
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DEST="${IQ9075_ASR_MODEL_DIR:-$PROJECT_ROOT/models/sensevoice}"
mkdir -p "$DEST"

TARBALL="sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2"
BASE="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/$TARBALL"
URLS=(
  "https://gh-proxy.com/$BASE"
  "https://ghfast.top/$BASE"
  "$BASE"
)

if [ -f "$DEST/model.onnx" ] && [ -f "$DEST/tokens.txt" ]; then
  echo "ASR 模型已存在: $DEST"
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

# sherpa 压缩包解压后自带一个同名目录，把内容平铺到 DEST
EXTRACTED="sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
if [ -d "$EXTRACTED" ]; then
  cp -f "$EXTRACTED"/model.onnx "$DEST/" 2>/dev/null || true
  cp -f "$EXTRACTED"/tokens.txt "$DEST/" 2>/dev/null || true
  rm -rf "$EXTRACTED"
fi

echo "done: $DEST"
ls -lh "$DEST/model.onnx" "$DEST/tokens.txt"
