#!/usr/bin/env sh
# Download the GGUF model for the llm container into ./models/gguf.
# Override with: LLM_REPO=... LLM_FILE=... scripts/fetch_llm.sh
set -eu
LLM_REPO="${LLM_REPO:-Qwen/Qwen3-8B-GGUF}"
LLM_FILE="${LLM_FILE:-Qwen3-8B-Q4_K_M.gguf}"
DEST="$(dirname "$0")/../models/gguf"
mkdir -p "$DEST"
if [ -f "$DEST/$LLM_FILE" ]; then
  echo "$DEST/$LLM_FILE already present"; exit 0
fi
echo "downloading $LLM_REPO/$LLM_FILE (~5 GB)"
curl -fL --retry 3 -o "$DEST/$LLM_FILE.part" "https://huggingface.co/$LLM_REPO/resolve/main/$LLM_FILE"
mv "$DEST/$LLM_FILE.part" "$DEST/$LLM_FILE"
echo "saved to $DEST/$LLM_FILE"
