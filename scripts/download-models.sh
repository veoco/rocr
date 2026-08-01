#!/usr/bin/env bash
# Download PP-OCRv6 model weights for rocr development / local use.
#
# Models are NOT bundled with the repository. This script downloads them into
# a local directory (default: <repo>/dev-models, which is gitignored).
#
# Requires the huggingface_hub CLI:
#     pip install -U huggingface_hub
# (provides either the `hf` or `huggingface-cli` command).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${ROCR_MODELS_DIR:-$ROOT/dev-models}"
mkdir -p "$DEST"

TIERS=(tiny small medium)
KINDS=(det rec)

download() {
    local repo="$1"
    local out="$DEST/$(basename "$repo")"
    echo "==> $repo  ->  $out"
    if command -v hf >/dev/null 2>&1; then
        hf download "$repo" --local-dir "$out"
    elif command -v huggingface-cli >/dev/null 2>&1; then
        huggingface-cli download "$repo" --local-dir "$out"
    else
        echo "ERROR: need 'hf' or 'huggingface-cli' (pip install -U huggingface_hub)" >&2
        return 1
    fi
}

for t in "${TIERS[@]}"; do
    for k in "${KINDS[@]}"; do
        download "PaddlePaddle/PP-OCRv6_${t}_${k}_safetensors" || exit 1
        download "PaddlePaddle/PP-OCRv6_${t}_${k}_onnx" || exit 1
    done
done

# Text-line orientation classifier (optional module).
download "PaddlePaddle/PP-LCNet_x0_25_textline_ori_safetensors" || echo "warning: optional orient model download failed"

# Document orientation classifier (optional module).
download "PaddlePaddle/PP-LCNet_x1_0_doc_ori_safetensors" || echo "warning: optional doc-orient model download failed"

echo
echo "Done. Models are in: $DEST"
echo "Pass the relevant directory to rocr via --model-dir (see README)."
