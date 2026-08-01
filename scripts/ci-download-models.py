#!/usr/bin/env python3
"""Download the models needed by the CI oracle job into `dev-models/`.

The small tier (default) models and the ONNX exports used as numeric oracles
are enough to run every oracle test: det/rec (ONNX + safetensors), text-line
orientation (x1_0), document orientation and UVDoc unwarping.

Requires: pip install huggingface_hub
"""
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "dev-models"

REPOS = [
    "PaddlePaddle/PP-OCRv6_small_det_safetensors",
    "PaddlePaddle/PP-OCRv6_small_rec_safetensors",
    "PaddlePaddle/PP-OCRv6_small_det_onnx",
    "PaddlePaddle/PP-OCRv6_small_rec_onnx",
    "PaddlePaddle/PP-LCNet_x1_0_textline_ori_safetensors",
    "PaddlePaddle/PP-LCNet_x1_0_doc_ori_safetensors",
    "PaddlePaddle/UVDoc_safetensors",
    "PaddlePaddle/UVDoc_onnx",
]


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    for repo in REPOS:
        out = DEST / repo.split("/")[1]
        if out.exists() and any(out.iterdir()):
            print(f"skip (already present): {repo}")
            continue
        print(f"downloading {repo} ...")
        snapshot_download(repo, local_dir=out)
    print(f"models ready under {DEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
