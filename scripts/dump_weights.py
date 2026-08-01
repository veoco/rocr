#!/usr/bin/env python
"""Dump all tensor names / shapes / dtypes from a PP-OCRv6 safetensors model.

Usage: python scripts/dump_weights.py <model_repo_name>
  e.g. python scripts/dump_weights.py PP-OCRv6_small_det_safetensors
"""
import json
import struct
import sys
import urllib.request


def inspect(repo, max_bytes=8 << 20):
    url = f"https://huggingface.co/PaddlePaddle/{repo}/resolve/main/model.safetensors"
    req = urllib.request.Request(url, headers={"Range": f"bytes=0-{max_bytes}"})
    data = urllib.request.urlopen(req, timeout=120).read()
    ln = struct.unpack("<Q", data[:8])[0]
    header = json.loads(data[8 : 8 + ln])
    print(f"# {repo}: {len(header)} tensors")
    for k, v in header.items():
        print(f"{k}\t{v['dtype']}\t{'x'.join(map(str, v['shape']))}")


if __name__ == "__main__":
    inspect(sys.argv[1] if len(sys.argv) > 1 else "PP-OCRv6_small_det_safetensors")
