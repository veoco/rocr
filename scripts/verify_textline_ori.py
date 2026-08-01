#!/usr/bin/env python3
"""Generate numeric reference fixtures for the text-line orientation model.

Uses the PaddleOCR transformers engine (same safetensors weights) as the
numeric oracle — the same approach as `verify_paddle_det.py`.

Outputs to dev-models/reference/:
  paddle_orient_input_{name}.npy   preprocessed input tensor [1,3,80,160]
  paddle_orient_output_{name}.npy  classification logits [1,2]
"""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT / "dev-models" / "PP-LCNet_x0_25_textline_ori_safetensors"
OUT = ROOT / "dev-models" / "reference"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    proc = AutoImageProcessor.from_pretrained(REPO, local_files_only=True)
    model = AutoModelForImageClassification.from_pretrained(REPO, local_files_only=True)
    model.eval()

    cases = {
        "cn": ROOT / "assets" / "line_cn.png",
        "en": ROOT / "assets" / "line_en.png",
    }
    with torch.no_grad():
        for name, path in cases.items():
            img = Image.open(path)
            inputs = proc(images=img, return_tensors="pt")
            x = inputs["pixel_values"]
            logits = model(pixel_values=x).last_hidden_state
            np.save(OUT / f"paddle_orient_input_{name}.npy", x.detach().cpu().numpy())
            np.save(OUT / f"paddle_orient_output_{name}.npy", logits.detach().cpu().numpy())
            label = int(logits.argmax(-1).item())
            print(f"{name}: {path.name} {img.size} -> logits={logits.tolist()} label={label}")

            # Also a 180°-rotated copy to sanity-check the classifier.
            rot = img.rotate(180)
            xr = proc(images=rot, return_tensors="pt")["pixel_values"]
            lr = model(pixel_values=xr).last_hidden_state
            np.save(OUT / f"paddle_orient_input_{name}_rot.npy", xr.detach().cpu().numpy())
            np.save(OUT / f"paddle_orient_output_{name}_rot.npy", lr.detach().cpu().numpy())
            print(f"  rot180 -> logits={lr.tolist()} label={int(lr.argmax(-1).item())}")

    print(f"fixtures written to {OUT}")


if __name__ == "__main__":
    sys.exit(main())
