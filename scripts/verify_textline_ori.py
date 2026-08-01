#!/usr/bin/env python3
"""Generate numeric reference fixtures for the text-line orientation models.

Uses the transformers PPLCNet image-classification engine (same safetensors
weights) as the numeric oracle — the same approach as `verify_paddle_det.py`.

Covers both variants: x1_0 (the PaddleOCR default) and x0_25 (lighter).

Outputs to dev-models/reference/:
  paddle_orient_input_{name}.npy    preprocessed input tensor [1,3,80,160]
  paddle_orient_output_{name}.npy   classification logits [1,2]
  paddle_orient_x1_input_{name}.npy / paddle_orient_x1_output_{name}.npy (x1_0)
"""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "dev-models" / "reference"
OUT.mkdir(parents=True, exist_ok=True)

# (repository dir, fixture tag)
MODELS = [
    ("PP-LCNet_x0_25_textline_ori_safetensors", ""),
    ("PP-LCNet_x1_0_textline_ori_safetensors", "x1_"),
]


def main():
    cases = {
        "cn": ROOT / "assets" / "line_cn.png",
        "en": ROOT / "assets" / "line_en.png",
    }
    for repo, tag in MODELS:
        repodir = ROOT / "dev-models" / repo
        if not repodir.exists():
            print(f"skip: {repo} not present")
            continue
        proc = AutoImageProcessor.from_pretrained(repodir, local_files_only=True)
        model = AutoModelForImageClassification.from_pretrained(repodir, local_files_only=True)
        model.eval()
        with torch.no_grad():
            for name, path in cases.items():
                img = Image.open(path)
                inputs = proc(images=img, return_tensors="pt")
                x = inputs["pixel_values"]
                logits = model(pixel_values=x).last_hidden_state
                np.save(OUT / f"paddle_orient_{tag}input_{name}.npy", x.detach().cpu().numpy())
                np.save(OUT / f"paddle_orient_{tag}output_{name}.npy", logits.detach().cpu().numpy())
                label = int(logits.argmax(-1).item())
                print(f"{repo} {name}: {path.name} {img.size} -> logits={logits.tolist()} label={label}")

                # Also a 180°-rotated copy to sanity-check the classifier.
                rot = img.rotate(180)
                xr = proc(images=rot, return_tensors="pt")["pixel_values"]
                lr = model(pixel_values=xr).last_hidden_state
                np.save(OUT / f"paddle_orient_{tag}input_{name}_rot.npy", xr.detach().cpu().numpy())
                np.save(OUT / f"paddle_orient_{tag}output_{name}_rot.npy", lr.detach().cpu().numpy())
                print(f"  rot180 -> logits={lr.tolist()} label={int(lr.argmax(-1).item())}")

    print(f"fixtures written to {OUT}")


if __name__ == "__main__":
    sys.exit(main())
