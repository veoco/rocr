#!/usr/bin/env python3
"""Generate numeric reference fixtures for the document-orientation model.

Uses the transformers engine (same safetensors weights) as the oracle.
Outputs to dev-models/reference/:
  paddle_docori_input_{tag}.npy    preprocessed input [1,3,224,224]
  paddle_docori_output_{tag}.npy   classification logits [1,4]
with tags: orig, rot90, rot180, rot270.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT / "dev-models" / "PP-LCNet_x1_0_doc_ori_safetensors"
OUT = ROOT / "dev-models" / "reference"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    proc = AutoImageProcessor.from_pretrained(REPO, local_files_only=True)
    model = AutoModelForImageClassification.from_pretrained(REPO, local_files_only=True)
    model.eval()

    src = Image.open(ROOT / "assets" / "doc.png")
    cases = {
        "orig": src,
        "rot90": src.rotate(90, expand=True),  # counter-clockwise 90°
        "rot180": src.rotate(180, expand=True),
        "rot270": src.rotate(270, expand=True),
    }
    with torch.no_grad():
        for tag, img in cases.items():
            x = proc(images=img, return_tensors="pt")["pixel_values"]
            logits = model(pixel_values=x).last_hidden_state
            np.save(OUT / f"paddle_docori_input_{tag}.npy", x.detach().cpu().numpy())
            np.save(OUT / f"paddle_docori_output_{tag}.npy", logits.detach().cpu().numpy())
            print(f"{tag}: {img.size} -> logits={logits.tolist()} label={int(logits.argmax(-1).item())}")

    print(f"fixtures written to {OUT}")


if __name__ == "__main__":
    sys.exit(main())
