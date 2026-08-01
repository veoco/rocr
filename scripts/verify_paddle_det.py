#!/usr/bin/env python
"""Generate the detection reference fixtures using PaddleOCR's transformers engine.

The official `*_det_onnx` export uses a different checkpoint from the
`*_det_safetensors` model, so it cannot serve as a numeric oracle. Instead, the
PaddleOCR transformers engine loads the SAME safetensors weights and is used to
produce the reference outputs stored under `dev-models/reference/paddle_det_*`.

Requirements: pip install paddleocr transformers torch torchvision
"""
import os
import sys

import numpy as np

os.environ.setdefault("HF_HOME", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".hf-cache"))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.path.join(ROOT, "dev-models", "reference")
TIER = sys.argv[1] if len(sys.argv) > 1 else "small"
IMAGE = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "assets", "doc.png")


def main():
    import warnings

    warnings.filterwarnings("ignore")
    from PIL import Image
    from paddleocr import TextDetection

    os.makedirs(REF, exist_ok=True)
    model_dir = os.path.join(ROOT, "dev-models", f"PP-OCRv6_{TIER}_det_safetensors")
    model = TextDetection(
        model_name=f"PP-OCRv6_{TIER}_det",
        model_dir=model_dir,
        engine="transformers",
    )
    p = model.paddlex_predictor
    img = np.array(Image.open(IMAGE).convert("RGB"))
    pix = p.preprocess_images([img])
    pv = pix["pixel_values"]
    out = p.infer(pv)
    suffix = "" if TIER == "small" else f"_{TIER}"
    np.save(os.path.join(REF, f"paddle_det_input{suffix}.npy"), np.array(pv))
    np.save(
        os.path.join(REF, f"paddle_det_last_hidden_state{suffix}.npy"),
        out.last_hidden_state.detach().numpy(),
    )
    print("saved paddle_det fixtures for tier", TIER)


if __name__ == "__main__":
    main()
