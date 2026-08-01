#!/usr/bin/env python
"""ONNX oracle tool for rocr development.

Loads the official PP-OCRv6 ONNX models, applies the reference preprocessing
(confirmed empirically against the PaddleOCR transformers pipeline), and saves
numeric fixtures to dev-models/reference/:

  rec_input.npy   — the preprocessed rec input tensor  (N,3,48,W)
  rec_logits.npy  — the official ONNX rec output        (N,T,18710)
  det_input.npy   — the preprocessed det input tensor   (N,3,H,W)
  det_output.npy  — the official ONNX det output        (N,1,H',W')

The Rust oracle tests assert that rocr's candle output matches these fixtures
within a small tolerance, and that rocr's preprocessing reproduces the input
tensors exactly.

Requires: pip install onnxruntime numpy pillow
"""
import argparse
import os
import sys

import numpy as np

try:
    import cv2
    import onnxruntime as ort
    from PIL import Image
except ImportError as e:  # pragma: no cover
    sys.exit(f"missing dependency: {e} (pip install onnxruntime numpy pillow opencv-python-headless)")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "dev-models")
REF = os.path.join(MODELS, "reference")

# PP-OCRv6 rec preprocessing (transformers engine).
REC_HEIGHT = 48
REC_MAX_WIDTH = 3200
REC_MEAN = [0.5, 0.5, 0.5]
REC_STD = [0.5, 0.5, 0.5]

# PP-OCRv6 det preprocessing.
DET_LIMIT_SIDE_LEN = 736
DET_LIMIT_TYPE = "min"
DET_MAX_SIDE = 4000
DET_MEAN = [0.406, 0.456, 0.485]  # BGR order
DET_STD = [0.225, 0.224, 0.229]


def cv2_resize_rgb(img_rgb, w, h):
    """cv2.resize INTER_LINEAR on a PIL RGB image, returned as RGB ndarray."""
    arr = np.asarray(img_rgb)  # HWC RGB uint8
    return cv2.resize(arr, (w, h), interpolation=cv2.INTER_LINEAR)


def rec_preprocess(img_rgb):
    """Resize to height 48 (aspect-preserving), RGB, normalize to [-1,1]."""
    w, h = img_rgb.size
    tw = int(round(REC_HEIGHT * w / h))
    if tw > REC_MAX_WIDTH:
        tw = REC_MAX_WIDTH
    img = cv2_resize_rgb(img_rgb, tw, REC_HEIGHT)
    x = img.astype(np.float32) / 255.0
    x = (x - np.asarray(REC_MEAN, np.float32)) / np.asarray(REC_STD, np.float32)
    return x.transpose(2, 0, 1)[None].astype(np.float32)  # N,C,H,W


def det_preprocess(img_rgb):
    """DetResizeForTest (limit_side_len=736, min) + BGR normalize + CHW."""
    w0, h0 = img_rgb.size
    ratio = 1.0
    if min(h0, w0) < DET_LIMIT_SIDE_LEN:
        ratio = float(DET_LIMIT_SIDE_LEN) / min(h0, w0)
    if max(h0, w0) * ratio > DET_MAX_SIDE:
        ratio = float(DET_MAX_SIDE) / max(h0, w0)
    new_w = int(round(w0 * ratio))
    new_h = int(round(h0 * ratio))
    # round to a multiple of 32 (det network requirement)
    new_w = max(32, int(round(new_w / 32)) * 32)
    new_h = max(32, int(round(new_h / 32)) * 32)
    img = cv2_resize_rgb(img_rgb, new_w, new_h)
    x = img.astype(np.float32)[:, :, ::-1] / 255.0  # RGB -> BGR
    x = (x - np.asarray(DET_MEAN, np.float32)) / np.asarray(DET_STD, np.float32)
    return x.transpose(2, 0, 1)[None].astype(np.float32), (new_h, new_w), (h0, w0)


def run_onnx(repo, x):
    sess = ort.InferenceSession(
        os.path.join(MODELS, repo, "inference.onnx"),
        providers=["CPUExecutionProvider"],
    )
    return sess.run(None, {sess.get_inputs()[0].name: x})[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tier", default="small", choices=["tiny", "small", "medium"])
    ap.add_argument("--rec", action="store_true", help="generate rec fixtures")
    ap.add_argument("--det", action="store_true", help="generate det fixtures")
    ap.add_argument("--image-dir", default=os.path.join(ROOT, "assets"))
    args = ap.parse_args()
    os.makedirs(REF, exist_ok=True)

    tag = args.tier
    if args.rec:
        rec = os.path.join(args.image_dir, "line_long.png")
        img = Image.open(rec).convert("RGB")
        x = rec_preprocess(img)
        logits = run_onnx(f"PP-OCRv6_{tag}_rec_onnx", x)
        np.save(os.path.join(REF, f"rec_input_{tag}.npy"), x)
        np.save(os.path.join(REF, f"rec_logits_{tag}.npy"), logits)
        print(f"rec: input {x.shape} -> logits {logits.shape}")

    if args.det:
        doc = os.path.join(args.image_dir, "doc.png")
        img = Image.open(doc).convert("RGB")
        x, new_shape, orig = det_preprocess(img)
        out = run_onnx(f"PP-OCRv6_{tag}_det_onnx", x)
        np.save(os.path.join(REF, f"det_input_{tag}.npy"), x)
        np.save(os.path.join(REF, f"det_output_{tag}.npy"), out)
        print(f"det: input {x.shape} -> probmap {out.shape} (resized {new_shape} from {orig})")


if __name__ == "__main__":
    main()
