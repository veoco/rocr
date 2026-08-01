#!/usr/bin/env python
"""Run the OFFICIAL PaddleOCR pipeline over every example image from the
PaddleOCR repo (docs/images + tests/test_files) and record per-image text and
wall-clock timing.

Outputs JSON lines to stdout:
  {"image": <rel path>, "texts": [...], "time_s": <float>}

Environment:
  HF_HOME  set to the repo's .hf-cache by default (model download cache).
  THREADS  if set, torch.set_num_threads(<n>); default 0 = torch default.

Usage:
  python scripts/bench_examples_official.py <images_dir> [tier]
"""
import json
import os
import sys
import time

os.environ.setdefault("HF_HOME", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".hf-cache"))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

THREADS = int(os.environ.get("THREADS", "0"))


def main():
    import warnings

    warnings.filterwarnings("ignore")
    if THREADS > 0:
        import torch

        torch.set_num_threads(THREADS)
        print(f"torch threads -> {THREADS}", file=sys.stderr)

    images_dir = sys.argv[1]
    tier = sys.argv[2] if len(sys.argv) > 2 else "small"

    from paddleocr import PaddleOCR

    ocr = PaddleOCR(
        text_detection_model_name=f"PP-OCRv6_{tier}_det",
        text_recognition_model_name=f"PP-OCRv6_{tier}_rec",
        lang="ch",
        engine="transformers",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=True,
    )

    # Model load happens lazily on the first predict; warm it up so per-image
    # timings exclude one-off model-loading cost.
    t0 = time.perf_counter()
    ocr.predict(os.path.join(images_dir, sorted(os.listdir(images_dir))[0]))
    print(f"warmup done in {time.perf_counter() - t0:.2f}s", file=sys.stderr)

    for name in sorted(os.listdir(images_dir)):
        path = os.path.join(images_dir, name)
        if not name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        t0 = time.perf_counter()
        try:
            result = ocr.predict(path)
            texts = sorted(t for r in result for t in r.get("rec_texts") or [])
            err = None
        except Exception as e:  # noqa: BLE001 - keep going over the whole set
            texts, err = [], str(e)
        dt = time.perf_counter() - t0
        out = {"image": f"{os.path.basename(images_dir)}/{name}", "texts": texts, "time_s": round(dt, 4)}
        if err:
            out["error"] = err
        print(json.dumps(out, ensure_ascii=False))
        sys.stdout.flush()


if __name__ == "__main__":
    main()
