#!/usr/bin/env python
"""Generate an end-to-end golden reference with PaddleOCR's transformers engine.

Runs the official engine on a page image and saves the recognized text lines
(sorted) so `crates/rocr/tests/oracle.rs` can compare rocr's pipeline output
against it. This is a soft check (text lines, not polygons / confidence).

Outputs to dev-models/reference/golden_<tier>_<name>.txt

Requirements: pip install paddleocr transformers torch torchvision
"""
import os
import sys

os.environ.setdefault("HF_HOME", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".hf-cache"))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.path.join(ROOT, "dev-models", "reference")
TIER = sys.argv[1] if len(sys.argv) > 1 else "small"
IMAGE = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "assets", "doc.png")


def main():
    import warnings

    warnings.filterwarnings("ignore")
    from paddleocr import PaddleOCR

    os.makedirs(REF, exist_ok=True)
    ocr = PaddleOCR(
        text_detection_model_name=f"PP-OCRv6_{TIER}_det",
        text_recognition_model_name=f"PP-OCRv6_{TIER}_rec",
        lang="ch",
        engine="transformers",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=True,
    )
    result = ocr.predict(IMAGE)
    lines = sorted(t for r in result for t in r.get("rec_texts") or [])
    name = os.path.splitext(os.path.basename(IMAGE))[0]
    out = os.path.join(REF, f"golden_{TIER}_{name}.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"golden -> {out}")
    for line in lines:
        print(" ", line)


if __name__ == "__main__":
    main()
