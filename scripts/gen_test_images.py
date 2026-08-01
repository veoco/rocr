#!/usr/bin/env python
"""Generate reproducible test images for rocr development and tests.

Outputs (written under the given output dir):
  doc.png        — a full-page document with several English + Chinese lines
  line_en.png    — a single English text line crop
  line_cn.png    — a single Chinese text line crop
  line_long.png  — a long English text line crop (wider than 320px after resize)
"""
import sys

from PIL import Image, ImageDraw, ImageFont

FONT_EN = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_CJK = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"


def cjk_font(size):
    # NotoSansCJK .ttc index: 0=JP, 1=KR, 2=SC, 3=TC, 4=HK (varies); try SC then fall back.
    for idx in (2, 0):
        try:
            return ImageFont.truetype(FONT_CJK, size, index=idx)
        except Exception:
            continue
    return ImageFont.truetype(FONT_EN, size)


def render_doc(path):
    W, H = 900, 700
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    y = 40
    f_en = ImageFont.truetype(FONT_EN, 34)
    f_cn = cjk_font(34)
    lines = [
        ("Hello World — rocr OCR", f_en, (0, 0, 0)),
        ("你好，世界", f_cn, (0, 0, 0)),
        ("rocr is a pure Rust OCR library", f_en, (0, 0, 0)),
        ("深度学习文本识别技术", f_cn, (0, 0, 0)),
        ("PP-OCRv6 supports 50 languages", f_en, (0, 0, 0)),
        ("Rust 与 PaddleOCR 的完美结合", f_cn, (0, 0, 0)),
    ]
    for text, font, color in lines:
        d.text((30, y), text, font=font, fill=color)
        y += 80
    img.save(path)


def render_line(path, text, font_resolver, font_size, canvas_w):
    img = Image.new("RGB", (canvas_w, 64), (255, 255, 255))
    d = ImageDraw.Draw(img)
    f = font_resolver(font_size)
    bbox = d.textbbox((0, 0), text, font=f)
    d.text((4 - bbox[0], 4 - bbox[1]), text, fill=(0, 0, 0), font=f)
    img.save(path)


def main(outdir):
    render_doc(f"{outdir}/doc.png")
    render_line(f"{outdir}/line_en.png", "Hello World OCR",
                lambda s: ImageFont.truetype(FONT_EN, s), 40, 480)
    render_line(f"{outdir}/line_cn.png", "你好世界", cjk_font, 40, 320)
    render_line(f"{outdir}/line_long.png", "hello world hello world hello",
                lambda s: ImageFont.truetype(FONT_EN, s), 40, 700)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "assets")
