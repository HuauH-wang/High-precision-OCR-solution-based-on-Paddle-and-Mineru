from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from lp.types import BBox, TextLine


def map_text_to_blank(
    src_size: tuple[int, int],
    lines: list[TextLine],
    dst: Path,
    *,
    background: tuple[int, int, int] = (255, 255, 255),
) -> Path:
    w, h = src_size
    img = Image.new("RGB", (w, h), color=background)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", size=14)
    except OSError:
        font = ImageFont.load_default()
    for line in lines:
        x0, y0, _, _ = line.bbox.as_int_tuple()
        draw.text((x0, max(0, y0)), line.text, fill=(0, 0, 0), font=font)
    dst.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst, format="PNG")
    return dst


def det_boxes_to_lines(image_path: Path, boxes: list[BBox], rec_fn) -> list[TextLine]:
    lines: list[TextLine] = []
    for box in boxes:
        text = rec_fn(image_path, box)
        if text.strip():
            lines.append(TextLine(text=text.strip(), bbox=box))
    return lines
