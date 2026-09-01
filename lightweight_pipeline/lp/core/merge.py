from __future__ import annotations

from lp.types import BBox, BlockResult, LayoutRegion, TextLine


def reading_order_key(bbox: BBox) -> tuple[float, float]:
    return (bbox.y0, bbox.x0)


def sort_regions(regions: list[LayoutRegion]) -> list[LayoutRegion]:
    return sorted(regions, key=lambda r: reading_order_key(r.bbox))


def sort_lines(lines: list[TextLine]) -> list[TextLine]:
    return sorted(lines, key=lambda l: reading_order_key(l.bbox))


def fill_nontable_blocks(
    layout: list[LayoutRegion],
    lines: list[TextLine],
    *,
    table_label: str,
    text_labels: list[str],
) -> list[BlockResult]:
    text_labels_set = set(text_labels)
    blocks: list[BlockResult] = []
    for reg in sort_regions(layout):
        if reg.label.lower() == table_label.lower():
            continue
        if text_labels_set and reg.label.lower() not in {t.lower() for t in text_labels_set}:
            continue
        picked = [ln for ln in sort_lines(lines) if _iou_or_center_in(ln.bbox, reg.bbox)]
        content = "\n".join(ln.text for ln in picked)
        if content.strip():
            blocks.append(BlockResult(bbox=reg.bbox, content=content.strip(), kind="text"))
    return blocks


def fill_nontable_text(
    layout: list[LayoutRegion],
    lines: list[TextLine],
    *,
    table_label: str,
    text_labels: list[str],
) -> str:
    return merge_blocks_to_markdown(
        fill_nontable_blocks(layout, lines, table_label=table_label, text_labels=text_labels)
    )


def fallback_markdown_block(md_text: str, page_bbox: BBox) -> BlockResult | None:
    if not md_text.strip():
        return None
    return BlockResult(bbox=page_bbox, content=md_text.strip(), kind="text")


def offset_bbox(inner: BBox, origin: BBox) -> BBox:
    return BBox(
        inner.x0 + origin.x0,
        inner.y0 + origin.y0,
        inner.x1 + origin.x0,
        inner.y1 + origin.y0,
        label=inner.label,
        score=inner.score,
    )


def sort_blocks(blocks: list[BlockResult]) -> list[BlockResult]:
    return sorted(blocks, key=lambda b: reading_order_key(b.bbox))


def merge_blocks_to_markdown(blocks: list[BlockResult]) -> str:
    ordered = sorted(blocks, key=lambda b: reading_order_key(b.bbox))
    parts: list[str] = []
    for b in ordered:
        if b.kind == "table":
            parts.append(b.content)
        else:
            parts.append(b.content)
    return "\n\n".join(parts).strip()


def merge_page_blocks(blocks: list[BlockResult]) -> str:
    return merge_blocks_to_markdown(blocks)


def _center_in(inner: BBox, outer: BBox) -> bool:
    cx = (inner.x0 + inner.x1) / 2
    cy = (inner.y0 + inner.y1) / 2
    return outer.x0 <= cx <= outer.x1 and outer.y0 <= cy <= outer.y1


def _iou_or_center_in(inner: BBox, outer: BBox) -> bool:
    if _center_in(inner, outer):
        return True
    ix0, iy0 = max(inner.x0, outer.x0), max(inner.y0, outer.y0)
    ix1, iy1 = min(inner.x1, outer.x1), min(inner.y1, outer.y1)
    if ix1 <= ix0 or iy1 <= iy0:
        return False
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_i = inner.width * inner.height or 1.0
    return inter / area_i > 0.3
