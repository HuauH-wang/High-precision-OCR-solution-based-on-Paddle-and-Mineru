from __future__ import annotations

from typing import Any

from lp.types import BBox, LayoutRegion


def _extract_regions_from_item(item: dict[str, Any], *, min_score: float) -> list[LayoutRegion]:
    regions: list[LayoutRegion] = []
    candidates: list[Any] = []

    for key in (
        "layoutDetResults",
        "layout_det_res",
        "region_det_res",
        "parsing_res_list",
        "layout_boxes",
        "boxes",
    ):
        if key in item and item[key]:
            val = item[key]
            candidates.extend(val if isinstance(val, list) else [val])

    pruned = item.get("prunedResult")
    if isinstance(pruned, dict):
        for key in ("parsing_res_list", "layout_det_res", "boxes"):
            if key in pruned:
                val = pruned[key]
                candidates.extend(val if isinstance(val, list) else [val])
    elif isinstance(pruned, list):
        candidates.extend(pruned)

    for c in candidates:
        reg = _dict_to_region(c, min_score=min_score)
        if reg:
            regions.append(reg)
    return regions


def _deep_find_regions(obj: Any, *, min_score: float, depth: int = 0) -> list[LayoutRegion]:
    if depth > 12:
        return []
    found: list[LayoutRegion] = []
    if isinstance(obj, dict):
        reg = _dict_to_region(obj, min_score=min_score)
        if reg:
            found.append(reg)
        for v in obj.values():
            found.extend(_deep_find_regions(v, min_score=min_score, depth=depth + 1))
    elif isinstance(obj, list):
        for v in obj:
            found.extend(_deep_find_regions(v, min_score=min_score, depth=depth + 1))
    return found


def _dict_to_region(d: Any, *, min_score: float) -> LayoutRegion | None:
    if not isinstance(d, dict):
        return None
    label = (
        d.get("label")
        or d.get("block_label")
        or d.get("type")
        or d.get("category")
        or d.get("region_label")
    )
    box = (
        d.get("bbox")
        or d.get("block_bbox")
        or d.get("box")
        or d.get("coordinate")
        or d.get("poly")
    )
    if not label or not box:
        return None
    score = float(d.get("score") or d.get("confidence") or d.get("prob") or 1.0)
    if score < min_score:
        return None
    if isinstance(box[0], (list, tuple)):
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    else:
        if len(box) < 4:
            return None
        x0, y0, x1, y1 = map(float, box[:4])
    label_s = str(label).lower()
    if label_s in {"table", "table_body", "wired_table", "wireless_table"}:
        label_s = "table"
    return LayoutRegion(bbox=BBox(x0, y0, x1, y1, label=label_s, score=score), label=label_s, score=score)
