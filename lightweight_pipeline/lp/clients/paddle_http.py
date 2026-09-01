from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import requests

from lp.settings import ApiSection
from lp.types import BBox, LayoutRegion, TextLine


class ApiNotConfiguredError(RuntimeError):
    pass


class PaddleHttpClient:
    """可配置的 Paddle HTTP API 客户端；具体 URL/鉴权/响应解析由 YAML + 子类实现。"""

    def __init__(self, section: ApiSection, name: str):
        self.section = section
        self.name = name
        self.session = requests.Session()

    def _ensure(self) -> None:
        if not self.section.enabled:
            raise ApiNotConfiguredError(
                f"API [{self.name}] 未启用。请在 config.yaml 的 apis.{self.name} 中填写并设置 enabled: true"
            )
        if not self.section.base_url:
            raise ApiNotConfiguredError(
                f"API [{self.name}] base_url 为空。请按助手提问填写调用方式。"
            )

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json", **self.section.extra_headers}
        if self.section.token:
            h["Authorization"] = f"Bearer {self.section.token}"
        return h

    def _post_image(self, image_path: str | Path, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        self._ensure()
        path = Path(image_path)
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        payload: dict[str, Any] = {
            "image": b64,
            "image_base64": b64,
            "file": b64,
            **self.section.extra_params,
        }
        if extra:
            payload.update(extra)
        resp = self.session.post(
            self.section.base_url,
            headers=self._headers(),
            json=payload,
            timeout=self.section.timeout_sec,
        )
        resp.raise_for_status()
        try:
            return resp.json()
        except json.JSONDecodeError:
            return {"raw_text": resp.text}


class PaddleLayoutClient(PaddleHttpClient):
    """PP-DocLayoutV3 等价版面 API — 响应解析可在 extra_params 中配置或后续按用户提供的格式扩展。"""

    def detect(self, image_path: str | Path, *, min_score: float = 0.2) -> list[LayoutRegion]:
        data = self._post_image(image_path)
        return _parse_layout_regions(data, min_score=min_score)


class PaddleDetectionClient(PaddleHttpClient):
    def detect_text(
        self,
        image_path: str | Path,
        *,
        threshold: float = 0.5,
    ) -> list[BBox]:
        data = self._post_image(image_path, extra={"threshold": threshold, "det_thresh": threshold})
        return _parse_det_boxes(data)


class PaddleRecognitionClient(PaddleHttpClient):
    def recognize_crop(self, image_path: str | Path, bbox: BBox) -> str:
        x0, y0, x1, y1 = bbox.as_int_tuple()
        data = self._post_image(
            image_path,
            extra={
                "bbox": [x0, y0, x1, y1],
                "crop": [x0, y0, x1, y1],
            },
        )
        return _parse_rec_text(data)

    def recognize_boxes(self, image_path: str | Path, boxes: list[BBox]) -> list[TextLine]:
        lines: list[TextLine] = []
        for box in boxes:
            text = self.recognize_crop(image_path, box)
            if text.strip():
                lines.append(TextLine(text=text.strip(), bbox=box))
        return lines


def _parse_layout_regions(data: dict[str, Any], *, min_score: float) -> list[LayoutRegion]:
    """通用解析：兼容多种常见 JSON 结构，用户可提供真实格式后收紧。"""
    items: list[Any] = []
    if "regions" in data:
        items = data["regions"]
    elif "data" in data and isinstance(data["data"], list):
        items = data["data"]
    elif "result" in data and isinstance(data["result"], list):
        items = data["result"]
    elif "layout" in data and isinstance(data["layout"], list):
        items = data["layout"]

    regions: list[LayoutRegion] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        label = str(it.get("label") or it.get("type") or it.get("category") or "text")
        score = float(it.get("score") or it.get("confidence") or it.get("prob") or 0.0)
        if score < min_score:
            continue
        box = it.get("bbox") or it.get("box") or it.get("coordinate")
        if not box or len(box) < 4:
            continue
        x0, y0, x1, y1 = map(float, box[:4])
        regions.append(LayoutRegion(bbox=BBox(x0, y0, x1, y1, label=label, score=score), label=label, score=score))
    return regions


def _parse_det_boxes(data: dict[str, Any]) -> list[BBox]:
    items: list[Any] = []
    if "boxes" in data:
        items = data["boxes"]
    elif "data" in data and isinstance(data["data"], list):
        items = data["data"]
    elif "result" in data and isinstance(data["result"], list):
        items = data["result"]

    boxes: list[BBox] = []
    for it in items:
        if isinstance(it, (list, tuple)) and len(it) >= 4:
            x0, y0, x1, y1 = map(float, it[:4])
            boxes.append(BBox(x0, y0, x1, y1))
        elif isinstance(it, dict):
            box = it.get("bbox") or it.get("box") or it.get("points")
            if box and len(box) >= 4:
                x0, y0, x1, y1 = map(float, box[:4])
                boxes.append(BBox(x0, y0, x1, y1, score=float(it.get("score", 0))))
    return boxes


def _parse_rec_text(data: dict[str, Any]) -> str:
    if "text" in data:
        return str(data["text"])
    if "data" in data:
        d = data["data"]
        if isinstance(d, str):
            return d
        if isinstance(d, dict) and "text" in d:
            return str(d["text"])
        if isinstance(d, list) and d:
            first = d[0]
            if isinstance(first, dict) and "text" in first:
                return str(first["text"])
            if isinstance(first, str):
                return first
    if "result" in data:
        return str(data["result"])
    return ""
