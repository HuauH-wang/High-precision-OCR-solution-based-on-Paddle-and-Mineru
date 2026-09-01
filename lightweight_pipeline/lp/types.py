from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BBox:
    x0: float
    y0: float
    x1: float
    y1: float
    label: str = ""
    score: float = 0.0

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    def as_int_tuple(self) -> tuple[int, int, int, int]:
        return int(self.x0), int(self.y0), int(self.x1), int(self.y1)


@dataclass
class TextLine:
    text: str
    bbox: BBox
    score: float = 0.0


@dataclass
class LayoutRegion:
    bbox: BBox
    label: str
    score: float


@dataclass
class PageContext:
    doc_id: str
    page_index: int
    image_path: str
    width: int
    height: int
    layout_regions: list[LayoutRegion] = field(default_factory=list)
    text_lines: list[TextLine] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)


@dataclass
class MinerUTaskMeta:
    artifact_id: str
    doc_id: str
    page_index: int
    local_path: str
    role: str
    table_index: int = 0
    table_bbox: BBox | None = None


@dataclass
class BlockResult:
    bbox: BBox
    content: str
    kind: str


@dataclass
class PageResult:
    doc_id: str
    page_index: int
    blocks: list[BlockResult]
    markdown: str = ""
