from __future__ import annotations

import hashlib
import uuid
from pathlib import Path

from lp.clients.layout_factory import create_layout_client, create_spotting_client
from lp.clients.paddle_http import (
    PaddleDetectionClient,
    PaddleRecognitionClient,
)
from lp.core.blank_map import det_boxes_to_lines, map_text_to_blank
from lp.core.image_utils import crop_image, load_image_size, pad_image_ratio
from lp.core.merge import (
    BlockResult,
    fallback_markdown_block,
    fill_nontable_blocks,
    offset_bbox,
    sort_blocks,
)
from lp.settings import Settings
from lp.types import BBox, LayoutRegion, MinerUTaskMeta, PageContext, TextLine


class LightweightOcrEngine:
    def __init__(self, settings: Settings):
        self.s = settings
        self.layout_client = create_layout_client(settings)
        self.spotting_client = create_spotting_client(settings)
        self.det = PaddleDetectionClient(settings.paddle_detection, "paddle_detection")
        self.rec = PaddleRecognitionClient(settings.paddle_recognition, "paddle_recognition")
        self.table_label = str(settings.ocr.get("table_label", "table"))
        self.text_labels = list(settings.ocr.get("text_labels", ["text"]))
        self.min_layout_score = settings.layout_min_score()
        self.pad_ratio = float(settings.ocr.get("round2_padding_ratio", 0.05))

    def det_rec_lines(self, image_path: Path) -> list[TextLine]:
        if self.spotting_client:
            res = self.spotting_client.spot(image_path)
            if res.lines:
                return res.lines
            if res.markdown_text:
                return [TextLine(text=res.markdown_text, bbox=BBox(0, 0, 100, 100))]
        th = self.s.det_threshold()
        boxes = self.det.detect_text(image_path, threshold=th)
        return det_boxes_to_lines(image_path, boxes, self.rec.recognize_crop)

    def spotting_subimage(self, image_path: Path, blank_out: Path) -> tuple[list[TextLine], str]:
        """spotting → 空白映射子图；返回 (lines, markdown_text)。"""
        w, h = load_image_size(image_path)
        md_text = ""
        if self.spotting_client:
            res = self.spotting_client.spot(image_path, save_blank_to=blank_out)
            md_text = res.markdown_text
            if res.blank_image_saved and res.blank_image_saved.exists():
                return res.lines, md_text
            map_text_to_blank((w, h), res.lines, blank_out)
            return res.lines, md_text
        lines = self.det_rec_lines(image_path)
        map_text_to_blank((w, h), lines, blank_out)
        return lines, md_text

    def layout(self, image_path: Path) -> list[LayoutRegion]:
        return self.layout_client.detect(image_path, min_score=self.min_layout_score)

    def _nontable_blocks(
        self,
        layout: list[LayoutRegion],
        lines: list[TextLine],
        *,
        spot_md: str,
        fallback_bbox: BBox,
    ) -> list[BlockResult]:
        blocks = fill_nontable_blocks(
            layout,
            lines,
            table_label=self.table_label,
            text_labels=self.text_labels,
        )
        if not blocks:
            fb = fallback_markdown_block(spot_md, fallback_bbox)
            if fb:
                blocks = [fb]
        return blocks

    def process_page(
        self,
        doc_id: str,
        page_index: int,
        image_path: Path,
        work_dir: Path,
        parse_path: str,
    ) -> tuple[PageContext, list[MinerUTaskMeta], list[BlockResult]]:
        w, h = load_image_size(image_path)
        page_bbox = BBox(0, 0, w, h)
        ctx = PageContext(
            doc_id=doc_id,
            page_index=page_index,
            image_path=str(image_path),
            width=w,
            height=h,
        )
        page_work = work_dir / doc_id / f"page_{page_index:04d}"
        page_work.mkdir(parents=True, exist_ok=True)

        mineru_tasks: list[MinerUTaskMeta] = []
        blocks: list[BlockResult] = []

        if parse_path == "P0":
            aid = _artifact_id(doc_id, page_index, "p0")
            mineru_tasks.append(
                MinerUTaskMeta(
                    artifact_id=aid,
                    doc_id=doc_id,
                    page_index=page_index,
                    local_path=str(image_path),
                    role="p0_page",
                    table_bbox=page_bbox,
                )
            )
            return ctx, mineru_tasks, blocks

        # --- 零轮 ---
        layout0 = self.layout(image_path)
        ctx.layout_regions = layout0
        z_spot = page_work / "zero_spotting.png"
        lines0, spot_md0 = self.spotting_subimage(image_path, z_spot)
        ctx.text_lines = lines0
        ctx.artifacts["zero_spotting"] = str(z_spot)

        zero_blocks = self._nontable_blocks(layout0, lines0, spot_md=spot_md0, fallback_bbox=page_bbox)
        ctx.artifacts["zero_nontable_md"] = "\n\n".join(b.content for b in zero_blocks)

        table_regions = [r for r in layout0 if r.label.lower() == self.table_label.lower()]
        round1_tables: list[tuple[int, Path, BBox]] = []
        for ti, reg in enumerate(table_regions):
            dst = page_work / f"round1_table_{ti:02d}.png"
            crop_image(image_path, reg.bbox.as_int_tuple(), dst)
            round1_tables.append((ti, dst, reg.bbox))

        round1_nontable_blocks: list[BlockResult] = []
        round2_tasks_meta: list[tuple[int, Path, BBox, BBox, str]] = []

        for ti, r1_path, r1_bbox in round1_tables:
            r1_spot = page_work / f"round1_spotting_{ti:02d}.png"
            lines1, spot_md1 = self.spotting_subimage(r1_path, r1_spot)

            layout1 = self.layout(r1_spot)
            part_blocks = self._nontable_blocks(
                layout1, lines1, spot_md=spot_md1, fallback_bbox=r1_bbox
            )
            for b in part_blocks:
                round1_nontable_blocks.append(
                    BlockResult(
                        bbox=offset_bbox(b.bbox, r1_bbox),
                        content=b.content,
                        kind=b.kind,
                    )
                )

            sub_tables = [r for r in layout1 if r.label.lower() == self.table_label.lower()]
            for si, sub in enumerate(sub_tables):
                r2 = page_work / f"round2_table_{ti:02d}_{si:02d}.png"
                crop_image(r1_path, sub.bbox.as_int_tuple(), r2)
                round2_tasks_meta.append((ti, r2, sub.bbox, r1_bbox, "p2_table"))

        ctx.artifacts["round1_nontable_md"] = "\n\n".join(b.content for b in round1_nontable_blocks)

        if parse_path == "P1":
            for ti, r1_path, r1_bbox in round1_tables:
                aid = _artifact_id(doc_id, page_index, f"p1_t{ti}")
                mineru_tasks.append(
                    MinerUTaskMeta(
                        artifact_id=aid,
                        doc_id=doc_id,
                        page_index=page_index,
                        local_path=str(r1_path),
                        role="p1_table",
                        table_index=ti,
                        table_bbox=r1_bbox,
                    )
                )
            blocks = sort_blocks(list(zero_blocks))
            return ctx, mineru_tasks, blocks

        if parse_path in {"P2-M1", "P2-M2"}:
            for ti, r2_path, r2_bbox, r1_bbox, _ in round2_tasks_meta:
                page_table_bbox = offset_bbox(r2_bbox, r1_bbox)
                if parse_path == "P2-M1":
                    r2_spot = page_work / f"round2_spotting_{ti:02d}.png"
                    self.spotting_subimage(r2_path, r2_spot)
                    r2_exp = page_work / f"round2_spotting_expand_{ti:02d}.png"
                    pad_image_ratio(r2_spot, self.pad_ratio, r2_exp)
                    upload_path = r2_exp
                    role = "p2_spot_expand"
                else:
                    upload_path = r2_path
                    role = "p2_table"
                aid = _artifact_id(doc_id, page_index, f"{role}_{ti}")
                mineru_tasks.append(
                    MinerUTaskMeta(
                        artifact_id=aid,
                        doc_id=doc_id,
                        page_index=page_index,
                        local_path=str(upload_path),
                        role=role,
                        table_index=ti,
                        table_bbox=page_table_bbox,
                    )
                )
            blocks = sort_blocks(list(zero_blocks) + list(round1_nontable_blocks))
            return ctx, mineru_tasks, blocks

        raise ValueError(f"unknown parse_path: {parse_path}")


def _artifact_id(doc_id: str, page_index: int, suffix: str) -> str:
    """MinerU data_id 上限 128 字符，使用短 hash。"""
    h = hashlib.sha1(doc_id.encode("utf-8")).hexdigest()[:10]
    return f"{h}_p{page_index:03d}_{suffix}_{uuid.uuid4().hex[:8]}"
