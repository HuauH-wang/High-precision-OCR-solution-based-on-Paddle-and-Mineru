from __future__ import annotations

import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from lp.clients.mineru_client import MinerUClientWrapper
from lp.core.entry import normalize_input
from lp.core.md_unify import (
    assemble_document_markdown,
    assemble_page_markdown,
    from_mineru_zip_markdown,
    merge_blocks_markdown,
)
from lp.core.batch_utils import write_failed_csv
from lp.core.merge import merge_page_blocks, sort_blocks
from lp.core.pdf_md import extract_pdf_to_json, probe_pdf_text, save_json_and_md
from lp.core.rounds import LightweightOcrEngine
from lp.settings import Settings
from lp.types import BlockResult, PageResult


class LightweightBatchPipeline:
    def __init__(self, settings: Settings):
        self.s = settings
        self.engine = LightweightOcrEngine(settings)
        self.mineru = MinerUClientWrapper(settings.mineru, settings.batch)
        out_cfg = settings.raw.get("output", {})
        self.output_format = str(out_cfg.get("format", "markdown")).lower()
        self.unified_md = bool(out_cfg.get("unified_markdown", True))
        self.include_frontmatter = bool(out_cfg.get("include_frontmatter", True))
        self.write_doc_md = bool(out_cfg.get("write_document_md", True))
        self.keep_json = bool(out_cfg.get("keep_json_intermediate", False))
        self.s.output_dir.mkdir(parents=True, exist_ok=True)
        self.s.work_dir.mkdir(parents=True, exist_ok=True)

    def run_files(self, inputs: list[Path]) -> list[PageResult | dict]:
        results: list[PageResult | dict] = []
        max_docs = int(self.s.batch.get("max_parallel_docs", 4))
        with ThreadPoolExecutor(max_workers=max_docs) as ex:
            futs = {ex.submit(self._process_document, p): p for p in inputs}
            for fut in as_completed(futs):
                results.extend(fut.result())
        return results

    def _process_document(self, path: Path) -> list[PageResult | dict]:
        doc_id = path.stem
        ext = path.suffix.lower()

        if ext == ".pdf":
            n = int(self.s.entry.get("pdf_probe_pages", 3))
            th = int(self.s.entry.get("pdf_text_threshold", 50))
            if probe_pdf_text(path, n) >= th:
                data = extract_pdf_to_json(path, self.s.pdfplumber.get("table_settings"))
                from lp.core.pdf_md import json_to_markdown

                md_body = json_to_markdown(data)
                out_dir = self.s.output_dir / doc_id
                out_dir.mkdir(parents=True, exist_ok=True)
                md_path = out_dir / f"{doc_id}.md"
                full_md = assemble_page_markdown(
                    doc_id=doc_id,
                    page_index=0,
                    body=md_body,
                    parse_path="pdfplumber",
                    source="text_layer_pdf",
                    include_frontmatter=self.include_frontmatter,
                )
                md_path.write_text(full_md, encoding="utf-8")
                if self.keep_json:
                    save_json_and_md(data, out_dir, doc_id)
                return [
                    {
                        "doc_id": doc_id,
                        "type": "text_layer",
                        "format": "markdown",
                        "markdown_path": str(md_path),
                    }
                ]

        zoom = float(self.s.entry.get("pymupdf_zoom", 2.0))
        heif_bin = str(self.s.entry.get("heif_converter_bin", "heif-converter"))
        if ext == ".pdf":
            pages = normalize_input(path, self.s.work_dir, heif_bin=heif_bin, zoom=zoom)
        else:
            pages = normalize_input(path, self.s.work_dir / doc_id, heif_bin=heif_bin, zoom=zoom)

        return self._process_scan_pages(doc_id, pages)

    def _process_scan_pages(self, doc_id: str, pages: list[Path]) -> list[PageResult]:
        parse_path = self.s.parse_path
        max_pages = int(self.s.batch.get("max_parallel_pages", 8))

        page_blocks: dict[int, list[BlockResult]] = {}
        all_tasks: list = []
        failed_rows: list[dict] = []

        def _one(page_index: int, page_path: Path):
            try:
                _ctx, tasks, blocks = self.engine.process_page(
                    doc_id, page_index, page_path, self.s.work_dir, parse_path
                )
                return page_index, tasks, blocks, None
            except Exception as e:
                return page_index, [], [], str(e)

        with ThreadPoolExecutor(max_workers=max_pages) as ex:
            futs = [ex.submit(_one, i, p) for i, p in enumerate(pages)]
            for fut in as_completed(futs):
                page_index, tasks, blocks, err = fut.result()
                if err:
                    failed_rows.append(
                        {
                            "doc_id": doc_id,
                            "page_index": page_index,
                            "stage": "ocr_rounds",
                            "error": err,
                        }
                    )
                    continue
                all_tasks.extend(tasks)
                page_blocks[page_index] = blocks

        mineru_md: dict[str, str] = {}
        mineru_raw: dict[str, dict] = {}
        if all_tasks:
            try:
                mineru_raw = self.mineru.batch_upload_and_wait(all_tasks)
            except Exception as e:
                for t in all_tasks:
                    failed_rows.append(
                        {
                            "doc_id": doc_id,
                            "artifact_id": t.artifact_id,
                            "page_index": t.page_index,
                            "stage": "mineru_batch",
                            "error": str(e),
                        }
                    )
            for t in all_tasks:
                item = mineru_raw.get(t.artifact_id, {})
                md = ""
                if item.get("state") == "done":
                    md = from_mineru_zip_markdown(_fetch_mineru_markdown(item))
                mineru_md[t.artifact_id] = md
                if not md:
                    failed_rows.append(
                        {
                            "doc_id": doc_id,
                            "artifact_id": t.artifact_id,
                            "page_index": t.page_index,
                            "role": t.role,
                            "stage": "mineru_result",
                            "error": item.get("err_msg") or item.get("state") or "empty markdown",
                        }
                    )

        out_pages: list[PageResult] = []
        page_mds: list[str] = []
        for page_index in sorted(page_blocks.keys()):
            blocks = list(page_blocks[page_index])
            for t in all_tasks:
                if t.page_index != page_index:
                    continue
                md = mineru_md.get(t.artifact_id, "")
                if md and t.table_bbox is not None:
                    blocks.append(
                        BlockResult(
                            bbox=t.table_bbox,
                            content=md,
                            kind="table",
                        )
                    )
            blocks = sort_blocks(blocks)
            body = (
                merge_blocks_markdown(blocks)
                if self.unified_md
                else merge_page_blocks(blocks)
            )
            full_md = assemble_page_markdown(
                doc_id=doc_id,
                page_index=page_index,
                body=body,
                parse_path=parse_path,
                source="scan",
                include_frontmatter=self.include_frontmatter,
            )
            out_path = self.s.output_dir / doc_id / f"page_{page_index:04d}.md"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(full_md, encoding="utf-8")
            page_mds.append(full_md)
            out_pages.append(
                PageResult(doc_id=doc_id, page_index=page_index, blocks=blocks, markdown=full_md)
            )

        if self.write_doc_md and page_mds:
            doc_md = assemble_document_markdown(
                doc_id=doc_id,
                page_markdowns=page_mds,
                include_frontmatter=self.include_frontmatter,
            )
            doc_path = self.s.output_dir / doc_id / f"{doc_id}.md"
            doc_path.write_text(doc_md, encoding="utf-8")

        if failed_rows:
            write_failed_csv(self.s.output_dir / "failed_tasks.csv", failed_rows)

        return out_pages


def _fetch_mineru_markdown(item: dict) -> str:
    if item.get("state") != "done":
        return ""
    zip_url = item.get("full_zip_url")
    if not zip_url:
        return ""
    tmp_path = Path(tempfile.mkstemp(suffix=".zip")[1])
    try:
        r = requests.get(zip_url, timeout=120)
        r.raise_for_status()
        tmp_path.write_bytes(r.content)
        return MinerUClientWrapper.extract_markdown_from_zip(tmp_path)
    except Exception:
        return ""
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
