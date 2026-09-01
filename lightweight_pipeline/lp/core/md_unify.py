from __future__ import annotations

from typing import Any

from lp.types import BlockResult


def normalize_api_markdown(text: str) -> str:
    """各 API 返回的 markdown 片段清洗后合并用。"""
    if not text:
        return ""
    t = text.replace("\r\n", "\n").strip()
    while t.endswith("\n\n\n"):
        t = t[:-1]
    return t


def from_aistudio_item(item: dict[str, Any]) -> str:
    """AI Studio layoutParsingResults 单条 → Markdown。"""
    md = item.get("markdown") or {}
    if isinstance(md, dict) and md.get("text"):
        return normalize_api_markdown(str(md["text"]))
    return ""


def from_aistudio_records(records: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for rec in records:
        result = rec.get("result") or rec
        for item in result.get("layoutParsingResults") or []:
            part = from_aistudio_item(item)
            if part:
                parts.append(part)
    return normalize_api_markdown("\n\n".join(parts))


def from_mineru_zip_markdown(md: str) -> str:
    return normalize_api_markdown(md)


def from_pdfplumber_markdown(md: str) -> str:
    return normalize_api_markdown(md)


def block_to_markdown(block: BlockResult, *, table_index: int | None = None) -> str:
    body = normalize_api_markdown(block.content)
    if not body:
        return ""
    if block.kind == "table":
        idx = f" {table_index}" if table_index is not None else ""
        return f"### 表格{idx}\n\n{body}"
    if block.kind == "text":
        return f"### 正文\n\n{body}"
    return body


def merge_blocks_markdown(blocks: list[BlockResult]) -> str:
    """按 block 顺序（调用方已按 bbox 排序）合并为统一 Markdown。"""
    parts: list[str] = []
    table_i = 0
    for b in blocks:
        if b.kind == "table":
            table_i += 1
            part = block_to_markdown(b, table_index=table_i)
        else:
            part = block_to_markdown(b)
        if part:
            parts.append(part)
    return normalize_api_markdown("\n\n---\n\n".join(parts))


def assemble_page_markdown(
    *,
    doc_id: str,
    page_index: int,
    body: str,
    parse_path: str = "",
    source: str = "scan",
    include_frontmatter: bool = True,
) -> str:
    """单页统一 Markdown 文档（含可选 YAML front matter）。"""
    body = normalize_api_markdown(body)
    if not include_frontmatter:
        header = f"## 第 {page_index + 1} 页\n\n"
        return normalize_api_markdown(header + body)

    fm = (
        "---\n"
        f'doc_id: "{_escape_yaml(doc_id)}"\n'
        f"page: {page_index + 1}\n"
        f'source: "{source}"\n'
        f'parse_path: "{parse_path}"\n'
        "format: markdown\n"
        "---\n\n"
    )
    return normalize_api_markdown(fm + f"## 第 {page_index + 1} 页\n\n{body}")


def assemble_document_markdown(
    *,
    doc_id: str,
    page_markdowns: list[str],
    include_frontmatter: bool = True,
) -> str:
    """多页合并为单一 Markdown 文件。"""
    pages = [normalize_api_markdown(p) for p in page_markdowns if p.strip()]
    if not pages:
        return ""
    if include_frontmatter:
        head = (
            "---\n"
            f'doc_id: "{_escape_yaml(doc_id)}"\n'
            f"pages: {len(pages)}\n"
            "format: markdown\n"
            "---\n\n"
            f"# {doc_id}\n\n"
        )
    else:
        head = f"# {doc_id}\n\n"
    return normalize_api_markdown(head + "\n\n---\n\n".join(pages))


def _escape_yaml(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')
