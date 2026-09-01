from __future__ import annotations

import json
from pathlib import Path

import pdfplumber


def probe_pdf_text(pdf_path: Path, first_n_pages: int) -> int:
    total = 0
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages[:first_n_pages]:
            total += len((page.extract_text() or "").strip())
    return total


def extract_pdf_to_json(pdf_path: Path, table_settings: dict | None = None) -> dict:
    table_settings = table_settings or {}
    pages_out: list[dict] = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            tables = page.extract_tables(table_settings) or []
            pages_out.append(
                {
                    "page_index": i,
                    "text": page.extract_text() or "",
                    "tables": tables,
                    "width": float(page.width),
                    "height": float(page.height),
                }
            )
    return {"source": str(pdf_path), "pages": pages_out}


def json_to_markdown(data: dict) -> str:
    parts: list[str] = []
    for page in data.get("pages", []):
        idx = page.get("page_index", 0)
        parts.append(f"## Page {idx + 1}\n")
        text = (page.get("text") or "").strip()
        if text:
            parts.append(text + "\n")
        for ti, table in enumerate(page.get("tables") or []):
            parts.append(f"\n### Table {ti + 1}\n")
            parts.append(_table_to_md(table))
        parts.append("\n---\n")
    return "\n".join(parts).strip()


def _table_to_md(table: list[list]) -> str:
    if not table:
        return ""
    rows = [[(c or "").replace("\n", " ").strip() for c in row] for row in table]
    if not rows:
        return ""
    header = rows[0]
    md = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for row in rows[1:]:
        if len(row) < len(header):
            row = row + [""] * (len(header) - len(row))
        md.append("| " + " | ".join(row[: len(header)]) + " |")
    return "\n".join(md)


def save_json_and_md(data: dict, out_dir: Path, stem: str) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{stem}.json"
    md_path = out_dir / f"{stem}.md"
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(json_to_markdown(data), encoding="utf-8")
    return json_path, md_path
