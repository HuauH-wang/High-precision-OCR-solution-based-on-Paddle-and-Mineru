from __future__ import annotations

import fitz
from pathlib import Path

from lp.core.image_utils import ensure_png


def rasterize_pdf(pdf_path: Path, out_dir: Path, zoom: float = 2.0) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    matrix = fitz.Matrix(zoom, zoom)
    paths: list[Path] = []
    for i in range(len(doc)):
        page = doc.load_page(i)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        p = out_dir / f"page_{i:04d}.png"
        pix.save(str(p))
        paths.append(p)
    doc.close()
    return paths


def normalize_input(path: Path, work_dir: Path, heif_bin: str, zoom: float = 2.0) -> list[Path]:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return rasterize_pdf(path, work_dir / path.stem / "pages", zoom=zoom)
    png = ensure_png(path, heif_bin=heif_bin)
    return [png]
