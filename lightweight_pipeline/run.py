#!/usr/bin/env python3
"""轻量产线 CLI。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lp.batch_pipeline import LightweightBatchPipeline
from lp.settings import Settings


def main() -> None:
    parser = argparse.ArgumentParser(description="轻量级文档解析产线（全 API + 批量并行）")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent / "config.yaml"),
        help="YAML 配置文件路径",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="输入文件或目录（pdf/png/jpg/jpeg/heif/heic）",
    )
    parser.add_argument(
        "--parse-path",
        choices=["P0", "P1", "P2-M1", "P2-M2"],
        help="覆盖 config 中的 pipeline.parse_path",
    )
    parser.add_argument(
        "--blur-sensitive",
        action="store_true",
        help="启用模糊文档低检测阈值",
    )
    args = parser.parse_args()

    settings = Settings.load(args.config)
    if args.parse_path:
        settings.pipeline["parse_path"] = args.parse_path
    if args.blur_sensitive:
        settings.pipeline["blur_sensitive"] = True

    files = _collect_inputs(args.inputs, settings)
    if not files:
        raise SystemExit("未找到可处理文件")

    pipeline = LightweightBatchPipeline(settings)
    results = pipeline.run_files(files)

    summary = []
    for r in results:
        if isinstance(r, dict):
            summary.append(r)
        else:
            summary.append(
                {
                    "doc_id": r.doc_id,
                    "page_index": r.page_index,
                    "markdown_len": len(r.markdown),
                }
            )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _collect_inputs(items: list[str], settings: Settings) -> list[Path]:
    exts = {f".{e.lower().lstrip('.')}" for e in settings.entry.get("supported_extensions", [])}
    out: list[Path] = []
    for item in items:
        p = Path(item)
        if p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.suffix.lower() in exts:
                    out.append(f)
        elif p.is_file() and p.suffix.lower() in exts:
            out.append(p)
    return out


if __name__ == "__main__":
    main()
