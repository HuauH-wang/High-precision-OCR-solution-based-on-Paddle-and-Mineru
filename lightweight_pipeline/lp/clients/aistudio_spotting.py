from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from lp.clients.aistudio_job import AiStudioJobClient
from lp.core.md_unify import from_aistudio_records, normalize_api_markdown
from lp.types import BBox, TextLine


@dataclass
class SpottingResult:
    lines: list[TextLine]
    markdown_text: str = ""
    blank_image_saved: Path | None = None


class AiStudioSpottingClient:
    """
    AI Studio spotting 模式（promptLabel=spotting, useLayoutDetection=false）
    替代轻量产线中的「文字检测 + 识别」两步 API。
    """

    def __init__(self, job: AiStudioJobClient):
        self.job = job

    @classmethod
    def from_config(
        cls,
        *,
        job_url: str,
        token: str,
        model: str,
        optional_payload: dict[str, Any],
        poll_interval_sec: float,
        poll_timeout_sec: float,
        request_timeout_sec: float,
        rate_limiter: RateLimiter | None = None,
        retry_attempts: int = 1,
        retry_delay_sec: float = 2.0,
    ) -> AiStudioSpottingClient:
        return cls(
            AiStudioJobClient(
                job_url=job_url,
                token=token,
                model=model,
                optional_payload=optional_payload,
                poll_interval_sec=poll_interval_sec,
                poll_timeout_sec=poll_timeout_sec,
                request_timeout_sec=request_timeout_sec,
                rate_limiter=rate_limiter,
                retry_attempts=retry_attempts,
                retry_delay_sec=retry_delay_sec,
            )
        )

    def spot(
        self,
        image_path: str | Path,
        *,
        save_blank_to: Path | None = None,
    ) -> SpottingResult:
        records = self.job.run_local_file(image_path)
        md_unified = from_aistudio_records(records)
        items = self.job.iter_layout_parsing_results(records)
        lines: list[TextLine] = []
        md_parts: list[str] = []
        blank_saved: Path | None = None

        for item in items:
            md = item.get("markdown") or {}
            if isinstance(md, dict) and md.get("text"):
                md_parts.append(str(md["text"]))

            lines.extend(_extract_text_lines(item))
            if not lines:
                lines.extend(_deep_find_text_lines(item))

            if save_blank_to and blank_saved is None:
                blank_saved = _download_blank_image(item, save_blank_to, self.job.request_timeout_sec)

        if not lines and md_unified:
            lines = _markdown_to_single_block([md_unified])
        elif not md_unified and md_parts:
            md_unified = normalize_api_markdown("\n\n".join(md_parts))

        return SpottingResult(
            lines=lines,
            markdown_text=md_unified or normalize_api_markdown("\n\n".join(md_parts)).strip(),
            blank_image_saved=blank_saved,
        )


def _extract_text_lines(item: dict[str, Any]) -> list[TextLine]:
    lines: list[TextLine] = []
    pools: list[Any] = []
    for key in (
        "spottingResults",
        "spotting_res",
        "ocrResult",
        "ocr_result",
        "text_det",
        "dt_polys",
        "rec_texts",
        "parsing_res_list",
    ):
        if key in item and item[key]:
            val = item[key]
            pools.extend(val if isinstance(val, list) else [val])

    pruned = item.get("prunedResult")
    if isinstance(pruned, dict):
        for key in ("parsing_res_list", "ocrResult", "spottingResults"):
            if key in pruned:
                val = pruned[key]
                pools.extend(val if isinstance(val, list) else [val])
    elif isinstance(pruned, list):
        pools.extend(pruned)

    for p in pools:
        tl = _dict_to_textline(p)
        if tl:
            lines.append(tl)
    return lines


def _deep_find_text_lines(obj: Any, depth: int = 0) -> list[TextLine]:
    if depth > 14:
        return []
    found: list[TextLine] = []
    if isinstance(obj, dict):
        tl = _dict_to_textline(obj)
        if tl:
            found.append(tl)
        for v in obj.values():
            found.extend(_deep_find_text_lines(v, depth + 1))
    elif isinstance(obj, list):
        for v in obj:
            found.extend(_deep_find_text_lines(v, depth + 1))
    return found


def _dict_to_textline(d: Any) -> TextLine | None:
    if not isinstance(d, dict):
        return None
    text = d.get("text") or d.get("rec_text") or d.get("content") or d.get("transcription")
    if not text or not str(text).strip():
        return None
    box = d.get("bbox") or d.get("box") or d.get("block_bbox") or d.get("poly") or d.get("dt_poly")
    if not box:
        return None
    if isinstance(box[0], (list, tuple)):
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    else:
        if len(box) < 4:
            return None
        x0, y0, x1, y1 = map(float, box[:4])
    score = float(d.get("score") or d.get("confidence") or 1.0)
    return TextLine(text=str(text).strip(), bbox=BBox(x0, y0, x1, y1, score=score), score=score)


def _markdown_to_single_block(parts: list[str]) -> list[TextLine]:
    text = "\n\n".join(parts).strip()
    if not text:
        return []
    return [TextLine(text=text, bbox=BBox(0, 0, 1000, 1000))]


def _download_blank_image(item: dict[str, Any], save_path: Path, timeout: float) -> Path | None:
    images = item.get("outputImages") or {}
    if not isinstance(images, dict) or not images:
        return None
    preferred = None
    for name, url in images.items():
        lname = str(name).lower()
        if any(k in lname for k in ("spot", "layout", "ocr", "preprocess")):
            preferred = url
            break
    url = preferred or next(iter(images.values()))
    if not url:
        return None
    resp = requests.get(url, timeout=timeout)
    if resp.status_code != 200:
        return None
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_bytes(resp.content)
    return save_path
