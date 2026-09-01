from __future__ import annotations

from lp.clients.aistudio_job import AiStudioJobClient
from lp.clients.aistudio_layout import _deep_find_regions, _extract_regions_from_item
from lp.clients.aistudio_spotting import AiStudioSpottingClient
from lp.core.batch_utils import RateLimiter
from lp.settings import AiStudioJobConfig, Settings
from lp.types import LayoutRegion


def _job_from_cfg(cfg: AiStudioJobConfig, settings: Settings, *, qps: float) -> AiStudioJobClient:
    limiter = RateLimiter(qps) if qps > 0 else None
    return AiStudioJobClient(
        job_url=cfg.base_url or "https://paddleocr.aistudio-app.com/api/v2/ocr/jobs",
        token=cfg.token,
        model=cfg.model,
        optional_payload=settings.job_optional_payload(cfg),
        poll_interval_sec=cfg.poll_interval_sec,
        poll_timeout_sec=cfg.poll_timeout_sec,
        request_timeout_sec=cfg.timeout_sec,
        rate_limiter=limiter,
        retry_attempts=settings.batch_retry_attempts(),
        retry_delay_sec=settings.batch_retry_delay_sec(),
    )


def create_layout_client(settings: Settings):
    cfg = settings.paddle_layout
    if not cfg.enabled:
        raise RuntimeError("版面 API 未启用：apis.paddle_layout.enabled")

    if cfg.provider == "aistudio" or "aistudio-app.com" in cfg.base_url:
        if not cfg.token:
            raise RuntimeError("apis.paddle_layout.token 为空")
        return _LayoutAdapter(_job_from_cfg(cfg, settings, qps=settings.layout_qps()))
    from lp.clients.paddle_http import PaddleLayoutClient

    return PaddleLayoutClient(cfg, "paddle_layout")


def create_spotting_client(settings: Settings) -> AiStudioSpottingClient | None:
    cfg = settings.paddle_spotting
    if not cfg.enabled:
        return None
    if not cfg.token:
        raise RuntimeError("apis.paddle_spotting.token 为空")
    return AiStudioSpottingClient.from_config(
        job_url=cfg.base_url or "https://paddleocr.aistudio-app.com/api/v2/ocr/jobs",
        token=cfg.token,
        model=cfg.model,
        optional_payload=settings.job_optional_payload(cfg),
        poll_interval_sec=cfg.poll_interval_sec,
        poll_timeout_sec=cfg.poll_timeout_sec,
        request_timeout_sec=cfg.timeout_sec,
        rate_limiter=RateLimiter(settings.spotting_qps()) if settings.spotting_qps() > 0 else None,
        retry_attempts=settings.batch_retry_attempts(),
        retry_delay_sec=settings.batch_retry_delay_sec(),
    )


class _LayoutAdapter:
    """将 AiStudioJobClient 适配为 layout.detect 接口。"""

    def __init__(self, job: AiStudioJobClient):
        self.job = job

    def detect(self, image_path, *, min_score: float = 0.2) -> list[LayoutRegion]:
        records = self.job.run_local_file(image_path)
        regions: list[LayoutRegion] = []
        for item in self.job.iter_layout_parsing_results(records):
            regions.extend(_extract_regions_from_item(item, min_score=min_score))
        if not regions and records:
            last = records[-1].get("result") or records[-1]
            regions.extend(_deep_find_regions(last, min_score=min_score))
        return regions
