from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _env_first(*names: str) -> str:
    """返回第一个非空环境变量值。"""
    for name in names:
        val = os.environ.get(name, "").strip()
        if val:
            return val
    return ""


@dataclass
class ApiSection:
    enabled: bool = False
    base_url: str = ""
    token: str = ""
    api_key: str = ""
    secret_key: str = ""
    timeout_sec: float = 60.0
    extra_headers: dict[str, str] = field(default_factory=dict)
    extra_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class MinerUApiConfig(ApiSection):
    model_version: str = "vlm"
    is_ocr: bool = True
    enable_formula: bool = True
    enable_table: bool = True
    language: str = "ch"
    request_timeout_sec: float = 60.0


@dataclass
class AiStudioJobConfig(ApiSection):
    """Paddle AI Studio 异步 Job API 配置（版面 / spotting 共用）。"""

    provider: str = "aistudio"
    model: str = "PaddleOCR-VL-1.5"
    optional_payload: dict[str, Any] = field(default_factory=dict)
    poll_interval_sec: float = 5.0
    poll_timeout_sec: float = 600.0


# 兼容旧名
AiStudioLayoutConfig = AiStudioJobConfig


@dataclass
class Settings:
    raw: dict[str, Any]
    pipeline: dict[str, Any]
    batch: dict[str, Any]
    entry: dict[str, Any]
    pdfplumber: dict[str, Any]
    ocr: dict[str, Any]
    detection: dict[str, Any]
    mineru: MinerUApiConfig
    paddle_layout: AiStudioJobConfig
    paddle_spotting: AiStudioJobConfig
    paddle_detection: ApiSection
    paddle_recognition: ApiSection

    @classmethod
    def load(cls, path: str | Path) -> Settings:
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        apis = raw.get("apis", {})
        return cls(
            raw=raw,
            pipeline=raw.get("pipeline", {}),
            batch=raw.get("batch", {}),
            entry=raw.get("entry", {}),
            pdfplumber=raw.get("pdfplumber", {}),
            ocr=raw.get("ocr", {}),
            detection=raw.get("detection", {}),
            mineru=_mineru_from(apis.get("mineru", {})),
            paddle_layout=_aistudio_job_from(
                apis.get("paddle_layout", {}),
                token_env=("AISTUDIO_LAYOUT_TOKEN", "AISTUDIO_TOKEN"),
            ),
            paddle_spotting=_aistudio_job_from(
                apis.get("paddle_spotting", {}),
                token_env=("AISTUDIO_SPOTTING_TOKEN", "AISTUDIO_TOKEN"),
            ),
            paddle_detection=_api_from(apis.get("paddle_detection", {})),
            paddle_recognition=_api_from(apis.get("paddle_recognition", {})),
        )

    @property
    def output_dir(self) -> Path:
        return Path(self.pipeline.get("output_dir", "./output"))

    @property
    def work_dir(self) -> Path:
        return Path(self.pipeline.get("work_dir", "./work"))

    @property
    def parse_path(self) -> str:
        return str(self.pipeline.get("parse_path", "P1"))

    @property
    def blur_sensitive(self) -> bool:
        return bool(self.pipeline.get("blur_sensitive", False))

    def det_threshold(self) -> float:
        if self.blur_sensitive:
            return float(self.detection.get("blur_threshold", 0.25))
        return float(self.detection.get("normal_threshold", 0.5))

    def layout_min_score(self) -> float:
        base = float(self.ocr.get("layout_confidence_min", 0.2))
        if self.blur_sensitive:
            return float(self.ocr.get("layout_confidence_min_blur", max(0.05, base * 0.5)))
        return base

    def job_optional_payload(self, cfg: AiStudioJobConfig) -> dict[str, Any]:
        payload = dict(cfg.optional_payload)
        if self.blur_sensitive:
            blur = cfg.extra_params.get("blur_optional_payload")
            if isinstance(blur, dict):
                payload.update(blur)
        return payload

    def batch_retry_attempts(self) -> int:
        return int(self.batch.get("retry_attempts", 3))

    def batch_retry_delay_sec(self) -> float:
        return float(self.batch.get("retry_delay_sec", 2.0))

    def layout_qps(self) -> float:
        return float(self.batch.get("paddle_layout_qps", 0))

    def spotting_qps(self) -> float:
        return float(self.batch.get("paddle_spotting_qps", self.batch.get("paddle_layout_qps", 0)))


def _api_from(d: dict[str, Any], *, token_env: tuple[str, ...] = ()) -> ApiSection:
    file_token = str(d.get("token", "") or "")
    env_token = _env_first(*token_env) if token_env else ""
    return ApiSection(
        enabled=bool(d.get("enabled", False)),
        base_url=str(d.get("base_url", "")),
        token=env_token or file_token,
        api_key=str(d.get("api_key", "")),
        secret_key=str(d.get("secret_key", "")),
        timeout_sec=float(d.get("timeout_sec", 60.0)),
        extra_headers=dict(d.get("extra_headers") or {}),
        extra_params=dict(d.get("extra_params") or {}),
    )


def _aistudio_job_from(
    d: dict[str, Any],
    *,
    token_env: tuple[str, ...] = ("AISTUDIO_TOKEN",),
) -> AiStudioJobConfig:
    base = _api_from(d, token_env=token_env)
    extra = dict(base.extra_params)
    if d.get("blur_optional_payload"):
        extra["blur_optional_payload"] = dict(d["blur_optional_payload"])
    return AiStudioJobConfig(
        enabled=base.enabled,
        base_url=base.base_url,
        token=base.token,
        api_key=base.api_key,
        secret_key=base.secret_key,
        timeout_sec=base.timeout_sec,
        extra_headers=base.extra_headers,
        extra_params=extra,
        provider=str(d.get("provider", "aistudio")),
        model=str(d.get("model", "PaddleOCR-VL-1.5")),
        optional_payload=dict(d.get("optional_payload") or {}),
        poll_interval_sec=float(d.get("poll_interval_sec", 5.0)),
        poll_timeout_sec=float(d.get("poll_timeout_sec", 600.0)),
    )


def _mineru_from(d: dict[str, Any]) -> MinerUApiConfig:
    base = _api_from(d, token_env=("MINERU_TOKEN",))
    return MinerUApiConfig(
        enabled=base.enabled,
        base_url=base.base_url or "https://mineru.net",
        token=base.token,
        api_key=base.api_key,
        secret_key=base.secret_key,
        timeout_sec=base.timeout_sec,
        extra_headers=base.extra_headers,
        extra_params=base.extra_params,
        model_version=str(d.get("model_version", "vlm")),
        is_ocr=bool(d.get("is_ocr", True)),
        enable_formula=bool(d.get("enable_formula", True)),
        enable_table=bool(d.get("enable_table", True)),
        language=str(d.get("language", "ch")),
        request_timeout_sec=float(d.get("request_timeout_sec", 60.0)),
    )
