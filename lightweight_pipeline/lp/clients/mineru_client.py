from __future__ import annotations

import sys
import zipfile
from pathlib import Path

from lp.core.retry import retry_call
from lp.settings import MinerUApiConfig
from lp.types import MinerUTaskMeta

# 复用工作区已有 MinerU 客户端
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_ROOT.parent))

from mineru_precision_api import ExtractOptions, MinerUConfig, MinerUPrecisionClient  # noqa: E402


class MinerUClientWrapper:
    def __init__(self, cfg: MinerUApiConfig, batch_cfg: dict):
        if not cfg.enabled:
            raise RuntimeError("MinerU API 未启用")
        if not cfg.token:
            raise RuntimeError("请在 config.yaml 的 apis.mineru.token 填写 MinerU Token")
        self.cfg = cfg
        self.batch_cfg = batch_cfg
        self.client = MinerUPrecisionClient(
            MinerUConfig(
                token=cfg.token,
                base_url=cfg.base_url,
                poll_interval=float(batch_cfg.get("mineru_poll_interval_sec", 3.0)),
                poll_timeout=float(batch_cfg.get("mineru_poll_timeout_sec", 600.0)),
                request_timeout=cfg.request_timeout_sec,
            )
        )

    def default_options(self) -> ExtractOptions:
        return ExtractOptions(
            model_version=self.cfg.model_version,
            is_ocr=self.cfg.is_ocr,
            enable_formula=self.cfg.enable_formula,
            enable_table=self.cfg.enable_table,
            language=self.cfg.language,
        )

    def batch_upload_and_wait(
        self,
        tasks: list[MinerUTaskMeta],
    ) -> dict[str, dict]:
        if not tasks:
            return {}
        batch_size = int(self.batch_cfg.get("mineru_batch_size", 50))
        options = self.default_options()
        out: dict[str, dict] = {}

        for i in range(0, len(tasks), batch_size):
            chunk = tasks[i : i + batch_size]
            paths = [t.local_path for t in chunk]
            data_ids = [t.artifact_id for t in chunk]

            def _upload_batch() -> str:
                return self.client.batch_upload_files(paths, options=options, data_ids=data_ids)

            batch_id = retry_call(
                _upload_batch,
                attempts=int(self.batch_cfg.get("retry_attempts", 3)),
                delay_sec=float(self.batch_cfg.get("retry_delay_sec", 2.0)),
            )

            def _wait_batch() -> list:
                return self.client.wait_batch(batch_id)

            results = retry_call(
                _wait_batch,
                attempts=int(self.batch_cfg.get("retry_attempts", 3)),
                delay_sec=float(self.batch_cfg.get("retry_delay_sec", 2.0)),
            )
            by_id = {r.get("data_id"): r for r in results if r.get("data_id")}
            for t in chunk:
                item = by_id.get(t.artifact_id)
                if item:
                    out[t.artifact_id] = item
                else:
                    out[t.artifact_id] = {"state": "failed", "err_msg": "missing in batch result"}
        return out

    @staticmethod
    def extract_markdown_from_zip(zip_path: Path) -> str:
        if not zip_path.exists():
            return ""
        md_parts: list[str] = []
        with zipfile.ZipFile(zip_path, "r") as zf:
            for name in sorted(zf.namelist()):
                if name.lower().endswith(".md"):
                    md_parts.append(zf.read(name).decode("utf-8", errors="replace"))
        return "\n\n".join(md_parts)
