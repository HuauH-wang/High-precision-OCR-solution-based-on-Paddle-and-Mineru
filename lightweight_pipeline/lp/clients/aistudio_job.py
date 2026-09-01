from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import requests

from lp.core.batch_utils import RateLimiter
from lp.core.retry import retry_call


class AiStudioJobError(RuntimeError):
    pass


class AiStudioJobClient:
    """Paddle AI Studio 异步 OCR Job API 公共客户端。"""

    def __init__(
        self,
        *,
        job_url: str,
        token: str,
        model: str = "PaddleOCR-VL-1.5",
        optional_payload: dict[str, Any] | None = None,
        poll_interval_sec: float = 5.0,
        poll_timeout_sec: float = 600.0,
        request_timeout_sec: float = 120.0,
        rate_limiter: RateLimiter | None = None,
        retry_attempts: int = 1,
        retry_delay_sec: float = 2.0,
    ):
        self.job_url = job_url.rstrip("/")
        self.token = token
        self.model = model
        self.optional_payload = optional_payload or {}
        self.poll_interval_sec = poll_interval_sec
        self.poll_timeout_sec = poll_timeout_sec
        self.request_timeout_sec = request_timeout_sec
        self.rate_limiter = rate_limiter
        self.retry_attempts = max(1, int(retry_attempts))
        self.retry_delay_sec = retry_delay_sec

    def _headers(self, *, json_mode: bool = False) -> dict[str, str]:
        h = {"Authorization": f"bearer {self.token}"}
        if json_mode:
            h["Content-Type"] = "application/json"
        return h

    def submit_local_file(self, file_path: str | Path) -> str:
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        data = {
            "model": self.model,
            "optionalPayload": json.dumps(self.optional_payload, ensure_ascii=False),
        }
        with open(path, "rb") as f:
            resp = requests.post(
                self.job_url,
                headers=self._headers(),
                data=data,
                files={"file": f},
                timeout=self.request_timeout_sec,
            )
        return self._parse_job_id(resp)

    def submit_file_url(self, file_url: str) -> str:
        payload = {
            "fileUrl": file_url,
            "model": self.model,
            "optionalPayload": self.optional_payload,
        }
        resp = requests.post(
            self.job_url,
            json=payload,
            headers=self._headers(json_mode=True),
            timeout=self.request_timeout_sec,
        )
        return self._parse_job_id(resp)

    def _parse_job_id(self, resp: requests.Response) -> str:
        if resp.status_code != 200:
            raise AiStudioJobError(f"submit failed HTTP {resp.status_code}: {resp.text}")
        body = resp.json()
        if "data" not in body or "jobId" not in body["data"]:
            raise AiStudioJobError(f"unexpected submit response: {body}")
        return str(body["data"]["jobId"])

    def wait_job(self, job_id: str) -> dict[str, Any]:
        deadline = time.time() + self.poll_timeout_sec
        while time.time() < deadline:
            resp = requests.get(
                f"{self.job_url}/{job_id}",
                headers=self._headers(),
                timeout=self.request_timeout_sec,
            )
            if resp.status_code != 200:
                raise AiStudioJobError(f"poll failed HTTP {resp.status_code}: {resp.text}")
            data = resp.json()["data"]
            state = data.get("state")
            if state == "done":
                return data
            if state == "failed":
                raise AiStudioJobError(f"job failed: {data.get('errorMsg', data)}")
            time.sleep(self.poll_interval_sec)
        raise TimeoutError(f"job {job_id} timeout after {self.poll_timeout_sec}s")

    def fetch_jsonl_records(self, job_data: dict[str, Any]) -> list[dict[str, Any]]:
        json_url = job_data.get("resultUrl", {}).get("jsonUrl")
        if not json_url:
            raise AiStudioJobError("missing resultUrl.jsonUrl in job result")
        resp = requests.get(json_url, timeout=self.request_timeout_sec)
        resp.raise_for_status()
        records: list[dict[str, Any]] = []
        for line in resp.text.strip().split("\n"):
            line = line.strip()
            if line:
                records.append(json.loads(line))
        return records

    def run_local_file(self, file_path: str | Path) -> list[dict[str, Any]]:
        def _once() -> list[dict[str, Any]]:
            if self.rate_limiter:
                self.rate_limiter.wait()
            job_id = self.submit_local_file(file_path)
            job_data = self.wait_job(job_id)
            return self.fetch_jsonl_records(job_data)

        return retry_call(
            _once,
            attempts=self.retry_attempts,
            delay_sec=self.retry_delay_sec,
        )

    def iter_layout_parsing_results(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for rec in records:
            result = rec.get("result") or rec
            for item in result.get("layoutParsingResults") or []:
                items.append(item)
        return items
