"""
MinerU 精准解析 API（v4）Python 客户端

官网申请 Token: https://mineru.net
文档要点:
  - 单文件 URL: POST /api/v4/extract/task → GET /api/v4/extract/task/{task_id}
  - 本地批量: POST /api/v4/file-urls/batch → PUT 上传 → GET 批量结果
  - URL 批量: POST /api/v4/extract/task/batch → GET 批量结果
  - 不支持直接 multipart 上传；单文件须公网可访问 URL
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import requests

BASE_URL = "https://mineru.net"


class ModelVersion(str, Enum):
    PIPELINE = "pipeline"
    VLM = "vlm"
    HTML = "MinerU-HTML"


class TaskState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    CONVERTING = "converting"
    DONE = "done"
    FAILED = "failed"
    WAITING_FILE = "waiting-file"


@dataclass
class MinerUConfig:
    token: str
    base_url: str = BASE_URL
    poll_interval: float = 3.0
    poll_timeout: float = 600.0
    request_timeout: float = 60.0


@dataclass
class ExtractOptions:
    """创建解析任务时的可选参数（单文件 / 批量共用逻辑）"""

    model_version: str = ModelVersion.VLM.value
    is_ocr: bool = False
    enable_formula: bool = True
    enable_table: bool = True
    language: str = "ch"
    data_id: str | None = None
    page_ranges: str | None = None
    extra_formats: list[str] | None = None  # docx, html, latex
    callback: str | None = None
    seed: str | None = None
    no_cache: bool = False
    cache_tolerance: int = 900

    def to_api_dict(self, *, for_file_item: bool = False) -> dict[str, Any]:
        """for_file_item=True 时仅返回可放在 files[] 元素内的字段"""
        if for_file_item:
            d: dict[str, Any] = {}
            if self.data_id is not None:
                d["data_id"] = self.data_id
            if self.page_ranges is not None:
                d["page_ranges"] = self.page_ranges
            if self.is_ocr:
                d["is_ocr"] = self.is_ocr
            return d

        d = {
            "model_version": self.model_version,
            "enable_formula": self.enable_formula,
            "enable_table": self.enable_table,
            "language": self.language,
            "no_cache": self.no_cache,
            "cache_tolerance": self.cache_tolerance,
        }
        if self.is_ocr:
            d["is_ocr"] = self.is_ocr
        if self.data_id is not None:
            d["data_id"] = self.data_id
        if self.page_ranges is not None:
            d["page_ranges"] = self.page_ranges
        if self.extra_formats:
            d["extra_formats"] = self.extra_formats
        if self.callback:
            d["callback"] = self.callback
            if self.seed:
                d["seed"] = self.seed
        return d


@dataclass
class TaskResult:
    task_id: str
    state: str
    full_zip_url: str | None = None
    err_msg: str | None = None
    data_id: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def done(self) -> bool:
        return self.state == TaskState.DONE.value

    @property
    def failed(self) -> bool:
        return self.state == TaskState.FAILED.value


class MinerUPrecisionClient:
    """MinerU 精准解析 API 封装"""

    def __init__(self, config: MinerUConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {config.token}",
                "Content-Type": "application/json",
                "Accept": "*/*",
            }
        )

    def _url(self, path: str) -> str:
        return f"{self.config.base_url.rstrip('/')}{path}"

    def _check_response(self, resp: requests.Response) -> dict[str, Any]:
        resp.raise_for_status()
        body = resp.json()
        if body.get("code") != 0:
            raise MinerUAPIError(
                code=body.get("code"),
                msg=body.get("msg", "unknown"),
                trace_id=body.get("trace_id"),
                raw=body,
            )
        return body

    # ------------------------------------------------------------------ #
    # 1. 单文件解析（公网 URL）
    # ------------------------------------------------------------------ #

    def create_extract_task(
        self,
        file_url: str,
        options: ExtractOptions | None = None,
    ) -> str:
        """
        创建单文件解析任务（文件须为可访问 URL，不支持直传文件体）。

        Returns:
            task_id
        """
        options = options or ExtractOptions()
        payload = {"url": file_url, **options.to_api_dict()}
        data = self._check_response(
            self.session.post(
                self._url("/api/v4/extract/task"),
                json=payload,
                timeout=self.config.request_timeout,
            )
        )
        return data["data"]["task_id"]

    def get_task(self, task_id: str) -> TaskResult:
        """查询单任务状态与结果"""
        data = self._check_response(
            self.session.get(
                self._url(f"/api/v4/extract/task/{task_id}"),
                timeout=self.config.request_timeout,
            )
        )
        d = data["data"]
        return TaskResult(
            task_id=d.get("task_id", task_id),
            state=d.get("state", ""),
            full_zip_url=d.get("full_zip_url"),
            err_msg=d.get("err_msg"),
            data_id=d.get("data_id"),
            raw=d,
        )

    def extract_url_and_wait(
        self,
        file_url: str,
        options: ExtractOptions | None = None,
    ) -> TaskResult:
        """提交 URL 并轮询直到 done / failed"""
        task_id = self.create_extract_task(file_url, options)
        return self.wait_task(task_id)

    def wait_task(self, task_id: str) -> TaskResult:
        """轮询单任务直到完成或失败"""
        deadline = time.time() + self.config.poll_timeout
        while time.time() < deadline:
            result = self.get_task(task_id)
            if result.done or result.failed:
                if result.failed:
                    raise MinerUTaskFailedError(result)
                return result
            time.sleep(self.config.poll_interval)
        raise TimeoutError(f"task {task_id} not finished within {self.config.poll_timeout}s")

    # ------------------------------------------------------------------ #
    # 2. 批量：本地文件上传
    # ------------------------------------------------------------------ #

    def batch_upload_files(
        self,
        local_paths: list[str | Path],
        options: ExtractOptions | None = None,
        data_ids: list[str] | None = None,
    ) -> str:
        """
        申请上传链接 → PUT 本地文件 → 返回 batch_id（系统自动提交解析）。

        限制: 单次最多 50 个文件；上传链接 24 小时内有效。
        """
        if len(local_paths) > 50:
            raise ValueError("单次最多 50 个文件")
        options = options or ExtractOptions()
        files_payload = []
        for i, p in enumerate(local_paths):
            p = Path(p)
            item = {"name": p.name, **options.to_api_dict(for_file_item=True)}
            if data_ids and i < len(data_ids):
                item["data_id"] = data_ids[i]
            files_payload.append(item)

        top = options.to_api_dict()
        for k in ("data_id", "page_ranges", "is_ocr"):
            top.pop(k, None)
        payload = {"files": files_payload, **top}

        data = self._check_response(
            self.session.post(
                self._url("/api/v4/file-urls/batch"),
                json=payload,
                timeout=self.config.request_timeout,
            )
        )
        batch_id = data["data"]["batch_id"]
        upload_urls: list[str] = data["data"]["file_urls"]

        for path, upload_url in zip(local_paths, upload_urls):
            with open(path, "rb") as f:
                r = requests.put(upload_url, data=f, timeout=self.config.request_timeout)
                if r.status_code != 200:
                    raise RuntimeError(f"upload failed {path}: HTTP {r.status_code}")

        return batch_id

    # ------------------------------------------------------------------ #
    # 3. 批量：URL 列表
    # ------------------------------------------------------------------ #

    def batch_extract_urls(
        self,
        file_urls: list[str],
        options: ExtractOptions | None = None,
        data_ids: list[str] | None = None,
    ) -> str:
        """批量提交公网 URL 解析任务，返回 batch_id"""
        if len(file_urls) > 50:
            raise ValueError("单次最多 50 个 URL")
        options = options or ExtractOptions()
        files_payload = []
        for i, url in enumerate(file_urls):
            item = {"url": url, **options.to_api_dict(for_file_item=True)}
            if data_ids and i < len(data_ids):
                item["data_id"] = data_ids[i]
            files_payload.append(item)

        top = options.to_api_dict()
        for k in ("data_id", "page_ranges", "is_ocr"):
            top.pop(k, None)
        payload = {"files": files_payload, **top}

        data = self._check_response(
            self.session.post(
                self._url("/api/v4/extract/task/batch"),
                json=payload,
                timeout=self.config.request_timeout,
            )
        )
        return data["data"]["batch_id"]

    def get_batch_results(self, batch_id: str) -> dict[str, Any]:
        """查询批量任务进度与结果"""
        data = self._check_response(
            self.session.get(
                self._url(f"/api/v4/extract-results/batch/{batch_id}"),
                timeout=self.config.request_timeout,
            )
        )
        return data["data"]

    def wait_batch(self, batch_id: str) -> list[dict[str, Any]]:
        """轮询批量任务直到全部 done 或 failed"""
        deadline = time.time() + self.config.poll_timeout
        while time.time() < deadline:
            data = self.get_batch_results(batch_id)
            results = data.get("extract_result", [])
            if not results:
                time.sleep(self.config.poll_interval)
                continue
            terminal = {"done", "failed"}
            if all(r.get("state") in terminal for r in results):
                return results
            time.sleep(self.config.poll_interval)
        raise TimeoutError(f"batch {batch_id} not finished within {self.config.poll_timeout}s")

    # ------------------------------------------------------------------ #
    # 工具
    # ------------------------------------------------------------------ #

    @staticmethod
    def verify_callback_checksum(uid: str, seed: str, content: str, checksum: str) -> bool:
        """校验 callback 推送的 checksum（SHA256(uid + seed + content)）"""
        raw = f"{uid}{seed}{content}"
        expected = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return expected == checksum

    def download_zip(self, zip_url: str, save_path: str | Path) -> Path:
        """下载 full_zip_url 到本地"""
        save_path = Path(save_path)
        r = requests.get(zip_url, stream=True, timeout=300)
        r.raise_for_status()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        return save_path


class MinerUAPIError(Exception):
    def __init__(self, code: Any, msg: str, trace_id: str | None = None, raw: dict | None = None):
        self.code = code
        self.msg = msg
        self.trace_id = trace_id
        self.raw = raw or {}
        super().__init__(f"[{code}] {msg} (trace_id={trace_id})")


class MinerUTaskFailedError(Exception):
    def __init__(self, result: TaskResult):
        self.result = result
        super().__init__(result.err_msg or "task failed")


# ====================================================================== #
# 使用示例（将 TOKEN 替换为官网申请值）
# ====================================================================== #

def example_single_url():
    """单文件 URL 解析（推荐 model_version=vlm）"""
    client = MinerUPrecisionClient(MinerUConfig(token="YOUR_TOKEN"))

    options = ExtractOptions(
        model_version=ModelVersion.VLM.value,
        enable_formula=True,
        enable_table=True,
        extra_formats=["docx"],  # 额外导出 docx；md/json 默认在 zip 内
    )

    result = client.extract_url_and_wait(
        "https://cdn-mineru.openxlab.org.cn/demo/example.pdf",
        options=options,
    )
    print("state:", result.state)
    print("zip:", result.full_zip_url)
    if result.full_zip_url:
        client.download_zip(result.full_zip_url, "output/example_result.zip")


def example_single_html():
    """HTML 文件须指定 MinerU-HTML"""
    client = MinerUPrecisionClient(MinerUConfig(token="YOUR_TOKEN"))
    options = ExtractOptions(model_version=ModelVersion.HTML.value)
    task_id = client.create_extract_task("https://your-domain.com/page.html", options)
    result = client.wait_task(task_id)
    print(result.full_zip_url)


def example_batch_local():
    """本地 PDF 批量：申请链接 → PUT 上传 → 轮询 batch"""
    client = MinerUPrecisionClient(MinerUConfig(token="YOUR_TOKEN"))
    options = ExtractOptions(model_version=ModelVersion.VLM.value)

    batch_id = client.batch_upload_files(
        ["demo1.pdf", "demo2.pdf"],
        options=options,
        data_ids=["biz_id_1", "biz_id_2"],
    )
    results = client.wait_batch(batch_id)
    for item in results:
        print(item["file_name"], item["state"], item.get("full_zip_url"))


def example_batch_urls():
    """多个公网 URL 批量解析"""
    client = MinerUPrecisionClient(MinerUConfig(token="YOUR_TOKEN"))
    options = ExtractOptions(model_version=ModelVersion.VLM.value)

    batch_id = client.batch_extract_urls(
        [
            "https://cdn-mineru.openxlab.org.cn/demo/example.pdf",
        ],
        options=options,
    )
    results = client.wait_batch(batch_id)
    print(json.dumps(results, indent=2, ensure_ascii=False))


def example_low_level_single():
    """底层：手动创建任务 + 轮询（与官方文档一致）"""
    token = "YOUR_TOKEN"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }
    create_url = "https://mineru.net/api/v4/extract/task"
    body = {
        "url": "https://cdn-mineru.openxlab.org.cn/demo/example.pdf",
        "model_version": "vlm",
    }
    res = requests.post(create_url, headers=headers, json=body, timeout=60)
    res.raise_for_status()
    task_id = res.json()["data"]["task_id"]
    print("task_id:", task_id)

    while True:
        r = requests.get(
            f"https://mineru.net/api/v4/extract/task/{task_id}",
            headers=headers,
            timeout=60,
        )
        r.raise_for_status()
        data = r.json()["data"]
        state = data["state"]
        print("state:", state)
        if state == "done":
            print("zip:", data.get("full_zip_url"))
            break
        if state == "failed":
            print("err:", data.get("err_msg"))
            break
        time.sleep(3)


if __name__ == "__main__":
    # 取消注释需要运行的示例
    # example_single_url()
    # example_batch_local()
    # example_batch_urls()
    print("请设置 TOKEN 后调用 example_* 函数，或: from mineru_precision_api import MinerUPrecisionClient")
