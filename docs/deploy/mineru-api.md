# MinerU 精准解析 API · Python 使用说明

> 云端 API（`https://mineru.net`），需官网申请 **Token**。  
> 完整可运行客户端：`mineru_precision_api.py`

---

## 1. 与 Agent 轻量 API 的区别

| 维度 | 精准解析 API（本文） | Agent 轻量 API |
|------|----------------------|----------------|
| Token | 需要 | 不需要（IP 限频） |
| 基础路径 | `/api/v4/...` | `/api/v1/agent/...` |
| 模型 | pipeline / **vlm** / MinerU-HTML | 固定 pipeline 轻量 |
| 单文件大小 | ≤ 200MB | ≤ 10MB |
| 页数 | ≤ 200 页 | ≤ 20 页 |
| 批量 | 支持（≤200 个任务规模见官网） | 单文件 |
| 输出 | Zip（Markdown、JSON，可 docx/html/latex） | 仅 Markdown CDN 链接 |

---

## 2. 安装依赖

```bash
pip install requests
```

---

## 3. 快速开始（推荐封装客户端）

```python
from mineru_precision_api import (
    MinerUConfig,
    MinerUPrecisionClient,
    ExtractOptions,
    ModelVersion,
)

client = MinerUPrecisionClient(MinerUConfig(token="官网申请的_TOKEN"))

# 单文件 URL → 轮询 → 下载 zip
result = client.extract_url_and_wait(
    "https://cdn-mineru.openxlab.org.cn/demo/example.pdf",
    options=ExtractOptions(model_version=ModelVersion.VLM.value),
)
print(result.full_zip_url)
client.download_zip(result.full_zip_url, "output/result.zip")
```

---

## 4. 接口一览

| 场景 | 方法 | HTTP |
|------|------|------|
| 单文件（URL） | `create_extract_task` | `POST /api/v4/extract/task` |
| 查单任务 | `get_task` / `wait_task` | `GET /api/v4/extract/task/{task_id}` |
| 本地文件批量 | `batch_upload_files` | `POST /api/v4/file-urls/batch` + `PUT` 上传链接 |
| URL 批量 | `batch_extract_urls` | `POST /api/v4/extract/task/batch` |
| 查批量 | `get_batch_results` / `wait_batch` | `GET /api/v4/extract-results/batch/{batch_id}` |

**请求头（所有接口）**

```text
Authorization: Bearer <TOKEN>
Content-Type: application/json
```

---

## 5. 单文件解析

### 5.1 限制

- 单文件 ≤ **200MB**，≤ **200 页**
- **不支持** 请求体直接上传文件，必须提供 **公网可访问 URL**
- GitHub、AWS 等部分国外 URL 可能超时
- 每日前 **1000 页** 高优先级额度（超出后优先级降低）

### 5.1 官方风格最小示例

```python
import time
import requests

token = "YOUR_TOKEN"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {token}",
}

# 创建任务
res = requests.post(
    "https://mineru.net/api/v4/extract/task",
    headers=headers,
    json={
        "url": "https://cdn-mineru.openxlab.org.cn/demo/example.pdf",
        "model_version": "vlm",  # pipeline | vlm | MinerU-HTML
    },
)
res.raise_for_status()
task_id = res.json()["data"]["task_id"]

# 轮询
while True:
    r = requests.get(
        f"https://mineru.net/api/v4/extract/task/{task_id}",
        headers=headers,
    )
    data = r.json()["data"]
    state = data["state"]  # pending | running | converting | done | failed
    if state == "done":
        print(data["full_zip_url"])
        break
    if state == "failed":
        print(data["err_msg"])
        break
    time.sleep(3)
```

### 5.2 HTML 文件

`model_version` 必须为 **`MinerU-HTML`**：

```python
ExtractOptions(model_version="MinerU-HTML")
```

### 5.3 常用请求参数

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `url` | string | 必填 | 文件 URL |
| `model_version` | string | pipeline | pipeline / **vlm** / MinerU-HTML |
| `is_ocr` | bool | false | 强制 OCR，仅 pipeline/vlm |
| `enable_formula` | bool | true | 公式识别；vlm 主要影响行内公式 |
| `enable_table` | bool | true | 表格识别 |
| `language` | string | ch | 文档语言 |
| `page_ranges` | string | 全页 | 如 `"2,4-6"`、`"2--2"` |
| `extra_formats` | list | - | 额外 `docx`/`html`/`latex`（md/json 已在 zip 内） |
| `data_id` | string | - | 业务侧唯一 ID，≤128 字符 |
| `callback` + `seed` | string | - | 结果回调；用 callback 时 **seed 必填** |
| `no_cache` | bool | false | 忽略 URL 缓存 |
| `cache_tolerance` | int | 900 | 缓存有效秒数 |

### 5.4 任务状态 `state`

| 值 | 含义 |
|----|------|
| `pending` | 排队 |
| `running` | 解析中（可看 `extract_progress`） |
| `converting` | 格式转换中 |
| `done` | 完成，`full_zip_url` 可用 |
| `failed` | 失败，看 `err_msg` |

---

## 6. 批量解析

### 6.1 本地文件上传

流程：**申请上传 URL → PUT 文件 → 自动提交任务 → 轮询 batch**

```python
batch_id = client.batch_upload_files(
    ["a.pdf", "b.pdf"],
    options=ExtractOptions(model_version="vlm"),
    data_ids=["id_a", "id_b"],
)
results = client.wait_batch(batch_id)
```

注意：

- 单次申请链接 **≤ 50** 个
- 上传链接 **24 小时** 有效
- PUT 时 **不要** 设置 `Content-Type`
- 上传完成后 **无需** 再调「提交任务」接口

### 6.2 URL 批量

```python
batch_id = client.batch_extract_urls(
    ["https://cdn-mineru.openxlab.org.cn/demo/example.pdf"],
    options=ExtractOptions(model_version="vlm"),
)
```

等价请求体：

```json
{
  "files": [{"url": "https://...", "data_id": "abcd"}],
  "model_version": "vlm"
}
```

### 6.3 批量结果字段

`extract_result[]` 每项含：`file_name`、`state`、`full_zip_url`、`err_msg`、`data_id`、`extract_progress`。

批量 `state` 可能含 **`waiting-file`**（等待上传完成）。

---

## 7. 结果 Zip 说明

`full_zip_url` 下载后常见内容（非 HTML）：

| 文件 | 说明 |
|------|------|
| `full.md` | Markdown 结果 |
| `layout.json` | 中间结果（middle.json） |
| `*_model.json` | 模型推理（model.json） |
| `*_content_list.json` | 内容列表 |

HTML 源文件：`full.md` + `main.html`。

详见：https://opendatalab.github.io/MinerU/reference/output_files/

---

## 8. Callback 校验（可选）

若配置 `callback`，MinerU 会 POST `checksum` 与 `content`：

```python
MinerUPrecisionClient.verify_callback_checksum(uid, seed, content_json_str, checksum)
# checksum = SHA256(uid + seed + content)
```

服务端需返回 **HTTP 200**；失败最多重试 **5** 次。

---

## 9. 常见错误码

| 错误码 | 说明 | 建议 |
|--------|------|------|
| A0202 | Token 错误 | 检查 Bearer 前缀与 Token |
| A0211 | Token 过期 | 重新申请 |
| -60005 | 文件过大 | ≤200MB |
| -60006 | 页数超限 | 拆分文件 |
| -60008 | URL 读取超时 | 换国内 CDN 或自建可访问链接 |
| -60012 | 找不到任务 | 检查 task_id |
| -60018 | 每日额度用尽 | 次日再试 |

---

## 10. 文件清单

| 文件 | 说明 |
|------|------|
| `mineru_precision_api.py` | `MinerUPrecisionClient` 封装 + 示例函数 |
| `MinerU_精准解析API_Python.md` | 本文档 |

运行示例前将 `YOUR_TOKEN` 替换为 MinerU 官网 Token。
