# PaddleOCR-VL · RTX 5090 本地 Docker 部署完整教程

> **文档定位**：面向 **NVIDIA RTX 5090**（Blackwell 架构，Compute Capability **sm120**）用户，仅使用 **Docker / Docker Compose** 在本地完成 PaddleOCR-VL 的部署与调用。  
> **无需阅读其他教程**：下文已包含环境准备、推理验证、VLM 加速服务、生产级 API 服务及常见问题处理。  
> **模型说明**：文中 “PaddleOCR-VL” 指系列模型（如 **PaddleOCR-VL-1.5**）；默认产线版本为 **v1.5**。

---

## 目录

1. [PaddleOCR-VL 是什么](#1-paddleocr-vl-是什么)
2. [RTX 5090 部署前准备](#2-rtx-5090-部署前准备)
3. [三种部署形态怎么选](#3-三种部署形态怎么选)
4. [环境准备：推理镜像 Docker](#4-环境准备推理镜像-docker)
5. [快速开始：本地直接推理](#5-快速开始本地直接推理)
6. [VLM 推理服务（Docker 加速）](#6-vlm-推理服务docker-加速)
7. [完整 API 服务（Docker Compose）](#7-完整-api-服务docker-compose)
8. [产线配置调整](#8-产线配置调整)
9. [常见问题与排查](#9-常见问题与排查)
10. [附录：镜像与文件速查](#10-附录镜像与文件速查)

---

## 1. PaddleOCR-VL 是什么

PaddleOCR-VL 是面向 **文档解析** 的视觉语言模型产线，可把扫描件、照片、PDF 等转为结构化内容（Markdown、JSON、Word 等），支持 **109 种语言**，可识别正文、表格、公式、图表、印章等。

### 1.1 工作流程（必须理解）

完整解析 **不是** 只跑一个大模型，而是两阶段协作：

```
输入图像/PDF ──► 版面分析 ──► 裁剪各元素子图 ──► VLM 识别每个子图 ──► 按阅读顺序合并 ──► Markdown / JSON 等
```

| 阶段 | 作用 | 典型模型（v1.5） |
|------|------|------------------|
| 版面分析 | 检测标题、段落、表格、图片等区域并排序 | PP-DocLayout 系列 |
| VLM 识别 | 对每个区域子图生成文本/表格/公式等 | PaddleOCR-VL-1.5-0.9B |

**重要**：

- 只调用 vLLM 的 OpenAI 兼容接口、或只加载 VLM 权重，**不等于**完整 PaddleOCR-VL，容易出现幻觉或精度异常。
- 本文中的 **「VLM 推理服务」** 只负责第二阶段；**「完整 API 服务」** 才提供端到端文档解析。

### 1.2 RTX 5090 与官方镜像

RTX 5090 属于 **Blackwell** 架构，需使用带 **`sm120`** 后缀的镜像（内置 CUDA 12.9+ 与对应 PaddlePaddle）：

| 镜像用途 | 镜像名（后缀部分） |
|----------|-------------------|
| 本地推理 / CLI / Python | `paddleocr-vl:latest-nvidia-gpu-sm120` |
| VLM vLLM 服务 | `paddleocr-genai-vllm-server:latest-nvidia-gpu-sm120` |
| Compose 产线 + VLM | 见第 7 节，标签为 `latest-nvidia-gpu-sm120-offline` |

> 官方在 RTX 5070 上做过验证；RTX 5090 理论兼容同架构镜像，若遇问题可参考第 9 节排查并反馈社区。

---

## 2. RTX 5090 部署前准备

### 2.1 硬件与驱动

| 项目 | 要求 |
|------|------|
| GPU | RTX 5090（建议显存 ≥ 16GB；大 PDF、高并发建议 24GB+） |
| NVIDIA 驱动 | 支持 **CUDA 12.9 及以上**（Blackwell 硬性要求） |
| 磁盘 | 单个在线推理镜像约 **10GB**；VLM 服务镜像约 **13GB**；Compose 需同时拉取 API + VLM 镜像，建议预留 **40GB+** |
| 内存 | 建议 **32GB+**（Compose 默认 `shm_size: 64g`） |

在 PowerShell 或 CMD 中检查驱动与 GPU：

```powershell
nvidia-smi
```

确认右上角 **CUDA Version** 显示为 **12.9** 或更高。

### 2.2 安装 Docker 与 GPU 支持

**Windows（本机常见场景）**

1. 安装 [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/)（WSL2 后端）。
2. 在 Docker Desktop → **Settings → Resources → WSL Integration** 中启用你的 Linux 发行版。
3. 安装 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)（Windows 下通常随驱动 + WSL2 环境配置；若 `docker run --gpus all` 失败，需在 WSL2 内按 Linux 文档安装 toolkit）。

验证 Docker 能访问 GPU：

```powershell
docker run --rm --gpus all nvidia/cuda:12.9.0-base-ubuntu22.04 nvidia-smi
```

应能看到 RTX 5090 信息。

**Linux**

```bash
# 安装 Docker（略，按发行版官方文档）
# 安装 NVIDIA Container Toolkit 后执行：
docker run --rm --gpus all nvidia/cuda:12.9.0-base-ubuntu22.04 nvidia-smi
```

### 2.3 Docker 版本

官方要求 **Docker ≥ 19.03**。生产环境建议使用当前稳定版 Docker / Docker Compose V2。

```powershell
docker version
docker compose version
```

---

## 3. 三种部署形态怎么选

| 形态 | 适用场景 | 对外端口 | 本文章节 |
|------|----------|----------|----------|
| **A. 容器内直接推理** | 本机试效果、脚本批处理、开发调试 | 无（命令行/Python） | §4 + §5 |
| **B. 客户端 + VLM 服务** | 版面分析在本地，VLM 用 vLLM 加速，吞吐更高 | VLM：**8118** | §4 + §5 + §6 |
| **C. 完整 HTTP API** | 对外提供 REST、多语言调用、生产部署 | API：**8080** | §7 + §8 |

```
形态 A（单容器推理）
  paddleocr-vl 镜像 ──► paddleocr doc_parser / Python API

形态 B（客户端 + 分离 VLM）
  paddleocr-vl 客户端 ──► genai-vllm-server（:8118）

形态 C（Docker Compose 产线）
  浏览器/HTTP 客户端 ──► paddleocr-vl-api（:8080）──► paddleocr-vlm-server（内部）
```

**推荐路径**：先完成 **形态 A** 确认环境与精度 → 需要更快 VLM 时用 **形态 B** → 需要标准 HTTP 接口时用 **形态 C**。

---

## 4. 环境准备：推理镜像 Docker

本节拉取 **`paddleocr-vl`** 镜像，用于 §5 本地推理，也可作为 §6 的 **客户端容器**（与 VLM 服务容器分开运行）。

### 4.1 在线环境：启动交互式容器

```bash
docker run \
    -it \
    --gpus all \
    --network host \
    --user root \
    ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-vl:latest-nvidia-gpu-sm120 \
    /bin/bash
```

**参数说明**：

| 参数 | 含义 |
|------|------|
| `-it` | 交互式终端，便于手动执行命令 |
| `--gpus all` | 将 GPU 暴露给容器（RTX 5090） |
| `--network host` | 与宿主机共享网络栈，访问本机 `localhost:8118` 等服务更方便 |
| `--user root` | 官方示例使用 root，避免权限问题 |
| 镜像 `...-sm120` | Blackwell / RTX 5090 专用构建 |

进入容器后提示符类似 `root@xxx:/#`，即可运行 `paddleocr` 或 Python。

**Windows 注意**：若 `--network host` 在 Docker Desktop 上行为异常，可改为端口映射模式（示例）：

```powershell
docker run -it --gpus all -p 8888:8888 `
  -v C:\Users\你的用户名\ocr_data:/data `
  ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-vl:latest-nvidia-gpu-sm120 `
  /bin/bash
```

将待测图片放到宿主机 `ocr_data`，在容器内路径为 `/data`。

### 4.2 挂载本地目录（推荐）

避免每次拷文件进容器：

```bash
docker run -it --gpus all --network host --user root \
  -v /path/on/host/work:/work \
  ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-vl:latest-nvidia-gpu-sm120 \
  /bin/bash
```

Windows 示例：

```powershell
docker run -it --gpus all --network host --user root `
  -v C:\Users\<YOUR_USER>\Desktop\ocr_data:/work `
  ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-vl:latest-nvidia-gpu-sm120 `
  /bin/bash
```

### 4.3 离线环境

无法联网时，使用 **offline** 镜像（约 **12GB**，比在线版略大，已打包依赖与模型）：

```text
ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-vl:latest-nvidia-gpu-sm120-offline
```

**离线迁移步骤**（在能联网的机器上执行）：

```bash
# 1. 拉取
docker pull ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-vl:latest-nvidia-gpu-sm120-offline

# 2. 导出为 tar
docker save ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-vl:latest-nvidia-gpu-sm120-offline \
  -o paddleocr-vl-sm120-offline.tar

# 3. 将 tar 拷贝到 RTX 5090 机器后导入
docker load -i paddleocr-vl-sm120-offline.tar

# 4. 按 §4.1 方式 run，仅把镜像名换成 offline 版本
```

### 4.4 更新与固定版本

- **更新到最新 `latest`**：

```bash
docker pull ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-vl:latest-nvidia-gpu-sm120
```

- **固定 PaddleOCR 版本**（示例 3.3）：

```text
paddleocr-vl:paddleocr3.3-nvidia-gpu-sm120-offline
```

---

## 5. 快速开始：本地直接推理

以下命令均在 **§4 启动的 `paddleocr-vl` 容器内** 执行（已映射 `/work` 时，先把测试图放到宿主机对应目录）。

> **说明**：本节为「形态 A」，VLM 与版面分析都在同一容器、同一进程链路中完成，适合验证功能；**生产环境**更推荐 §6 或 §7 以获得更好吞吐与稳定性。

### 5.1 准备测试图片

**方式一：容器内下载官方样例**

```bash
cd /work
curl -L -o paddleocr_vl_demo.png \
  https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/paddleocr_vl_demo.png
```

**方式二**：将你自己的 `jpg/png/pdf` 放到宿主机挂载目录，在容器内使用 `/work/你的文件`。

首次运行会自动下载版面分析与 VLM 相关权重（**在线镜像**需要网络；**offline 镜像**已内置，无需下载）。

### 5.2 命令行（CLI）推理

RTX 5090 上显式指定 GPU：

```bash
paddleocr doc_parser \
  -i /work/paddleocr_vl_demo.png \
  --device gpu \
  --save_path /work/output
```

**常用扩展**：

```bash
# 使用 v1.5 产线（默认即为 v1.5，可省略）
paddleocr doc_parser -i /work/demo.png --pipeline_version v1.5 --device gpu --save_path /work/output

# 开启文档方向分类
paddleocr doc_parser -i /work/demo.png --device gpu \
  --use_doc_orientation_classify True --save_path /work/output

# 开启弯曲/透视矫正
paddleocr doc_parser -i /work/demo.png --device gpu \
  --use_doc_unwarping True --save_path /work/output

# 关闭版面分析（仅对整图做单一类型 VLM 识别，特殊场景用）
paddleocr doc_parser -i /work/demo.png --device gpu \
  --use_layout_detection False --save_path /work/output

# 开启图表识别
paddleocr doc_parser -i /work/demo.png --device gpu \
  --use_chart_recognition True --save_path /work/output

# 开启印章识别（v1.5）
paddleocr doc_parser -i /work/demo.png --device gpu \
  --use_seal_recognition True --save_path /work/output
```

**指定 GPU 卡号**（多卡机器）：

```bash
paddleocr doc_parser -i /work/demo.png --device gpu:0 --save_path /work/output
```

**PDF 文件**：

```bash
paddleocr doc_parser -i /work/document.pdf --device gpu --save_path /work/output
```

执行成功后：

- 终端会打印结构化 JSON 摘要；
- `/work/output` 下会生成与输入同名的结果文件（如 Markdown、JSON 等，取决于保存选项）。

### 5.3 Python API 推理

在容器内创建脚本 `/work/run_ocr.py`：

```python
from pathlib import Path

from paddleocr import PaddleOCRVL

input_file = "/work/paddleocr_vl_demo.png"
output_dir = Path("/work/output")
output_dir.mkdir(parents=True, exist_ok=True)

# RTX 5090：使用 GPU；默认 pipeline_version 为 v1.5
pipeline = PaddleOCRVL(device="gpu")

output = pipeline.predict(input_file)

for res in output:
    res.print()                          # 终端打印
    res.save_to_json(save_path=output_dir)
    res.save_to_markdown(save_path=output_dir)
    res.save_to_word(save_path=output_dir)  # 需要 word 依赖时可用
```

运行：

```bash
python /work/run_ocr.py
```

**处理 PDF 并合并多页**（跨页表格、标题层级等）：

```python
from pathlib import Path
from paddleocr import PaddleOCRVL

pipeline = PaddleOCRVL(device="gpu")
output = pipeline.predict(input="/work/document.pdf")
pages_res = list(output)

# 合并跨页表格、重建标题、合并为单页 Markdown
output = pipeline.restructure_pages(
    pages_res,
    merge_tables=True,
    relevel_titles=True,
    concatenate_pages=True,
)

output_dir = Path("/work/output")
for res in output:
    res.save_to_markdown(save_path=output_dir)
```

**批量处理多张图**（效率高于循环单张）：

```python
pipeline = PaddleOCRVL(device="gpu")
output = pipeline.predict("/work/images")           # 目录
# 或
output = pipeline.predict([
    "/work/images/a.png",
    "/work/images/b.png",
])
```

### 5.4 输出结果说明

每个 `predict` 结果对象支持：

| 方法 | 作用 |
|------|------|
| `print()` | 在终端查看结构化结果 |
| `save_to_json()` | 保存 JSON（含版面框、类别、文本等） |
| `save_to_markdown()` | 保存 Markdown |
| `save_to_word()` | 保存 Word |
| `save_to_img()` | 保存可视化中间图 |

JSON / 打印结果中常见字段：

- **`layout_det_res` / `parsing_res_list`**：版面区域列表（坐标、标签如 `text`/`table`/`formula`、识别内容）；
- **`block_label`**：区域类型；
- **`block_content`**：该区域识别出的文本或结构化内容；
- **`block_order`**：阅读顺序。

若结果与预期差距大，请先确认使用的是 **完整 doc_parser 产线**，而非单独调用 VLM。

### 5.5 形态 A 的性能说明

默认在容器内用 PaddlePaddle 直接跑 VLM，**速度、显存占用未必最优**。若处理大批量 PDF 或需要稳定 SLA，请继续阅读 §6（分离 VLM）或 §7（HTTP 产线）。

---

## 6. VLM 推理服务（Docker 加速）

### 6.1 服务是什么、不是什么

| 项目 | 说明 |
|------|------|
| **是什么** | 仅运行 **PaddleOCR-VL-1.5-0.9B** 的 vLLM 加速服务，供客户端把裁剪后的子图发来做识别 |
| **不是什么** | 不是完整文档 OCR API；不要用 OpenAI 客户端把整页文档图直接丢进去当「完整解析」 |
| **监听地址** | 默认 `http://0.0.0.0:8118`，OpenAI 兼容路径一般为 `http://localhost:8118/v1` |

架构关系（时序）：

```
1. paddleocr-vl 客户端容器 ──整图──► 版面分析（本地 GPU）
2. 版面分析 ──子图列表──► 客户端
3. 客户端 ──并发请求──► genai-vllm-server（:8118）
4. VLM 服务 ──文本/表格等──► 客户端
5. 客户端 ──合并──► 最终 Markdown
```

### 6.2 启动 VLM 服务容器

在 **宿主机** 新开一个终端（与 §4 推理容器分开）：

```bash
docker run \
    -it \
    --gpus all \
    --network host \
    ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-vllm-server:latest-nvidia-gpu-sm120 \
    paddleocr genai_server \
      --model_name PaddleOCR-VL-1.5-0.9B \
      --host 0.0.0.0 \
      --port 8118 \
      --backend vllm
```

**首次启动**可能持续数分钟：加载权重、编译 CUDA 内核。日志中出现服务就绪、监听 8118 后再测客户端。

**离线镜像**（约 15GB）：

```text
ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-vllm-server:latest-nvidia-gpu-sm120-offline
```

**后台运行示例**：

```bash
docker run -d --name paddleocr-vlm --gpus all --network host \
  ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-vllm-server:latest-nvidia-gpu-sm120 \
  paddleocr genai_server --model_name PaddleOCR-VL-1.5-0.9B --host 0.0.0.0 --port 8118 --backend vllm

docker logs -f paddleocr-vlm
```

### 6.3 自定义 vLLM 配置（显存、并发）

在宿主机创建 `vllm_config.yml`（路径示例：`C:\Users\<YOUR_USER>\Desktop\ocr_data\vllm_config.yml`）：

```yaml
# 显存占用比例（0~1），5090 显存大时可适当提高
gpu-memory-utilization: 0.85
# 最大并发序列数，影响吞吐与显存
max-num-seqs: 128
```

挂载并启动：

```bash
docker run -it --rm --gpus all --network host \
  -v C:/Users/<YOUR_USER>/Desktop/ocr_data/vllm_config.yml:/tmp/vllm_config.yml \
  ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-vllm-server:latest-nvidia-gpu-sm120 \
  paddleocr genai_server \
    --model_name PaddleOCR-VL-1.5-0.9B \
    --host 0.0.0.0 --port 8118 --backend vllm \
    --backend_config /tmp/vllm_config.yml
```

Linux 将 `-v` 左侧换为 `/home/user/vllm_config.yml` 即可。

更多 vLLM 参数见 [vLLM 官方文档](https://docs.vllm.ai/)（如 `max-model-len`、`tensor-parallel-size` 等，按 YAML 键名写入配置文件）。

### 6.4 客户端如何连接 VLM 服务

客户端仍需 **`paddleocr-vl` 镜像**（§4），在其中执行；版面分析在客户端容器内用 GPU 完成，VLM 请求发往 `8118`。

#### 6.4.1 CLI

```bash
paddleocr doc_parser \
  --input /work/paddleocr_vl_demo.png \
  --device gpu \
  --vl_rec_backend vllm-server \
  --vl_rec_server_url http://localhost:8118/v1 \
  --save_path /work/output
```

#### 6.4.2 Python

```python
from pathlib import Path
from paddleocr import PaddleOCRVL

pipeline = PaddleOCRVL(
    device="gpu",
    vl_rec_backend="vllm-server",
    vl_rec_server_url="http://localhost:8118/v1",
)

output = pipeline.predict("/work/paddleocr_vl_demo.png")
for res in output:
    res.save_to_markdown(save_path="/work/output")
```

若 VLM 服务在另一台机器，将 `localhost` 改为该机器 IP，并保证防火墙放行 **8118**。

#### 6.4.3 客户端并发调优

客户端会对多张裁剪子图 **并发** 请求 VLM。可通过 CLI / Python 调整：

```bash
paddleocr doc_parser \
  --input /work/demo.png --device gpu \
  --vl_rec_backend vllm-server \
  --vl_rec_server_url http://localhost:8118/v1 \
  --vl_rec_max_concurrency 16 \
  --save_path /work/output
```

```python
pipeline = PaddleOCRVL(
    device="gpu",
    vl_rec_backend="vllm-server",
    vl_rec_server_url="http://localhost:8118/v1",
    vl_rec_max_concurrency=16,
)
```

| 场景 | 建议 |
|------|------|
| 单客户端、5090 独占 | 可适当提高 `vl_rec_max_concurrency` 与 `max-num-seqs` |
| 多客户端共享一台 VLM | 降低并发，避免 OOM 或超时 |
| 显存不足 | 降低 `gpu-memory-utilization`、`max-num-seqs`、客户端并发 |

---

## 7. 完整 API 服务（Docker Compose）

「形态 C」：一条命令启动 **VLM 服务 + 产线 API**，对外提供 HTTP **8080**，适合对接业务系统。

### 7.1 下载 Compose 与 .env（RTX 5090 / sm120 专用）

在宿主机创建工作目录，例如 `C:\Users\<YOUR_USER>\Desktop\paddleocr-compose`，下载以下两个文件到**同一目录**：

| 文件 | 下载地址 |
|------|----------|
| `compose.yaml` | https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/deploy/paddleocr_vl_docker/accelerators/nvidia-gpu-sm120/compose.yaml |
| `.env` | https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/deploy/paddleocr_vl_docker/accelerators/nvidia-gpu-sm120/.env |

**PowerShell 下载示例**：

```powershell
cd C:\Users\<YOUR_USER>\Desktop\paddleocr-compose
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/deploy/paddleocr_vl_docker/accelerators/nvidia-gpu-sm120/compose.yaml" -OutFile compose.yaml
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/deploy/paddleocr_vl_docker/accelerators/nvidia-gpu-sm120/.env" -OutFile .env
```

默认 `.env` 内容为：

```env
API_IMAGE_TAG_SUFFIX=latest-nvidia-gpu-sm120-offline
VLM_BACKEND=vllm
VLM_IMAGE_TAG_SUFFIX=latest-nvidia-gpu-sm120-offline
```

即：API 与 VLM 均使用 **sm120 离线 GPU 镜像**，无需启动后再拉模型。

### 7.2 启动服务

```bash
cd C:\Users\<YOUR_USER>\Desktop\paddleocr-compose

# 建议先拉取最新镜像
docker compose pull

# 前台启动（可看日志）
docker compose up

# 或后台启动
docker compose up -d
```

**启动顺序**（由 Compose 依赖与健康检查控制）：

1. **`paddleocr-vlm-server`**：内部 vLLM，默认端口 8118（容器内，不必须映射到宿主机）；
2. **`paddleocr-vl-api`**：产线服务，映射 **宿主机 8080 → 容器 8080**。

成功日志示例：

```text
paddleocr-vl-api  | INFO:     Uvicorn running on http://0.0.0.0:8080
```

**健康检查**（可选）：

```powershell
curl http://localhost:8080/health
```

### 7.3 Compose 文件结构说明

官方 `compose.yaml` 核心逻辑（便于你改配置）：

| 服务名 | 镜像 | 作用 |
|--------|------|------|
| `paddleocr-vlm-server` | `paddleocr-genai-vllm-server:${VLM_IMAGE_TAG_SUFFIX}` | VLM vLLM 后端 |
| `paddleocr-vl-api` | `paddleocr-vl:${API_IMAGE_TAG_SUFFIX}` | 版面分析 + 调用 VLM + HTTP API |

- 两个服务默认都使用 **`device_ids: ["0"]`**（第一块 GPU）。RTX 5090 单卡机器无需修改。
- `shm_size: 64g`：增大共享内存，避免大模型 / 多进程 OOM。
- API 启动命令：`paddlex --serve --pipeline /home/paddleocr/pipeline_config_vllm.yaml`（已配置好连接内部 VLM）。

### 7.4 修改端口、GPU、VLM 参数

**改对外端口**（例如改为 8111）：

编辑 `compose.yaml` 中 `paddleocr-vl-api.ports`：

```yaml
ports:
  - 8111:8080   # 宿主机 8111 → 容器 8080
```

**指定 GPU**（多卡时改用卡 1）：

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          device_ids: ["1"]
          capabilities: [gpu]
```

`paddleocr-vl-api` 与 `paddleocr-vlm-server` **建议改为同一张卡**（或按显存拆卡，需自行评估显存）。

**挂载 VLM 调优配置**：

在 `paddleocr-vlm-server` 下增加：

```yaml
volumes:
  - ./vlm_server_config.yaml:/home/paddleocr/vlm_server_config.yaml
command: paddleocr genai_server --model_name PaddleOCR-VL-1.5-0.9B --host 0.0.0.0 --port 8118 --backend vllm --backend_config /home/paddleocr/vlm_server_config.yaml
```

### 7.5 离线部署 Compose

在**能联网**的机器上：

```bash
cd paddleocr-compose
docker compose pull
docker save $(docker images -q ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-vl) -o paddleocr-vl-api.tar
docker save $(docker images -q ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-vllm-server) -o paddleocr-vlm.tar
```

将 tar、`compose.yaml`、`.env` 拷到 RTX 5090 离线机后：

```bash
docker load -i paddleocr-vl-api.tar
docker load -i paddleocr-vlm.tar
docker compose up -d
```

### 7.6 HTTP API 调用说明

服务启动后，基址为 **`http://localhost:8080`**（若改了端口则用新端口）。

#### 7.6.1 版面解析 `POST /layout-parsing`

**请求**：JSON，`Content-Type: application/json`

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `file` | string | 是 | 图片/PDF 的 **Base64**，或可访问的 **URL** |
| `fileType` | int | 否 | `0`=PDF，`1`=图像；不传则自动推断 |
| `useLayoutDetection` | bool | 否 | 是否版面分析，对应 Python `use_layout_detection` |
| `useSealRecognition` | bool | 否 | 是否印章识别 |
| `useChartRecognition` | bool | 否 | 是否图表解析 |
| `visualize` | bool | 否 | 是否返回可视化图（Base64） |

**Python 调用完整示例**（保存 Markdown）：

```python
import base64
import pathlib
import requests

BASE_URL = "http://localhost:8080"
image_path = "./demo.jpg"

with open(image_path, "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode("ascii")

payload = {
    "file": image_b64,
    "fileType": 1,
}

resp = requests.post(f"{BASE_URL}/layout-parsing", json=payload, timeout=600)
resp.raise_for_status()
result = resp.json()["result"]

for i, page in enumerate(result["layoutParsingResults"]):
    md_text = page["markdown"]["text"]
    out = pathlib.Path("markdown") / f"page_{i}.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text(md_text, encoding="utf-8")
    print(f"已保存 {out}")
```

**curl 示例**（需先将图片转为 Base64 写入文件或使用脚本生成）：

```bash
# 建议用 Python 脚本发请求；纯 curl 处理大 Base64 不便
curl -X POST http://localhost:8080/layout-parsing \
  -H "Content-Type: application/json" \
  -d "{\"fileType\":1,\"file\":\"<BASE64_STRING>\"}"
```

**成功响应结构（摘要）**：

```json
{
  "logId": "...",
  "errorCode": 0,
  "errorMsg": "Success",
  "result": {
    "layoutParsingResults": [
      {
        "prunedResult": { },
        "markdown": { "text": "...", "images": { } },
        "outputImages": { }
      }
    ],
    "dataInfo": { }
  }
}
```

#### 7.6.2 多页重构 `POST /restructure-pages`

对 PDF 多页结果做跨页表格合并、标题分级、合并为一页 Markdown：

```python
import requests

BASE_URL = "http://localhost:8080"

# pages 来自 layout-parsing 的 prunedResult 与 markdown.images
pages = [
    {"prunedResult": res["prunedResult"], "markdownImages": res["markdown"].get("images")}
    for res in layout_parsing_results
]

payload = {
    "pages": pages,
    "mergeTables": True,
    "relevelTitles": True,
    "concatenatePages": True,
}

resp = requests.post(f"{BASE_URL}/restructure-pages", json=payload, timeout=600)
resp.raise_for_status()
merged = resp.json()["result"]["layoutParsingResults"][0]
pathlib.Path("doc.md").write_text(merged["markdown"]["text"], encoding="utf-8")
```

### 7.7 停止与重启

```bash
cd C:\Users\<YOUR_USER>\Desktop\paddleocr-compose
docker compose down      # 停止并删除容器
docker compose up -d     # 再次启动
docker compose logs -f   # 查看日志
```

---

## 8. 产线配置调整

仅在使用 **§7 Compose** 或需要自定义服务行为时需要本节。

### 8.1 获取配置文件

vLLM 后端产线配置（与 Compose 默认一致）：

- 下载：https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/deploy/paddleocr_vl_docker/accelerators/nvidia-gpu-sm120/pipeline_config_vllm.yaml  

（若仓库路径有变，可在 GitHub 搜索 `pipeline_config_vllm.yaml` + `nvidia-gpu-sm120`。）

### 8.2 常用修改示例

**指向外部 VLM 服务**（不用 Compose 内置 VLM 时）：

```yaml
VLRecognition:
  genai_config:
    backend: vllm-server
    server_url: http://localhost:8118/v1
```

**启用文档预处理**（方向分类、矫正）：

```yaml
# 在配置中设置
use_doc_preprocessor: True
```

**关闭默认可视化（降低开销）**：

```yaml
Serving:
  visualize: False
```

**限制 PDF 最大页数**：

```yaml
Serving:
  extra:
    max_num_input_imgs: 100
```

**提高 VLM 并发（服务侧）**：

在配置中查找 `VLRecognition.genai_config.max_concurrency` 并调整（与 §6.4.3 客户端并发配合）。

### 8.3 应用到 Compose

将 `pipeline_config_vllm.yaml` 放在 `paddleocr-compose` 目录，修改 `compose.yaml`：

```yaml
services:
  paddleocr-vl-api:
    volumes:
      - ./pipeline_config_vllm.yaml:/home/paddleocr/pipeline_config_vllm.yaml
```

确保 `command` 仍指向 `/home/paddleocr/pipeline_config_vllm.yaml`，然后：

```bash
docker compose up -d --force-recreate
```

---

## 9. 常见问题与排查

| 现象 | 可能原因 | 处理办法 |
|------|----------|----------|
| `docker run --gpus all` 报错 | 未装 NVIDIA Container Toolkit / WSL2 未配 GPU | 按 §2.2 重装 toolkit；重启 Docker |
| 驱动 CUDA 版本 &lt; 12.9 | Blackwell 不支持旧 CUDA | 升级 NVIDIA 驱动到支持 CUDA 12.9+ |
| 容器内 `nvidia-smi` 看不到 5090 | GPU 未透传 | 检查 Docker Desktop GPU 支持、WSL2 |
| 首次推理很慢 | 在线镜像下载模型 | 换 `offline` 镜像或提前 `docker pull` |
| 显存 OOM | 默认 vLLM 占满显存或并发过高 | 降低 `gpu-memory-utilization`、`max-num-seqs`、`vl_rec_max_concurrency` |
| 结果乱码 / 大量幻觉 | 只调了 VLM、未走完整产线 | 使用 `doc_parser` 或 `/layout-parsing`，勿整页直投 VLM |
| 连不上 `8118` | VLM 未启动或网络模式不对 | `docker logs` 查 VLM 容器；`--network host` 或正确端口映射 |
| Compose API 一直 Starting | VLM 健康检查未通过 | `docker compose logs paddleocr-vlm-server`；等待模型加载完成（可达数分钟） |
| Windows 路径挂载失败 | 反斜杠或盘符格式 | 使用 `-v C:/Users/...:/work` 形式 |

**查看容器日志**：

```bash
docker logs -f <容器名或ID>
docker compose logs -f paddleocr-vl-api
docker compose logs -f paddleocr-vlm-server
```

---

## 10. 附录：镜像与文件速查

### 10.1 镜像完整地址

仓库前缀：

```text
ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/
```

| 用途 | 在线标签 | 离线标签 |
|------|----------|----------|
| 推理 / 客户端 | `paddleocr-vl:latest-nvidia-gpu-sm120` | `...-sm120-offline` |
| VLM vLLM | `paddleocr-genai-vllm-server:latest-nvidia-gpu-sm120` | `...-sm120-offline` |

### 10.2 端口

| 服务 | 端口 |
|------|------|
| VLM genai_server | 8118 |
| 产线 HTTP API | 8080 |

### 10.3 关键命令一览

```bash
# 推理容器
docker run -it --gpus all --network host --user root \
  -v /work:/work \
  ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-vl:latest-nvidia-gpu-sm120 /bin/bash

# 容器内 CLI
paddleocr doc_parser -i /work/demo.png --device gpu --save_path /work/output

# VLM 服务
docker run -it --gpus all --network host \
  ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-vllm-server:latest-nvidia-gpu-sm120 \
  paddleocr genai_server --model_name PaddleOCR-VL-1.5-0.9B --host 0.0.0.0 --port 8118 --backend vllm

# Compose 产线
docker compose pull && docker compose up -d
```

### 10.4 官方文档（可选延伸阅读）

- Blackwell 专题：https://www.paddleocr.ai/latest/version3.x/pipeline_usage/PaddleOCR-VL-NVIDIA-Blackwell.html  
- PaddleOCR-VL 总览：https://www.paddleocr.ai/latest/version3.x/pipeline_usage/PaddleOCR-VL.html  

---

**文档版本**：面向 RTX 5090 / sm120 / Docker-only 本地部署；若镜像标签或 GitHub 路径随 PaddleOCR 更新发生变化，请以 [PaddleOCR GitHub `nvidia-gpu-sm120`](https://github.com/PaddlePaddle/PaddleOCR/tree/main/deploy/paddleocr_vl_docker/accelerators/nvidia-gpu-sm120) 目录为准。
