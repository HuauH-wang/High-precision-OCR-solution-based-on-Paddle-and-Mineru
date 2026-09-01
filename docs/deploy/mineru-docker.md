# MinerU Docker 部署教程

> 本文整理 MinerU 官方 Docker 部署说明，涵盖镜像构建、容器启动、Docker Compose 多服务配置及常用访问方式。

---

## 目录

1. [部署前须知](#1-部署前须知)
2. [使用 Dockerfile 构建镜像](#2-使用-dockerfile-构建镜像)
3. [Docker 镜像说明](#3-docker-镜像说明)
4. [启动 Docker 容器（交互式）](#4-启动-docker-容器交互式)
5. [通过 Docker Compose 启动服务](#5-通过-docker-compose-启动服务)
6. [各服务 profile 说明](#6-各服务-profile-说明)
7. [端口与服务速查](#7-端口与服务速查)

---

## 1. 部署前须知

### 1.1 Apple Silicon 限制

在 **Docker** 中部署 MinerU 时，**无法调用 macOS 上的 MPS 和 MLX 加速**。因此 **Apple Silicon（M 系列）设备** 通过该 Docker 方案 **无法获得** 在原生 macOS 上使用 MLX 时的预期加速效果。

若你使用 Mac 且依赖 MLX/MPS，请考虑非 Docker 的本地安装路径；本文以下均以 **带 NVIDIA GPU 的 Linux / Windows（WSL2 + GPU）** 为主要场景。

### 1.2 使用 vLLM 加速的前提（重要）

MinerU 的 Docker 基于 **`vllm/vllm-openai`** 镜像，默认集成 **vLLM** 及依赖。在满足条件时，可直接用 vLLM 加速 VLM 推理。

| 条件 | 要求 |
|------|------|
| GPU 架构 | **Volta 及以后**（如 V100、T4、RTX 20/30/40/50 等） |
| 显存 | 可用显存 **≥ 8GB** |
| 驱动 | 物理机 NVIDIA 驱动支持 **CUDA 12.9.1+**（`nvidia-smi` 查看） |
| 容器 GPU | Docker 能访问宿主机 GPU（`docker run --gpus all` 等） |

### 1.3 环境检查

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.9.0-base-ubuntu22.04 nvidia-smi
docker version
docker compose version
```

---

## 2. 使用 Dockerfile 构建镜像

在空目录或项目目录中执行：

```bash
# 下载国内加速源 Dockerfile（官方 china 路径）
wget https://gcore.jsdelivr.net/gh/opendatalab/MinerU@master/docker/china/Dockerfile

# 构建镜像
docker build -t mineru:latest -f Dockerfile .
```

构建完成后本地镜像名为 **`mineru:latest`**，后续 `docker run` / Compose 可引用该标签（若 Compose 使用其他镜像名，需与构建标签一致或修改 compose 配置）。

---

## 3. Docker 镜像说明

- **基础镜像**：`vllm/vllm-openai`
- **已包含**：vLLM 推理加速框架及运行 MinerU 所需的主要依赖
- **能力**：在 GPU 与驱动满足 §1.2 时，容器内可直接使用 **vLLM** 加速 VLM，无需再单独拼装 vLLM 环境

---

## 4. 启动 Docker 容器（交互式）

进入容器交互终端，并映射常用服务端口：

```bash
docker run --gpus all \
  --shm-size 32g \
  -p 30000:30000 -p 7860:7860 -p 8000:8000 -p 8002:8002 \
  --ipc=host \
  -it mineru:latest \
  /bin/bash
```

| 参数 | 含义 |
|------|------|
| `--gpus all` | 使用全部 GPU |
| `--shm-size 32g` | 增大共享内存，避免大模型 / vLLM OOM |
| `-p 30000:30000` 等 | 映射 OpenAI 兼容服务、Gradio、API、Router 等端口（见 §7） |
| `--ipc=host` | 与宿主机共享 IPC，利于多进程 / vLLM |
| `/bin/bash` | 进入 shell，在容器内手动执行 `mineru` 命令 |

进入容器后，可直接运行 MinerU CLI。若要以 **服务方式** 启动（而非交互 shell），可将末尾 **`/bin/bash`** 替换为官方文档中的 **服务启动命令**（参见 MinerU 文档「通过命令启动服务」）。

---

## 5. 通过 Docker Compose 启动服务

### 5.1 获取 compose 文件

```bash
wget https://gcore.jsdelivr.net/gh/opendatalab/MinerU@master/docker/compose.yaml
```

### 5.2 使用说明

- `compose.yaml` 内含 **多种服务配置**，通过 **`--profile`** 选择启动哪一类服务。
- 不同 profile 可能有 **额外环境变量或命令行参数**，需在 `compose.yaml` 中查看并按需修改。
- **显存注意**：vLLM 会 **预分配显存**，同一台机器上通常 **不宜同时跑多个 vLLM 服务**。启动 **`openai-server`** 或使用 **`vlm-vllm-engine`** 后端前，请先停止其他占用 GPU 显存的服务。

### 5.3 通用命令格式

```bash
docker compose -f compose.yaml --profile <profile名> up -d
```

查看日志：

```bash
docker compose -f compose.yaml --profile <profile名> logs -f
```

停止：

```bash
docker compose -f compose.yaml --profile <profile名> down
```

---

## 6. 各服务 profile 说明

### 6.1 OpenAI 兼容接口（`openai-server`）

**启动**

```bash
docker compose -f compose.yaml --profile openai-server up -d
```

**作用**：在容器内提供 **OpenAI 兼容的 VLM HTTP 服务**（默认相关端口含 **30000**，以 compose 为准）。

**客户端连接（vlm-http-client）**

在 **另一终端** 使用 MinerU，以 HTTP 客户端后端连接上述服务（该客户端侧 **只需 CPU 与网络**，不要求本机安装 vLLM）：

```bash
mineru -p <input_path> -o <output_path> -b vlm-http-client -u http://<server_ip>:30000
```

| 占位符 | 说明 |
|--------|------|
| `<input_path>` | 待解析 PDF/文档路径 |
| `<output_path>` | 输出目录 |
| `<server_ip>` | 运行 Compose 的机器 IP；本机可为 `127.0.0.1` |

---

### 6.2 Web API 服务（`api`）

**启动**

```bash
docker compose -f compose.yaml --profile api up -d
```

**访问**

浏览器打开：

```text
http://<server_ip>:8000/docs
```

查看并调试 **REST API**（Swagger 文档）。

---

### 6.3 MinerU Router 服务（`router`）

**启动**

```bash
docker compose -f compose.yaml --profile router up -d
```

**默认行为**

- 以 **`--local-gpus auto`** 在容器内 **自动拉起本地 worker**。
- 统一入口文档：**`http://<server_ip>:8002/docs`**。

**聚合已有 API（不启本地 worker）**

若已有独立的 `mineru-api` 实例，希望 Router 只做转发，可在 `compose.yaml` 中查看 **`mineru-router`** 服务下的 **注释示例**，改为使用 **`--upstream-url`** 指向已有服务地址。

---

### 6.4 Gradio WebUI（`gradio`）

**启动**

```bash
docker compose -f compose.yaml --profile gradio up -d
```

**访问**

浏览器打开：

```text
http://<server_ip>:7860
```

使用 **Gradio Web 界面** 上传文件并查看解析结果。

---

## 7. 端口与服务速查

| 端口（默认映射） | 常见用途 |
|------------------|----------|
| **30000** | OpenAI 兼容 VLM 服务（`vlm-http-client` 连接 `-u http://...:30000`） |
| **7860** | Gradio WebUI |
| **8000** | Web API（`/docs`） |
| **8002** | MinerU Router（`/docs`） |

> 实际端口以 `compose.yaml` 与 `docker run -p` 配置为准；若修改映射，访问 URL 中的端口需同步修改。

---

## 8. 推荐部署路径

```
需要完整 GPU 解析（容器内 vLLM）
  → docker build → docker run --gpus all（手动 mineru）
  或 compose --profile openai-server / api / gradio / router

仅需轻量客户端、GPU 在远端
  → compose --profile openai-server
  → 本机/其他机器：mineru -b vlm-http-client -u http://<server>:30000
```

---

## 9. 常见问题

| 现象 | 可能原因 | 处理 |
|------|----------|------|
| 容器内无 GPU | 未加 `--gpus all` 或未装 NVIDIA Container Toolkit | 安装 toolkit 后重试 |
| vLLM OOM | 显存不足或多服务同时占用 | 只启一个 vLLM profile；减小并发或换更大显存卡 |
| Mac Docker 很慢 | 无 CUDA/vLLM 加速 | 预期行为；考虑非 Docker 或远程 Linux GPU 机 |
| 连不上 30000 | 防火墙 / 端口未映射 | 检查 `docker ps` 端口映射与 `<server_ip>` |

---

## 10. 参考资源

| 资源 | 地址 |
|------|------|
| MinerU 仓库 | https://github.com/opendatalab/MinerU |
| Dockerfile（china） | https://github.com/opendatalab/MinerU/blob/master/docker/china/Dockerfile |
| compose.yaml | https://github.com/opendatalab/MinerU/blob/master/docker/compose.yaml |

---

**文档说明**：内容来源于 MinerU 官方 Docker 部署说明的整理，便于本地查阅；若上游仓库路径或 profile 名称变更，请以 GitHub `opendatalab/MinerU` 仓库 `docker/` 目录为准。
