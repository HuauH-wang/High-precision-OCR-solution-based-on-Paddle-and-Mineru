# PaddleOCR 模块列表 · 本地部署汇总

> 对应官网 **模块列表** 侧边栏全部条目（PaddleOCR 3.x）。  
> **本地部署**指在本机完成推理（非官网在线 API）；支持 **pip 安装** 与 **Docker 镜像** 两种方式。  
> 文档页基址：`https://www.paddleocr.ai/latest/version3.x/module_usage/`

---

## 一、模块概述（概念）

**模块**是 PaddleOCR 的最小功能单元：通常 **一个模块 = 一个模型 + 一套 API**，完成单一任务（如文本检测、版面分析）。多个模块可组合成 **产线**（如通用 OCR、PP-StructureV3、PaddleOCR-VL）。

下文 **§三** 起为各模块的本地部署要点；**§二** 为所有模块共用的环境步骤，只需做一次。

---

## 二、统一环境准备（所有模块共用）

### 2.1 方式 A：pip 本地安装（推荐单模块调试）

**Python**：3.9–3.13（使用 `doc-parser` / `all` 等扩展组时建议 ≥3.9）。

```bash
# 1. 飞桨 GPU（CUDA 按本机选择，示例 cu126；RTX 5090/Blackwell 用 cu129）
python -m pip install paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu129/

# 2. PaddleOCR
# 仅 OCR + 文档预处理类模块（§3.1–3.2、3.11–3.13）
python -m pip install -U paddleocr

# 含版面/表格/公式/印章/图表/DocVLM 等（§3.3–3.10、3.14）
python -m pip install -U "paddleocr[doc-parser]"

# 或一次装全
# python -m pip install -U "paddleocr[all]"
```

**可选**：使用 Transformers 引擎的模块，另装 `transformers>=5.8.0` 及对应 torch。

**模型下载**：首次推理自动拉取权重。访问 HuggingFace 不便时：

```bash
set PADDLE_PDX_MODEL_SOURCE=BOS    # Windows CMD
# export PADDLE_PDX_MODEL_SOURCE=BOS   # Linux
```

**指定 GPU**（各模块 CLI / Python 通用）：

```bash
--device gpu          # 或 gpu:0
```

---

### 2.2 方式 B：Docker 本地运行（GPU 整机环境一致）

适用于 NVIDIA GPU（含 RTX 5090 / Blackwell，使用 **sm120** 镜像）。

```bash
docker run -it --gpus all --network host --user root \
  -v /path/to/work:/work \
  ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-vl:latest-nvidia-gpu-sm120 \
  /bin/bash
```

进入容器后，环境与 pip 安装等价，直接执行下文各模块的 `paddleocr ...` 命令（路径改为 `/work/...`）。

| 镜像 | 用途 |
|------|------|
| `paddleocr-vl:latest-nvidia-gpu-sm120` | 在线，约 10GB |
| `paddleocr-vl:latest-nvidia-gpu-sm120-offline` | 离线，约 12GB |

---

### 2.3 三种本地使用形态

| 形态 | 说明 | 适用 |
|------|------|------|
| **CLI** | `paddleocr <子命令> -i <输入>` | 快速验证、脚本 |
| **Python API** | `from paddleocr import Xxx` → `predict()` | 业务集成 |
| **产线组合** | 不单独调模块，用 `paddleocr ocr` / `pp_structurev3` / `doc_parser` 等 | 端到端场景 |

单模块 **不提供** 独立 Docker Compose；若需 HTTP 服务，用 **产线级** `paddlex --serve`（见 §四）。

---

### 2.4 推理引擎与依赖组对照

| 引擎 | 适用模块（默认） |
|------|------------------|
| `paddle_static` | 多数检测/识别/分类/表格/公式/印章模块 |
| `paddle_dynamic` | 文档类 VLM、图表解析 |
| `transformers` | 上述模块均可通过 `--engine transformers` 切换（需额外安装） |

| pip 依赖组 | 覆盖模块 |
|------------|----------|
| `paddleocr`（默认） | 文档方向分类、文本图像矫正、文本行方向、文本检测、文本识别 |
| `paddleocr[doc-parser]` | 另含版面检测/分析、表格三模块、公式、印章、DocVLM、图表解析 |
| `paddleocr[all]` | 全部可选能力 |

---

## 三、各模块本地部署（与官网列表一一对应）

### 总览表

| # | 官网名称 | CLI 子命令 | Python 类 | 默认模型 | 默认引擎 | pip 依赖 |
|---|----------|------------|-----------|----------|----------|----------|
| 1 | 文档图像方向分类 | `doc_img_orientation_classification` | `DocImgOrientationClassification` | `PP-LCNet_x1_0_doc_ori` | paddle_static | 基础包 |
| 2 | 文档类视觉语言模型 | `doc_vlm` | `DocVLM` | `PP-DocBee2-3B`（未指定时文档写 `PP-DocBee-2B`） | paddle_dynamic | doc-parser |
| 3 | 公式识别 | `formula_recognition` | `FormulaRecognition` | `PP-FormulaNet_plus-M` | paddle_static | doc-parser |
| 4 | 版面区域检测 | `layout_detection` | `LayoutDetection` | `PP-DocLayout_plus-L` | paddle_static | doc-parser |
| 5 | 版面分析 | `layout_detection` + `--model_name PP-DocLayoutV3` | `LayoutDetection` | `PP-DocLayoutV3` | paddle_static | doc-parser |
| 6 | 印章文本检测 | `seal_text_detection` | `SealTextDetection` | `PP-OCRv4_server_seal_det` | paddle_static | doc-parser |
| 7 | 表格单元格检测 | `table_cells_detection` | `TableCellsDetection` | `RT-DETR-L_wired_table_cell_det` | paddle_static | doc-parser |
| 8 | 表格分类 | `table_classification` | `TableClassification` | `PP-LCNet_x1_0_table_cls` | paddle_static | doc-parser |
| 9 | 表格结构识别 | `table_structure_recognition` | `TableStructureRecognition` | `SLANet` / 常用 `SLANet_plus` | paddle_static | doc-parser |
| 10 | 文本检测 | `text_detection` | `TextDetection` | `PP-OCRv5_server_det` | paddle_static | 基础包 |
| 11 | 文本图像矫正 | `text_image_unwarping` | `TextImageUnwarping` | `UVDoc` | paddle_static | 基础包 |
| 12 | 文本行方向分类 | `textline_orientation_classification` | `TextLineOrientationClassification` | `PP-LCNet_x0_25_textline_ori` | paddle_static | 基础包 |
| 13 | 文本识别 | `text_recognition` | `TextRecognition` | `PP-OCRv5_server_rec` | paddle_static | 基础包 |
| 14 | 图表解析 | `chart_parsing` | `ChartParsing` | `PP-Chart2Table` | paddle_dynamic | doc-parser |

---

### 3.1 文档图像方向分类模块

| 项 | 内容 |
|----|------|
| **作用** | 判断文档图 0°/90°/180°/270°，便于后续 OCR |
| **文档** | [doc_img_orientation_classification](https://www.paddleocr.ai/latest/version3.x/module_usage/doc_img_orientation_classification.html) |

**CLI**

```bash
paddleocr doc_img_orientation_classification -i ./img.jpg --device gpu
paddleocr doc_img_orientation_classification -i ./img.jpg --model_name PP-LCNet_x1_0_doc_ori --device gpu
```

**Python**

```python
from paddleocr import DocImgOrientationClassification
model = DocImgOrientationClassification(model_name="PP-LCNet_x1_0_doc_ori", device="gpu")
for res in model.predict("./img.jpg", batch_size=1):
    res.print()
    res.save_to_json("./output")
```

**要点**：`--model_dir` 可指向自训练权重；训练走 PaddleX 文档方向分类教程。

---

### 3.2 文档类视觉语言模型模块

| 项 | 内容 |
|----|------|
| **作用** | 图像 + 文本问题 → 文档理解/表格描述等（多模态） |
| **文档** | [doc_vlm](https://www.paddleocr.ai/latest/version3.x/module_usage/doc_vlm.html) |
| **可选模型** | `PP-DocBee-2B`、`PP-DocBee-7B`、`PP-DocBee2-3B` |

**CLI**（`-i` 为 Python 字典字符串）

```bash
paddleocr doc_vlm -i "{'image': './table.png', 'query': '识别这份表格的内容, 以markdown格式输出'}" --device gpu
paddleocr doc_vlm -i "{'image': './table.png', 'query': '...'}" --model_name PP-DocBee2-3B --device gpu
```

**Python**

```python
from paddleocr import DocVLM
model = DocVLM(model_name="PP-DocBee2-3B", device="gpu")
out = model.predict(input={"image": "table.png", "query": "识别这份表格的内容, 以markdown格式输出"}, batch_size=1)
for res in out:
    res.print()
    res.save_to_json("./output/res.json")
```

**要点**：输入必须为 `{'image': 路径或URL, 'query': 文本}`；暂不支持在本模块文档路径内微调。

---

### 3.3 公式识别模块

| 项 | 内容 |
|----|------|
| **作用** | 公式图 → LaTeX 等 |
| **文档** | [formula_recognition](https://www.paddleocr.ai/latest/version3.x/module_usage/formula_recognition.html) |

**CLI**

```bash
paddleocr formula_recognition -i ./formula.png --device gpu
paddleocr formula_recognition -i ./formula.png --model_name PP-FormulaNet_plus-M --device gpu
```

**Python**

```python
from paddleocr import FormulaRecognition
model = FormulaRecognition(model_name="PP-FormulaNet_plus-M", device="gpu")
for res in model.predict("./formula.png", batch_size=1):
    res.print()
```

**要点**：可训练替换模型（PaddleOCR 检测类训练流程 + `PP-FormulaNet` 配置）；纳入 **PP-StructureV3** 产线。

---

### 3.4 版面区域检测模块

| 项 | 内容 |
|----|------|
| **作用** | 检测文档中标题、正文、图、表等区域框（不排序） |
| **文档** | [layout_detection](https://www.paddleocr.ai/latest/version3.x/module_usage/layout_detection.html) |

**CLI**

```bash
paddleocr layout_detection -i ./layout.jpg --device gpu
paddleocr layout_detection -i ./layout.jpg --model_name PP-DocLayout_plus-L --device gpu
```

**Python**

```python
from paddleocr import LayoutDetection
model = LayoutDetection(model_name="PP-DocLayout_plus-L", device="gpu")
for res in model.predict("./layout.jpg", batch_size=1):
    res.print()
    res.save_to_img("./output")
```

**要点**：官方有多档 PP-DocLayout 系列模型，见文档模型表；支持 `transformers` 引擎。

---

### 3.5 版面分析模块

| 项 | 内容 |
|----|------|
| **作用** | 区域检测 + **阅读顺序**（PaddleOCR-VL / PP-Structure 核心） |
| **文档** | [layout_analysis](https://www.paddleocr.ai/latest/version3.x/module_usage/layout_analysis.html) |
| **实现说明** | 与版面区域检测 **同一 CLI/API**，指定分析模型名 |

**CLI**

```bash
paddleocr layout_detection -i ./layout.jpg --model_name PP-DocLayoutV3 --device gpu
```

**Python**

```python
from paddleocr import LayoutDetection
model = LayoutDetection(model_name="PP-DocLayoutV3", device="gpu")
for res in model.predict("./layout.jpg", batch_size=1):
    res.print()
```

**要点**：PaddleOCR-VL 产线默认使用 PP-DocLayout 系列做版面阶段。

---

### 3.6 印章文本检测模块

| 项 | 内容 |
|----|------|
| **作用** | 检测印章内文字区域 |
| **文档** | [seal_text_detection](https://www.paddleocr.ai/latest/version3.x/module_usage/seal_text_detection.html) |

**CLI**

```bash
paddleocr seal_text_detection -i ./seal.png --device gpu
paddleocr seal_text_detection -i ./seal.png --model_name PP-OCRv4_server_seal_det --device gpu
```

**Python**

```python
from paddleocr import SealTextDetection
model = SealTextDetection(model_name="PP-OCRv4_server_seal_det", device="gpu")
for res in model.predict("./seal.png", batch_size=1):
    res.print()
```

**要点**：检测后需配合 **文本识别** 得到字符；PaddleOCR-VL 可用 `--use_seal_recognition True`。

---

### 3.7 表格单元格检测模块

| 项 | 内容 |
|----|------|
| **作用** | 检测表格内每个单元格框 |
| **文档** | [table_cells_detection](https://www.paddleocr.ai/latest/version3.x/module_usage/table_cells_detection.html) |

**CLI**

```bash
paddleocr table_cells_detection -i ./table.jpg --device gpu
paddleocr table_cells_detection -i ./table.jpg --model_name RT-DETR-L_wired_table_cell_det --device gpu
```

**Python**

```python
from paddleocr import TableCellsDetection
model = TableCellsDetection(model_name="RT-DETR-L_wired_table_cell_det", device="gpu")
for res in model.predict("./table.jpg", batch_size=1):
    res.print()
```

**要点**：有线/无线表格有不同模型名，见官网模型列表。

---

### 3.8 表格分类模块

| 项 | 内容 |
|----|------|
| **作用** | 判断表格有线/无线等类型 |
| **文档** | [table_classification](https://www.paddleocr.ai/latest/version3.x/module_usage/table_classification.html) |

**CLI**

```bash
paddleocr table_classification -i ./table.jpg --device gpu
```

**Python**

```python
from paddleocr import TableClassification
model = TableClassification(model_name="PP-LCNet_x1_0_table_cls", device="gpu")
for res in model.predict("./table.jpg", batch_size=1):
    res.print()
```

---

### 3.9 表格结构识别模块

| 项 | 内容 |
|----|------|
| **作用** | 表格图 → HTML/结构（单元格逻辑关系） |
| **文档** | [table_structure_recognition](https://www.paddleocr.ai/latest/version3.x/module_usage/table_structure_recognition.html) |

**CLI**

```bash
paddleocr table_structure_recognition -i ./table.jpg --device gpu
paddleocr table_structure_recognition -i ./table.jpg --model_name SLANet_plus --device gpu
```

**Python**

```python
from paddleocr import TableStructureRecognition
model = TableStructureRecognition(model_name="SLANet_plus", device="gpu")
for res in model.predict("./table.jpg", batch_size=1):
    res.print()
```

**要点**：常与 **单元格检测 + 文本识别** 组合；PP-StructureV3 产线内置表格链路。

---

### 3.10 文本检测模块

| 项 | 内容 |
|----|------|
| **作用** | 检测文本行/词区域框 |
| **文档** | [text_detection](https://www.paddleocr.ai/latest/version3.x/module_usage/text_detection.html) |

**CLI**

```bash
paddleocr text_detection -i ./doc.png --device gpu
paddleocr text_detection -i ./doc.png --model_name PP-OCRv5_server_det --device gpu
paddleocr text_detection -i ./images/ --model_name PP-OCRv5_mobile_det --device gpu
```

**Python**

```python
from paddleocr import TextDetection
model = TextDetection(model_name="PP-OCRv5_server_det", device="gpu")
for res in model.predict("./doc.png", batch_size=1):
    res.print()
    res.save_to_img("./output")
```

**要点**：通用 OCR 产线第一步；支持目录批量输入。

---

### 3.11 文本图像矫正模块

| 项 | 内容 |
|----|------|
| **作用** | 矫正拍摄/扫描的几何扭曲 |
| **文档** | [text_image_unwarping](https://www.paddleocr.ai/latest/version3.x/module_usage/text_image_unwarping.html) |

**CLI**

```bash
paddleocr text_image_unwarping -i ./warped.jpg --device gpu
paddleocr text_image_unwarping -i ./warped.jpg --model_name UVDoc --device gpu
```

**Python**

```python
from paddleocr import TextImageUnwarping
model = TextImageUnwarping(model_name="UVDoc", device="gpu")
for res in model.predict("./warped.jpg", batch_size=1):
    res.print()
    res.save_to_img("./output")
```

**要点**：与 **文档方向分类** 同属「文档图像预处理产线」：`paddleocr doc_preprocessor`。

---

### 3.12 文本行方向分类模块

| 项 | 内容 |
|----|------|
| **作用** | 判断文本行 0°/180° 等并矫正 |
| **文档** | [textline_orientation_classification](https://www.paddleocr.ai/latest/version3.x/module_usage/textline_orientation_classification.html) |

**CLI**

```bash
paddleocr textline_orientation_classification -i ./line.jpg --device gpu
```

**Python**

```python
from paddleocr import TextLineOrientationClassification
model = TextLineOrientationClassification(model_name="PP-LCNet_x0_25_textline_ori", device="gpu")
for res in model.predict("./line.jpg", batch_size=1):
    res.print()
```

**要点**：通用 OCR 产线中对应 `--use_textline_orientation` / `use_textline_orientation`。

---

### 3.13 文本识别模块

| 项 | 内容 |
|----|------|
| **作用** | 文本行图像 → 字符序列 |
| **文档** | [text_recognition](https://www.paddleocr.ai/latest/version3.x/module_usage/text_recognition.html) |

**CLI**

```bash
paddleocr text_recognition -i ./crop.png --device gpu
paddleocr text_recognition -i ./crop.png --model_name PP-OCRv5_server_rec --device gpu
```

**Python**

```python
from paddleocr import TextRecognition
model = TextRecognition(model_name="PP-OCRv5_server_rec", device="gpu")
for res in model.predict("./crop.png", batch_size=1):
    res.print()
```

**要点**：多语言有多套 rec 模型（简中、英文、日文等），见文档模型表；与检测组合为 `paddleocr ocr`。

---

### 3.14 图表解析模块

| 项 | 内容 |
|----|------|
| **作用** | 柱状图/折线图/饼图等 → 数据表（Markdown/文本） |
| **文档** | [chart_parsing](https://www.paddleocr.ai/latest/version3.x/module_usage/chart_parsing.html) |

**CLI**

```bash
paddleocr chart_parsing -i "{'image': './chart.png'}" --device gpu
paddleocr chart_parsing -i "{'image': './chart.png'}" --model_name PP-Chart2Table --device gpu
```

**Python**

```python
from paddleocr import ChartParsing
model = ChartParsing(model_name="PP-Chart2Table", device="gpu")
for res in model.predict(input={"image": "chart.png"}, batch_size=1):
    res.print()
    res.save_to_json("./output/res.json")
```

**要点**：输入格式 `{'image': path}`；PaddleOCR-VL / PP-StructureV3 可用 `--use_chart_recognition True` 走产线。

---

## 四、模块与产线 / 服务化关系（避免重复部署）

多数模块已嵌入产线，**端到端场景优先用产线**，不必单独起 14 个服务。

| 产线 CLI | 包含的典型模块 | 本地命令示例 |
|----------|----------------|--------------|
| **通用 OCR** | 文本检测 + 文本识别 + 可选方向/矫正/行方向 | `paddleocr ocr -i ./img.png --device gpu` |
| **文档图像预处理** | 文档方向分类 + 文本图像矫正 | `paddleocr doc_preprocessor -i ./img.png --device gpu` |
| **PP-StructureV3** | 版面、表格、公式、图表等组合 | `paddleocr pp_structurev3 -i ./page.png --device gpu` |
| **PaddleOCR-VL** | 版面分析 + VLM 识别 | `paddleocr doc_parser -i ./page.png --device gpu` |

**HTTP 服务化（产线级，非单模块）**

```bash
pip install paddlex   # 随 paddleocr 安装
paddlex --install serving
paddlex --serve --pipeline PaddleOCR-VL    # 或其他产线名
# 默认 http://0.0.0.0:8080 ，接口见产线文档
```

GPU Docker 下一键产线 + VLM：使用 `paddleocr_vl_docker` 的 `compose.yaml`（见《PaddleOCR-VL_RTX5090_Docker本地部署.md》）。

---

## 五、通用参数（所有模块 CLI / Python 均适用）

| 参数 | 说明 |
|------|------|
| `-i` / `input` | 图像路径、URL、目录、numpy、列表 |
| `--device` / `device` | `gpu`、`gpu:0`、`cpu` 等 |
| `--model_name` / `model_name` | 切换模型表中的名称 |
| `--model_dir` / `model_dir` | 本地推理权重目录 |
| `--engine` / `engine` | `paddle_static`（默认多数）、`paddle_dynamic`、`transformers` |
| `batch_size` | Python `predict()` 批大小 |
| `predict_iter()` | 与 `predict()` 相同，返回生成器，省内存 |

**结果对象**：`res.print()`、`res.save_to_json()`；检测/版面类常另有 `save_to_img()`。

---

## 六、官网文档链接索引

| 模块 | 链接 |
|------|------|
| 模块概述 | https://www.paddleocr.ai/latest/version3.x/module_usage/module_overview.html |
| 文档图像方向分类 | https://www.paddleocr.ai/latest/version3.x/module_usage/doc_img_orientation_classification.html |
| 文档类视觉语言模型 | https://www.paddleocr.ai/latest/version3.x/module_usage/doc_vlm.html |
| 公式识别 | https://www.paddleocr.ai/latest/version3.x/module_usage/formula_recognition.html |
| 版面区域检测 | https://www.paddleocr.ai/latest/version3.x/module_usage/layout_detection.html |
| 版面分析 | https://www.paddleocr.ai/latest/version3.x/module_usage/layout_analysis.html |
| 印章文本检测 | https://www.paddleocr.ai/latest/version3.x/module_usage/seal_text_detection.html |
| 表格单元格检测 | https://www.paddleocr.ai/latest/version3.x/module_usage/table_cells_detection.html |
| 表格分类 | https://www.paddleocr.ai/latest/version3.x/module_usage/table_classification.html |
| 表格结构识别 | https://www.paddleocr.ai/latest/version3.x/module_usage/table_structure_recognition.html |
| 文本检测 | https://www.paddleocr.ai/latest/version3.x/module_usage/text_detection.html |
| 文本图像矫正 | https://www.paddleocr.ai/latest/version3.x/module_usage/text_image_unwarping.html |
| 文本行方向分类 | https://www.paddleocr.ai/latest/version3.x/module_usage/textline_orientation_classification.html |
| 文本识别 | https://www.paddleocr.ai/latest/version3.x/module_usage/text_recognition.html |
| 图表解析 | https://www.paddleocr.ai/latest/version3.x/module_usage/chart_parsing.html |
| 安装 | https://www.paddleocr.ai/latest/version3.x/installation.html |
| 快速开始 | https://www.paddleocr.ai/latest/quick_start.html |

---

**说明**：各模块完整模型列表、耗时、训练导出步骤以官网对应页面为准；本文汇总 **本地部署所需的最小完整路径**（环境 → CLI → Python → 默认模型 → 产线关系），便于对照侧边栏逐项落地。
