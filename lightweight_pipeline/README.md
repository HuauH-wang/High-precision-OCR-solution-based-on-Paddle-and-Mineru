# Lightweight Document OCR Pipeline

轻量级文档解析产线 — 全云端 API、批量并行、统一 Markdown 输出。对应论文方法第 4 章。

支持 PDF / 图片多格式入口，扫描版复杂表格 **零/一/二轮 OCR**，四条解析路径 **P0–P2**，最终合并为按阅读顺序排列的 Markdown。

更完整的项目说明见仓库根目录 [`README.md`](../README.md)。

---

## 目录结构

```
lightweight_pipeline/
├── run.py
├── config.example.yaml   # 脱敏模板（提交 git）
├── config.yaml           # 本地密钥配置（gitignore，勿提交）
├── requirements.txt
├── README.md
└── lp/
    ├── settings.py              # YAML + 环境变量加载
    ├── batch_pipeline.py        # 批处理主流程
    ├── types.py
    ├── clients/                 # MinerU / AI Studio API
    └── core/                    # 入口、多轮 OCR、合并
```

上级依赖（同仓库根目录）：

- [`../mineru_precision_api.py`](../mineru_precision_api.py) — MinerU 客户端

运行时目录（自动生成，已 gitignore）：

- `work/` — 中间产物
- `output/` — Markdown 与 `failed_tasks.csv`

---

## 安装

```bash
cd lightweight_pipeline
pip install -r requirements.txt
cp config.example.yaml config.yaml
```

填写 `config.yaml` 中的 Token，或设置环境变量 `MINERU_TOKEN` / `AISTUDIO_TOKEN`。

---

## 使用

```bash
python run.py --config config.yaml ../samples/diagram-flowchart.png
python run.py --config config.yaml --parse-path P1 scan.pdf
python run.py --config config.yaml --blur-sensitive scan.pdf
```

| 路径 | 说明 |
|------|------|
| `P0` | 整页直调 MinerU |
| `P1` | 一轮表格 → MinerU + 零轮非表格（默认） |
| `P2-M1` | 二轮 spotting 扩边 → MinerU（偏文本） |
| `P2-M2` | 二轮表图 → MinerU（偏结构） |

---

## 已知限制

- 高精度本地产线（`mode: high_precision`）尚未实现
- PP-TableMagic 未接入
- 混合 PDF 不在设计范围内
- 检测/识别 HTTP 默认关闭，由 spotting 替代
