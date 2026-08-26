# 📈 Finance Agent — 金融投研 Agentic RAG 系统

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](#)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green)](#)
[![Milvus](https://img.shields.io/badge/Milvus-2.4-blueviolet)](#)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agent-orange)](#)
[![React](https://img.shields.io/badge/React-19-61dafb)](#)
[![CI](https://github.com/sherxlg-gif/Finance-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/sherxlg-gif/Finance-agent/actions/workflows/ci.yml)

面向金融财报的 **Agentic RAG 系统**，支持 PDF 入库、混合检索、财务计算、对话记忆和原文引用。

---

## ✨ 核心特性

### 🔗 PDF 原文引用

检索结果保留命中的文件、页码和原文片段。回答下方的来源标签支持悬停预览；点击后可在 PDF 查看器中打开对应页面。

### 🧠 ReAct Agent + 进程级代码沙盒

Agent 基于 **LangGraph** 的 ReAct 流程运行，根据问题选择财报检索、长期记忆或 Python 计算工具。

财务计算在 **multiprocessing 子进程**中执行：
- 运行时间超过 10 秒时终止子进程
- 仅允许 math / json / datetime / collections / itertools / decimal
- 阻止 `import os`、`open()` 和 `exec()` 等操作

### 📊 Dense / Sparse 召回 + RRF 融合 + Rerank 重排

`HybridSearchEngine` 使用 Milvus 的 Dense 和 Sparse 索引并行召回，再进行 RRF 融合：

```
Dense (1024d 语义向量) ─┐
                         ├→ RRF 倒数排序融合 → gte-rerank-v2 精排 → Top-N 父块
Sparse (持久化 BM25) ───┘
```

财报 Child 入库和查询共用保存在 `data/processed/finance_bm25.json` 的 BM25 词表。模型缺失、损坏或查询向量为空时，检索会回退到 `dense_only`，日志记录实际检索模式和 Sparse `nnz`。已有 Milvus 数据可用以下命令重建 Sparse 向量：

```bash
python -m app.scripts.rebuild_sparse
```

### 📦 Parent-Child 存储解耦

Parent 按 Markdown 标题切分，单个 Parent 超过 40,000 字符时再按长度拆分。Child 使用 `500` 字符和 `50` 字符重叠。Parent 存入 **PostgreSQL**，Child 及 Dense、Sparse 向量存入 **Milvus**。命中 Child 后按检索顺序回填 Parent，并保留命中 Child 的文本、页码和排序信息。

### 💾 三层记忆体系

- **短期记忆**：当前对话最近 10 条消息
- **对话持久化**：对话内容存入 PostgreSQL JSONB，可从侧边栏切换
- **长期记忆**：用户明确提到“上次、以前、其他对话”时，通过 Dense 向量检索 Milvus 中的历史问答；正常回答完成后写入，相同问答不会重复保存

### 🎨 前端功能

- **Markdown 渲染**：支持表格、代码、列表和标题
- **主题切换**：通过 localStorage 保存明暗主题
- **对话管理**：自动生成标题，支持重命名和历史对话切换
- **文件管理**：上传、删除和预览 PDF，并显示解析进度
- **响应式布局**：适配桌面和移动端

### 🛡️ 工程与测试

- GitHub Actions 在 PR 和 `main` Push 时运行后端测试、前端 lint/build 和 Docker Compose 配置检查
- PDF 使用 MD5 指纹去重
- Milvus 写入失败时回滚本次写入 PostgreSQL 的 Parent
- Agent 最大执行步数为 10
- 对话接口通过 SSE 返回回答、工具状态和来源

---

## 🛠 技术栈

| 层级 | 选型 |
|:---|:---|
| Agent 编排 | LangGraph (ReAct), LangChain |
| 大模型 | qwen3.7-max / qwen-turbo / text-embedding-v4 (1024d) / gte-rerank-v2 |
| 向量数据库 | Milvus 2.4 (AUTOINDEX + SPARSE_INVERTED_INDEX) |
| 关系型数据库 | PostgreSQL 15 (SQLAlchemy ORM, JSONB) |
| 文档解析 | pypdf (数字 PDF) + Docling (扫描件兜底) |
| 后端 | FastAPI · Pydantic V2 · SSE · multiprocessing |
| 前端 | React 19 · TypeScript · Zustand · Shadcn/ui · Tailwind · react-pdf · react-markdown |
| 部署 | Docker Compose 一键 8 服务 |

---

## 🏗 系统架构

```
PDF 上传
  → pypdf 提取逐页文本
  → 扫描件或低文本 PDF 使用 Docling 分批解析
  → Parent 按标题切分 / Child 按 500 字符切分
  → Child 文本匹配 PDF 页码
  → 文件名解析，必要时由 LLM 补充公司和年份
  → 双库落盘: Parent → PostgreSQL / Child + Dense、Sparse 向量 → Milvus
  → MD5 指纹登记（去重）

用户提问
  → LangGraph ReAct Agent
  → memory_retriever_tool    ← 跨对话长期记忆（Dense-only）
  → financial_retriever_tool ← Dense / Sparse + RRF + Rerank
  → python_repl_tool         ← 子进程沙盒精确计算
  → SSE 流式输出 + 来源引用（文件、页码、片段、hash）
  → 自动写入长期记忆
```

---

## 🚀 快速启动

### 1. 环境要求

Docker Desktop

### 2. 配置

```bash
cp .env.example .env
# 编辑 .env，填入 DASHSCOPE_API_KEY
# 申请地址：https://bailian.console.aliyun.com/
```

### 3. 一键启动

```bash
docker compose up -d
```

### 4. 访问

| 服务 | 地址 | 说明 |
|:---|:---|:---|
| React 前端 | http://localhost:8502 | 用户界面 |
| API 文档 (Swagger) | http://localhost:8000/docs | 后端接口调试 |
| Milvus 管理面板 (Attu) | http://localhost:8002 | 向量数据可视化 |
| pgAdmin | http://localhost:5050 | PostgreSQL 管理界面（admin@rag.com / admin） |

### 5. 运行测试

```bash
# 运行后端测试
docker compose exec backend-v2 pytest tests/ -v

# 仅沙盒安全测试
docker compose exec backend-v2 pytest tests/test_finance_repl.py -v

# 跳过高耗时测试
docker compose exec backend-v2 pytest tests/ -v -k "not timeout"
```

---

## 📁 项目结构

```
.
├── app/
│   ├── api/            # FastAPI 路由 (chat, upload, conversations)
│   ├── core/           # 配置, 鉴权中间件, 异常体系, 日志
│   ├── models/         # Pydantic schemas
│   ├── prompts/        # YAML Prompt 模板, 热加载
│   ├── services/       # ingestion, retrieval, hybrid_search, memory, progress
│   └── tools/          # Agent 工具: finance_repl, retriever_tool, memory_tool
├── frontend-react-v2/  # 当前使用的 React SPA
│   └── src/
│       ├── components/ # ChatMessage, ChatArea, Sidebar, PDFViewer, ...
│       ├── services/   # API 调用 + SSE 流式解析
│       └── store/      # Zustand 状态管理
├── frontend-react/     # 保留的旧版前端目录
├── tests/              # pytest 用例
├── evals/              # 评测数据与脚本
├── .github/workflows/  # GitHub Actions
├── docker-compose.yml  # 8 服务编排
└── Dockerfile
```

## 🧪 评测

新增 `evals/eval_dataset_sinopec.csv`，包含 15 道中国石化财报问题。每道题记录标准答案、来源文件、页码、证据原文和答案类型，覆盖数值定位、表格信息、业务描述、Python 计算和无答案拒答。

```bash
python evals/evaluate_rag.py --dataset evals/eval_dataset_sinopec.csv
python evals/evaluate_rag.py --dataset evals/eval_dataset_sinopec.csv --top-n 3
```

评测报告写入 `evals/reports/`，包含 LLM Judge、Recall@5、Recall@10、MRR@10、页码与证据原文命中率、Sparse `nnz`，并对比 Dense-only、Sparse-only 和 Hybrid 三种检索模式。`--top-n` 可用于测试不同的返回数量，系统默认值为 5。

---

> 📌 建议使用 `公司名+年份+报告类型.pdf` 作为文件名，例如 `中国石化2026年半年度报告.pdf`。文件名无法识别时，系统会从正文中提取公司和年份。
