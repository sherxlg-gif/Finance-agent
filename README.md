# Finance Agent

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](#)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green)](#)
[![Milvus](https://img.shields.io/badge/Milvus-2.4-blueviolet)](#)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agent-orange)](#)
[![React](https://img.shields.io/badge/React-19-61dafb)](#)
[![CI](https://github.com/sherxlg-gif/Finance-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/sherxlg-gif/Finance-agent/actions/workflows/ci.yml)

面向金融财报场景的 **Agentic RAG 系统**。从 AI 生成的答案出发，可逐级追溯至原始 PDF 的精确页码——让每一条结论都可核查。

---

## ✨ 核心特性

### 🔗 端到端溯源闭环

每条 Agent 回答的引用来源均附带**页码 + 原文片段**。鼠标悬停来源标签即可预览命中文字；点击标签一键打开 PDF 原文查看器，自动跳转到对应页码，页面顶部展示黄色命中标记卡。

> "这个数据从哪里来的？" → 1 次点击 → 原文第 42 页高亮位置。

### 🧠 ReAct Agent + 进程级代码沙盒

基于 **LangGraph** 的 ReAct 范式智能体，LLM 自主完成「意图识别 → 工具选择 → 观察 → 回答」的推理循环。

针对 LLM 财务计算幻觉，封装受限于 **multiprocessing 子进程**的 Python 执行环境：
- **10 秒超时自动 kill** —— 死循环不会卡死主进程
- **模块白名单** —— 仅允许 math / json / datetime / collections / itertools / decimal
- **安全拦截** —— `import os` / `open()` / `exec()` 全部被阻止
- **5 项安全边界测试**覆盖，子进程完全隔离

### 📊 Dense / Sparse 召回 + RRF 融合 + Rerank 重排

`HybridSearchEngine` 使用 Milvus 的 Dense 和 Sparse 索引并行召回，再进行 RRF 融合：

```
Dense (1024d 语义向量) ─┐
                         ├→ RRF 倒数排序融合 → gte-rerank-v2 精排 → Top-N 父块
Sparse (持久化 BM25) ───┘
```

财报检索优先使用统一持久化的 BM25 词表和 Dense 向量；BM25 模型缺失、损坏或查询向量为空时，会明确降级为 `dense_only`，不会伪装成双路检索。可用以下命令从现有 Milvus Child 数据重建 Sparse 向量：

```bash
python -m app.scripts.rebuild_sparse
```

### 📦 Parent-Child 存储解耦

Parent-Child 切分保持当前 `500` 字符块、`50` 字符重叠的参数。完整语义段落（Parent）存 **PostgreSQL**，细粒度子块（Child）及其向量存入 **Milvus**。检索命中 Child 后按命中顺序回填 Parent，并保留命中 Child 文本和实际页码，兼顾上下文完整性与引用可核验性。

### 💾 三层记忆体系

- **短期记忆** — 当前对话上下文窗口
- **对话持久化** — PostgreSQL JSONB 存储，侧边栏随时切换历史对话
- **长期记忆** — 只有明确引用“上次、以前、其他对话”时才调用；Q&A 片段以 Dense 向量存入 Milvus，跨对话进行语义检索。相同的规范化问答不会重复写入。

### 🎨 产品级交互体验

- **Markdown 渲染** — LLM 输出的表格/代码/列表/标题原生排版，金融数据表格带斑马条纹
- **暗色模式** — 一键切换，localStorage 持久化偏好
- **对话自动标题** — 首轮问答后用 qwen-turbo 自动生成 10 字摘要
- **对话重命名** — 双击侧边栏标题即可编辑
- **文件管理** — 上传/删除/预览 PDF，文件列表显示大小与入库日期
- **📱 响应式** — 桌面侧边栏常驻 + 移动端汉堡菜单浮层

### 🛡️ 工程基础

- **自动化测试**覆盖沙盒安全、来源解析、入库逻辑、检索管道、评测指标和记忆逻辑；每次 PR 与 `main` Push 由 GitHub Actions 执行后端测试。
- **MD5 指纹去重** —— 防止重复入库浪费 Token
- **分布式事务补偿** —— Milvus 写入失败 → PostgreSQL 自动回滚孤块
- **三级异常体系** —— 区分可恢复/不可恢复错误，精准降级
- **Recursion Limit** —— Agent 步数上限 10，防止无限循环
- **SSE 流式输出** —— 实时展示 LLM 逐字生成 + 工具调用状态

---

## 🛠 技术栈

| 层级 | 选型 |
|:---|:---|
| Agent 编排 | LangGraph (ReAct), LangChain |
| 大模型 | qwen3.7-max / qwen-turbo / text-embedding-v4 (1024d) / gte-rerank-v2 |
| 向量数据库 | Milvus 2.4 (HNSW + SPARSE_INVERTED_INDEX 双索引) |
| 关系型数据库 | PostgreSQL 15 (SQLAlchemy ORM, JSONB) |
| 文档解析 | Docling (版面分析) + pypdf (页码追踪) |
| 后端 | FastAPI · Pydantic V2 · SSE · multiprocessing |
| 前端 | React 19 · TypeScript · Zustand · Shadcn/ui · Tailwind · react-pdf · react-markdown |
| 部署 | Docker Compose 一键 8 服务 |

---

## 🏗 系统架构

```
PDF 上传
  → Docling 版面解析 → Markdown
  → pypdf 逐页文本提取 → chunk 页码匹配
  → 父子块切分 + LLM 元数据兜底
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
| pgAdmin | http://localhost:5050 | PostgreSQL 管理界面（凭据以本地配置为准） |

### 5. 运行测试

```bash
# 全部后端测试
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
├── docker-compose.yml  # 8 服务编排
└── Dockerfile
```

## 评测

默认评测仍使用原有的深信服数据集；该文件不会被中国石化数据集替换。中国石化证据集包含 15 道题，额外记录来源文件、页码和证据原文，支持确定性检索指标和原有 LLM Judge：

```bash
python evals/evaluate_rag.py
python evals/evaluate_rag.py --dataset evals/eval_dataset_sinopec.csv
python evals/evaluate_rag.py --dataset evals/eval_dataset_sinopec.csv --top-n 3
```

本地已完成的 14 道有明确证据题检索对比结果如下，具体报告写入被忽略的 `evals/reports/`：

| 模式 | Recall@10 |
|:---|---:|
| Dense-only | 0.3571 |
| Hybrid | 0.5714 |
| Sparse-only | 0.6429 |

同一轮验证中 Sparse 查询 `nnz` 最小值为 8。Top-N 仍可通过参数比较，当前默认值为 5，不把 Top-5 宣称为最优值。

---

> 📌 建议上传 `公司名+年份+报告类型.pdf` 格式的财报文件（如 `深信服2025年半年度报告.pdf`），以便系统自动提取年份与公司信息。命名不规范时，LLM 会从正文中自动提取。
