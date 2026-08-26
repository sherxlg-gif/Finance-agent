# Finance Agent 响应延迟小范围修复设计

## 目标

在不修改公开接口、数据库结构、检索参数和前端的前提下，减少简单财务问题的重复检索，消除 BM25 首次查询的分词冷启动，并修复长期记忆写入现有 Milvus Schema 时的空 Sparse 向量错误。

## 方案选择

### 重复检索

- 采用：收紧现有 Agent Prompt。简单事实题在一次成功检索后直接回答；只有多公司、多年份对比才按对象分别检索；返回非空证据时禁止同义词重试。
- 不采用：重构 LangGraph 或增加请求级工具中间件。它能做硬限制，但会扩大改动，并可能误伤合理的多对象检索。
- 不采用：直接降低递归步数。它不能区分重复检索和必要的检索、计算步骤。

### BM25 冷启动

- 采用：后端启动时加载持久化 BM25，并执行一次轻量查询编码；失败时只记录 `dense_only` 降级日志，不影响启动。
- 不采用：修改词表格式或检索服务生命周期。

### 长期记忆

- 采用：保持 Dense-only 检索和现有 `sparse_vector` 字段，写入一个固定、非零、不会被查询的 Sparse 占位向量。
- 不采用：迁移或重建 Milvus Collection。

## 验收边界

- `POST /api/v1/chat/stream`：简单事实题成功检索后不再用同义词重复检索；对比题仍可按对象多次检索。
- `PersistentBM25Encoder`：启动预热成功时查询 `nnz > 0`，模型不可用时继续允许 Dense-only。
- `MemoryService.store_episode()`：写入载荷包含合法非空 Sparse 向量，相同问答仍不会重复写入。
- 回归：完整 pytest 通过；真实请求保持 `retrieval_mode=hybrid`；后端日志不再出现 `empty sparse float vector row`。

## 风险控制

- Prompt 约束属于概率性行为，使用真实 SSE 请求统计工具调用次数作为验收，不宣称绝对硬限制。
- 预热只移动一次性成本，不改变每次检索的 Dense、Sparse、RRF 或 Rerank 流程。
- 占位 Sparse 向量只为兼容旧 Schema，长期记忆检索仍明确使用 Dense-only。
