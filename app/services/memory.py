"""
Agent 长期记忆服务。

每轮对话结束后自动将 Q&A 对存入 Milvus + PostgreSQL，
后续对话中 Agent 可通过 memory_retriever_tool 语义检索历史记忆。

存储架构（复用现有基础设施）：
- Milvus (long_term_memories): Dense(1024维)，用于语义检索
- PostgreSQL (long_term_memories 表): 存储原文，用于回填完整内容
"""
import hashlib
import logging
import re
from typing import Optional

from langchain_community.embeddings import DashScopeEmbeddings
from pymilvus import (
    connections,
    utility,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
)

from app.core.config import settings
from app.database import get_db_session, LongTermMemory

logger = logging.getLogger(__name__)


def _normalize_memory_part(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def make_memory_id(query: str, answer: str) -> str:
    """Create a stable ID for semantically identical whitespace-normalized Q&A."""
    normalized = f"{_normalize_memory_part(query)}\0{_normalize_memory_part(answer)}"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class MemoryService:
    """长期记忆服务：存储对话片段，支持跨对话语义检索。"""

    def __init__(self):
        self._collection: Optional[Collection] = None

        # 复用 Embedding 模型（与检索共用同一 API）
        self.embeddings = DashScopeEmbeddings(
            model=settings.EMBEDDING_MODEL,
            dashscope_api_key=settings.DASHSCOPE_API_KEY,
        )

        # 连接 Milvus（带重试，生产环境 Milvus 启动较慢需要更多次重试）
        for attempt in range(1, 8):
            try:
                connections.connect(
                    alias="default",
                    host=settings.MILVUS_HOST,
                    port=settings.MILVUS_PORT,
                )
                logger.info("🧠 MemoryService: Milvus 连接成功")
                break
            except Exception:
                if attempt < 7:
                    import time as _t; _t.sleep(2)
                else:
                    logger.warning("⚠️ MemoryService: Milvus 连接失败，长期记忆功能不可用")

    # ==========================================
    # 内部：Milvus Collection 惰性初始化
    # ==========================================

    def _get_collection(self) -> Collection:
        """获取记忆 Collection，不存在则自动创建（与 ingestion 建表逻辑一致）。"""
        if self._collection is not None:
            return self._collection

        collection_name = settings.MEMORY_COLLECTION_NAME

        if not utility.has_collection(collection_name):
            logger.info(f"🧠 创建长期记忆向量表: {collection_name}")

            fields = [
                FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, is_primary=True, max_length=65535),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=settings.EMBEDDING_DIM),
                FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
                FieldSchema(name="metadata", dtype=DataType.JSON),
            ]
            schema = CollectionSchema(fields, "Agent Long-Term Memory Store", enable_dynamic_field=True)
            collection = Collection(collection_name, schema)

            # Dense 向量索引
            collection.create_index("dense_vector", {
                "index_type": "AUTOINDEX",
                "metric_type": "L2",
                "params": {},
            })

            logger.info("✅ 长期记忆向量表及 Dense 索引创建完成！")
        else:
            collection = Collection(collection_name)

        collection.load()
        self._collection = collection
        return self._collection

    # ==========================================
    # 公开方法
    # ==========================================

    def store_episode(
        self,
        query: str,
        answer: str,
        metadata: Optional[dict] = None,
    ) -> bool:
        """
        将一轮 Q&A 存入长期记忆（Milvus 向量 + PostgreSQL 原文）。

        调用时机：每轮对话 SSE 流结束后，异步写入。
        失败不影响对话体验，仅记录 warning 日志。
        """
        if not settings.MEMORY_ENABLED:
            return False

        normalized_query = _normalize_memory_part(query)
        normalized_answer = _normalize_memory_part(answer)
        if not normalized_query or not normalized_answer:
            return False

        memory_text = f"[用户问题] {normalized_query}\n[AI回答] {normalized_answer}"
        meta = dict(metadata or {})
        memory_id = make_memory_id(normalized_query, normalized_answer)

        try:
            # --- 分支 1：写入 PostgreSQL 存储原文 ---
            with get_db_session() as db:
                existing = db.query(LongTermMemory).filter(LongTermMemory.id == memory_id).first()
                if existing is not None:
                    logger.debug("🧠 相同问答已存在，跳过重复写入: %s", memory_id[:12])
                    return False

                record = LongTermMemory(
                    id=memory_id,
                    user_query=normalized_query,
                    assistant_answer=normalized_answer,
                    memory_text=memory_text,
                    meta_data=meta,
                )
                db.add(record)
                db.commit()
                logger.debug(f"🧠 长期记忆已写入 PostgreSQL: {memory_id[:12]}")

            # --- 分支 2：写入 Milvus 存储向量 ---
            collection = self._get_collection()

            # 截断超长文本（Embedding API 有限制）
            trunc_text = memory_text[:settings.EMBEDDING_MAX_TEXT_LENGTH]

            # Dense 向量（云端 API）
            dense_vec = self.embeddings.embed_query(trunc_text)

            collection.insert([
                [memory_id],       # chunk_id
                [trunc_text],       # text
                [dense_vec],        # dense_vector
                [{}],               # 保留现有 schema，长期记忆不使用 Sparse
                [meta],             # metadata
            ])
            collection.flush()
            logger.debug(f"🧠 长期记忆已写入 Milvus: {memory_id[:12]}")

            return True

        except Exception as e:
            # 静默失败 — 记忆存储是辅助功能，不影响核心对话
            logger.warning(f"⚠️ 长期记忆存储失败: {e}")
            return False

    def search_memories(self, query: str, top_k: Optional[int] = None) -> list[dict]:
        """
        语义检索历史记忆（Dense-only）。

        返回格式: [{"text": str, "metadata": dict, "score": float}, ...]
        """
        if top_k is None:
            top_k = settings.MEMORY_SEARCH_TOP_K

        try:
            collection = self._get_collection()

            # 生成查询 Dense 向量
            query_dense_vec = self.embeddings.embed_query(
                query[:settings.EMBEDDING_MAX_TEXT_LENGTH]
            )

            raw_results = collection.search(
                data=[query_dense_vec],
                anns_field="dense_vector",
                param={"metric_type": "L2", "params": {}},
                limit=top_k,
                output_fields=["text", "metadata"],
            )
            return [
                {
                    "text": hit.entity.get("text"),
                    "metadata": hit.entity.get("metadata", {}),
                    "score": hit.distance,
                }
                for hit in raw_results[0]
            ]

        except Exception as e:
            logger.warning(f"⚠️ 记忆检索失败: {e}")
            return []
