import logging
from typing import List, Dict
from pymilvus import AnnSearchRequest, RRFRanker
from app.core.config import settings
from app.services.sparse_encoder import PersistentBM25Encoder

logger = logging.getLogger(__name__)


class HybridSearchEngine:
    def __init__(self, sparse_encoder=None):
        logger.info("🔌 初始化 Milvus 原生双路检索引擎...")
        self.sparse_encoder = sparse_encoder or PersistentBM25Encoder()
        self.last_search_mode = "dense_only"
        self.last_sparse_nnz = 0

    def execute_search(self, query: str, query_dense_vec: list, collection, expr: str, top_k: int = 15) -> List[Dict]:
        """
        组装双路请求，并直接交由 Milvus 数据库底层完成并行召回与 RRF 融合
        """
        # 1. 使用与入库一致的持久化词表生成 Sparse Query。
        try:
            query_sparse_matrix = self.sparse_encoder.encode_query(query)
            self.last_sparse_nnz = int(query_sparse_matrix.nnz)
        except Exception as exc:
            self.last_sparse_nnz = 0
            logger.warning("Sparse 查询不可用，将执行 dense_only: %s", exc)
            return self._execute_dense_only(query_dense_vec, collection, expr, top_k)

        if self.last_sparse_nnz == 0:
            logger.warning("Sparse 查询 nnz=0，将执行 dense_only")
            return self._execute_dense_only(query_dense_vec, collection, expr, top_k)

        # 读取底层 C 语言数组的指针 (indptr) 和数据 (data)，避开版本差异
        sparse_dict_list = []
        for i in range(query_sparse_matrix.shape[0]):
            # 找到第 i 行在底层一维数组中的起止位置
            start_idx = query_sparse_matrix.indptr[i]
            end_idx = query_sparse_matrix.indptr[i + 1]

            # 切片取出词的 ID 列表和对应的权重列表
            indices = query_sparse_matrix.indices[start_idx:end_idx]
            values = query_sparse_matrix.data[start_idx:end_idx]

            # 拼装成 Milvus 唯一认准的字典格式
            sparse_dict_list.append({int(k): float(v) for k, v in zip(indices, values)})

        # 2. 构建 Dense (语义) 召回请求
        req_dense = AnnSearchRequest(
            data=[query_dense_vec],
            anns_field="dense_vector",
            param={"metric_type": "L2"},
            limit=settings.HYBRID_DENSE_LIMIT,
            expr=expr
        )

        # 3. 构建 Sparse (字面量 BM25) 召回请求
        req_sparse = AnnSearchRequest(
            data=sparse_dict_list,
            anns_field="sparse_vector",
            param={"metric_type": "IP"},
            limit=settings.HYBRID_SPARSE_LIMIT,
            expr=expr
        )

        # 4. 底层 C++ 原生融合
        results = collection.hybrid_search(
            reqs=[req_dense, req_sparse],
            rerank=RRFRanker(k=settings.HYBRID_RRF_K),
            limit=top_k,
            output_fields=["text", "metadata"]
        )

        self.last_search_mode = "hybrid"
        logger.info("retrieval_mode=hybrid sparse_nnz=%d", self.last_sparse_nnz)

        return self._format_results(results)

    def _execute_dense_only(self, query_dense_vec, collection, expr, top_k):
        results = collection.search(
            data=[query_dense_vec],
            anns_field="dense_vector",
            param={"metric_type": "L2", "params": {}},
            limit=top_k,
            expr=expr,
            output_fields=["text", "metadata"],
        )
        self.last_search_mode = "dense_only"
        logger.info("retrieval_mode=dense_only sparse_nnz=%d", self.last_sparse_nnz)
        return self._format_results(results)

    @staticmethod
    def _format_results(results):
        # 组装结果返回给外层
        final_docs = []
        for hit in results[0]:  # results[0] 对应唯一的 query
            final_docs.append({
                "text": hit.entity.get("text"),
                "metadata": hit.entity.get("metadata", {}),
                "score": hit.distance  # 底层 RRF 给出的最终排名分
            })

        return final_docs
