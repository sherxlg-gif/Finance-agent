"""从 Milvus 现有 Child 数据重建共享 BM25 稀疏向量。"""

import logging
import os

from app.services.sparse_encoder import PersistentBM25Encoder

logger = logging.getLogger(__name__)


def rebuild_sparse_vectors(collection, batch_size: int = 500) -> dict[str, int]:
    """重算现有 Child 的 Sparse 向量，不触发 PDF 解析或 Dense Embedding。"""
    encoder = PersistentBM25Encoder()
    encoder.mark_dirty()
    temp_path = encoder.model_path.with_suffix(".json.tmp")

    rows = []
    iterator = None
    try:
        collection.load()
        iterator = collection.query_iterator(
            batch_size=batch_size,
            expr='metadata["doc_level"] == "child"',
            output_fields=["chunk_id", "text", "dense_vector", "metadata"],
        )
        while True:
            batch = iterator.next()
            if not batch:
                break
            rows.extend(batch)

        if not rows:
            logger.warning("BM25 重建未找到 Child 数据，检索保持 dense_only")
            return {"documents": 0, "upserted": 0}

        texts = [row["text"] for row in rows]
        encoder.fit(texts)
        sparse_vectors = encoder.encode_documents(texts)

        upserted = 0
        for start in range(0, len(rows), batch_size):
            end = min(start + batch_size, len(rows))
            batch_rows = rows[start:end]
            collection.upsert([
                [row["chunk_id"] for row in batch_rows],
                [row["text"] for row in batch_rows],
                [row["dense_vector"] for row in batch_rows],
                sparse_vectors[start:end],
                [row.get("metadata", {}) for row in batch_rows],
            ])
            upserted += len(batch_rows)

        collection.flush()
        encoder.save(temp_path)
        os.replace(temp_path, encoder.model_path)
        encoder.clear_dirty()
        logger.info("BM25 重建完成: documents=%d upserted=%d", len(rows), upserted)
        return {"documents": len(rows), "upserted": upserted}
    except Exception:
        logger.exception("BM25 重建失败，检索将保持 dense_only")
        raise
    finally:
        if iterator is not None:
            iterator.close()
        temp_path.unlink(missing_ok=True)
