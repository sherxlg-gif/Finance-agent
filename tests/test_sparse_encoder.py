"""持久化 BM25 编码器与检索降级测试。"""

from unittest.mock import MagicMock

import numpy as np

from app.services.hybrid_search import HybridSearchEngine
from app.services.sparse_encoder import PersistentBM25Encoder


def test_bm25_save_load_keeps_document_and_query_vectors(tmp_path):
    model_path = tmp_path / "finance_bm25.json"
    corpus = ["营业收入 石油炼制", "网络安全 研发投入"]

    encoder = PersistentBM25Encoder(model_path)
    encoder.fit(corpus)
    document_vectors = encoder.encode_documents(corpus).toarray()
    query_vector = encoder.encode_query("营业收入 研发投入").toarray()
    encoder.save()

    restored = PersistentBM25Encoder(model_path)
    assert restored.load() is True
    np.testing.assert_allclose(restored.encode_documents(corpus).toarray(), document_vectors)
    np.testing.assert_allclose(restored.encode_query("营业收入 研发投入").toarray(), query_vector)


def test_chinese_query_uses_vocabulary_fitted_from_all_documents(tmp_path):
    encoder = PersistentBM25Encoder(tmp_path / "finance_bm25.json")
    encoder.fit(["营业收入 石油炼制", "网络安全 研发投入"])

    document_vectors = encoder.encode_documents(["营业收入 石油炼制", "网络安全 研发投入"])
    query_vector = encoder.encode_query("营业收入 研发投入")

    assert document_vectors.shape[0] == 2
    assert query_vector.shape[1] == document_vectors.shape[1]
    assert query_vector.nnz > 0


def test_bm25_warmup_loads_persisted_model_and_returns_nonzero_nnz(tmp_path):
    model_path = tmp_path / "finance_bm25.json"
    trained = PersistentBM25Encoder(model_path)
    trained.fit(["中国石化 营业收入", "网络安全 研发投入"])
    trained.save()

    restored = PersistentBM25Encoder(model_path)

    assert restored.warmup("中国石化 营业收入") > 0
    assert restored.encode_query("营业收入").nnz > 0


def test_hybrid_search_falls_back_to_dense_when_sparse_query_is_empty():
    sparse_encoder = MagicMock()
    sparse_encoder.encode_query.return_value = MagicMock(nnz=0)
    engine = HybridSearchEngine(sparse_encoder=sparse_encoder)
    collection = MagicMock()
    collection.search.return_value = [[]]

    results = engine.execute_search(
        query="不存在于词表中的词",
        query_dense_vec=[0.1, 0.2],
        collection=collection,
        expr='metadata["doc_level"] == "child"',
        top_k=5,
    )

    assert results == []
    collection.search.assert_called_once()
    collection.hybrid_search.assert_not_called()
