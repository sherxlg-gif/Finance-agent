"""长期记忆的写入去重与路由测试。"""

from unittest.mock import MagicMock, patch

from app.services.memory import MemoryService, make_memory_id
from app.tools.memory_tool import memory_retriever_tool


def _memory_service() -> MemoryService:
    service = MemoryService.__new__(MemoryService)
    service.embeddings = MagicMock()
    service.embeddings.embed_query.return_value = [0.1] * 4
    service._collection = None
    return service


def test_memory_id_normalizes_equivalent_question_answer_text():
    assert make_memory_id("  营业收入是多少？\n", "答案  是  100。") == make_memory_id(
        "营业收入是多少？", "答案 是 100。"
    )


def test_store_episode_skips_duplicate_question_answer_without_vector_write():
    service = _memory_service()
    collection = MagicMock()
    service._get_collection = MagicMock(return_value=collection)

    db = MagicMock()
    db.query.return_value.filter.return_value.first.side_effect = [None, MagicMock()]
    context = MagicMock()
    context.__enter__.return_value = db

    with patch("app.services.memory.get_db_session", return_value=context):
        assert service.store_episode("营收是多少", "营收为100亿元") is True
        assert service.store_episode("营收是多少", "营收为100亿元") is False

    assert db.add.call_count == 1
    assert collection.insert.call_count == 1
    assert service.embeddings.embed_query.call_count == 1
    payload = collection.insert.call_args.args[0]
    assert len(payload[3]) == 1
    assert payload[3][0]
    assert all(value > 0 for value in payload[3][0].values())


def test_search_memories_uses_dense_vector_search_only():
    service = _memory_service()
    collection = MagicMock()
    collection.search.return_value = [[]]
    service._get_collection = MagicMock(return_value=collection)

    assert service.search_memories("上次讨论的营收") == []
    collection.search.assert_called_once()
    kwargs = collection.search.call_args.kwargs
    assert kwargs["anns_field"] == "dense_vector"
    assert kwargs["param"]["metric_type"] == "L2"


def test_current_turn_reference_does_not_query_long_term_memory():
    with patch("app.tools.memory_tool.get_memory_service") as get_memory_service:
        response = memory_retriever_tool.invoke({"query": "你刚才提到的营业收入是多少？"})

    assert "当前对话" in response
    get_memory_service.assert_not_called()
