"""请求级财报检索次数控制测试。"""

from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from app.tools.retriever_tool import (
    begin_retrieval_request,
    end_retrieval_request,
    financial_retriever_tool,
)


def _invoke(query: str, company: str | None = None, year: str | None = None) -> str:
    return financial_retriever_tool.invoke({
        "query": query,
        "company": company,
        "year": year,
    })


def test_simple_question_blocks_retrieval_after_first_success():
    service = MagicMock()
    service.run_pipeline.return_value = [Document("营业收入证据")]
    token = begin_retrieval_request("中国石化的年度营业收入是多少？")

    try:
        with patch("app.tools.retriever_tool.get_retrieval_service", return_value=service):
            first = _invoke("年度营业收入", company="中国石化")
            second = _invoke("全年营业收入", company="中国石化")
    finally:
        end_retrieval_request(token)

    assert "营业收入证据" in first
    assert "已有非空证据" in second
    assert service.run_pipeline.call_count == 1


def test_simple_question_allows_one_retry_after_empty_result():
    service = MagicMock()
    service.run_pipeline.side_effect = [[], [Document("重试命中的证据")]]
    token = begin_retrieval_request("中国石化的年度营业收入是多少？")

    try:
        with patch("app.tools.retriever_tool.get_retrieval_service", return_value=service):
            first = _invoke("年度营业收入", company="中国石化")
            second = _invoke("全年营业收入", company="中国石化")
    finally:
        end_retrieval_request(token)

    assert "未检索到" in first
    assert "重试命中的证据" in second
    assert service.run_pipeline.call_count == 2


def test_comparison_question_allows_distinct_targets_once_each():
    service = MagicMock()
    service.run_pipeline.return_value = [Document("对比证据")]
    token = begin_retrieval_request("对比中国石化和比亚迪的营业收入")

    try:
        with patch("app.tools.retriever_tool.get_retrieval_service", return_value=service):
            _invoke("营业收入", company="中国石化")
            _invoke("营业收入", company="比亚迪")
            duplicate = _invoke("全年营业收入", company="中国石化")
    finally:
        end_retrieval_request(token)

    assert "已有非空证据" in duplicate
    assert service.run_pipeline.call_count == 2


def test_retriever_output_separates_matched_text_from_full_context():
    """工具输出同时暴露命中子块（来源预览）和完整父块（LLM 上下文）。"""
    service = MagicMock()
    service.run_pipeline.return_value = [
        Document(
            page_content="完整父块上下文",
            metadata={
                "source": "报告.pdf",
                "matched_child_text": "命中子块原文",
                "matched_page_number": 58,
                "file_hash": "hash58",
                "rerank_score": 0.91,
            },
        )
    ]
    token = begin_retrieval_request("营业收入是多少？")
    try:
        with patch("app.tools.retriever_tool.get_retrieval_service", return_value=service):
            output = _invoke("营业收入", company="测试公司")
    finally:
        end_retrieval_request(token)

    assert "[来源文件] 报告.pdf" in output
    assert "[命中页码] 58" in output
    assert "[命中原文] 命中子块原文" in output
    assert "[完整上下文]" in output
    assert "完整父块上下文" in output
