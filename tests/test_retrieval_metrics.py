"""确定性检索评测指标测试。"""

import csv
from pathlib import Path

from evals.retrieval_metrics import calculate_retrieval_metrics, find_evidence_rank


def test_metrics_require_the_expected_source_and_page():
    results = [
        {"metadata": {"source": "另一份报告.pdf", "page_number": 6}},
        {"metadata": {"source": "中国石化2026年半年度报告.pdf", "page_number": 7}},
        {"metadata": {"source": "中国石化2026年半年度报告.pdf", "page_number": 6}},
    ]

    assert find_evidence_rank(results, "中国石化2026年半年度报告.pdf", 6) == 3


def test_evidence_rank_requires_the_expected_quote_when_supplied():
    results = [
        {
            "text": "营业收入 1,436,561 1,409,052 2.0",
            "metadata": {"source": "report.pdf", "page_number": 6},
        },
        {
            "text": "基本每股收益 0.212 0.177 19.8",
            "metadata": {"source": "report.pdf", "page_number": 6},
        },
    ]

    assert find_evidence_rank(results, "report.pdf", 6, "基本每股收益 0.212 0.177 19.8") == 2


def test_metrics_report_recall_mrr_and_requested_top_n_page_hit_rate():
    rows = [
        {
            "source_file": "report.pdf",
            "source_page": 6,
            "results": [{"metadata": {"source": "report.pdf", "page_number": 6}}],
        },
        {
            "source_file": "report.pdf",
            "source_page": 7,
            "results": [
                {"metadata": {"source": "report.pdf", "page_number": 1}},
                {"metadata": {"source": "report.pdf", "page_number": 2}},
                {"metadata": {"source": "report.pdf", "page_number": 3}},
                {"metadata": {"source": "report.pdf", "page_number": 4}},
                {"metadata": {"source": "report.pdf", "page_number": 5}},
                {"metadata": {"source": "report.pdf", "page_number": 6}},
                {"metadata": {"source": "report.pdf", "page_number": 7}},
            ],
        },
        {
            "source_file": "",
            "source_page": "",
            "results": [],
        },
    ]

    metrics = calculate_retrieval_metrics(rows, top_n=5)

    assert metrics["evaluated_questions"] == 2
    assert metrics["recall_at_5"] == 0.5
    assert metrics["recall_at_10"] == 1.0
    assert metrics["mrr_at_10"] == round((1 + 1 / 7) / 2, 4)
    assert metrics["page_hit_rate"] == 0.5
    assert metrics["evidence_quote_hit_rate_at_10"] == 1.0


def test_sinopec_evidence_dataset_has_the_required_schema_and_15_rows():
    dataset_path = Path(__file__).parents[1] / "evals" / "eval_dataset_sinopec.csv"
    with dataset_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))

    assert len(rows) == 15
    assert rows[0].keys() == {
        "question_id",
        "question",
        "ground_truth",
        "company",
        "year",
        "source_file",
        "source_page",
        "evidence_quote",
        "answer_type",
    }
    assert {row["answer_type"] for row in rows} >= {
        "direct_numeric",
        "table_lookup",
        "business_or_risk",
        "python_calculation",
        "unanswerable",
    }
