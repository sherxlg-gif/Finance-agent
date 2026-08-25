"""Deterministic evidence-page metrics for retrieval evaluation."""

from collections.abc import Iterable
import re
from typing import Any


def _page_number(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def find_page_rank(
    results: Iterable[dict],
    source_file: str,
    source_page: int | str,
    max_rank: int = 10,
) -> int | None:
    """Return the first exact source-and-page match rank, or None."""
    expected_page = _page_number(source_page)
    if not source_file or expected_page is None:
        return None

    for rank, result in enumerate(results, start=1):
        if rank > max_rank:
            break
        metadata = result.get("metadata") or {}
        if (
            metadata.get("source") == source_file
            and _page_number(metadata.get("page_number")) == expected_page
        ):
            return rank
    return None


def _normalize_evidence(text: Any) -> str:
    return re.sub(r"[\s|*_`]+", "", str(text or ""))


def find_evidence_rank(
    results: Iterable[dict],
    source_file: str,
    source_page: int | str,
    evidence_quote: str | None = None,
    max_rank: int = 10,
) -> int | None:
    """Return a source-page-evidence match rank, or None.

    Existing datasets without evidence quotes retain page-only behavior.
    """
    expected_evidence = _normalize_evidence(evidence_quote)
    if not expected_evidence:
        return find_page_rank(results, source_file, source_page, max_rank)

    expected_page = _page_number(source_page)
    if not source_file or expected_page is None:
        return None

    for rank, result in enumerate(results, start=1):
        if rank > max_rank:
            break
        metadata = result.get("metadata") or {}
        if (
            metadata.get("source") == source_file
            and _page_number(metadata.get("page_number")) == expected_page
            and expected_evidence in _normalize_evidence(result.get("text"))
        ):
            return rank
    return None


def calculate_retrieval_metrics(rows: Iterable[dict], top_n: int = 5) -> dict:
    """Aggregate Recall, MRR and exact-page hit rate for one search mode."""
    if top_n < 1:
        raise ValueError("top_n must be at least 1")

    page_ranks = []
    evidence_ranks = []
    for row in rows:
        source_file = row.get("source_file")
        source_page = _page_number(row.get("source_page"))
        if not source_file or source_page is None:
            continue
        page_ranks.append(
            find_page_rank(
                row.get("results") or [],
                source_file,
                source_page,
                max_rank=10,
            )
        )
        evidence_ranks.append(
            find_evidence_rank(
                row.get("results") or [],
                source_file,
                source_page,
                row.get("evidence_quote"),
                max_rank=10,
            )
        )

    count = len(evidence_ranks)
    if count == 0:
        return {
            "evaluated_questions": 0,
            "recall_at_5": 0.0,
            "recall_at_10": 0.0,
            "mrr_at_10": 0.0,
            "page_hit_rate": 0.0,
            "page_hit_top_n": top_n,
            "evidence_quote_hit_rate_at_10": 0.0,
        }

    return {
        "evaluated_questions": count,
        "recall_at_5": round(sum(rank is not None and rank <= 5 for rank in page_ranks) / count, 4),
        "recall_at_10": round(sum(rank is not None and rank <= 10 for rank in page_ranks) / count, 4),
        "mrr_at_10": round(sum(1 / rank if rank is not None and rank <= 10 else 0 for rank in page_ranks) / count, 4),
        "page_hit_rate": round(sum(rank is not None and rank <= top_n for rank in page_ranks) / count, 4),
        "page_hit_top_n": top_n,
        "evidence_quote_hit_rate_at_10": round(
            sum(rank is not None and rank <= 10 for rank in evidence_ranks) / count,
            4,
        ),
    }
