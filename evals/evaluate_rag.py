"""Run LLM-judge and deterministic evidence retrieval evaluation."""

import argparse
import datetime
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from tqdm import tqdm

from app.core.config import settings
from app.services.agent import FinancialAgentService
from app.services.retrieval import RetrievalService
from evals.retrieval_metrics import (
    calculate_retrieval_metrics,
    find_evidence_rank,
    find_page_rank,
)

logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


class RAGEvaluator:
    def __init__(self):
        print("正在初始化 RAG 评测流水线...")
        self.agent = FinancialAgentService()
        self.retrieval_service = None

        self.judge_llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            api_key=settings.DASHSCOPE_API_KEY,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            temperature=0.0,
        )
        self.eval_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个严苛的金融投研 RAG 系统评测专家。
你需要对比【用户问题】、【人工给出的标准答案】以及【AI 系统的实际回答】。

请从以下两个维度给出 0 到 10 的整数评分：
1. 准确性：AI 回答是否涵盖标准答案中的核心事实。正确且不矛盾的扩展信息不扣分。
2. 无幻觉性：AI 回答是否没有编造与标准答案或证据冲突的数据。

只输出合法 JSON，不要输出其他文字：
{{"accuracy": 8, "faithfulness": 10, "reason": "理由简述"}}"""),
            ("user", "【用户问题】: {question}\n\n【标准答案】: {ground_truth}\n\n【AI 实际回答】: {ai_answer}"),
        ])
        self.eval_chain = self.eval_prompt | self.judge_llm
        print("评测流水线准备就绪。")

    def _get_retrieval_service(self) -> RetrievalService:
        if self.retrieval_service is None:
            self.retrieval_service = RetrievalService()
        return self.retrieval_service

    @staticmethod
    def _format_hits(results) -> list[dict]:
        formatted = []
        for hit in results[0]:
            formatted.append({
                "text": hit.entity.get("text"),
                "metadata": hit.entity.get("metadata", {}),
                "score": hit.distance,
            })
        return formatted

    @staticmethod
    def _sparse_query_data(sparse_matrix) -> list[dict[int, float]]:
        rows = []
        for row_index in range(sparse_matrix.shape[0]):
            start = sparse_matrix.indptr[row_index]
            end = sparse_matrix.indptr[row_index + 1]
            rows.append({
                int(index): float(value)
                for index, value in zip(
                    sparse_matrix.indices[start:end],
                    sparse_matrix.data[start:end],
                )
            })
        return rows

    def _compare_retrieval_modes(self, query: str) -> dict:
        """Run Child-level Dense, Sparse and Hybrid retrieval for one question."""
        service = self._get_retrieval_service()
        collection = service.collection
        engine = service.hybrid_engine
        expr = 'metadata["doc_level"] == "child"'
        limit = 10

        collection.load()
        dense_vector = service.embeddings.embed_query(query)
        dense_results = collection.search(
            data=[dense_vector],
            anns_field="dense_vector",
            param={"metric_type": "L2", "params": {}},
            limit=limit,
            expr=expr,
            output_fields=["text", "metadata"],
        )

        sparse_results = []
        sparse_nnz = 0
        try:
            sparse_matrix = engine.sparse_encoder.encode_query(query)
            sparse_nnz = int(sparse_matrix.nnz)
            if sparse_nnz > 0:
                raw_sparse_results = collection.search(
                    data=self._sparse_query_data(sparse_matrix),
                    anns_field="sparse_vector",
                    param={"metric_type": "IP", "params": {}},
                    limit=limit,
                    expr=expr,
                    output_fields=["text", "metadata"],
                )
                sparse_results = self._format_hits(raw_sparse_results)
        except Exception as exc:
            logger.warning("Sparse-only evaluation unavailable: %s", exc)

        hybrid_results = engine.execute_search(
            query=query,
            query_dense_vec=dense_vector,
            collection=collection,
            expr=expr,
            top_k=limit,
        )

        return {
            "dense_only": self._format_hits(dense_results),
            "sparse_only": sparse_results,
            "hybrid": hybrid_results,
            "sparse_nnz": sparse_nnz,
            "hybrid_execution_mode": engine.last_search_mode,
        }

    @staticmethod
    def _read_dataset(csv_path: str | Path) -> pd.DataFrame:
        try:
            dataframe = pd.read_csv(csv_path, encoding="utf-8")
        except UnicodeDecodeError:
            dataframe = pd.read_csv(csv_path, encoding="gbk")
        dataframe.columns = dataframe.columns.str.strip().str.lower()
        required = {"question", "ground_truth"}
        if not required.issubset(dataframe.columns):
            raise ValueError(
                "CSV 必须包含 question 和 ground_truth，"
                f"实际表头为: {list(dataframe.columns)}"
            )
        return dataframe

    @staticmethod
    def _text_or_empty(value) -> str:
        return "" if pd.isna(value) else str(value).strip()

    @staticmethod
    def _retrieval_summary(retrieval_rows: dict[str, list], top_n: int) -> dict:
        summary = {
            mode: calculate_retrieval_metrics(rows, top_n=top_n)
            for mode, rows in retrieval_rows.items()
        }
        sparse_nnz_values = [
            row["sparse_nnz"]
            for row in retrieval_rows.get("hybrid", [])
            if row.get("sparse_nnz") is not None
        ]
        summary["sparse_query_nnz"] = {
            "questions": len(sparse_nnz_values),
            "average": round(sum(sparse_nnz_values) / len(sparse_nnz_values), 4)
            if sparse_nnz_values else 0.0,
            "minimum": min(sparse_nnz_values) if sparse_nnz_values else 0,
            "nonzero_rate": round(
                sum(value > 0 for value in sparse_nnz_values) / len(sparse_nnz_values),
                4,
            ) if sparse_nnz_values else 0.0,
        }
        return summary

    def run_evaluation(self, csv_path: str, output_path: str, top_n: int = 5):
        print(f"正在读取测试集: {csv_path}")
        dataframe = self._read_dataset(csv_path)
        has_evidence = {"source_file", "source_page"}.issubset(dataframe.columns)

        results = []
        retrieval_rows = {"dense_only": [], "sparse_only": [], "hybrid": []}
        total_accuracy = 0
        total_faithfulness = 0
        judged_count = 0

        print(f"开始评测，共 {len(dataframe)} 道测试题。")
        for index, row in tqdm(dataframe.iterrows(), total=len(dataframe)):
            question = str(row["question"])
            ground_truth = str(row["ground_truth"])
            record = row.to_dict()

            if has_evidence and self._text_or_empty(row.get("source_page")):
                try:
                    comparison = self._compare_retrieval_modes(question)
                    source_file = self._text_or_empty(row.get("source_file"))
                    source_page = row.get("source_page")
                    for mode in retrieval_rows:
                        mode_results = comparison[mode]
                        retrieval_rows[mode].append({
                            "source_file": source_file,
                            "source_page": source_page,
                            "evidence_quote": row.get("evidence_quote"),
                            "results": mode_results,
                            "sparse_nnz": comparison["sparse_nnz"],
                        })
                        record[f"{mode}_page_rank"] = find_page_rank(
                            mode_results,
                            source_file,
                            source_page,
                        )
                        record[f"{mode}_evidence_rank"] = find_evidence_rank(
                            mode_results,
                            source_file,
                            source_page,
                            row.get("evidence_quote"),
                        )
                    record["sparse_query_nnz"] = comparison["sparse_nnz"]
                    record["hybrid_execution_mode"] = comparison["hybrid_execution_mode"]
                except Exception as exc:
                    logger.warning("Question %s retrieval evaluation failed: %s", index + 1, exc)
                    record["retrieval_error"] = str(exc)

            try:
                ai_answer = self.agent.chat(question)
                judge_response = self.eval_chain.invoke({
                    "question": question,
                    "ground_truth": ground_truth,
                    "ai_answer": ai_answer,
                })
                score_dict = json.loads(
                    judge_response.content.replace("```json", "").replace("```", "").strip()
                )
                accuracy = score_dict.get("accuracy", 0)
                faithfulness = score_dict.get("faithfulness", 0)
                total_accuracy += accuracy
                total_faithfulness += faithfulness
                judged_count += 1
                record.update({
                    "ai_answer": ai_answer,
                    "accuracy_score": accuracy,
                    "faithfulness_score": faithfulness,
                    "judge_reason": score_dict.get("reason", ""),
                })
            except Exception as exc:
                logger.warning("Question %s LLM Judge failed: %s", index + 1, exc)
                record["judge_error"] = str(exc)

            results.append(record)

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(results).to_csv(output, index=False, encoding="utf-8-sig")

        summary = {
            "dataset": str(Path(csv_path).resolve()),
            "questions": len(dataframe),
            "top_n": top_n,
            "llm_judge": {
                "judged_questions": judged_count,
                "average_accuracy": round(total_accuracy / judged_count, 4) if judged_count else None,
                "average_faithfulness": round(total_faithfulness / judged_count, 4) if judged_count else None,
            },
        }
        if has_evidence:
            summary["retrieval_comparison"] = self._retrieval_summary(retrieval_rows, top_n)

        summary_path = output.with_name(f"{output.stem}_summary.json")
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        print("\n自动化评测报告总结")
        print(f"总测试题数: {len(dataframe)}")
        if judged_count:
            print(f"平均准确性: {total_accuracy / judged_count:.2f} / 10")
            print(f"平均无幻觉性: {total_faithfulness / judged_count:.2f} / 10")
        if has_evidence:
            print(json.dumps(summary["retrieval_comparison"], ensure_ascii=False, indent=2))
        print(f"详细报告: {output}")
        print(f"汇总报告: {summary_path}")
        return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Finance Agent RAG behavior")
    parser.add_argument(
        "--dataset",
        default=str(PROJECT_ROOT / "evals" / "eval_dataset.csv"),
        help="CSV dataset path",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=settings.RERANK_TOP_N,
        help="Top-N cutoff used by deterministic page-hit metrics",
    )
    args = parser.parse_args()
    if args.top_n < 1:
        parser.error("--top-n must be at least 1")
    return args


def main() -> None:
    args = parse_args()
    dataset = Path(args.dataset)
    if not dataset.is_absolute():
        dataset = PROJECT_ROOT / dataset

    reports_dir = PROJECT_ROOT / "evals" / "reports"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv = reports_dir / f"eval_report_{timestamp}.csv"

    evaluator = RAGEvaluator()
    evaluator.run_evaluation(str(dataset), str(output_csv), top_n=args.top_n)


if __name__ == "__main__":
    main()
