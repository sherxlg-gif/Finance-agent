import logging
import threading
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from langchain_core.tools import tool
from app.core.dependencies import get_retrieval_service

logger = logging.getLogger(__name__)


@dataclass
class _RetrievalRequestState:
    allow_multiple_targets: bool
    lock: threading.Lock = field(default_factory=threading.Lock)
    attempts: dict[tuple[str, str], int] = field(default_factory=dict)
    in_flight: set[tuple[str, str]] = field(default_factory=set)
    successful: set[tuple[str, str]] = field(default_factory=set)

    def _scope(self, company: str | None, year: str | None) -> tuple[str, str]:
        if not self.allow_multiple_targets:
            return ("__simple_question__", "")
        return ((company or "").strip(), (year or "").strip())

    def try_start(self, company: str | None, year: str | None) -> tuple[str, str] | None:
        scope = self._scope(company, year)
        with self.lock:
            if scope in self.successful or scope in self.in_flight:
                return None
            if self.attempts.get(scope, 0) >= 2:
                return None
            self.attempts[scope] = self.attempts.get(scope, 0) + 1
            self.in_flight.add(scope)
            return scope

    def finish(self, scope: tuple[str, str], success: bool) -> None:
        with self.lock:
            self.in_flight.discard(scope)
            if success:
                self.successful.add(scope)


_retrieval_request_state: ContextVar[_RetrievalRequestState | None] = ContextVar(
    "retrieval_request_state",
    default=None,
)
_COMPARISON_MARKERS = ("对比", "比较", "分别", "差异", "相比", "各自")


def begin_retrieval_request(user_query: str) -> Token:
    """为一次 Agent 请求建立检索次数状态。"""
    allow_multiple = any(marker in user_query for marker in _COMPARISON_MARKERS)
    return _retrieval_request_state.set(_RetrievalRequestState(allow_multiple))


def end_retrieval_request(token: Token) -> None:
    _retrieval_request_state.reset(token)


@tool
def financial_retriever_tool(query: str, company: str = None, year: str = None) -> str:
    """
    当你需要查询真实客观的上市公司的财务数据、业务情况、联系人等信息时，必须优先调用此工具。
    输入参数：
    - query: 具体的查询问题，例如"研发费用是多少？"、"联系人及联系方式是什么？"
    - company: 公司名称，例如"深信服"（可选，用户未指定时严禁自行猜测）
    - year: 年份，例如"2025"（可选。⚠️ 极度重要警告：如果用户提问中没有明确说出具体的年份，你必须将此参数保留为空(null/None)，绝对不允许使用默认值或自行猜测年份！）

    返回：
    包含上下文片段的纯文本，请仔细阅读返回的文本以提取事实。
    """

    request_state = _retrieval_request_state.get()
    scope = request_state.try_start(company, year) if request_state is not None else None
    if request_state is not None and scope is None:
        logger.info("检索调用已拦截：当前对象已有检索正在进行或已有非空证据")
        return "当前问题已有非空证据或相同对象的检索正在进行，请直接使用已有证据回答，不要再次检索。"

    logger.info(f"🛠️ Agent 决定调用检索工具 | 搜索词: {query} | 公司: {company} | 年份: {year}")

    try:
        retrieval_service = get_retrieval_service()
        docs = retrieval_service.run_pipeline(query=query, company=company, year=year)

        if not docs:
            if request_state is not None:
                request_state.finish(scope, success=False)
            return "数据库中未检索到相关财报信息。请告知用户没有查到，不要自行编造。"

        if request_state is not None:
            request_state.finish(scope, success=True)

        context_parts = []
        for i, d in enumerate(docs):
            source = d.metadata.get("source", "未知文件")
            score = d.metadata.get("rerank_score", "N/A")
            page = d.metadata.get("page_number", "")
            file_hash = d.metadata.get("file_hash", "")
            context_parts.append(
                f"--- 证据 {i + 1} [来源: {source}, 相关度: {score}"
                + (f", 页码: {page}" if page else "")
                + (f", hash: {file_hash}" if file_hash else "")
                + f"] ---\n{d.page_content}\n"
            )

        return "\n".join(context_parts)

    except Exception as e:
        if request_state is not None:
            request_state.finish(scope, success=False)
        logger.error(f"❌ 检索工具执行失败: {str(e)}", exc_info=True)
        return "检索系统发生内部错误，请稍后重试。"
