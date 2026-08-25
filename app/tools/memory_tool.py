"""
Agent 跨对话长期记忆检索工具。
"""
import logging
from langchain_core.tools import tool
from app.core.dependencies import get_memory_service

logger = logging.getLogger(__name__)

CURRENT_TURN_REFERENCES = ("刚才", "上面", "本轮", "这轮", "当前对话", "前面提到")


@tool
def memory_retriever_tool(query: str) -> str:
    """
    仅当你需要回顾其他对话中的信息时调用此工具。
    适用场景：
    - 用户说"上次提到的那个数据"、"以前分析过的公司"
    - 用户明确引用"之前其他对话"中的内容
    - 用户问"你还记得上次的结论吗"

    不适用："刚才"、"上面"、"本轮"等当前对话引用；这些内容必须直接
    从当前消息历史回答，不要调用本工具。

    输入参数：
    - query: 用于搜索记忆的关键词或问题，例如"深信服毛利率"、"上次分析的网络安全公司"

    返回：与 query 语义相关的历史对话片段。如果没有找到相关记忆，会返回空结果。
    """

    if any(reference in query for reference in CURRENT_TURN_REFERENCES):
        return "这是当前对话中的引用，请直接使用当前对话历史回答，不要查询长期记忆。"

    logger.info(f"🧠 Agent 调用记忆检索工具 | 搜索: {query}")

    try:
        memory_service = get_memory_service()
        results = memory_service.search_memories(query)

        if not results:
            return "（长期记忆库中没有找到与该问题相关的历史记录。）"

        parts = []
        for i, r in enumerate(results):
            parts.append(f"--- 历史记忆 {i + 1} [相关度: {r.get('score', 'N/A')}] ---\n{r.get('text', '')}\n")

        return "\n".join(parts)

    except Exception as e:
        logger.error(f"❌ 记忆检索工具执行失败: {str(e)}", exc_info=True)
        return "记忆检索系统发生内部错误，请告知用户你无法获取历史记忆，但仍可基于当前上下文回答。"
