"""Agent 检索调用策略测试。"""

from app.prompts.loader import load_prompt


def test_prompt_limits_successful_simple_fact_question_to_one_retrieval():
    prompt = load_prompt("financial_agent")["system_prompt"]

    assert "简单事实题最多调用一次" in prompt
    assert "已有非空证据后禁止同义改写重试" in prompt


def test_prompt_keeps_one_retrieval_per_comparison_target():
    prompt = load_prompt("financial_agent")["system_prompt"]

    assert "每个不同公司或年份各调用一次" in prompt
