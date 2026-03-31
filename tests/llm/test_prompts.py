"""Tests for src/llm/prompts.py"""

from src.llm.prompts import (
    answer_synthesis_prompt,
    intent_classification_prompt,
    sql_generation_prompt,
    AGENT_SYSTEM_PROMPT,
    FEW_SHOT_NL_TO_SQL,
)


def test_sql_generation_prompt_contains_question():
    prompt = sql_generation_prompt("How much did I spend last month?")
    assert "How much did I spend last month?" in prompt
    assert "transactions" in prompt          # schema is embedded
    assert "starbucks" in prompt.lower()     # few-shot examples present


def test_intent_classification_prompt():
    prompt = intent_classification_prompt("Show me my Uber charges")
    assert "Show me my Uber charges" in prompt
    assert "merchant_lookup" in prompt


def test_answer_synthesis_prompt():
    prompt = answer_synthesis_prompt(
        question="Top 3 categories?",
        data='[{"category": "Food", "total": 120.5}]',
    )
    assert "Top 3 categories?" in prompt
    assert "Food" in prompt


def test_agent_system_prompt_has_schema():
    assert "transactions" in AGENT_SYSTEM_PROMPT
    assert "amount" in AGENT_SYSTEM_PROMPT
    assert "category" in AGENT_SYSTEM_PROMPT


def test_few_shot_examples_present():
    assert "SELECT" in FEW_SHOT_NL_TO_SQL
    assert "GROUP BY" in FEW_SHOT_NL_TO_SQL
