"""
src.llm — Unified LLM client for the bank statement agent.

Primary provider: Gemini
Optional providers: Anthropic, OpenAI

Usage
-----
    from src.llm import LLMClient, load_config, Model

    client = LLMClient(load_config())
    response = client.complete("Summarise my spending last month.")
    print(response.text)
"""

from src.llm.client import LLMClient, LLMResponse, UsageSummary
from src.llm.config import LLMConfig, Model, Provider, load_config

__all__ = [
    "LLMClient",
    "LLMResponse",
    "UsageSummary",
    "LLMConfig",
    "Model",
    "Provider",
    "load_config",
]
