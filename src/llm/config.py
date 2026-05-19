"""
LLM configuration: supported models, pricing, and API key loading.

Supported providers: Gemini (primary), Anthropic, OpenAI

API key resolution order (same pattern as portfolio-advisor project):
  1. macOS Keychain  — security find-generic-password -s "portfolio_advisor" -a "<key_name>"
  2. Environment variable fallback — GEMINI_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY
  3. Empty string (caller should call cfg.validate() to catch missing keys early)

Keychain entries expected (add once via Terminal):
  security add-generic-password -a "gemini_api_key"    -s "portfolio_advisor" -w "YOUR_KEY"
  security add-generic-password -a "anthropic_api_key" -s "portfolio_advisor" -w "YOUR_KEY"
  security add-generic-password -a "openai_api_key"    -s "portfolio_advisor" -w "YOUR_KEY"
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum


# ---------------------------------------------------------------------------
# Keychain helper
# ---------------------------------------------------------------------------

_KEYCHAIN_SERVICE = "portfolio_advisor"


def _keychain_get(account: str) -> str:
    """
    Read a secret from the macOS Keychain.

    Returns the secret string, or an empty string if:
      - not running on macOS
      - the entry does not exist
      - the `security` command fails for any reason
    """
    if sys.platform != "darwin":
        return ""
    try:
        result = subprocess.run(
            [
                "security", "find-generic-password",
                "-s", _KEYCHAIN_SERVICE,
                "-a", account,
                "-w",           # print password only
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return ""


def _resolve_key(keychain_account: str, env_var: str) -> str:
    """Try Keychain first, fall back to environment variable."""
    return _keychain_get(keychain_account) or os.getenv(env_var, "")


# ---------------------------------------------------------------------------
# Provider & Model enums
# ---------------------------------------------------------------------------

class Provider(str, Enum):
    GEMINI = "gemini"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OLLAMA = "ollama"


class Model(str, Enum):
    # Gemini
    GEMINI_FLASH = "gemini-2.5-flash"
    GEMINI_PRO = "gemini-2.5-pro"

    # Anthropic
    CLAUDE_HAIKU = "claude-haiku-4-5-20251001"
    CLAUDE_SONNET = "claude-sonnet-4-6"

    # OpenAI
    GPT4O_MINI = "gpt-4o-mini"
    GPT4O = "gpt-4o"

    # Ollama (local)
    QWEN3_8B = "qwen3:8b"


# ---------------------------------------------------------------------------
# Cost table  (USD per 1 000 tokens)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelCost:
    input_per_1k: float   # $ per 1 000 input tokens
    output_per_1k: float  # $ per 1 000 output tokens


COST_TABLE: dict[Model, ModelCost] = {
    # Gemini  (approximate public pricing)
    Model.GEMINI_FLASH:   ModelCost(input_per_1k=0.000075, output_per_1k=0.0003),
    Model.GEMINI_PRO:     ModelCost(input_per_1k=0.00125,  output_per_1k=0.005),

    # Anthropic
    Model.CLAUDE_HAIKU:   ModelCost(input_per_1k=0.00025,  output_per_1k=0.00125),
    Model.CLAUDE_SONNET:  ModelCost(input_per_1k=0.003,    output_per_1k=0.015),

    # OpenAI
    Model.GPT4O_MINI:     ModelCost(input_per_1k=0.00015,  output_per_1k=0.0006),
    Model.GPT4O:          ModelCost(input_per_1k=0.0025,   output_per_1k=0.01),

    # Ollama — local inference, no cost
    Model.QWEN3_8B:       ModelCost(input_per_1k=0.0,      output_per_1k=0.0),
}

# Map each model to its provider
MODEL_PROVIDER: dict[Model, Provider] = {
    Model.GEMINI_FLASH:  Provider.GEMINI,
    Model.GEMINI_PRO:    Provider.GEMINI,
    Model.CLAUDE_HAIKU:  Provider.ANTHROPIC,
    Model.CLAUDE_SONNET: Provider.ANTHROPIC,
    Model.GPT4O_MINI:    Provider.OPENAI,
    Model.GPT4O:         Provider.OPENAI,

    # Ollama
    Model.QWEN3_8B:      Provider.OLLAMA,
}


# ---------------------------------------------------------------------------
# Runtime config (loaded once from environment)
# ---------------------------------------------------------------------------

@dataclass
class LLMConfig:
    # Which model to use by default
    default_model: Model = Model.GEMINI_FLASH

    # API keys — Keychain primary, env var fallback
    gemini_api_key: str = field(
        default_factory=lambda: _resolve_key("gemini_api_key", "GEMINI_API_KEY")
    )
    anthropic_api_key: str = field(
        default_factory=lambda: _resolve_key("anthropic_api_key", "ANTHROPIC_API_KEY")
    )
    openai_api_key: str = field(
        default_factory=lambda: _resolve_key("openai_api_key", "OPENAI_API_KEY")
    )

    # Ollama — local server, no API key needed
    ollama_base_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_HOST", "http://localhost:11434")
    )

    # Generation defaults
    max_output_tokens: int = 2048
    temperature: float = 0.0        # deterministic for SQL / structured output

    # Simple in-process response cache (prompt hash → response text)
    enable_cache: bool = True

    # Tier routing: use a cheap model for intent classification
    routing_model: Model = Model.GEMINI_FLASH
    reasoning_model: Model = Model.GEMINI_PRO

    def api_key_for(self, provider: Provider) -> str:
        mapping = {
            Provider.GEMINI:    self.gemini_api_key,
            Provider.ANTHROPIC: self.anthropic_api_key,
            Provider.OPENAI:    self.openai_api_key,
            Provider.OLLAMA:    "",   # no API key for local models
        }
        return mapping[provider]

    def validate(self) -> None:
        """Raise ValueError if the required API key for the default model is missing."""
        provider = MODEL_PROVIDER[self.default_model]
        if provider == Provider.OLLAMA:
            return  # local model — no API key required
        key = self.api_key_for(provider)
        if not key:
            account = f"{provider.value}_api_key"
            env_var  = f"{provider.value.upper()}_API_KEY"
            raise ValueError(
                f"Missing API key for provider '{provider.value}'. "
                f"Add it to macOS Keychain:\n"
                f"  security add-generic-password -a \"{account}\" "
                f"-s \"portfolio_advisor\" -w \"YOUR_KEY\"\n"
                f"Or set the environment variable: {env_var}"
            )


def load_config(**overrides) -> LLMConfig:
    """
    Build an LLMConfig, resolving API keys from macOS Keychain first,
    then environment variables as fallback.

    Keychain service name: "portfolio_advisor" (shared with portfolio-advisor project).

    Example:
        cfg = load_config(default_model=Model.GEMINI_PRO, temperature=0.2)
    """
    cfg = LLMConfig()
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise ValueError(f"Unknown LLMConfig field: {k!r}")
        setattr(cfg, k, v)
    return cfg