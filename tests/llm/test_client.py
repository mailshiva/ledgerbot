"""
Tests for src/llm/client.py and src/llm/config.py

All tests are fully offline — LLM provider calls are mocked so no API
key is required and no money is spent running the suite.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.llm.client import LLMClient, LLMResponse, UsageSummary, _estimate_cost
from src.llm.config import (
    COST_TABLE,
    LLMConfig,
    Model,
    Provider,
    load_config,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cfg() -> LLMConfig:
    """A config that points at Gemini Flash but doesn't need a real key."""
    return LLMConfig(
        default_model=Model.GEMINI_FLASH,
        gemini_api_key="fake-gemini-key",
        enable_cache=True,
    )


@pytest.fixture
def client(cfg: LLMConfig) -> LLMClient:
    return LLMClient(cfg)


def _mock_gemini_response(text: str, input_tokens: int = 10, output_tokens: int = 5):
    """Build a fake Gemini response object."""
    raw = MagicMock()
    raw.parts = [MagicMock(text=text, function_call=MagicMock(name=""))]
    raw.usage_metadata = MagicMock(
        prompt_token_count=input_tokens,
        candidates_token_count=output_tokens,
    )
    return raw


# ---------------------------------------------------------------------------
# config.py tests
# ---------------------------------------------------------------------------

class TestLLMConfig:
    def test_load_config_defaults(self):
        # Patch Keychain so tests are hermetic (no real macOS dependency)
        with patch("src.llm.config._keychain_get", return_value=""):
            cfg = load_config()
        assert cfg.default_model == Model.GEMINI_FLASH
        assert cfg.temperature == 0.0

    def test_load_config_overrides(self):
        with patch("src.llm.config._keychain_get", return_value=""):
            cfg = load_config(temperature=0.5, default_model=Model.GEMINI_PRO)
        assert cfg.temperature == 0.5
        assert cfg.default_model == Model.GEMINI_PRO

    def test_load_config_unknown_field_raises(self):
        with patch("src.llm.config._keychain_get", return_value=""):
            with pytest.raises(ValueError, match="Unknown LLMConfig field"):
                load_config(nonexistent_field=True)

    def test_validate_missing_api_key_raises(self):
        cfg = LLMConfig(default_model=Model.GEMINI_FLASH, gemini_api_key="")
        with pytest.raises(ValueError, match="portfolio_advisor"):
            cfg.validate()

    def test_validate_with_key_passes(self):
        cfg = LLMConfig(default_model=Model.GEMINI_FLASH, gemini_api_key="abc")
        cfg.validate()  # should not raise

    def test_keychain_key_takes_precedence_over_env(self, monkeypatch):
        """Keychain value should win over environment variable."""
        monkeypatch.setenv("GEMINI_API_KEY", "env-key")
        with patch("src.llm.config._keychain_get", return_value="keychain-key"):
            cfg = load_config()
        assert cfg.gemini_api_key == "keychain-key"

    def test_env_var_fallback_when_keychain_empty(self, monkeypatch):
        """Env var should be used when Keychain returns nothing."""
        monkeypatch.setenv("GEMINI_API_KEY", "env-only-key")
        with patch("src.llm.config._keychain_get", return_value=""):
            cfg = load_config()
        assert cfg.gemini_api_key == "env-only-key"

    def test_api_key_for_provider(self):
        cfg = LLMConfig(
            gemini_api_key="g-key",
            anthropic_api_key="a-key",
            openai_api_key="o-key",
        )
        assert cfg.api_key_for(Provider.GEMINI) == "g-key"
        assert cfg.api_key_for(Provider.ANTHROPIC) == "a-key"
        assert cfg.api_key_for(Provider.OPENAI) == "o-key"


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------

class TestCostEstimation:
    def test_known_model_cost(self):
        cost = _estimate_cost(Model.GEMINI_FLASH, input_tokens=1000, output_tokens=1000)
        expected = COST_TABLE[Model.GEMINI_FLASH].input_per_1k + COST_TABLE[Model.GEMINI_FLASH].output_per_1k
        assert abs(cost - expected) < 1e-9

    def test_zero_tokens(self):
        assert _estimate_cost(Model.GEMINI_FLASH, 0, 0) == 0.0

    def test_unknown_model_returns_zero(self):
        fake_model = MagicMock(spec=Model)
        fake_model.value = "unknown-model"
        assert _estimate_cost(fake_model, 100, 100) == 0.0


# ---------------------------------------------------------------------------
# LLMResponse
# ---------------------------------------------------------------------------

class TestLLMResponse:
    def test_total_tokens(self):
        r = LLMResponse(text="hi", model=Model.GEMINI_FLASH, input_tokens=10, output_tokens=5)
        assert r.total_tokens == 15

    def test_as_json_valid(self):
        r = LLMResponse(text='{"key": "value"}', model=Model.GEMINI_FLASH)
        assert r.as_json() == {"key": "value"}

    def test_as_json_strips_fences(self):
        text = "```json\n{\"a\": 1}\n```"
        r = LLMResponse(text=text, model=Model.GEMINI_FLASH)
        assert r.as_json() == {"a": 1}

    def test_as_json_invalid_raises(self):
        r = LLMResponse(text="not json at all", model=Model.GEMINI_FLASH)
        with pytest.raises(ValueError, match="not valid JSON"):
            r.as_json()


# ---------------------------------------------------------------------------
# UsageSummary
# ---------------------------------------------------------------------------

class TestUsageSummary:
    def test_record_accumulates(self):
        summary = UsageSummary()
        r1 = LLMResponse("a", Model.GEMINI_FLASH, input_tokens=10, output_tokens=5, cost_usd=0.001)
        r2 = LLMResponse("b", Model.GEMINI_FLASH, input_tokens=20, output_tokens=8, cost_usd=0.002)
        summary.record(r1)
        summary.record(r2)
        assert summary.total_calls == 2
        assert summary.total_input_tokens == 30
        assert summary.total_output_tokens == 13
        assert abs(summary.total_cost_usd - 0.003) < 1e-9

    def test_cache_hit_is_free(self):
        summary = UsageSummary()
        r = LLMResponse("a", Model.GEMINI_FLASH, input_tokens=10, output_tokens=5, cost_usd=0.0, cached=True)
        summary.record(r)
        assert summary.cache_hits == 1
        assert summary.total_cost_usd == 0.0

    def test_str_representation(self):
        summary = UsageSummary(total_calls=3, total_cost_usd=0.000123)
        assert "Calls: 3" in str(summary)
        assert "0.000123" in str(summary)


# ---------------------------------------------------------------------------
# LLMClient — core completion (mocked Gemini)
# ---------------------------------------------------------------------------

class TestLLMClientComplete:
    @patch("src.llm.client._complete_gemini")
    def test_complete_returns_response(self, mock_gemini, client):
        mock_gemini.return_value = LLMResponse(
            text="Paris", model=Model.GEMINI_FLASH,
            input_tokens=10, output_tokens=3, cost_usd=0.0001,
        )
        r = client.complete("What is the capital of France?")
        assert r.text == "Paris"
        assert r.model == Model.GEMINI_FLASH
        mock_gemini.assert_called_once()

    @patch("src.llm.client._complete_gemini")
    def test_usage_tracked(self, mock_gemini, client):
        mock_gemini.return_value = LLMResponse(
            text="ok", model=Model.GEMINI_FLASH,
            input_tokens=5, output_tokens=2, cost_usd=0.0001,
        )
        client.complete("hello")
        assert client.usage.total_calls == 1
        assert client.usage.total_input_tokens == 5

    @patch("src.llm.client._complete_gemini")
    def test_cache_hit_on_second_call(self, mock_gemini, client):
        mock_gemini.return_value = LLMResponse(
            text="cached", model=Model.GEMINI_FLASH,
            input_tokens=5, output_tokens=2, cost_usd=0.0001,
        )
        r1 = client.complete("same prompt")
        r2 = client.complete("same prompt")

        assert mock_gemini.call_count == 1   # only one real API call
        assert r2.cached is True
        assert client.usage.cache_hits == 1

    @patch("src.llm.client._complete_gemini")
    def test_cache_bypass(self, mock_gemini, client):
        mock_gemini.return_value = LLMResponse(
            text="fresh", model=Model.GEMINI_FLASH,
            input_tokens=5, output_tokens=2, cost_usd=0.0001,
        )
        client.complete("prompt", use_cache=False)
        client.complete("prompt", use_cache=False)
        assert mock_gemini.call_count == 2

    @patch("src.llm.client._complete_gemini")
    def test_clear_cache(self, mock_gemini, client):
        mock_gemini.return_value = LLMResponse(
            text="x", model=Model.GEMINI_FLASH,
            input_tokens=1, output_tokens=1, cost_usd=0.0,
        )
        client.complete("p")
        client.clear_cache()
        client.complete("p")
        assert mock_gemini.call_count == 2


# ---------------------------------------------------------------------------
# LLMClient — complete_json
# ---------------------------------------------------------------------------

class TestLLMClientCompleteJSON:
    @patch("src.llm.client._complete_gemini")
    def test_complete_json_parses(self, mock_gemini, client):
        mock_gemini.return_value = LLMResponse(
            text='{"total": 42.5}', model=Model.GEMINI_FLASH,
            input_tokens=10, output_tokens=5, cost_usd=0.0,
        )
        data, response = client.complete_json("Return JSON with total spending.")
        assert data == {"total": 42.5}

    @patch("src.llm.client._complete_gemini")
    def test_complete_json_raises_on_bad_output(self, mock_gemini, client):
        mock_gemini.return_value = LLMResponse(
            text="Sure! Here is the answer: 42", model=Model.GEMINI_FLASH,
            input_tokens=10, output_tokens=5, cost_usd=0.0,
        )
        with pytest.raises(ValueError, match="not valid JSON"):
            client.complete_json("Give me JSON.")


# ---------------------------------------------------------------------------
# LLMClient — tier routing helpers
# ---------------------------------------------------------------------------

class TestLLMClientRouting:
    @patch("src.llm.client._complete_gemini")
    def test_classify_uses_routing_model(self, mock_gemini, client):
        mock_gemini.return_value = LLMResponse(
            text="spending_summary", model=Model.GEMINI_FLASH,
            input_tokens=5, output_tokens=2, cost_usd=0.0,
        )
        r = client.classify("How much did I spend?")
        # verify model passed to adapter matches routing_model
        _, _, model_arg, *_ = mock_gemini.call_args.args
        assert model_arg == client.cfg.routing_model

    @patch("src.llm.client._complete_gemini")
    def test_reason_uses_reasoning_model(self, mock_gemini, client):
        mock_gemini.return_value = LLMResponse(
            text="answer", model=Model.GEMINI_PRO,
            input_tokens=20, output_tokens=10, cost_usd=0.0,
        )
        client.reason("Complex question requiring deep thought.")
        _, _, model_arg, *_ = mock_gemini.call_args.args
        assert model_arg == client.cfg.reasoning_model


# ---------------------------------------------------------------------------
# LLMClient — reset usage
# ---------------------------------------------------------------------------

class TestLLMClientReset:
    @patch("src.llm.client._complete_gemini")
    def test_reset_usage(self, mock_gemini, client):
        mock_gemini.return_value = LLMResponse(
            text="hi", model=Model.GEMINI_FLASH,
            input_tokens=5, output_tokens=2, cost_usd=0.001,
        )
        client.complete("hello")
        assert client.usage.total_calls == 1
        client.reset_usage()
        assert client.usage.total_calls == 0
        assert client.usage.total_cost_usd == 0.0


# ---------------------------------------------------------------------------
# LLMClient — Ollama (local model, no API key, stdlib urllib)
# ---------------------------------------------------------------------------

class TestLLMClientOllama:
    @pytest.fixture
    def ollama_cfg(self) -> LLMConfig:
        return LLMConfig(
            default_model=Model.QWEN3_8B,
            ollama_base_url="http://localhost:11434",
            enable_cache=False,
        )

    @pytest.fixture
    def ollama_client(self, ollama_cfg: LLMConfig) -> LLMClient:
        return LLMClient(ollama_cfg)

    @patch("src.llm.client._complete_ollama")
    def test_ollama_complete(self, mock_ollama, ollama_client):
        mock_ollama.return_value = LLMResponse(
            text="42", model=Model.QWEN3_8B,
            input_tokens=8, output_tokens=2, cost_usd=0.0,
        )
        r = ollama_client.complete("What is 6 x 7?")
        assert r.text == "42"
        assert r.cost_usd == 0.0        # local model is always free
        mock_ollama.assert_called_once()

    @patch("src.llm.client._complete_ollama")
    def test_ollama_usage_tracked(self, mock_ollama, ollama_client):
        mock_ollama.return_value = LLMResponse(
            text="ok", model=Model.QWEN3_8B,
            input_tokens=5, output_tokens=3, cost_usd=0.0,
        )
        ollama_client.complete("hello")
        assert ollama_client.usage.total_calls == 1
        assert ollama_client.usage.total_cost_usd == 0.0   # free

    def test_ollama_no_api_key_required(self):
        """Ollama config should be valid even with no API keys set."""
        cfg = LLMConfig(default_model=Model.QWEN3_8B)
        cfg.validate()  # must not raise

    def test_ollama_default_base_url(self):
        cfg = LLMConfig()
        assert "11434" in cfg.ollama_base_url

    @patch("urllib.request.urlopen")
    def test_ollama_adapter_parses_response(self, mock_urlopen):
        """Unit-test the adapter directly with a mocked HTTP response."""
        import json
        from io import BytesIO
        from unittest.mock import MagicMock
        from src.llm.client import _complete_ollama

        fake_response_body = json.dumps({
            "message": {"role": "assistant", "content": "Paris"},
            "prompt_eval_count": 10,
            "eval_count": 3,
        }).encode()

        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=MagicMock(read=MagicMock(return_value=fake_response_body)))
        mock_cm.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_cm

        cfg = LLMConfig(default_model=Model.QWEN3_8B, ollama_base_url="http://localhost:11434")
        r = _complete_ollama("What is the capital of France?", None, Model.QWEN3_8B, cfg)

        assert r.text == "Paris"
        assert r.input_tokens == 10
        assert r.output_tokens == 3
        assert r.cost_usd == 0.0

    @patch("urllib.request.urlopen")
    def test_ollama_adapter_connection_error(self, mock_urlopen):
        """Adapter should raise ConnectionError with helpful message when Ollama is down."""
        import urllib.error
        from src.llm.client import _complete_ollama

        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

        cfg = LLMConfig(default_model=Model.QWEN3_8B, ollama_base_url="http://localhost:11434")
        with pytest.raises(ConnectionError, match="ollama serve"):
            _complete_ollama("hello", None, Model.QWEN3_8B, cfg)

    @patch("urllib.request.urlopen")
    def test_ollama_adapter_tool_calls_serialized(self, mock_urlopen):
        """When model returns tool_calls, they should be serialized as JSON text."""
        import json
        from unittest.mock import MagicMock
        from src.llm.client import _complete_ollama

        fake_response_body = json.dumps({
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "query_db", "arguments": {"sql": "SELECT 1"}}}
                ],
            },
            "prompt_eval_count": 15,
            "eval_count": 10,
        }).encode()

        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=MagicMock(read=MagicMock(return_value=fake_response_body)))
        mock_cm.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_cm

        cfg = LLMConfig(default_model=Model.QWEN3_8B, ollama_base_url="http://localhost:11434")
        r = _complete_ollama("Run a query", None, Model.QWEN3_8B, cfg)

        parsed = json.loads(r.text)
        assert parsed[0]["name"] == "query_db"
        assert parsed[0]["args"]["sql"] == "SELECT 1"


# ---------------------------------------------------------------------------
# parse_tool_calls
# ---------------------------------------------------------------------------

class TestParseToolCalls:
    def test_parses_valid_tool_call(self):
        r = LLMResponse(
            text='[{"name": "get_categories", "args": {}}]',
            model=Model.GEMINI_FLASH,
        )
        calls = LLMClient.parse_tool_calls(r)
        assert len(calls) == 1
        assert calls[0]["name"] == "get_categories"
        assert calls[0]["args"] == {}

    def test_parses_tool_call_with_args(self):
        r = LLMResponse(
            text='[{"name": "find_transactions", "args": {"merchant": "Netflix", "limit": 5}}]',
            model=Model.GEMINI_FLASH,
        )
        calls = LLMClient.parse_tool_calls(r)
        assert calls[0]["args"]["merchant"] == "Netflix"

    def test_parses_multiple_tool_calls(self):
        r = LLMResponse(
            text='[{"name": "get_categories", "args": {}}, {"name": "get_merchants", "args": {"limit": 10}}]',
            model=Model.GEMINI_FLASH,
        )
        calls = LLMClient.parse_tool_calls(r)
        assert len(calls) == 2

    def test_plain_text_returns_empty(self):
        r = LLMResponse(text="You spent $150 last month.", model=Model.GEMINI_FLASH)
        assert LLMClient.parse_tool_calls(r) == []

    def test_empty_text_returns_empty(self):
        r = LLMResponse(text="", model=Model.GEMINI_FLASH)
        assert LLMClient.parse_tool_calls(r) == []

    def test_invalid_json_returns_empty(self):
        r = LLMResponse(text="[not valid json", model=Model.GEMINI_FLASH)
        assert LLMClient.parse_tool_calls(r) == []

    def test_json_without_name_key_returns_empty(self):
        r = LLMResponse(text='[{"tool": "get_categories"}]', model=Model.GEMINI_FLASH)
        assert LLMClient.parse_tool_calls(r) == []


# ---------------------------------------------------------------------------
# run_agentic_loop
# ---------------------------------------------------------------------------

class TestRunAgenticLoop:
    """
    Tests for the agentic loop. All LLM and DB calls are mocked — fully
    offline, no API keys, no real SQLite file.
    """

    @pytest.fixture
    def mock_db(self):
        """Minimal mock DatabaseManager."""
        db = MagicMock()
        db.conn = MagicMock()
        return db

    @patch("src.llm.client._complete_gemini")
    def test_single_shot_no_tool_call(self, mock_gemini, client, mock_db):
        """LLM answers directly — no tool calls, loop exits after 1 iteration."""
        mock_gemini.return_value = LLMResponse(
            text="You spent $150 last month.",
            model=Model.GEMINI_FLASH,
            input_tokens=20, output_tokens=10, cost_usd=0.0,
        )
        result = client.run_agentic_loop("How much did I spend?", mock_db)
        assert result.text == "You spent $150 last month."
        assert mock_gemini.call_count == 1

    @patch("src.agent.tools.execute_tool")
    @patch("src.llm.client._complete_gemini")
    def test_one_tool_call_then_answer(self, mock_gemini, mock_execute, client, mock_db):
        """LLM calls one tool, gets result, then answers."""
        tool_response = LLMResponse(
            text='[{"name": "get_categories", "args": {}}]',
            model=Model.GEMINI_FLASH,
            input_tokens=20, output_tokens=10, cost_usd=0.0,
        )
        final_response = LLMResponse(
            text="Your top category is Food & Dining.",
            model=Model.GEMINI_FLASH,
            input_tokens=30, output_tokens=15, cost_usd=0.0,
        )
        mock_gemini.side_effect = [tool_response, final_response]
        mock_execute.return_value = [{"category": "Food & Dining", "total_spent": 120.0}]

        result = client.run_agentic_loop("What is my top category?", mock_db)

        assert result.text == "Your top category is Food & Dining."
        assert mock_gemini.call_count == 2
        mock_execute.assert_called_once_with("get_categories", mock_db)

    @patch("src.agent.tools.execute_tool")
    @patch("src.llm.client._complete_gemini")
    def test_max_iterations_triggers_summary(self, mock_gemini, mock_execute, client, mock_db):
        """When LLM keeps calling tools past max_iterations, a summary is requested."""
        tool_response = LLMResponse(
            text='[{"name": "get_categories", "args": {}}]',
            model=Model.GEMINI_FLASH,
            input_tokens=10, output_tokens=5, cost_usd=0.0,
        )
        summary_response = LLMResponse(
            text="Based on the data, you spent $500 total.",
            model=Model.GEMINI_FLASH,
            input_tokens=40, output_tokens=20, cost_usd=0.0,
        )
        mock_gemini.side_effect = [tool_response] * 3 + [summary_response]
        mock_execute.return_value = []

        result = client.run_agentic_loop(
            "Summarise my spending", mock_db, max_iterations=3
        )
        assert "500" in result.text
        # 3 tool iterations + 1 summary call = 4 total
        assert mock_gemini.call_count == 4

    @patch("src.agent.tools.execute_tool")
    @patch("src.llm.client._complete_gemini")
    def test_tool_error_does_not_crash_loop(self, mock_gemini, mock_execute, client, mock_db):
        """A failing tool call returns an error dict; loop continues to final answer."""
        tool_response = LLMResponse(
            text='[{"name": "bad_tool", "args": {}}]',
            model=Model.GEMINI_FLASH,
            input_tokens=10, output_tokens=5, cost_usd=0.0,
        )
        final_response = LLMResponse(
            text="I could not retrieve that data.",
            model=Model.GEMINI_FLASH,
            input_tokens=20, output_tokens=8, cost_usd=0.0,
        )
        mock_gemini.side_effect = [tool_response, final_response]
        mock_execute.side_effect = KeyError("Unknown tool: 'bad_tool'")

        result = client.run_agentic_loop("anything", mock_db)
        assert result.text == "I could not retrieve that data."

    @patch("src.agent.tools.execute_tool")
    @patch("src.llm.client._complete_gemini")
    def test_usage_accumulated_across_iterations(self, mock_gemini, mock_execute, client, mock_db):
        """Token counts and costs are accumulated over all iterations."""
        mock_gemini.side_effect = [
            LLMResponse(text='[{"name": "get_categories", "args": {}}]',
                        model=Model.GEMINI_FLASH, input_tokens=10, output_tokens=5, cost_usd=0.001),
            LLMResponse(text="Done.", model=Model.GEMINI_FLASH,
                        input_tokens=20, output_tokens=8, cost_usd=0.002),
        ]
        mock_execute.return_value = []
        client.run_agentic_loop("question", mock_db)

        assert client.usage.total_input_tokens == 30
        assert client.usage.total_output_tokens == 13
        assert abs(client.usage.total_cost_usd - 0.003) < 1e-9


# ---------------------------------------------------------------------------
# _history_to_prompt
# ---------------------------------------------------------------------------

class TestHistoryToPrompt:
    def test_single_user_message(self):
        history = [{"role": "user", "content": "Hello"}]
        prompt = LLMClient._history_to_prompt(history)
        assert "User: Hello" in prompt

    def test_multi_turn_ordering(self):
        history = [
            {"role": "user",      "content": "Question"},
            {"role": "assistant", "content": "Tool call"},
            {"role": "tool",      "content": '{"result": []}'},
        ]
        prompt = LLMClient._history_to_prompt(history)
        assert prompt.index("User:") < prompt.index("Assistant:") < prompt.index("Tool results:")

    def test_unknown_role_ignored(self):
        history = [
            {"role": "user",   "content": "Hi"},
            {"role": "system", "content": "You are helpful"},
        ]
        prompt = LLMClient._history_to_prompt(history)
        assert "User: Hi" in prompt
        assert "system" not in prompt.lower()