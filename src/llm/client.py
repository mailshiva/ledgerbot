"""
Unified LLM client supporting Gemini (primary), Anthropic, OpenAI, and Ollama (local).

Key features
------------
- Single `.complete()` interface regardless of provider
- Structured JSON output via `.complete_json()`
- Tool / function calling via `.complete_with_tools()`
- Per-call token usage tracking and cost estimation
- Simple in-process cache (prompt hash → response) to avoid duplicate API calls
- Model-tier routing helpers (cheap model for classification, smart model for reasoning)

Quick start
-----------
    from src.llm.client import LLMClient
    from src.llm.config import load_config

    cfg = load_config()                        # reads GEMINI_API_KEY from env
    client = LLMClient(cfg)

    response = client.complete("What is 2 + 2?")
    print(response.text)                       # "4"
    print(response.cost_usd)                   # e.g. 0.000001
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from src.llm.config import COST_TABLE, MODEL_PROVIDER, LLMConfig, Model, Provider, load_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Response dataclass
# ---------------------------------------------------------------------------

@dataclass
class LLMResponse:
    text: str                          # raw text content
    model: Model                       # model that produced the response
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    cached: bool = False               # True if served from in-process cache
    latency_ms: float = 0.0
    raw: Any = field(default=None, repr=False)  # original SDK response object

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def as_json(self) -> dict:
        """Parse .text as JSON. Raises ValueError on failure."""
        text = self.text.strip()
        # Strip markdown fences if present
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM response is not valid JSON: {exc}\n---\n{self.text}") from exc


# ---------------------------------------------------------------------------
# Session-level usage accumulator
# ---------------------------------------------------------------------------

@dataclass
class UsageSummary:
    total_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    cache_hits: int = 0

    def record(self, response: LLMResponse) -> None:
        self.total_calls += 1
        self.total_input_tokens += response.input_tokens
        self.total_output_tokens += response.output_tokens
        self.total_cost_usd += response.cost_usd
        if response.cached:
            self.cache_hits += 1

    def __str__(self) -> str:
        return (
            f"Calls: {self.total_calls} | "
            f"Tokens in/out: {self.total_input_tokens}/{self.total_output_tokens} | "
            f"Cost: ${self.total_cost_usd:.6f} | "
            f"Cache hits: {self.cache_hits}"
        )


# ---------------------------------------------------------------------------
# Internal cost helper
# ---------------------------------------------------------------------------

def _estimate_cost(model: Model, input_tokens: int, output_tokens: int) -> float:
    cost = COST_TABLE.get(model)
    if cost is None:
        return 0.0
    return (input_tokens / 1000) * cost.input_per_1k + (output_tokens / 1000) * cost.output_per_1k


# ---------------------------------------------------------------------------
# Provider-specific adapters
# ---------------------------------------------------------------------------

def _complete_gemini(
        prompt: str,
        system: str | None,
        model: Model,
        cfg: LLMConfig,
        tools: list[dict] | None = None,
) -> LLMResponse:
    """Call the Gemini API using the google.genai SDK (0.4.x)."""
    try:
        import google.genai as genai  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "google-genai is required for Gemini. "
            "Install it with: pip install google-genai"
        ) from exc

    # Create client with API key
    client = genai.Client(api_key=cfg.gemini_api_key)

    # Build config dict with basic settings
    config_dict = {
        "temperature": cfg.temperature,
        "max_output_tokens": cfg.max_output_tokens,
    }

    # Add system instruction if provided
    if system:
        config_dict["system_instruction"] = system

    # Add tools if provided — convert OpenAI-style to Gemini FunctionDeclaration format
    if tools:
        gemini_functions = []
        for t in tools:
            fn = t.get("function", t)
            params = fn.get("parameters", {"type": "object", "properties": {}})
            gemini_functions.append(
                genai.types.FunctionDeclaration(
                    name=fn["name"],
                    description=fn.get("description", ""),
                    parameters=params,
                )
            )
        config_dict["tools"] = [genai.types.Tool(function_declarations=gemini_functions)]

    # Create GenerateContentConfig
    generate_config = genai.types.GenerateContentConfig(**config_dict)

    # Call the API
    t0 = time.perf_counter()
    raw = client.models.generate_content(
        model=model.value,
        contents=prompt,
        config=generate_config,
    )
    latency_ms = (time.perf_counter() - t0) * 1000

    # Extract text and tool calls from response
    tool_calls = []

    # Check parts for tool calls only; use raw.text for the aggregated text
    # (raw.text already combines all text parts — don't add part.text on top)
    if hasattr(raw, 'parts') and raw.parts:
        for part in raw.parts:
            if hasattr(part, "function_call") and part.function_call and part.function_call.name:
                fc = part.function_call
                tool_calls.append({"name": fc.name, "args": dict(fc.args)})

    # Only access raw.text when there are no tool calls — accessing it on a
    # function-call response triggers a Gemini SDK warning about non-text parts.
    text = "" if tool_calls else ((raw.text or "") if hasattr(raw, 'text') else "")

    # Get token counts
    input_tokens = 0
    output_tokens = 0

    if hasattr(raw, 'usage_metadata') and raw.usage_metadata:
        usage = raw.usage_metadata
        input_tokens = getattr(usage, "prompt_token_count", 0) or 0
        output_tokens = getattr(usage, "candidates_token_count", 0) or 0

    # If tool calls present, serialize them as JSON text for uniform handling
    if tool_calls and not text:
        text = json.dumps(tool_calls)

    return LLMResponse(
        text=text,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=_estimate_cost(model, input_tokens, output_tokens),
        latency_ms=latency_ms,
        raw=raw,
    )


def _complete_anthropic(
    prompt: str,
    system: str | None,
    model: Model,
    cfg: LLMConfig,
    tools: list[dict] | None = None,
) -> LLMResponse:
    """Call the Anthropic API using the anthropic SDK."""
    try:
        import anthropic  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "anthropic SDK is required. Install it with: pip install anthropic"
        ) from exc

    client = anthropic.Anthropic(api_key=cfg.anthropic_api_key)

    kwargs: dict[str, Any] = dict(
        model=model.value,
        max_tokens=cfg.max_output_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    if system:
        kwargs["system"] = system
    if tools:
        # Convert OpenAI-style {"type":"function","function":{...}} to Anthropic format
        anthropic_tools = []
        for t in tools:
            fn = t.get("function", t)  # handle both wrapped and bare dicts
            anthropic_tools.append({
                "name": fn["name"],
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
            })
        kwargs["tools"] = anthropic_tools

    t0 = time.perf_counter()
    raw = client.messages.create(**kwargs)
    latency_ms = (time.perf_counter() - t0) * 1000

    text = "".join(
        block.text for block in raw.content if hasattr(block, "text")
    )

    # Serialize tool_use blocks to text so parse_tool_calls() can detect them
    tool_calls_list = []
    for block in raw.content:
        if block.type == "tool_use":
            tool_calls_list.append({"name": block.name, "args": dict(block.input)})
    if tool_calls_list:
        text = json.dumps(tool_calls_list)

    input_tokens = raw.usage.input_tokens
    output_tokens = raw.usage.output_tokens

    return LLMResponse(
        text=text,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=_estimate_cost(model, input_tokens, output_tokens),
        latency_ms=latency_ms,
        raw=raw,
    )


def _complete_openai(
    prompt: str,
    system: str | None,
    model: Model,
    cfg: LLMConfig,
    tools: list[dict] | None = None,
) -> LLMResponse:
    """Call the OpenAI API using the openai SDK."""
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "openai SDK is required. Install it with: pip install openai"
        ) from exc

    client = OpenAI(api_key=cfg.openai_api_key)

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    kwargs: dict[str, Any] = dict(
        model=model.value,
        messages=messages,
        max_tokens=cfg.max_output_tokens,
        temperature=cfg.temperature,
    )
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    t0 = time.perf_counter()
    raw = client.chat.completions.create(**kwargs)
    latency_ms = (time.perf_counter() - t0) * 1000

    message = raw.choices[0].message
    text = message.content or ""

    input_tokens = raw.usage.prompt_tokens
    output_tokens = raw.usage.completion_tokens

    return LLMResponse(
        text=text,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=_estimate_cost(model, input_tokens, output_tokens),
        latency_ms=latency_ms,
        raw=raw,
    )



def _complete_ollama(
    prompt: str,
    system: str | None,
    model: Model,
    cfg: LLMConfig,
    tools: list[dict] | None = None,
) -> LLMResponse:
    """
    Call a local Ollama model via its OpenAI-compatible REST API.

    Ollama exposes POST /api/chat at http://localhost:11434 (or OLLAMA_HOST).
    No SDK required — uses the standard library urllib so there are zero
    extra dependencies.

    Tool calling is supported for models that declare tool support (qwen3
    does); falls back to plain text for models that don't.
    """
    import json as _json
    import urllib.error
    import urllib.request

    base_url = cfg.ollama_base_url.rstrip("/")
    url = f"{base_url}/api/chat"

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload: dict[str, Any] = {
        "model": model.value,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": cfg.temperature,
            "num_predict": cfg.max_output_tokens,
        },
    }
    if tools:
        payload["tools"] = tools

    body = _json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            raw_bytes = resp.read()
    except urllib.error.URLError as exc:
        raise ConnectionError(
            f"Could not reach Ollama at {base_url}. "
            "Make sure Ollama is running: `ollama serve`"
        ) from exc
    latency_ms = (time.perf_counter() - t0) * 1000

    raw = _json.loads(raw_bytes)

    message = raw.get("message", {})
    text = message.get("content", "")

    # Handle tool_calls returned by the model
    tool_calls = message.get("tool_calls", [])
    if tool_calls and not text:
        # Serialize to JSON so the agent loop can parse them uniformly
        text = _json.dumps([
            {"name": tc["function"]["name"], "args": tc["function"]["arguments"]}
            for tc in tool_calls
        ])

    # Ollama returns token counts in the top-level response object
    input_tokens  = raw.get("prompt_eval_count", 0) or 0
    output_tokens = raw.get("eval_count", 0) or 0

    return LLMResponse(
        text=text,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=0.0,          # local inference — always free
        latency_ms=latency_ms,
        raw=raw,
    )

# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------
#
# We look up adapter functions by name at call time (via getattr on the
# module) rather than capturing references in a dict at import time.
# This ensures unittest.mock.patch() -- which rebinds the module-level name
# -- is always respected during testing.

_ADAPTER_NAMES: dict[Provider, str] = {
    Provider.GEMINI:    "_complete_gemini",
    Provider.ANTHROPIC: "_complete_anthropic",
    Provider.OPENAI:    "_complete_openai",
    Provider.OLLAMA:    "_complete_ollama",
}


def _get_adapter(provider: Provider):
    """Return the adapter callable for *provider*, honouring any active mock."""
    import src.llm.client as _self
    return getattr(_self, _ADAPTER_NAMES[provider])


# ---------------------------------------------------------------------------
# Public client
# ---------------------------------------------------------------------------

class LLMClient:
    """
    Unified LLM client.

    Parameters
    ----------
    config:
        An ``LLMConfig`` instance (from ``load_config()``).  If omitted, one
        is created from environment variables.

    Examples
    --------
    >>> client = LLMClient()
    >>> r = client.complete("What is the capital of France?")
    >>> print(r.text)
    Paris
    >>> print(client.usage)
    Calls: 1 | Tokens in/out: 12/3 | Cost: $0.000001 | Cache hits: 0
    """

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.cfg = config or load_config()
        self.usage = UsageSummary()
        self._cache: dict[str, LLMResponse] = {}

    # ------------------------------------------------------------------
    # Core completion
    # ------------------------------------------------------------------

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: Model | None = None,
        use_cache: bool | None = None,
    ) -> LLMResponse:
        """
        Send a single-turn prompt and return an LLMResponse.

        Parameters
        ----------
        prompt:
            The user-facing message.
        system:
            Optional system / instruction prompt.
        model:
            Override the default model for this call.
        use_cache:
            Override the config-level cache setting for this call.
        """
        model = model or self.cfg.default_model
        should_cache = use_cache if use_cache is not None else self.cfg.enable_cache

        cache_key = self._cache_key(prompt, system, model)
        if should_cache and cache_key in self._cache:
            cached = self._cache[cache_key]
            cached_copy = LLMResponse(
                text=cached.text,
                model=cached.model,
                input_tokens=cached.input_tokens,
                output_tokens=cached.output_tokens,
                cost_usd=0.0,          # cache hits are free
                cached=True,
                latency_ms=0.0,
                raw=cached.raw,
            )
            self.usage.record(cached_copy)
            return cached_copy

        provider = MODEL_PROVIDER[model]
        adapter = _get_adapter(provider)

        logger.debug("LLM call | model=%s provider=%s prompt_len=%d", model.value, provider.value, len(prompt))

        response = adapter(prompt, system, model, self.cfg)
        self.usage.record(response)

        if should_cache:
            self._cache[cache_key] = response

        logger.debug(
            "LLM response | tokens=%d/%d cost=$%.6f latency=%.0fms",
            response.input_tokens, response.output_tokens, response.cost_usd, response.latency_ms,
        )
        return response

    # ------------------------------------------------------------------
    # Structured JSON output
    # ------------------------------------------------------------------

    def complete_json(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: Model | None = None,
    ) -> tuple[dict, LLMResponse]:
        """
        Ask the LLM to return JSON and parse the response automatically.

        Returns a (parsed_dict, response) tuple.
        Raises ValueError if the output cannot be parsed as JSON.
        """
        json_system = (system or "") + (
            "\n\nIMPORTANT: Respond ONLY with valid JSON. "
            "No markdown fences, no explanation, no preamble."
        )
        response = self.complete(prompt, system=json_system.strip(), model=model)
        return response.as_json(), response

    # ------------------------------------------------------------------
    # Tool / function calling
    # ------------------------------------------------------------------

    def complete_with_tools(
        self,
        prompt: str,
        tools: list[dict],
        *,
        system: str | None = None,
        model: Model | None = None,
    ) -> LLMResponse:
        """
        Send a prompt together with tool definitions.

        ``tools`` should follow the OpenAI function-calling schema:
        [{"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}]

        The returned ``LLMResponse.text`` contains either the model's plain-text
        answer or a JSON-serialised list of tool call dicts when the model
        chose to invoke a tool.
        """
        model = model or self.cfg.default_model
        provider = MODEL_PROVIDER[model]
        adapter = _get_adapter(provider)
        response = adapter(prompt, system, model, self.cfg, tools=tools)
        self.usage.record(response)
        return response

    # ------------------------------------------------------------------
    # Model-tier routing helpers
    # ------------------------------------------------------------------

    def classify(self, prompt: str, *, system: str | None = None) -> LLMResponse:
        """Use the cheap routing model for simple classification tasks."""
        return self.complete(prompt, system=system, model=self.cfg.routing_model)

    def reason(self, prompt: str, *, system: str | None = None) -> LLMResponse:
        """Use the full reasoning model for complex tasks."""
        return self.complete(prompt, system=system, model=self.cfg.reasoning_model)


    # ------------------------------------------------------------------
    # Tool call parsing
    # ------------------------------------------------------------------

    @staticmethod
    def parse_tool_calls(response: "LLMResponse") -> list[dict]:
        """
        Extract tool call(s) from an LLMResponse.

        All four provider adapters serialise tool calls as a JSON list in
        response.text when the model chose to invoke a tool:
            [{"name": "get_categories", "args": {}}, ...]

        Returns an empty list if the response is plain text (final answer).
        """
        text = response.text.strip()
        if not text or not text.startswith("["):
            return []
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list) and all("name" in item for item in parsed):
                return parsed
        except (json.JSONDecodeError, KeyError):
            pass
        return []

    # ------------------------------------------------------------------
    # Agentic loop
    # ------------------------------------------------------------------

    def run_agentic_loop(
        self,
        question: str,
        db,
        *,
        system: str | None = None,
        model: "Model | None" = None,
        max_iterations: int = 5,
        prior_history: list[dict] | None = None,
    ) -> "LLMResponse":
        """
        Run a full tool-use loop until the LLM returns a plain-text answer
        or max_iterations is reached.

        Flow for each iteration
        -----------------------
        1. Call the LLM with the current message history + TOOL_DEFINITIONS.
        2. If the response contains tool calls → execute each tool via
           execute_tool(), append results to history, go to step 1.
        3. If the response is plain text → return it as the final answer.

        Parameters
        ----------
        question       : The user's natural-language question.
        db             : DatabaseManager instance (passed to execute_tool).
        system         : Optional system prompt (defaults to AGENT_SYSTEM_PROMPT).
        model          : Override model (defaults to config default_model).
        max_iterations : Safety cap to prevent infinite loops (default 5).

        Returns the final LLMResponse containing the agent's plain-text answer.
        """
        #from src.agent.tools import TOOL_DEFINITIONS, execute_tool
        from src.agent.unified_tools import ALL_TOOL_DEFINITIONS as TOOL_DEFINITIONS, execute_any_tool as execute_tool
        from src.llm.prompts import AGENT_SYSTEM_PROMPT

        system_prompt = system or AGENT_SYSTEM_PROMPT
        model = model or self.cfg.default_model

        # Build a running message history (OpenAI-style role/content dicts)
        # that we accumulate across iterations.
        history: list[dict] = list(prior_history) if prior_history else []
        history.append({"role": "user", "content": question})

        for iteration in range(max_iterations):
            logger.debug("Agent iteration %d/%d", iteration + 1, max_iterations)

            # Flatten history into a single prompt string for providers that
            # don't natively support multi-turn (Ollama, Gemini via our adapter).
            prompt = self._history_to_prompt(history)

            response = self.complete_with_tools(
                prompt,
                TOOL_DEFINITIONS,
                system=system_prompt,
                model=model,
            )

            tool_calls = self.parse_tool_calls(response)

            # ── No tool calls → LLM produced a final answer ──────────────
            if not tool_calls:
                logger.debug("Agent finished after %d iteration(s)", iteration + 1)
                return response

            # ── Execute each requested tool ───────────────────────────────
            tool_results = []
            for call in tool_calls:
                tool_name = call.get("name", "")
                tool_args  = call.get("args", {})
                logger.debug("Executing tool: %s(%s)", tool_name, tool_args)

                try:
                    result = execute_tool(tool_name, db, **tool_args)
                except KeyError as exc:
                    result = {"error": str(exc)}
                except Exception as exc:
                    result = {"error": f"Tool execution failed: {exc}"}

                tool_results.append({
                    "tool": tool_name,
                    "args": tool_args,
                    "result": result,
                })

            # Append the assistant's tool-call turn + tool results to history
            history.append({
                "role": "assistant",
                "content": response.text,
            })
            history.append({
                "role": "tool",
                "content": json.dumps(tool_results, default=str),
            })

        # Safety fallback: max iterations reached — ask LLM to summarise
        # whatever it has so far rather than returning raw tool JSON.
        logger.warning("Agent reached max_iterations=%d, requesting summary", max_iterations)
        history.append({
            "role": "user",
            "content": (
                "You have reached the maximum number of tool calls. "
                "Please summarise the data collected so far and answer "
                "the original question as best you can."
            ),
        })
        prompt = self._history_to_prompt(history)
        return self.complete(prompt, system=system_prompt, model=model)

    # ------------------------------------------------------------------
    # History formatting helper
    # ------------------------------------------------------------------

    @staticmethod
    def _history_to_prompt(history: list[dict]) -> str:
        """
        Flatten a multi-turn message history into a single prompt string.

        Providers that natively support multi-turn (e.g. Anthropic) can
        receive history directly in future; for now a flat string works
        across all four adapters.

        Format:
            User: ...
            Assistant: ...
            Tool results: ...
            User: ...
        """
        parts = []
        for msg in history:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
            elif role == "tool":
                parts.append(f"Tool results: {content}")
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def clear_cache(self) -> None:
        """Evict all cached responses."""
        self._cache.clear()

    def reset_usage(self) -> None:
        """Reset the session-level usage counters."""
        self.usage = UsageSummary()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _cache_key(prompt: str, system: str | None, model: Model) -> str:
        payload = f"{model.value}|||{system or ''}|||{prompt}"
        return hashlib.sha256(payload.encode()).hexdigest()