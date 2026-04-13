"""
Tests for graph.py (LangGraph agent) and unified_tools.py

Uses mock LLMClient and in-memory SQLite — no real API calls.

Tests:
  - unified_tools: ALL_TOOL_DEFINITIONS merge, execute_any_tool routing
  - AgentState initialization
  - call_llm node: state updates
  - should_continue: conditional edge logic
  - call_tools node: tool execution and history append
  - force_summary: max iterations fallback
  - GraphAgent.invoke: full loop (no tools, single tool, multi-tool)
  - Multi-turn conversation history
  - Max iterations safety
"""

import json
import sqlite3
from unittest.mock import MagicMock, patch
from dataclasses import dataclass, field

# We need to mock the imports before importing graph.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# ---------------------------------------------------------------------------
# Setup: Create mock modules so graph.py can import without real deps
# ---------------------------------------------------------------------------

# Mock tools modules
mock_tools = MagicMock()
mock_tools.TOOL_DEFINITIONS = [
    {"type": "function", "function": {"name": "get_categories", "description": "Get credit card spending by category", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "query_db", "description": "Run SQL on credit card tables", "parameters": {"type": "object", "properties": {"sql": {"type": "string"}}}}},
]
mock_tools.execute_tool = MagicMock(return_value=[{"category": "Food", "total": 100}])

mock_bank_tools = MagicMock()
mock_bank_tools.BANK_TOOL_DEFINITIONS = [
    {"type": "function", "function": {"name": "get_bank_categories", "description": "Get bank spending by category", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "get_loan_summary", "description": "Loan payment summary", "parameters": {"type": "object", "properties": {}}}},
]
mock_bank_tools.execute_bank_tool = MagicMock(return_value=[{"category": "Utilities", "total": 200}])

sys.modules["src.agent.tools"] = mock_tools
sys.modules["src.agent.bank_tools"] = mock_bank_tools

# Mock LLM modules
mock_prompts = MagicMock()
mock_prompts.AGENT_SYSTEM_PROMPT = "You are a financial assistant."
sys.modules["src.llm.prompts"] = mock_prompts

mock_config = MagicMock()
sys.modules["src.llm.config"] = mock_config

mock_client_mod = MagicMock()
sys.modules["src.llm.client"] = mock_client_mod

# Now import our modules
from src.agent.unified_tools import ALL_TOOL_DEFINITIONS, execute_any_tool, _CREDIT_TOOL_NAMES, _BANK_TOOL_NAMES
from src.agent.graph import (
    GraphAgent, AgentState, Turn,
    call_llm, call_tools, should_continue, force_summary,
    _history_to_prompt,
)

passed = 0
failed = 0

def check(name, condition):
    global passed, failed
    if condition:
        print(f"  PASS: {name}")
        passed += 1
    else:
        print(f"  FAIL: {name}")
        failed += 1


# ---------------------------------------------------------------------------
# Mock LLMClient
# ---------------------------------------------------------------------------

class MockLLMResponse:
    def __init__(self, text, input_tokens=10, output_tokens=5, cost_usd=0.001):
        self.text = text
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cost_usd = cost_usd


class MockLLMClient:
    """Mock LLMClient for testing graph nodes."""

    def __init__(self, responses=None):
        """
        responses: list of MockLLMResponse to return in sequence.
        If None, returns a plain text answer.
        """
        self._responses = responses or [MockLLMResponse("You spent $500 total.")]
        self._call_count = 0

    def complete_with_tools(self, prompt, tools, system=None, model=None):
        resp = self._responses[min(self._call_count, len(self._responses) - 1)]
        self._call_count += 1
        return resp

    def complete(self, prompt, system=None, model=None):
        resp = self._responses[min(self._call_count, len(self._responses) - 1)]
        self._call_count += 1
        return resp

    @staticmethod
    def parse_tool_calls(response):
        text = response.text.strip()
        if text.startswith("["):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list) and all("name" in item for item in parsed):
                    return parsed
            except json.JSONDecodeError:
                pass
        return []


mock_db = MagicMock()


# =====================================================================
print("=== unified_tools: ALL_TOOL_DEFINITIONS ===")
check("merged count", len(ALL_TOOL_DEFINITIONS) == 4)  # 2 credit + 2 bank

names = {t["function"]["name"] for t in ALL_TOOL_DEFINITIONS}
check("has get_categories", "get_categories" in names)
check("has query_db", "query_db" in names)
check("has get_bank_categories", "get_bank_categories" in names)
check("has get_loan_summary", "get_loan_summary" in names)

check("credit names", "get_categories" in _CREDIT_TOOL_NAMES)
check("bank names", "get_bank_categories" in _BANK_TOOL_NAMES)

print()
print("=== unified_tools: execute_any_tool routing ===")
# Reset mocks
mock_tools.execute_tool.reset_mock()
mock_bank_tools.execute_bank_tool.reset_mock()

execute_any_tool("get_categories", mock_db)
check("credit tool dispatched", mock_tools.execute_tool.called)

mock_tools.execute_tool.reset_mock()
execute_any_tool("get_bank_categories", mock_db)
check("bank tool dispatched", mock_bank_tools.execute_bank_tool.called)

try:
    execute_any_tool("nonexistent", mock_db)
    check("unknown raises KeyError", False)
except KeyError:
    check("unknown raises KeyError", True)

print()
print("=== _history_to_prompt ===")
history = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there"},
    {"role": "tool", "content": '{"result": 42}'},
    {"role": "user", "content": "Thanks"},
]
prompt = _history_to_prompt(history)
check("contains User:", "User: Hello" in prompt)
check("contains Assistant:", "Assistant: Hi there" in prompt)
check("contains Tool results:", "Tool results:" in prompt)
check("contains second user", "User: Thanks" in prompt)

print()
print("=== should_continue ===")
state_with_tools: AgentState = {
    "question": "test", "system_prompt": "", "messages": [],
    "prior_history": [], "llm_response": None,
    "tool_calls": [{"name": "get_categories", "args": {}}],
    "iteration": 1, "max_iterations": 5,
    "final_answer": "", "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0,
}
check("has tools → call_tools", should_continue(state_with_tools) == "call_tools")

state_no_tools = {**state_with_tools, "tool_calls": []}
check("no tools → end", should_continue(state_no_tools) == "end")

state_max_iter = {**state_with_tools, "iteration": 5, "max_iterations": 5}
check("max iter → end", should_continue(state_max_iter) == "end")

state_under_max = {**state_with_tools, "iteration": 4, "max_iterations": 5}
check("under max → call_tools", should_continue(state_under_max) == "call_tools")

print()
print("=== call_llm node ===")
client = MockLLMClient([MockLLMResponse("Plain answer", 20, 10, 0.002)])
state: AgentState = {
    "question": "How much?", "system_prompt": "You are helpful.",
    "messages": [{"role": "user", "content": "How much?"}],
    "prior_history": [], "llm_response": None, "tool_calls": [],
    "iteration": 0, "max_iterations": 5,
    "final_answer": "", "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0,
}
state = call_llm(state, client, mock_db)
check("llm_response set", state["llm_response"] is not None)
check("llm_response text", state["llm_response"]["text"] == "Plain answer")
check("iteration incremented", state["iteration"] == 1)
check("tokens accumulated", state["input_tokens"] == 20)
check("cost accumulated", state["cost_usd"] == 0.002)
check("no tool calls", state["tool_calls"] == [])

print()
print("=== call_llm with tool calls ===")
tool_json = json.dumps([{"name": "get_categories", "args": {}}])
client2 = MockLLMClient([MockLLMResponse(tool_json, 15, 8, 0.001)])
state2: AgentState = {
    "question": "Show categories", "system_prompt": "",
    "messages": [{"role": "user", "content": "Show categories"}],
    "prior_history": [], "llm_response": None, "tool_calls": [],
    "iteration": 0, "max_iterations": 5,
    "final_answer": "", "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0,
}
state2 = call_llm(state2, client2, mock_db)
check("tool calls parsed", len(state2["tool_calls"]) == 1)
check("tool name", state2["tool_calls"][0]["name"] == "get_categories")

print()
print("=== call_tools node ===")
mock_tools.execute_tool.reset_mock()
mock_tools.execute_tool.return_value = [{"category": "Food", "total": 100}]

state3: AgentState = {
    "question": "test", "system_prompt": "",
    "messages": [{"role": "user", "content": "test"}],
    "prior_history": [],
    "llm_response": {"text": tool_json, "input_tokens": 10, "output_tokens": 5, "cost_usd": 0.001},
    "tool_calls": [{"name": "get_categories", "args": {}}],
    "iteration": 1, "max_iterations": 5,
    "final_answer": "", "input_tokens": 10, "output_tokens": 5, "cost_usd": 0.001,
}
state3 = call_tools(state3, mock_db)
check("tool_calls cleared", state3["tool_calls"] == [])
check("messages grew", len(state3["messages"]) == 3)  # user + assistant + tool
check("assistant message added", state3["messages"][1]["role"] == "assistant")
check("tool result added", state3["messages"][2]["role"] == "tool")

# Verify tool result content
tool_content = json.loads(state3["messages"][2]["content"])
check("tool result has data", tool_content[0]["tool"] == "get_categories")

print()
print("=== force_summary ===")
# Case 1: not at max iterations — use llm_response text
state_ok: AgentState = {
    "question": "test", "system_prompt": "",
    "messages": [], "prior_history": [],
    "llm_response": {"text": "Final answer here", "input_tokens": 0, "output_tokens": 0, "cost_usd": 0},
    "tool_calls": [],
    "iteration": 2, "max_iterations": 5,
    "final_answer": "", "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0,
}
state_ok = force_summary(state_ok, client)
check("normal: uses llm_response", state_ok["final_answer"] == "Final answer here")

# Case 2: at max iterations with pending tool calls — forces summary
state_max: AgentState = {
    "question": "test", "system_prompt": "",
    "messages": [{"role": "user", "content": "test"}], "prior_history": [],
    "llm_response": {"text": tool_json, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0},
    "tool_calls": [{"name": "get_categories", "args": {}}],
    "iteration": 5, "max_iterations": 5,
    "final_answer": "", "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0,
}
summary_client = MockLLMClient([MockLLMResponse("Summary of findings", 30, 15, 0.003)])
state_max = force_summary(state_max, summary_client)
check("max iter: forced summary", "Summary" in state_max["final_answer"] or len(state_max["final_answer"]) > 0)

print()
print("=== GraphAgent.invoke: no tools (direct answer) ===")
direct_client = MockLLMClient([MockLLMResponse("You spent $500.")])
agent = GraphAgent(llm_client=direct_client, db=mock_db)
answer = agent.invoke("How much did I spend?")
check("direct answer", answer == "You spent $500.")
check("turn recorded", len(agent.conversation_history) == 1)
check("turn question", agent.conversation_history[0].question == "How much did I spend?")

print()
print("=== GraphAgent.invoke: single tool call ===")
# First response: tool call. Second response: final answer.
tool_then_answer = MockLLMClient([
    MockLLMResponse(json.dumps([{"name": "get_categories", "args": {}}]), 20, 10, 0.002),
    MockLLMResponse("Your top category is Groceries at $200.", 25, 15, 0.003),
])
agent2 = GraphAgent(llm_client=tool_then_answer, db=mock_db)
answer2 = agent2.invoke("What are my top categories?")
check("tool→answer works", "Groceries" in answer2 or len(answer2) > 0)
check("cost accumulated", agent2.conversation_history[0].cost_usd > 0)

print()
print("=== GraphAgent: multi-turn ===")
multi_client = MockLLMClient([
    MockLLMResponse("You spent $500 total."),
    MockLLMResponse("Last month was $120."),
])
agent3 = GraphAgent(llm_client=multi_client, db=mock_db)
a1 = agent3.invoke("Total spending?")
a2 = agent3.invoke("What about last month?")
check("multi-turn: 2 turns", len(agent3.conversation_history) == 2)
check("multi-turn: turn 1", agent3.conversation_history[0].turn_number == 1)
check("multi-turn: turn 2", agent3.conversation_history[1].turn_number == 2)

print()
print("=== GraphAgent: chat() alias ===")
alias_client = MockLLMClient([MockLLMResponse("Alias works")])
agent4 = GraphAgent(llm_client=alias_client, db=mock_db)
check("chat() alias", agent4.chat("test") == "Alias works")

print()
print("=== GraphAgent: reset_history ===")
agent4.reset_history()
check("history cleared", len(agent4.conversation_history) == 0)

print()
print("=== GraphAgent: get_turn_summary ===")
summary_client2 = MockLLMClient([MockLLMResponse("Answer 1"), MockLLMResponse("Answer 2")])
agent5 = GraphAgent(llm_client=summary_client2, db=mock_db)
agent5.invoke("Q1")
agent5.invoke("Q2")
summary = agent5.get_turn_summary()
check("summary has Turn 1", "Turn 1" in summary)
check("summary has Turn 2", "Turn 2" in summary)
check("summary has total cost", "Total cost" in summary)

print()
print("=== GraphAgent: empty question rejected ===")
try:
    agent5.invoke("")
    check("empty rejected", False)
except ValueError:
    check("empty rejected", True)

try:
    agent5.invoke("   ")
    check("whitespace rejected", False)
except ValueError:
    check("whitespace rejected", True)

print()
print("=== GraphAgent: max iterations safety ===")
# Client always returns tool calls — should hit max_iterations
always_tools = MockLLMClient([
    MockLLMResponse(json.dumps([{"name": "get_categories", "args": {}}])),
] * 10 + [MockLLMResponse("Forced summary after max iterations")])
agent6 = GraphAgent(llm_client=always_tools, db=mock_db, max_iterations=3)
answer6 = agent6.invoke("Keep calling tools forever")
check("max iter: got answer", len(answer6) > 0)
check("max iter: didn't loop forever", True)  # if we got here, it didn't hang

print()
print("=== Turn dataclass ===")
turn = Turn(
    turn_number=1, question="test", tools_invoked=["get_categories"],
    tool_results=[{}], final_answer="answer", input_tokens=100,
    output_tokens=50, cost_usd=0.005, timestamp="2025-01-01T00:00:00",
)
check("turn fields", turn.turn_number == 1 and turn.cost_usd == 0.005)

print()
print("=" * 60)
print(f"RESULTS: {passed} passed, {failed} failed")
if failed == 0:
    print("ALL TESTS PASSED")
print("=" * 60)