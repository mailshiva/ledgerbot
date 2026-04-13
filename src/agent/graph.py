"""
src/agent/graph.py

LangGraph-based agent for financial data queries.

Replaces the manual run_agentic_loop() in LLMClient with a declarative
StateGraph. Same tool-calling behaviour, but with:
  - Explicit state management (AgentState TypedDict)
  - Conditional edges (tool call → execute → loop back)
  - Configurable max iterations via state
  - Clean separation between routing, execution, and generation
  - Drop-in replacement: graph_agent.invoke(question) → str

Graph structure:
    ┌──────────┐
    │  START   │
    └────┬─────┘
         │
    ┌────▼─────┐
    │  call_llm │◄──────────────────────┐
    └────┬─────┘                        │
         │                              │
    ┌────▼──────────┐    ┌──────────┐   │
    │ should_continue├───►│call_tools├───┘
    └────┬──────────┘    └──────────┘
         │ (no tools)
    ┌────▼─────┐
    │   END    │
    └──────────┘

Usage:
    from src.agent.graph import GraphAgent

    agent = GraphAgent(llm_client=client, db=db)
    answer = agent.invoke("How much did I spend on groceries?")

    # Multi-turn
    answer2 = agent.invoke("What about last month?")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, TypedDict, Literal

try:
    from src.llm.client import LLMClient, LLMResponse
    from src.llm.prompts import AGENT_SYSTEM_PROMPT
    from src.agent.unified_tools import ALL_TOOL_DEFINITIONS, execute_any_tool
except ImportError:
    # Standalone / test mode
    from unified_tools import ALL_TOOL_DEFINITIONS, execute_any_tool
    LLMClient = None
    LLMResponse = None
    AGENT_SYSTEM_PROMPT = "You are a helpful financial assistant."

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    """State passed between graph nodes."""
    # Input
    question: str
    system_prompt: str

    # Conversation
    messages: list[dict]           # OpenAI-style role/content dicts
    prior_history: list[dict]      # prior turns for multi-turn context

    # LLM interaction
    llm_response: dict | None      # serialised LLMResponse fields
    tool_calls: list[dict]         # parsed tool calls from LLM

    # Control
    iteration: int
    max_iterations: int

    # Output
    final_answer: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


# ---------------------------------------------------------------------------
# Turn tracking (same as agent.py)
# ---------------------------------------------------------------------------

@dataclass
class Turn:
    """Single conversational turn."""
    turn_number: int
    question: str
    tools_invoked: list[str]
    tool_results: list[dict]
    final_answer: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    timestamp: str


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

def call_llm(state: AgentState, llm_client: LLMClient, db) -> AgentState:
    """
    Node: Call the LLM with current message history + tool definitions.
    Parses tool calls from the response.
    """
    # Build prompt from message history
    messages = state["messages"]
    prompt = _history_to_prompt(messages)

    response = llm_client.complete_with_tools(
        prompt,
        ALL_TOOL_DEFINITIONS,
        system=state["system_prompt"],
    )

    tool_calls = llm_client.parse_tool_calls(response)

    state["llm_response"] = {
        "text": response.text,
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
        "cost_usd": response.cost_usd,
    }
    state["tool_calls"] = tool_calls
    state["input_tokens"] = state.get("input_tokens", 0) + response.input_tokens
    state["output_tokens"] = state.get("output_tokens", 0) + response.output_tokens
    state["cost_usd"] = state.get("cost_usd", 0.0) + response.cost_usd
    state["iteration"] = state.get("iteration", 0) + 1

    logger.debug(
        "call_llm: iteration=%d, tool_calls=%d",
        state["iteration"], len(tool_calls),
    )

    return state


def call_tools(state: AgentState, db) -> AgentState:
    """
    Node: Execute each tool call and append results to message history.
    """
    tool_calls = state["tool_calls"]
    messages = state["messages"]

    # Append the assistant's tool-call turn
    messages.append({
        "role": "assistant",
        "content": state["llm_response"]["text"],
    })

    # Execute tools
    tool_results = []
    tools_invoked = []
    for call in tool_calls:
        tool_name = call.get("name", "")
        tool_args = call.get("args", {})
        logger.debug("Executing tool: %s(%s)", tool_name, tool_args)

        try:
            result = execute_any_tool(tool_name, db, **tool_args)
        except KeyError as exc:
            result = {"error": str(exc)}
        except Exception as exc:
            result = {"error": f"Tool execution failed: {exc}"}

        tool_results.append({
            "tool": tool_name,
            "args": tool_args,
            "result": result,
        })
        tools_invoked.append(tool_name)

    # Append tool results to message history
    messages.append({
        "role": "tool",
        "content": json.dumps(tool_results, default=str),
    })

    state["messages"] = messages
    state["tool_calls"] = []  # clear for next iteration

    return state


def should_continue(state: AgentState) -> Literal["call_tools", "end"]:
    """
    Conditional edge: decide whether to execute tools or finish.

    Returns "call_tools" if the LLM requested tools and we haven't
    hit max iterations. Returns "end" otherwise.
    """
    if state["tool_calls"] and state["iteration"] < state["max_iterations"]:
        return "call_tools"
    return "end"


def force_summary(state: AgentState, llm_client: LLMClient) -> AgentState:
    """
    Called when max iterations reached with pending tool calls.
    Asks the LLM to summarize what it has so far.
    """
    if state["iteration"] >= state["max_iterations"] and state.get("tool_calls"):
        logger.warning(
            "Max iterations (%d) reached, forcing summary",
            state["max_iterations"],
        )
        state["messages"].append({
            "role": "user",
            "content": (
                "You have reached the maximum number of tool calls. "
                "Please summarise the data collected so far and answer "
                "the original question as best you can."
            ),
        })
        prompt = _history_to_prompt(state["messages"])
        response = llm_client.complete(prompt, system=state["system_prompt"])
        state["final_answer"] = response.text
        state["input_tokens"] += response.input_tokens
        state["output_tokens"] += response.output_tokens
        state["cost_usd"] += response.cost_usd
    else:
        state["final_answer"] = state["llm_response"]["text"]

    return state


# ---------------------------------------------------------------------------
# Prompt helper (same as LLMClient._history_to_prompt)
# ---------------------------------------------------------------------------

def _history_to_prompt(history: list[dict]) -> str:
    """Flatten multi-turn message history into a single prompt string."""
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


# ---------------------------------------------------------------------------
# GraphAgent — high-level wrapper
# ---------------------------------------------------------------------------

class GraphAgent:
    """
    LangGraph-based agent for financial data queries.

    Drop-in replacement for Agent.chat() — same interface,
    declarative graph execution underneath.

    Parameters
    ----------
    llm_client : LLMClient instance
    db : DatabaseManager or DualWriteManager
    max_iterations : max tool-call loops per question (default 5)
    system_prompt : override system prompt
    """

    def __init__(
        self,
        llm_client: LLMClient,
        db,
        max_iterations: int = 5,
        system_prompt: str | None = None,
    ):
        self.llm_client = llm_client
        self.db = db
        self.max_iterations = max_iterations
        self.system_prompt = system_prompt or AGENT_SYSTEM_PROMPT
        self.conversation_history: list[Turn] = []

    def invoke(self, question: str) -> str:
        """
        Execute a question through the graph and return the final answer.

        This is the main entry point — equivalent to Agent.chat().
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")

        turn_num = len(self.conversation_history) + 1

        # Build prior context from conversation history (last 6 turns)
        prior_history = []
        for turn in self.conversation_history[-6:]:
            prior_history.append({"role": "user", "content": turn.question})
            prior_history.append({"role": "assistant", "content": turn.final_answer})

        # Initialize state
        messages = list(prior_history)
        messages.append({"role": "user", "content": question})

        state: AgentState = {
            "question": question,
            "system_prompt": self.system_prompt,
            "messages": messages,
            "prior_history": prior_history,
            "llm_response": None,
            "tool_calls": [],
            "iteration": 0,
            "max_iterations": self.max_iterations,
            "final_answer": "",
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
        }

        # Run the graph loop
        state = self._run_graph(state)

        # Record turn
        turn = Turn(
            turn_number=turn_num,
            question=question,
            tools_invoked=[],  # could extract from state if needed
            tool_results=[],
            final_answer=state["final_answer"],
            input_tokens=state["input_tokens"],
            output_tokens=state["output_tokens"],
            cost_usd=state["cost_usd"],
            timestamp=datetime.now().isoformat(),
        )
        self.conversation_history.append(turn)

        logger.info(
            "Turn %d: %s... | Cost: $%.4f | Iterations: %d",
            turn.turn_number, question[:50], turn.cost_usd, state["iteration"],
        )

        return state["final_answer"]

    def _run_graph(self, state: AgentState) -> AgentState:
        """
        Execute the state graph loop.

        Equivalent to LLMClient.run_agentic_loop() but with explicit
        state transitions:

            call_llm → should_continue?
                → "call_tools" → call_tools → call_llm (loop)
                → "end" → force_summary → done
        """
        while True:
            # Node: call_llm
            state = call_llm(state, self.llm_client, self.db)

            # Edge: should_continue?
            decision = should_continue(state)

            if decision == "call_tools":
                # Node: call_tools
                state = call_tools(state, self.db)
                # Loop back to call_llm
                continue
            else:
                # Node: end — extract final answer
                state = force_summary(state, self.llm_client)
                break

        return state

    def chat(self, question: str) -> str:
        """Alias for invoke() — matches Agent.chat() interface."""
        return self.invoke(question)

    def reset_history(self) -> None:
        """Clear conversation history for a fresh session."""
        self.conversation_history = []

    def get_turn_summary(self) -> str:
        """Return human-readable summary of all turns."""
        if not self.conversation_history:
            return "(No conversation history)"

        lines = []
        total_cost = 0.0
        total_in = 0
        total_out = 0

        for turn in self.conversation_history:
            lines.append(f"Turn {turn.turn_number}: \"{turn.question}\"")
            lines.append(f"  Answer: {turn.final_answer[:100]}...")
            lines.append(
                f"  Cost: ${turn.cost_usd:.6f} | "
                f"Tokens: {turn.input_tokens} in, {turn.output_tokens} out"
            )
            total_cost += turn.cost_usd
            total_in += turn.input_tokens
            total_out += turn.output_tokens

        lines.append("---")
        lines.append(f"Total cost: ${total_cost:.6f} USD")
        lines.append(f"Total tokens: {total_in} in, {total_out} out")
        return "\n".join(lines)