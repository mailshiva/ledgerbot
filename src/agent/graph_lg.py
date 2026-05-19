"""
src/agent/graph_lg.py

LangGraph-based agent for financial data queries.

Uses the real LangGraph StateGraph (langgraph>=1.0) to replace the manual
while-loop in the previous graph.py. The LLM provider is still your custom
LLMClient — this is NOT wired to LangChain's BaseChatModel, so LangGraph's
ToolNode/create_react_agent are intentionally not used. What LangGraph
provides here is:

  - StateGraph  — declared nodes + edges instead of while True
  - add_conditional_edges  — explicit routing logic
  - compile()   — validates the graph and returns a runnable
  - invoke()    — single entry point that runs the full graph

Graph structure:
    ┌──────────┐
    │  START   │
    └────┬─────┘
         │
    ┌────▼─────┐
    │ call_llm │ ◄──────────────────────┐
    └────┬─────┘                        │
         │                              │
    ┌────▼──────────┐    ┌──────────┐   │
    │should_continue├───►│call_tools├───┘
    └────┬──────────┘    └──────────┘
         │ (no tools / max iterations)
    ┌────▼──────────┐
    │ force_summary │
    └────┬──────────┘
         │
    ┌────▼─────┐
    │   END    │
    └──────────┘

Install:
    pip install langgraph

Usage:
    from src.agent.graph_lg import GraphAgent

    agent = GraphAgent(llm_client=client, db=db)
    answer = agent.invoke("How much did I spend on groceries?")

    # Multi-turn — history is carried automatically
    answer2 = agent.invoke("What about last month?")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

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
    """
    State passed between graph nodes.

    LangGraph merges return values from each node into this dict.
    Every node returns only the keys it modifies — unchanged keys are
    preserved automatically by LangGraph's state reducer.
    """
    # Input
    question: str
    system_prompt: str

    # Conversation
    messages: list[dict]        # OpenAI-style role/content dicts
    prior_history: list[dict]   # prior turns injected at invoke() time

    # LLM interaction
    llm_response: dict | None   # serialised LLMResponse fields
    tool_calls: list[dict]      # parsed tool calls from current LLM response

    # Control
    iteration: int
    max_iterations: int

    # Accumulated output
    final_answer: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


# ---------------------------------------------------------------------------
# Turn record (for conversation_history on GraphAgent)
# ---------------------------------------------------------------------------

@dataclass
class Turn:
    """Single conversational turn recorded after each invoke()."""
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
# Node functions
# ---------------------------------------------------------------------------

def call_llm(state: AgentState, llm_client: LLMClient) -> dict:
    """
    Node: call the LLM with the current message history and all tool
    definitions. Parses any tool calls from the response.

    Returns only the state keys this node changes.
    """
    prompt = _history_to_prompt(state["messages"])

    response = llm_client.complete_with_tools(
        prompt,
        ALL_TOOL_DEFINITIONS,
        system=state["system_prompt"],
    )

    tool_calls = llm_client.parse_tool_calls(response)

    logger.debug(
        "call_llm: iteration=%d, tool_calls=%d",
        state["iteration"] + 1,
        len(tool_calls),
    )

    return {
        "llm_response": {
            "text": response.text,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "cost_usd": response.cost_usd,
        },
        "tool_calls": tool_calls,
        "input_tokens": state["input_tokens"] + response.input_tokens,
        "output_tokens": state["output_tokens"] + response.output_tokens,
        "cost_usd": state["cost_usd"] + response.cost_usd,
        "iteration": state["iteration"] + 1,
    }


def call_tools(state: AgentState, db) -> dict:
    """
    Node: execute each tool call and append results to the message history.

    Returns only the state keys this node changes.
    """
    tool_calls = state["tool_calls"]

    # Append the assistant's tool-call turn to history
    messages = list(state["messages"])
    messages.append({
        "role": "assistant",
        "content": state["llm_response"]["text"],
    })

    # Execute tools
    tool_results = []
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
        logger.debug("Tool %s result: %s", tool_name, str(result)[:1000])

    # Append all tool results as a single tool message
    messages.append({
        "role": "tool",
        "content": json.dumps(tool_results, default=str),
    })

    return {
        "messages": messages,
        "tool_calls": [],   # cleared — LLM will repopulate on next call_llm
    }


def force_summary(state: AgentState, llm_client: LLMClient) -> dict:
    """
    Node: extract the final answer from state.

    If max_iterations was hit while tool calls were still pending, ask the
    LLM to synthesise what it has collected so far. Otherwise just surface
    the last LLM response text.
    """
    hit_limit = (
        state["iteration"] >= state["max_iterations"]
        and bool(state.get("tool_calls"))
    )

    if hit_limit:
        logger.warning(
            "Max iterations (%d) reached — forcing summary",
            state["max_iterations"],
        )
        messages = list(state["messages"])
        messages.append({
            "role": "user",
            "content": (
                "You have reached the maximum number of tool calls. "
                "Please summarise the data collected so far and answer "
                "the original question as best you can."
            ),
        })
        prompt = _history_to_prompt(messages)
        response = llm_client.complete(prompt, system=state["system_prompt"])
        return {
            "final_answer": response.text,
            "input_tokens": state["input_tokens"] + response.input_tokens,
            "output_tokens": state["output_tokens"] + response.output_tokens,
            "cost_usd": state["cost_usd"] + response.cost_usd,
        }

    return {"final_answer": state["llm_response"]["text"]}


# ---------------------------------------------------------------------------
# Conditional edge
# ---------------------------------------------------------------------------

def should_continue(state: AgentState) -> Literal["call_tools", "force_summary"]:
    """
    Edge function: route to call_tools if the LLM requested tools and we
    haven't hit the iteration ceiling; otherwise route to force_summary.
    """
    if state["tool_calls"] and state["iteration"] < state["max_iterations"]:
        return "call_tools"
    return "force_summary"


# ---------------------------------------------------------------------------
# Prompt helper
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
# Graph factory
# ---------------------------------------------------------------------------

def _build_graph(llm_client: LLMClient, db) -> Any:
    """
    Declare and compile the StateGraph.

    Node closures capture llm_client and db so the node functions stay
    pure (state-in → dict-out) with no global side effects.
    """
    # Wrap nodes as closures so they only receive `state`
    def _call_llm(state: AgentState) -> dict:
        return call_llm(state, llm_client)

    def _call_tools(state: AgentState) -> dict:
        return call_tools(state, db)

    def _force_summary(state: AgentState) -> dict:
        return force_summary(state, llm_client)

    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("call_llm", _call_llm)
    graph.add_node("call_tools", _call_tools)
    graph.add_node("force_summary", _force_summary)

    # Edges
    graph.add_edge(START, "call_llm")
    graph.add_conditional_edges(
        "call_llm",
        should_continue,
        {
            "call_tools": "call_tools",
            "force_summary": "force_summary",
        },
    )
    graph.add_edge("call_tools", "call_llm")   # loop back
    graph.add_edge("force_summary", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# GraphAgent — public interface (drop-in for old GraphAgent)
# ---------------------------------------------------------------------------

class GraphAgent:
    """
    Financial data agent backed by a real LangGraph StateGraph.

    Identical external interface to the previous hand-rolled GraphAgent:
      - chat(question) → str
      - invoke(question) → str
      - conversation_history: list[Turn]
      - reset_history()
      - get_turn_summary() → str

    Internally, _run_graph() is now graph.invoke(state) — LangGraph owns
    the node execution loop.

    Parameters
    ----------
    llm_client : LLMClient
        Your custom LLM wrapper (Gemini / Anthropic / Ollama).
    db : DatabaseManager | DualWriteManager
        Database handle passed to tool execution.
    max_iterations : int
        Maximum tool-call cycles per question. Default 25.
    system_prompt : str | None
        Override the default AGENT_SYSTEM_PROMPT.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        db,
        max_iterations: int = 25,
        system_prompt: str | None = None,
    ):
        self.llm_client = llm_client
        self.db = db
        self.max_iterations = max_iterations
        self.system_prompt = system_prompt or AGENT_SYSTEM_PROMPT
        self.conversation_history: list[Turn] = []

        # Compile the graph once; reused across all invoke() calls
        self._graph = _build_graph(llm_client, db)
        logger.info("GraphAgent initialised (max_iterations=%d)", max_iterations)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def invoke(self, question: str) -> str:
        """
        Run the question through the LangGraph and return the final answer.

        Multi-turn context: the last 6 turns from conversation_history are
        prepended to the message list before each invoke.
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")

        turn_num = len(self.conversation_history) + 1

        # Build prior context from conversation history (last 6 turns)
        prior_history: list[dict] = []
        for turn in self.conversation_history[-6:]:
            prior_history.append({"role": "user", "content": turn.question})
            prior_history.append({"role": "assistant", "content": turn.final_answer})

        messages = list(prior_history)
        messages.append({"role": "user", "content": question})

        # Initial state handed to LangGraph
        initial_state: AgentState = {
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

        # LangGraph drives the loop
        final_state: AgentState = self._graph.invoke(initial_state)

        # Record turn for multi-turn context and session summary
        turn = Turn(
            turn_number=turn_num,
            question=question,
            tools_invoked=[],       # could enrich from final_state["messages"]
            tool_results=[],
            final_answer=final_state["final_answer"],
            input_tokens=final_state["input_tokens"],
            output_tokens=final_state["output_tokens"],
            cost_usd=final_state["cost_usd"],
            timestamp=datetime.now().isoformat(),
        )
        self.conversation_history.append(turn)

        logger.info(
            "Turn %d complete | cost=$%.4f | iterations=%d",
            turn_num, turn.cost_usd, final_state["iteration"],
        )

        return final_state["final_answer"]

    def chat(self, question: str) -> str:
        """Alias for invoke() — matches the original Agent.chat() interface."""
        return self.invoke(question)

    def reset_history(self) -> None:
        """Clear conversation history to start a fresh session."""
        self.conversation_history = []
        logger.debug("Conversation history cleared")

    def get_turn_summary(self) -> str:
        """Return a human-readable summary of all turns in this session."""
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
        lines.append(f"Total cost:   ${total_cost:.6f} USD")
        lines.append(f"Total tokens: {total_in:,} in, {total_out:,} out")
        return "\n".join(lines)
