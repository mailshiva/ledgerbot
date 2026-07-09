"""
Agent loop wrapper — maintains multi-turn conversation over a database.

Thin wrapper around LLMClient.run_agentic_loop() that:
- Tracks conversation history across multiple questions
- Logs each turn (question, tools, answer, cost)
- Provides a clean chat(question) -> str interface for the CLI
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.database.db_manager import DatabaseManager
from src.llm.client import LLMClient
#from src.llm.prompts import AGENT_SYSTEM_PROMPT
from src.agent.tools import TOOL_DEFINITIONS, execute_tool
from src.agent.bank_tools import BANK_TOOL_DEFINITIONS, execute_bank_tool

ALL_TOOLS = TOOL_DEFINITIONS + BANK_TOOL_DEFINITIONS


logger = logging.getLogger(__name__)


@dataclass
class Turn:
    """Single conversational turn (user question → agent answer)."""
    turn_number: int
    question: str
    tools_invoked: list[str]
    tool_results: list[dict]
    final_answer: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    timestamp: str  # ISO format


class Agent:
    """Multi-turn conversational agent over a financial database."""

    def __init__(
        self,
        db: DatabaseManager,
        llm_client: LLMClient,
        max_turns: int = 5,
        log_dir: Optional[str] = None,
    ):
        """
        Initialize the agent.

        Args:
            db: DatabaseManager instance (provides .conn for tools)
            llm_client: Configured LLMClient (handles all 4 providers)
            max_turns: Max iterations per question before force-exit
            log_dir: Optional directory for turn logs (JSON files)
        """
        self.db = db
        self.llm_client = llm_client
        self.max_turns = max_turns
        self.log_dir = Path(log_dir) if log_dir else None
        self.conversation_history: list[Turn] = []

        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)

    def chat(self, question: str) -> str:
        """
        Execute a question with proper tool calling and return final answer.

        Delegates to LLMClient.run_agentic_loop() which handles:
        1. Sending question + tools to LLM
        2. Executing tool calls and feeding results back
        3. Returning the final plain-text answer
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")

        try:
            turn_num = len(self.conversation_history) + 1

            # Build prior conversation context (last 6 turns max to cap token usage)
            prior_history = []
            for turn in self.conversation_history[-6:]:
                prior_history.append({"role": "user", "content": turn.question})
                prior_history.append({"role": "assistant", "content": turn.final_answer})

            response = self.llm_client.run_agentic_loop(
                question,
                db=self.db,
                max_iterations=self.max_turns,
                prior_history=prior_history or None,
            )

            final_answer = response.text

            turn = Turn(
                turn_number=turn_num,
                question=question,
                tools_invoked=[],
                tool_results=[],
                final_answer=final_answer,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                cost_usd=response.cost_usd,
                timestamp=datetime.now().isoformat(),
            )

            self.conversation_history.append(turn)

            if self.log_dir:
                self._log_turn(turn)

            logger.info(
                f"Turn {turn.turn_number}: {question[:50]}... "
                f"| Cost: ${turn.cost_usd:.4f}"
            )

            return final_answer

        except Exception as e:
            logger.error(f"Error in agent.chat(): {e}")
            raise RuntimeError(f"Failed to process question: {e}") from e


    def reset_history(self) -> None:
        """
        Clear conversation history.

        Use this to start a fresh conversation session without reinitializing
        the Agent instance.
        """
        self.conversation_history = []
        logger.info("Conversation history cleared")

    def get_turn_summary(self) -> str:
        """
        Return human-readable summary of all turns in the conversation.

        Returns:
            Multi-line string with turn details

        Example:
            Turn 1: "How much did I spend?" | Answer: "You spent $450."
            Turn 2: "What about food?" | Answer: "Food was $87.40."
            ---
            Total cost: $0.0035 USD
            Total tokens: 450 in, 120 out
        """
        if not self.conversation_history:
            return "(No conversation history)"

        lines = []
        total_cost = 0.0
        total_input_tokens = 0
        total_output_tokens = 0

        for turn in self.conversation_history:
            lines.append(f"Turn {turn.turn_number}: \"{turn.question}\"")
            lines.append(f"  Answer: {turn.final_answer[:100]}...")
            lines.append(f"  Cost: ${turn.cost_usd:.6f} | Tokens: {turn.input_tokens} in, {turn.output_tokens} out")

            total_cost += turn.cost_usd
            total_input_tokens += turn.input_tokens
            total_output_tokens += turn.output_tokens

        lines.append("---")
        lines.append(f"Total cost: ${total_cost:.6f} USD")
        lines.append(f"Total tokens: {total_input_tokens} in, {total_output_tokens} out")

        return "\n".join(lines)

    def _log_turn(self, turn: Turn) -> None:
        """
        Write turn details to a JSON log file in log_dir.

        File naming: turn_0001.json, turn_0002.json, etc.

        Args:
            turn: Turn object to log
        """
        if not self.log_dir:
            return

        filename = self.log_dir / f"turn_{turn.turn_number:04d}.json"
        try:
            with open(filename, "w") as f:
                json.dump(asdict(turn), f, indent=2)
            logger.debug(f"Logged turn to {filename}")
        except Exception as e:
            logger.warning(f"Failed to log turn to {filename}: {e}")
