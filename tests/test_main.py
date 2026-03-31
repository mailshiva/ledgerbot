# tests/cli/test_main.py

import pytest
from unittest.mock import MagicMock, patch, Mock
from pathlib import Path
import sys

from main import CLIAgent


# Fixtures
@pytest.fixture
def mock_db():
    """Mock DatabaseManager."""
    mock = MagicMock()
    mock.conn = None  # Not used by CLI
    return mock


@pytest.fixture
def mock_llm_client():
    """Mock LLMClient."""
    mock = MagicMock()
    return mock


@pytest.fixture
def mock_agent():
    """Mock Agent that returns deterministic Turn."""
    from collections import namedtuple
    Turn = namedtuple('Turn', ['question', 'answer', 'cost_usd', 'input_tokens', 'output_tokens'])

    mock = MagicMock()
    mock.chat = MagicMock(return_value="Mocked answer")
    mock.conversation_history = []

    # Simulate appending to history
    def mock_chat_side_effect(q):
        turn = Turn(
            question=q,
            answer="You spent $100 on food.",
            cost_usd=0.0001,
            input_tokens=100,
            output_tokens=50
        )
        mock.conversation_history.append(turn)
        return turn.answer

    mock.chat.side_effect = mock_chat_side_effect
    return mock


# Test classes
class TestCLIAgentInit:
    def test_init_default_provider(self, monkeypatch):
        """CLIAgent initializes with default provider (anthropic)."""
        # Patch LLMConfig, LLMClient, DatabaseManager, Agent
        # Assert self.provider == 'anthropic'

    def test_init_ollama_provider(self, monkeypatch):
        """CLIAgent initializes with Ollama provider."""
        # Similar, assert self.provider == 'ollama'


class TestCLIAgentChat:
    def test_chat_single_question(self, monkeypatch, mock_agent):
        """Single question → response + turn tracked."""
        # Mock Agent, call chat(), assert response printed, turn appended

    def test_chat_multi_turn(self, monkeypatch, mock_agent):
        """Multi-turn conversation accumulates turns."""
        # Ask 3 questions, assert len(self.turns) == 3


class TestInteractiveMode:
    def test_repl_loop(self, monkeypatch, mock_agent):
        """REPL loop processes questions until 'exit'."""
        # Mock input() to return ["How much?", "Top merchants?", "exit"]
        # Assert 2 turns processed, session summary printed

    def test_exit_command(self, monkeypatch, mock_agent):
        """'exit' command breaks loop."""
        # ...

    def test_interrupt_handling(self, monkeypatch, mock_agent):
        """Ctrl+C caught, prompts to exit."""
        # ...


class TestSessionSummary:
    def test_summary_metrics(self, monkeypatch, mock_agent):
        """Session summary shows correct turn count, cost, tokens, duration."""
        # Create 3 turns, call print_session_summary()
        # Assert output contains "Turns: 3", "Total cost: $X", etc.

# More tests for edge cases, piped input, errors, etc.