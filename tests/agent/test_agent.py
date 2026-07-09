"""
Tests for src/agent/agent.py

Fully offline — uses mocked LLMClient and in-memory SQLite database.
No real API calls, no network.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pytest

from src.agent.agent import Agent, Turn
from src.llm.client import LLMResponse, Model, LLMClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_db() -> MagicMock:
    """Build an in-memory SQLite DB and return a mock DatabaseManager."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row

    conn.executescript("""
        CREATE TABLE statements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            file_hash TEXT NOT NULL UNIQUE,
            bank_name TEXT,
            statement_date TEXT
        );
        CREATE TABLE transactions_raw (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            statement_id INTEGER,
            date TEXT NOT NULL,
            description TEXT NOT NULL,
            amount REAL NOT NULL,
            category TEXT DEFAULT 'Uncategorized'
        );
    """)

    conn.execute("""
        INSERT INTO statements (filename, file_hash, bank_name, statement_date)
        VALUES ('test.pdf', 'hash1', 'capital_one', '2025-01-31')
    """)

    conn.executemany("""
        INSERT INTO transactions_raw (statement_id, date, description, amount, category)
        VALUES (?, ?, ?, ?, ?)
    """, [
        (1, '2025-01-05', 'STARBUCKS', 8.50, 'Food & Dining'),
        (1, '2025-01-10', 'AMAZON', 55.99, 'Shopping'),
        (1, '2025-01-15', 'PAYCHECK', 1200.0, 'Income'),
    ])

    conn.commit()

    mock_db = MagicMock()
    mock_db.conn = conn
    return mock_db


@pytest.fixture
def mock_db():
    """In-memory DatabaseManager mock."""
    return _make_db()


@pytest.fixture
def mock_llm_client():
    """Mocked LLMClient."""
    client = MagicMock(spec=LLMClient)
    return client


@pytest.fixture
def agent(mock_db, mock_llm_client):
    """Agent instance with mocked dependencies."""
    return Agent(
        db=mock_db,
        llm_client=mock_llm_client,
        max_turns=5,
        log_dir=None,
    )


@pytest.fixture
def agent_with_logging(mock_db, mock_llm_client):
    """Agent instance with logging enabled."""
    with tempfile.TemporaryDirectory() as tmpdir:
        agent = Agent(
            db=mock_db,
            llm_client=mock_llm_client,
            max_turns=5,
            log_dir=tmpdir,
        )
        yield agent


# ---------------------------------------------------------------------------
# Agent initialization
# ---------------------------------------------------------------------------

class TestAgentInit:
    def test_init_basic(self, mock_db, mock_llm_client):
        agent = Agent(db=mock_db, llm_client=mock_llm_client)
        assert agent.db is mock_db
        assert agent.llm_client is mock_llm_client
        assert agent.max_turns == 5
        assert agent.conversation_history == []

    def test_init_with_custom_max_turns(self, mock_db, mock_llm_client):
        agent = Agent(db=mock_db, llm_client=mock_llm_client, max_turns=10)
        assert agent.max_turns == 10

    def test_init_creates_log_dir(self, mock_db, mock_llm_client):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "agent_logs"
            agent = Agent(db=mock_db, llm_client=mock_llm_client, log_dir=str(log_path))
            assert agent.log_dir.exists()

    def test_init_log_dir_none(self, mock_db, mock_llm_client):
        agent = Agent(db=mock_db, llm_client=mock_llm_client, log_dir=None)
        assert agent.log_dir is None


# ---------------------------------------------------------------------------
# chat() method
# ---------------------------------------------------------------------------

class TestAgentChat:
    def test_chat_single_turn(self, agent, mock_llm_client):
        """Test a single chat call."""
        mock_llm_client.run_agentic_loop.return_value = LLMResponse(
            text="You spent $450 this month on food.",
            model=Model.GEMINI_FLASH,
            input_tokens=100,
            output_tokens=25,
            cost_usd=0.001,
        )

        answer = agent.chat("How much did I spend on food?")

        assert answer == "You spent $450 this month on food."
        assert len(agent.conversation_history) == 1
        assert agent.conversation_history[0].turn_number == 1

    def test_chat_multi_turn(self, agent, mock_llm_client):
        """Test multiple sequential chat calls."""
        mock_llm_client.run_agentic_loop.side_effect = [
            LLMResponse(
                text="You spent $450 on food.",
                model=Model.GEMINI_FLASH,
                input_tokens=100,
                output_tokens=20,
                cost_usd=0.001,
            ),
            LLMResponse(
                text="Your top merchant is Amazon ($55.99).",
                model=Model.GEMINI_FLASH,
                input_tokens=110,
                output_tokens=25,
                cost_usd=0.0012,
            ),
        ]

        answer1 = agent.chat("How much on food?")
        answer2 = agent.chat("Top merchant?")

        assert len(agent.conversation_history) == 2
        assert agent.conversation_history[0].turn_number == 1
        assert agent.conversation_history[1].turn_number == 2
        assert answer1 == "You spent $450 on food."
        assert answer2 == "Your top merchant is Amazon ($55.99)."

    def test_chat_empty_question_raises(self, agent):
        """Test that empty question raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            agent.chat("")

    def test_chat_whitespace_only_raises(self, agent):
        """Test that whitespace-only question raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            agent.chat("   ")

    def test_chat_llm_failure_raises_runtime_error(self, agent, mock_llm_client):
        """Test that LLM exceptions are re-raised as RuntimeError."""
        mock_llm_client.run_agentic_loop.side_effect = ValueError("API error")

        with pytest.raises(RuntimeError, match="Failed to process question"):
            agent.chat("How much did I spend?")

    def test_chat_records_turn_details(self, agent, mock_llm_client):
        """Test that Turn objects are properly recorded."""
        mock_llm_client.run_agentic_loop.return_value = LLMResponse(
            text="Answer",
            model=Model.GEMINI_FLASH,
            input_tokens=50,
            output_tokens=15,
            cost_usd=0.0005,
        )

        agent.chat("Test question?")

        turn = agent.conversation_history[0]
        assert turn.question == "Test question?"
        assert turn.final_answer == "Answer"
        assert turn.input_tokens == 50
        assert turn.output_tokens == 15
        assert turn.cost_usd == 0.0005
        assert turn.timestamp is not None

    def test_chat_passes_db_to_llm_client(self, agent, mock_db, mock_llm_client):
        """Test that agent passes db to llm_client.run_agentic_loop."""
        mock_llm_client.run_agentic_loop.return_value = LLMResponse(
            text="Answer",
            model=Model.GEMINI_FLASH,
            input_tokens=10,
            output_tokens=5,
            cost_usd=0.0001,
        )

        agent.chat("Question?")

        # Verify db was passed
        mock_llm_client.run_agentic_loop.assert_called_once()
        call_kwargs = mock_llm_client.run_agentic_loop.call_args[1]
        assert call_kwargs["db"] is mock_db

    def test_chat_passes_max_iterations(self, agent, mock_llm_client):
        """Test that agent passes max_iterations to run_agentic_loop."""
        mock_llm_client.run_agentic_loop.return_value = LLMResponse(
            text="Answer",
            model=Model.GEMINI_FLASH,
            input_tokens=10,
            output_tokens=5,
            cost_usd=0.0001,
        )

        agent.chat("Question?")

        call_kwargs = mock_llm_client.run_agentic_loop.call_args[1]
        assert call_kwargs["max_iterations"] == 5


# ---------------------------------------------------------------------------
# reset_history()
# ---------------------------------------------------------------------------

class TestResetHistory:
    def test_reset_clears_history(self, agent, mock_llm_client):
        """Test that reset_history clears conversation_history."""
        mock_llm_client.run_agentic_loop.return_value = LLMResponse(
            text="Answer",
            model=Model.GEMINI_FLASH,
            input_tokens=10,
            output_tokens=5,
            cost_usd=0.0001,
        )

        agent.chat("Question 1?")
        agent.chat("Question 2?")
        assert len(agent.conversation_history) == 2

        agent.reset_history()
        assert len(agent.conversation_history) == 0

    def test_reset_allows_new_conversation(self, agent, mock_llm_client):
        """Test that reset enables starting a fresh conversation."""
        mock_llm_client.run_agentic_loop.return_value = LLMResponse(
            text="Answer",
            model=Model.GEMINI_FLASH,
            input_tokens=10,
            output_tokens=5,
            cost_usd=0.0001,
        )

        agent.chat("Q1")
        agent.reset_history()
        agent.chat("Q2")

        # After reset, only the new turn should exist
        assert len(agent.conversation_history) == 1
        assert agent.conversation_history[0].question == "Q2"
        assert agent.conversation_history[0].turn_number == 1


# ---------------------------------------------------------------------------
# get_turn_summary()
# ---------------------------------------------------------------------------

class TestGetTurnSummary:
    def test_summary_empty_history(self, agent):
        """Test summary with no turns."""
        summary = agent.get_turn_summary()
        assert "(No conversation history)" in summary

    def test_summary_single_turn(self, agent, mock_llm_client):
        """Test summary with one turn."""
        mock_llm_client.run_agentic_loop.return_value = LLMResponse(
            text="You spent $450.",
            model=Model.GEMINI_FLASH,
            input_tokens=100,
            output_tokens=20,
            cost_usd=0.001,
        )

        agent.chat("How much did I spend?")
        summary = agent.get_turn_summary()

        assert "Turn 1:" in summary
        assert "How much did I spend?" in summary
        assert "You spent $450." in summary
        assert "Total cost:" in summary

    def test_summary_multi_turn(self, agent, mock_llm_client):
        """Test summary with multiple turns."""
        mock_llm_client.run_agentic_loop.side_effect = [
            LLMResponse(
                text="$450",
                model=Model.GEMINI_FLASH,
                input_tokens=100,
                output_tokens=10,
                cost_usd=0.001,
            ),
            LLMResponse(
                text="Amazon $55.99",
                model=Model.GEMINI_FLASH,
                input_tokens=110,
                output_tokens=15,
                cost_usd=0.0012,
            ),
        ]

        agent.chat("Q1?")
        agent.chat("Q2?")
        summary = agent.get_turn_summary()

        assert "Turn 1:" in summary
        assert "Turn 2:" in summary
        assert "Total cost:" in summary
        # Total should be sum of both turns
        assert "0.0022" in summary

    def test_summary_format_is_readable(self, agent, mock_llm_client):
        """Test that summary is human-readable."""
        mock_llm_client.run_agentic_loop.return_value = LLMResponse(
            text="Test answer",
            model=Model.GEMINI_FLASH,
            input_tokens=50,
            output_tokens=10,
            cost_usd=0.0005,
        )

        agent.chat("Test?")
        summary = agent.get_turn_summary()

        # Should have multiple lines and be readable
        lines = summary.split("\n")
        assert len(lines) > 3
        assert any("Turn" in line for line in lines)
        assert any("Answer:" in line for line in lines)
        assert any("Cost:" in line for line in lines)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class TestLogging:
    def test_log_turn_writes_json(self, agent_with_logging, mock_llm_client):
        """Test that turns are logged to JSON files."""
        mock_llm_client.run_agentic_loop.return_value = LLMResponse(
            text="Answer",
            model=Model.GEMINI_FLASH,
            input_tokens=100,
            output_tokens=20,
            cost_usd=0.001,
        )

        agent_with_logging.chat("Question?")

        # Check that turn_0001.json was created
        log_file = agent_with_logging.log_dir / "turn_0001.json"
        assert log_file.exists()

        # Verify JSON content
        with open(log_file) as f:
            data = json.load(f)
        assert data["turn_number"] == 1
        assert data["question"] == "Question?"
        assert data["final_answer"] == "Answer"

    def test_log_multiple_turns(self, agent_with_logging, mock_llm_client):
        """Test logging multiple turns with correct numbering."""
        mock_llm_client.run_agentic_loop.return_value = LLMResponse(
            text="Answer",
            model=Model.GEMINI_FLASH,
            input_tokens=10,
            output_tokens=5,
            cost_usd=0.0001,
        )

        agent_with_logging.chat("Q1?")
        agent_with_logging.chat("Q2?")

        # Check both files exist with correct numbering
        log1 = agent_with_logging.log_dir / "turn_0001.json"
        log2 = agent_with_logging.log_dir / "turn_0002.json"

        assert log1.exists()
        assert log2.exists()

        with open(log1) as f:
            data1 = json.load(f)
        with open(log2) as f:
            data2 = json.load(f)

        assert data1["question"] == "Q1?"
        assert data2["question"] == "Q2?"

    def test_no_logging_when_log_dir_none(self, agent, mock_llm_client):
        """Test that no logging occurs when log_dir is None."""
        mock_llm_client.run_agentic_loop.return_value = LLMResponse(
            text="Answer",
            model=Model.GEMINI_FLASH,
            input_tokens=10,
            output_tokens=5,
            cost_usd=0.0001,
        )

        # Should not raise any file errors
        agent.chat("Question?")

        # log_dir should be None
        assert agent.log_dir is None


# ---------------------------------------------------------------------------
# Turn dataclass
# ---------------------------------------------------------------------------

class TestTurn:
    def test_turn_creation(self):
        """Test Turn object creation."""
        turn = Turn(
            turn_number=1,
            question="Test?",
            tools_invoked=["get_merchants"],
            tool_results=[{"merchant": "Amazon"}],
            final_answer="You spent $50",
            input_tokens=100,
            output_tokens=20,
            cost_usd=0.001,
            timestamp="2025-01-15T10:30:00",
        )

        assert turn.turn_number == 1
        assert turn.question == "Test?"
        assert turn.final_answer == "You spent $50"

    def test_turn_serializable(self):
        """Test that Turn can be serialized to JSON."""
        turn = Turn(
            turn_number=1,
            question="Test?",
            tools_invoked=[],
            tool_results=[],
            final_answer="Answer",
            input_tokens=10,
            output_tokens=5,
            cost_usd=0.0001,
            timestamp="2025-01-15T10:30:00",
        )

        # Should not raise
        json_str = json.dumps(turn.__dict__)
        assert "Test?" in json_str
        assert "Answer" in json_str
