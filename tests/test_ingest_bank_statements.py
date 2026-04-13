"""
Tests for ingest_bank_statements.py (dual-write version)

Verifies:
  - ensure_tables (idempotent)
  - _coerce type conversion for Supabase
  - _sqlite_savepoint rollback on failure
  - upsert_bank_statement: insert, duplicate detection
  - upsert_bank_transactions_bulk: insert + statement_id injection
  - upsert_loan_transactions_bulk: insert + statement_id injection
  - Savepoint rollback: SQLite rolled back when Supabase fails
  - FK integrity across tables
  - Full pipeline: 2 statements with mixed data
  - Aggregate queries (same ones the agent tools would run)

Uses in-memory SQLite, supa=None (sqlite-only mode). No Supabase needed.
"""

from __future__ import annotations

import sqlite3
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.migrate_bank_tables import SQLITE_DDL
from src.database.ingest_bank_statements import (
    ensure_tables,
    _coerce,
    _sqlite_savepoint,
    _sqlite_upsert,
    _sqlite_insert,
    upsert_bank_statement,
    upsert_bank_transactions_bulk,
    upsert_loan_transactions_bulk,
    _NUMERIC_COLS,
    _INTEGER_COLS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SQLITE_DDL)
    conn.commit()
    return conn


def sample_metadata(suffix: str = "1") -> dict:
    return {
        "filename": f"stmt_{suffix}.pdf",
        "original_path": f"/data/stmt_{suffix}.pdf",
        "file_hash": f"hash_{suffix}",
        "bank_name": "dcu",
        "statement_period_start": "2025-01-01",
        "statement_period_end": "2025-01-31",
        "total_transactions": 5,
        "total_debits": 230.40,
        "total_credits": 2400.00,
    }


def sample_bank_rows() -> list[dict]:
    return [
        {"bank_name": "dcu", "account_type": "checking", "date": "2025-01-05",
         "description": "STOP & SHOP", "amount": 85.40,
         "transaction_type": "DEBIT", "balance": 1200.00,
         "raw_text": "JAN05 STOP & SHOP 85.40 1,200.00"},
        {"bank_name": "dcu", "account_type": "checking", "date": "2025-01-10",
         "description": "PAYROLL DIRECT DEP", "amount": 2400.00,
         "transaction_type": "CREDIT", "balance": 3600.00,
         "raw_text": "JAN10 PAYROLL 2,400.00 3,600.00"},
        {"bank_name": "dcu", "account_type": "savings", "date": "2025-01-02",
         "description": "TRANSFER FROM CHECKING", "amount": 500.00,
         "transaction_type": "CREDIT", "balance": 5500.00,
         "raw_text": "JAN02 TRANSFER 500.00 5,500.00"},
    ]


def sample_loan_rows() -> list[dict]:
    return [
        {"bank_name": "dcu", "loan_identifier": "NEW VEHICLE LOAN# 142",
         "date": "2025-01-15", "description": "LOAN PAYMENT",
         "payment_amount": 450.00, "principal_amount": 320.00,
         "interest_amount": 130.00, "balance": 18500.00,
         "raw_text": "JAN15 PAYMENT 450.00 -320.00 18,500.00"},
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEnsureTables:
    def test_creates_tables(self):
        conn = sqlite3.connect(":memory:")
        ensure_tables(conn)
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        assert "bank_statements" in tables
        assert "bank_transactions_raw" in tables
        assert "loan_transactions" in tables

    def test_idempotent(self):
        conn = sqlite3.connect(":memory:")
        ensure_tables(conn)
        ensure_tables(conn)
        assert conn.execute("SELECT COUNT(*) FROM bank_statements").fetchone()[0] == 0


class TestCoerce:
    def test_numeric_cols_to_float(self):
        row = {"amount": "85.40", "balance": "1200", "description": "TEST"}
        out = _coerce(row)
        assert isinstance(out["amount"], float)
        assert isinstance(out["balance"], float)
        assert isinstance(out["description"], str)

    def test_integer_cols_to_int(self):
        row = {"id": "1", "statement_id": "5", "filename": "test.pdf"}
        out = _coerce(row)
        assert isinstance(out["id"], int)
        assert isinstance(out["statement_id"], int)

    def test_none_preserved(self):
        row = {"amount": None, "id": None, "description": None}
        out = _coerce(row)
        assert out["amount"] is None
        assert out["id"] is None
        assert out["description"] is None

    def test_loan_numeric_cols(self):
        row = {"payment_amount": "450.00", "principal_amount": "320",
               "interest_amount": "130.0"}
        out = _coerce(row)
        assert isinstance(out["payment_amount"], float)
        assert isinstance(out["principal_amount"], float)
        assert isinstance(out["interest_amount"], float)


class TestSavepointRollback:
    def test_successful_savepoint(self):
        conn = make_conn()
        with _sqlite_savepoint(conn):
            _sqlite_insert(conn, "bank_statements", sample_metadata())
        assert conn.execute("SELECT COUNT(*) FROM bank_statements").fetchone()[0] == 1

    def test_rollback_on_exception(self):
        conn = make_conn()
        try:
            with _sqlite_savepoint(conn):
                _sqlite_insert(conn, "bank_statements", sample_metadata())
                raise RuntimeError("Simulated Supabase failure")
        except RuntimeError:
            pass
        # SQLite should be rolled back
        assert conn.execute("SELECT COUNT(*) FROM bank_statements").fetchone()[0] == 0

    def test_rollback_preserves_earlier_data(self):
        conn = make_conn()
        # First insert succeeds
        with _sqlite_savepoint(conn):
            _sqlite_insert(conn, "bank_statements", sample_metadata("1"))
        # Second insert fails
        try:
            with _sqlite_savepoint(conn):
                _sqlite_insert(conn, "bank_statements", sample_metadata("2"))
                raise RuntimeError("Simulated failure")
        except RuntimeError:
            pass
        # Only first row should exist
        count = conn.execute("SELECT COUNT(*) FROM bank_statements").fetchone()[0]
        assert count == 1
        row = conn.execute("SELECT file_hash FROM bank_statements").fetchone()
        assert row["file_hash"] == "hash_1"


class TestUpsertBankStatement:
    def test_returns_id(self):
        conn = make_conn()
        stmt_id = upsert_bank_statement(conn, None, sample_metadata(), sqlite_only=True)
        assert stmt_id is not None
        assert stmt_id >= 1

    def test_duplicate_returns_none(self):
        conn = make_conn()
        upsert_bank_statement(conn, None, sample_metadata(), sqlite_only=True)
        result = upsert_bank_statement(conn, None, sample_metadata(), sqlite_only=True)
        assert result is None

    def test_different_hashes_both_succeed(self):
        conn = make_conn()
        id1 = upsert_bank_statement(conn, None, sample_metadata("a"), sqlite_only=True)
        id2 = upsert_bank_statement(conn, None, sample_metadata("b"), sqlite_only=True)
        assert id1 is not None and id2 is not None
        assert id1 != id2

    def test_data_persisted(self):
        conn = make_conn()
        upsert_bank_statement(conn, None, sample_metadata(), sqlite_only=True)
        row = conn.execute("SELECT * FROM bank_statements").fetchone()
        assert row["bank_name"] == "dcu"
        assert row["total_debits"] == 230.40


class TestUpsertBankTransactionsBulk:
    def test_inserts_all_rows(self):
        conn = make_conn()
        upsert_bank_statement(conn, None, sample_metadata(), sqlite_only=True)
        count = upsert_bank_transactions_bulk(
            conn, None, 1, sample_bank_rows(), sqlite_only=True,
        )
        assert count == 3
        total = conn.execute("SELECT COUNT(*) FROM bank_transactions_raw").fetchone()[0]
        assert total == 3

    def test_statement_id_injected(self):
        conn = make_conn()
        upsert_bank_statement(conn, None, sample_metadata(), sqlite_only=True)
        upsert_bank_transactions_bulk(
            conn, None, 1, sample_bank_rows(), sqlite_only=True,
        )
        rows = conn.execute("SELECT statement_id FROM bank_transactions_raw").fetchall()
        assert all(r["statement_id"] == 1 for r in rows)

    def test_account_types(self):
        conn = make_conn()
        upsert_bank_statement(conn, None, sample_metadata(), sqlite_only=True)
        upsert_bank_transactions_bulk(
            conn, None, 1, sample_bank_rows(), sqlite_only=True,
        )
        checking = conn.execute(
            "SELECT COUNT(*) FROM bank_transactions_raw WHERE account_type='checking'"
        ).fetchone()[0]
        savings = conn.execute(
            "SELECT COUNT(*) FROM bank_transactions_raw WHERE account_type='savings'"
        ).fetchone()[0]
        assert checking == 2
        assert savings == 1

    def test_empty_rows(self):
        conn = make_conn()
        count = upsert_bank_transactions_bulk(conn, None, 1, [], sqlite_only=True)
        assert count == 0


class TestUpsertLoanTransactionsBulk:
    def test_inserts_row(self):
        conn = make_conn()
        upsert_bank_statement(conn, None, sample_metadata(), sqlite_only=True)
        count = upsert_loan_transactions_bulk(
            conn, None, 1, sample_loan_rows(), sqlite_only=True,
        )
        assert count == 1

    def test_data_correct(self):
        conn = make_conn()
        upsert_bank_statement(conn, None, sample_metadata(), sqlite_only=True)
        upsert_loan_transactions_bulk(
            conn, None, 1, sample_loan_rows(), sqlite_only=True,
        )
        row = conn.execute("SELECT * FROM loan_transactions").fetchone()
        assert row["payment_amount"] == 450.00
        assert row["principal_amount"] == 320.00
        assert row["interest_amount"] == 130.00
        assert row["loan_identifier"] == "NEW VEHICLE LOAN# 142"
        assert row["statement_id"] == 1

    def test_empty_rows(self):
        conn = make_conn()
        count = upsert_loan_transactions_bulk(conn, None, 1, [], sqlite_only=True)
        assert count == 0


class TestFKIntegrity:
    def test_bank_txn_joins_statement(self):
        conn = make_conn()
        upsert_bank_statement(conn, None, sample_metadata(), sqlite_only=True)
        upsert_bank_transactions_bulk(conn, None, 1, sample_bank_rows(), sqlite_only=True)
        row = conn.execute("""
            SELECT s.bank_name, COUNT(*) as cnt
            FROM bank_transactions_raw t
            JOIN bank_statements s ON t.statement_id = s.id
            GROUP BY s.bank_name
        """).fetchone()
        assert row["bank_name"] == "dcu" and row["cnt"] == 3

    def test_loan_txn_joins_statement(self):
        conn = make_conn()
        upsert_bank_statement(conn, None, sample_metadata(), sqlite_only=True)
        upsert_loan_transactions_bulk(conn, None, 1, sample_loan_rows(), sqlite_only=True)
        row = conn.execute("""
            SELECT s.bank_name, COUNT(*) as cnt
            FROM loan_transactions t
            JOIN bank_statements s ON t.statement_id = s.id
            GROUP BY s.bank_name
        """).fetchone()
        assert row["bank_name"] == "dcu" and row["cnt"] == 1


class TestFullPipeline:
    def test_two_statements(self):
        conn = make_conn()
        # Statement 1
        id1 = upsert_bank_statement(conn, None, sample_metadata("jan"), sqlite_only=True)
        upsert_bank_transactions_bulk(conn, None, id1, sample_bank_rows(), sqlite_only=True)
        upsert_loan_transactions_bulk(conn, None, id1, sample_loan_rows(), sqlite_only=True)
        # Statement 2
        meta2 = sample_metadata("feb")
        meta2["statement_period_start"] = "2025-02-01"
        meta2["statement_period_end"] = "2025-02-28"
        id2 = upsert_bank_statement(conn, None, meta2, sqlite_only=True)
        upsert_bank_transactions_bulk(conn, None, id2, [
            {"bank_name": "dcu", "account_type": "checking", "date": "2025-02-05",
             "description": "WHOLE FOODS", "amount": 92.30,
             "transaction_type": "DEBIT", "balance": 1107.70,
             "raw_text": "FEB05 WHOLE FOODS"},
        ], sqlite_only=True)

        assert conn.execute("SELECT COUNT(*) FROM bank_statements").fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM bank_transactions_raw").fetchone()[0] == 4
        assert conn.execute("SELECT COUNT(*) FROM loan_transactions").fetchone()[0] == 1

    def test_duplicate_skipped(self):
        conn = make_conn()
        upsert_bank_statement(conn, None, sample_metadata("jan"), sqlite_only=True)
        upsert_bank_transactions_bulk(conn, None, 1, sample_bank_rows(), sqlite_only=True)

        dup = upsert_bank_statement(conn, None, sample_metadata("jan"), sqlite_only=True)
        assert dup is None
        assert conn.execute("SELECT COUNT(*) FROM bank_transactions_raw").fetchone()[0] == 3


class TestSupabaseRollback:
    """Verify SQLite rollback when Supabase mock raises."""

    def test_statement_rolled_back_on_supa_failure(self):
        conn = make_conn()
        mock_supa = MagicMock()
        mock_supa.table.return_value.upsert.return_value.execute.side_effect = (
            RuntimeError("Supabase down")
        )

        try:
            upsert_bank_statement(conn, mock_supa, sample_metadata(), sqlite_only=False)
        except RuntimeError:
            pass

        count = conn.execute("SELECT COUNT(*) FROM bank_statements").fetchone()[0]
        assert count == 0, "SQLite should be rolled back after Supabase failure"

    def test_bank_txns_rolled_back_on_supa_failure(self):
        conn = make_conn()
        # Insert statement successfully (sqlite-only)
        upsert_bank_statement(conn, None, sample_metadata(), sqlite_only=True)

        mock_supa = MagicMock()
        mock_supa.table.return_value.upsert.return_value.execute.side_effect = (
            RuntimeError("Supabase down")
        )

        try:
            upsert_bank_transactions_bulk(
                conn, mock_supa, 1, sample_bank_rows(), sqlite_only=False,
            )
        except RuntimeError:
            pass

        count = conn.execute("SELECT COUNT(*) FROM bank_transactions_raw").fetchone()[0]
        assert count == 0, "Bank txns should be rolled back after Supabase failure"

    def test_loan_txns_rolled_back_on_supa_failure(self):
        conn = make_conn()
        upsert_bank_statement(conn, None, sample_metadata(), sqlite_only=True)

        mock_supa = MagicMock()
        mock_supa.table.return_value.upsert.return_value.execute.side_effect = (
            RuntimeError("Supabase down")
        )

        try:
            upsert_loan_transactions_bulk(
                conn, mock_supa, 1, sample_loan_rows(), sqlite_only=False,
            )
        except RuntimeError:
            pass

        count = conn.execute("SELECT COUNT(*) FROM loan_transactions").fetchone()[0]
        assert count == 0, "Loan txns should be rolled back after Supabase failure"


class TestAggregateQueries:
    """Verify queries that agent tools would run against ingested data."""

    def _setup(self):
        conn = make_conn()
        upsert_bank_statement(conn, None, sample_metadata(), sqlite_only=True)
        upsert_bank_transactions_bulk(conn, None, 1, sample_bank_rows(), sqlite_only=True)
        upsert_loan_transactions_bulk(conn, None, 1, sample_loan_rows(), sqlite_only=True)
        return conn

    def test_spending_by_account_type(self):
        conn = self._setup()
        rows = conn.execute("""
            SELECT account_type,
                   SUM(CASE WHEN transaction_type='DEBIT' THEN amount ELSE 0 END) as spent,
                   SUM(CASE WHEN transaction_type='CREDIT' THEN amount ELSE 0 END) as received,
                   COUNT(*) as cnt
            FROM bank_transactions_raw GROUP BY account_type ORDER BY account_type
        """).fetchall()
        checking = dict(rows[0])
        assert checking["cnt"] == 2
        assert abs(checking["spent"] - 85.40) < 0.01
        assert abs(checking["received"] - 2400.00) < 0.01

    def test_loan_payment_totals(self):
        conn = self._setup()
        row = conn.execute("""
            SELECT SUM(payment_amount) as tp,
                   SUM(principal_amount) as tpr,
                   SUM(interest_amount) as ti
            FROM loan_transactions
        """).fetchone()
        assert row["tp"] == 450.00
        assert row["tpr"] == 320.00
        assert row["ti"] == 130.00

    def test_monthly_spending(self):
        conn = self._setup()
        row = conn.execute("""
            SELECT strftime('%Y-%m', date) as month,
                   SUM(CASE WHEN transaction_type='DEBIT' THEN amount ELSE 0 END) as spent
            FROM bank_transactions_raw GROUP BY month
        """).fetchone()
        assert row["month"] == "2025-01"
        assert abs(row["spent"] - 85.40) < 0.01