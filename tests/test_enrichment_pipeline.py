"""
tests/test_enrichment_pipeline.py
──────────────────────────────────
Tests for EnrichmentPipeline, EnrichedTransaction, and the DB helpers.

Each test that needs a pipeline gets its own isolated temp config.yaml
pointing to its own temp database – the same isolation pattern used
throughout the Week 1 test suite.

Run from project root (venv active):
    pytest tests/test_enrichment_pipeline.py -v
"""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from src.database.enriched_transaction import (
    EnrichedTransaction,
    apply_migration,
    bulk_upsert,
)
from src.nlp.enrichment_pipeline import EnrichmentPipeline, PipelineStats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_tmp_db() -> tuple[sqlite3.Connection, Path]:
    """Create an isolated SQLite DB (with both source tables) and return
    (connection, db_path).  Uses a NamedTemporaryFile so it survives the
    connection being closed and re-opened."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    db_path = Path(tmp.name)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE statements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            original_path TEXT NOT NULL,
            file_hash TEXT NOT NULL UNIQUE,
            bank_name TEXT,
            account_number TEXT,
            statement_date TEXT,
            import_date TEXT DEFAULT (datetime('now')),
            total_transactions INTEGER DEFAULT 0,
            total_debits REAL DEFAULT 0.0,
            total_credits REAL DEFAULT 0.0
        );

        CREATE TABLE transactions_raw (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            statement_id INTEGER,
            bank_name TEXT,
            date TEXT NOT NULL,
            description TEXT NOT NULL,
            amount REAL NOT NULL,
            transaction_type TEXT NOT NULL
                CHECK(transaction_type IN ('DEBIT','CREDIT','UNKNOWN')),
            merchant_name TEXT,
            category TEXT DEFAULT 'Uncategorized',
            balance REAL,
            confidence_score REAL DEFAULT 1.0,
            raw_text TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (statement_id) REFERENCES statements(id)
        );
    """)
    return conn, db_path


def make_tmp_config(db_path: Path) -> Path:
    """Write a minimal config.yaml in a temp dir pointing at db_path.
    Returns the path to the config file."""
    cfg_dir  = Path(tempfile.mkdtemp())
    cfg_file = cfg_dir / "config.yaml"
    cfg_file.write_text(yaml.dump({
        "database": {"path": str(db_path)},
        "parser":   {"output_dir": str(cfg_dir / "exports"),
                     "supported_banks": ["capital_one"]},
        "nlp":      {"spacy_model": "en_core_web_sm", "fuzzy_threshold": 80},
    }))
    return cfg_file


def seed_raw(conn: sqlite3.Connection, rows: list[dict]) -> list[int]:
    """Insert rows into transactions_raw and return their IDs."""
    ids = []
    for row in rows:
        cur = conn.execute(
            """
            INSERT INTO transactions_raw
                (statement_id, bank_name, date, description, amount,
                 transaction_type, balance)
            VALUES (:statement_id,:bank_name,:date,:description,:amount,
                    :transaction_type,:balance)
            """,
            row,
        )
        ids.append(cur.lastrowid)
    conn.commit()
    return ids


RAW_SAMPLES = [
    {
        "statement_id": None, "bank_name": "Capital One",
        "date": "2024-01-15", "description": "STARBUCKS #12345 SAN FRANCISCO CA",
        "amount": -5.75,  "transaction_type": "DEBIT",  "balance": 1200.00,
    },
    {
        "statement_id": None, "bank_name": "Capital One",
        "date": "2024-01-16", "description": "AMAZON.COM*1A2B3C4D5",
        "amount": -29.99, "transaction_type": "DEBIT",  "balance": 1170.01,
    },
    {
        "statement_id": None, "bank_name": "Citi",
        "date": "2024-01-17", "description": "DIRECT DEPOSIT EMPLOYER PAYROLL",
        "amount": 3500.00, "transaction_type": "CREDIT", "balance": 4670.01,
    },
]

MOCK_NLP_RESULT = {
    "clean_description":  "Starbucks",
    "merchant_name":      "Starbucks",
    "merchant_raw":       "STARBUCKS #12345",
    "category":           "Food & Dining",
    "subcategory":        "Coffee",
    "location":           "San Francisco, CA",
    "confidence_score":   0.95,
    "enrichment_method":  "substring",
}


# ---------------------------------------------------------------------------
# EnrichedTransaction unit tests
# ---------------------------------------------------------------------------

class TestEnrichedTransaction:
    def test_defaults(self):
        et = EnrichedTransaction(raw_id=1, date="2024-01-01",
                                 description="TEST", amount=-10.0)
        assert et.category         == "Uncategorized"
        assert et.confidence_score == 0.0
        assert et.transaction_type == "UNKNOWN"

    def test_to_db_row_has_required_keys(self):
        et = EnrichedTransaction(
            raw_id=42, date="2024-01-15", description="STARBUCKS",
            amount=-5.75, merchant_name="Starbucks",
            category="Food & Dining", subcategory="Coffee",
            location="San Francisco, CA", confidence_score=0.95,
            enrichment_method="substring",
        )
        row = et.to_db_row()
        for key in ("raw_id", "merchant_name", "category", "subcategory",
                    "location", "confidence_score", "enrichment_method"):
            assert key in row, f"Missing key: {key}"
        assert "id" not in row, "Primary key must be excluded from insert dict"

    def test_to_db_row_values(self):
        et = EnrichedTransaction(raw_id=7, date="2024-02-01",
                                 description="NETFLIX", amount=-15.99,
                                 category="Entertainment", subcategory="Streaming")
        row = et.to_db_row()
        assert row["raw_id"]     == 7
        assert row["subcategory"] == "Streaming"


# ---------------------------------------------------------------------------
# DB helper tests
# ---------------------------------------------------------------------------

class TestBulkUpsert:
    def test_inserts_new_rows(self):
        conn, _ = make_tmp_db()
        apply_migration(conn)
        rows = [
            EnrichedTransaction(raw_id=1, date="2024-01-15",
                                description="STARBUCKS", amount=-5.75,
                                merchant_name="Starbucks",
                                category="Food & Dining", subcategory="Coffee",
                                confidence_score=0.95),
            EnrichedTransaction(raw_id=2, date="2024-01-16",
                                description="AMAZON", amount=-29.99,
                                merchant_name="Amazon", category="Shopping"),
        ]
        bulk_upsert(conn, rows)
        count = conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
        assert count == 2

    def test_upsert_overwrites_existing(self):
        conn, _ = make_tmp_db()
        apply_migration(conn)

        bulk_upsert(conn, [EnrichedTransaction(
            raw_id=1, date="2024-01-15", description="STARBUCKS", amount=-5.75,
            confidence_score=0.5, category="Uncategorized",
        )])
        bulk_upsert(conn, [EnrichedTransaction(
            raw_id=1, date="2024-01-15", description="STARBUCKS", amount=-5.75,
            merchant_name="Starbucks", category="Food & Dining",
            subcategory="Coffee", confidence_score=0.95,
        )])

        row = conn.execute(
            "SELECT merchant_name, category, subcategory, confidence_score "
            "FROM transactions WHERE raw_id=1"
        ).fetchone()
        assert row["merchant_name"]  == "Starbucks"
        assert row["category"]       == "Food & Dining"
        assert row["subcategory"]    == "Coffee"
        assert row["confidence_score"] == pytest.approx(0.95)

    def test_empty_batch_is_no_op(self):
        conn, _ = make_tmp_db()
        apply_migration(conn)
        assert bulk_upsert(conn, []) == 0

    def test_apply_migration_is_idempotent(self):
        conn, _ = make_tmp_db()
        apply_migration(conn)
        apply_migration(conn)   # must not raise
        count = conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE name='transactions'"
        ).fetchone()[0]
        assert count == 1


# ---------------------------------------------------------------------------
# EnrichmentPipeline integration tests  (NLP mocked)
# ---------------------------------------------------------------------------

class TestEnrichmentPipeline:
    """
    Pipeline tests use an isolated temp config.yaml + temp DB per test,
    matching the Week 1 test isolation pattern.  TransactionProcessor is
    mocked so no spaCy model is needed.
    """

    def _make_pipeline(self, db_path: Path,
                       reprocess_all: bool = True) -> EnrichmentPipeline:
        cfg_path = make_tmp_config(db_path)
        mock_proc = MagicMock()
        mock_proc.process.return_value = MOCK_NLP_RESULT

        with patch(
            "src.nlp.enrichment_pipeline.TransactionProcessor",
            return_value=mock_proc,
        ):
            pipeline = EnrichmentPipeline(
                config_path   = cfg_path,
                reprocess_all = reprocess_all,
            )
        pipeline._processor = mock_proc
        return pipeline

    def test_db_path_comes_from_config(self):
        """Pipeline._open_db must use cfg.db_path, not a hard-coded path."""
        conn, db_path = make_tmp_db()
        conn.close()
        pipeline = self._make_pipeline(db_path)
        assert pipeline.cfg.db_path == db_path

    def test_run_processes_all_raw_rows(self):
        conn, db_path = make_tmp_db()
        seed_raw(conn, RAW_SAMPLES)
        conn.close()

        stats = self._make_pipeline(db_path).run()
        assert stats.total_raw == len(RAW_SAMPLES)
        assert stats.processed == len(RAW_SAMPLES)
        assert stats.errors    == 0

    def test_enriched_rows_written_to_db(self):
        conn, db_path = make_tmp_db()
        seed_raw(conn, RAW_SAMPLES)
        conn.close()

        self._make_pipeline(db_path).run()

        conn2 = sqlite3.connect(db_path)
        conn2.row_factory = sqlite3.Row
        rows = conn2.execute(
            "SELECT * FROM transactions ORDER BY raw_id"
        ).fetchall()
        assert len(rows) == len(RAW_SAMPLES)

        first = rows[0]
        assert first["merchant_name"]    == "Starbucks"
        assert first["category"]         == "Food & Dining"
        assert first["subcategory"]      == "Coffee"
        assert first["location"]         == "San Francisco, CA"
        assert first["confidence_score"] == pytest.approx(0.95)
        assert first["enrichment_method"] == "substring"
        conn2.close()

    def test_skip_already_enriched_rows(self):
        conn, db_path = make_tmp_db()
        seed_raw(conn, RAW_SAMPLES[:1])
        conn.close()

        # First run processes the 1 row
        stats1 = self._make_pipeline(db_path, reprocess_all=False).run()
        assert stats1.processed == 1

        # Second run should skip it
        stats2 = self._make_pipeline(db_path, reprocess_all=False).run()
        assert stats2.skipped  == 1
        assert stats2.processed == 0

    def test_statement_id_filter(self):
        conn, db_path = make_tmp_db()
        seed_raw(conn, [dict(r, statement_id=1) for r in RAW_SAMPLES[:2]])
        seed_raw(conn, [dict(r, statement_id=2) for r in RAW_SAMPLES[2:]])
        conn.close()

        stats = self._make_pipeline(db_path).run(statement_id=1)
        assert stats.total_raw == 2
        assert stats.processed == 2

    def test_nlp_error_increments_error_count(self):
        conn, db_path = make_tmp_db()
        seed_raw(conn, RAW_SAMPLES[:1])
        conn.close()

        pipeline = self._make_pipeline(db_path)
        pipeline._processor.process.side_effect = ValueError("NLP exploded")

        stats = pipeline.run()
        assert stats.errors    == 1
        assert stats.processed == 0

    def test_pipeline_is_idempotent(self):
        conn, db_path = make_tmp_db()
        seed_raw(conn, RAW_SAMPLES)
        conn.close()

        pipeline = self._make_pipeline(db_path, reprocess_all=True)
        pipeline.run()
        pipeline.run()   # second run must not duplicate rows

        conn2 = sqlite3.connect(db_path)
        count = conn2.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
        assert count == len(RAW_SAMPLES)
        conn2.close()
