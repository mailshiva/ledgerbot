"""
src/db/enriched_transaction.py
──────────────────────────────
Dataclass that mirrors the `transactions` table plus a helper that
bulk-upserts rows.  Kept separate from the NLP logic so the DB layer
has no spaCy / RapidFuzz imports.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class EnrichedTransaction:
    """
    One fully-enriched transaction row ready for the `transactions` table.

    Populated by EnrichmentPipeline after running the NLP stack over a
    transactions_raw row.
    """

    # --- identity / source ---
    raw_id: int                              # PK of the transactions_raw row
    statement_id: Optional[int] = None
    bank_name: Optional[str] = None

    # --- core financial fields (copied from raw) ---
    date: str = ""
    description: str = ""                   # original un-cleaned description
    amount: float = 0.0
    transaction_type: str = "UNKNOWN"
    balance: Optional[float] = None

    # --- NLP-enriched ---
    clean_description: Optional[str] = None
    merchant_name: Optional[str] = None
    merchant_raw: Optional[str] = None      # token before normalisation
    category: str = "Uncategorized"
    subcategory: Optional[str] = None
    location: Optional[str] = None
    confidence_score: float = 0.0

    # --- provenance ---
    enrichment_method: Optional[str] = None
    enriched_at: str = field(
        default_factory=lambda: datetime.utcnow().strftime("%Y-%m-%d %Human:%M:%S")
    )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def to_db_row(self) -> dict:
        """Return a dict whose keys exactly match the `transactions` columns."""
        d = asdict(self)
        # Remove the auto-gen primary key – SQLite handles it
        d.pop("id", None)
        return d


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

_UPSERT_SQL = """
INSERT INTO transactions (
    raw_id, statement_id, bank_name,
    date, description, clean_description,
    amount, transaction_type, balance,
    merchant_name, merchant_raw,
    category, subcategory, location,
    confidence_score, enrichment_method, enriched_at
) VALUES (
    :raw_id, :statement_id, :bank_name,
    :date, :description, :clean_description,
    :amount, :transaction_type, :balance,
    :merchant_name, :merchant_raw,
    :category, :subcategory, :location,
    :confidence_score, :enrichment_method, :enriched_at
)
ON CONFLICT(raw_id) DO UPDATE SET
    merchant_name      = excluded.merchant_name,
    merchant_raw       = excluded.merchant_raw,
    clean_description  = excluded.clean_description,
    category           = excluded.category,
    subcategory        = excluded.subcategory,
    location           = excluded.location,
    confidence_score   = excluded.confidence_score,
    enrichment_method  = excluded.enrichment_method,
    enriched_at        = excluded.enriched_at;
"""


def bulk_upsert(conn: sqlite3.Connection, rows: Sequence[EnrichedTransaction]) -> int:
    """
    Insert or update a batch of EnrichedTransaction rows.

    Uses ON CONFLICT(raw_id) DO UPDATE so the pipeline is safe to re-run
    against the same source data – it will overwrite previous enrichment
    results rather than duplicating rows.

    Returns the number of rows affected.
    """
    if not rows:
        return 0

    params = []
    for row in rows:
        d = row.to_db_row()
        params.append(d)

    cursor = conn.executemany(_UPSERT_SQL, params)
    conn.commit()
    return cursor.rowcount


def apply_migration(conn: sqlite3.Connection, sql_path: Optional[Path] = None) -> None:
    """
    Create the `transactions` table and indexes if they don't already exist.

    Pass sql_path to load from an external .sql file, or leave None to use
    the inline DDL (handy in tests).
    """
    if sql_path and sql_path.exists():
        ddl = sql_path.read_text()
    else:
        # Inline minimal DDL – mirrors migrate_schema.sql exactly
        ddl = _INLINE_DDL

    # sqlite3 executescript auto-commits, which is what we want for DDL
    conn.executescript(ddl)


_INLINE_DDL = """
CREATE TABLE IF NOT EXISTS transactions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    raw_id              INTEGER NOT NULL UNIQUE,
    statement_id        INTEGER,
    bank_name           TEXT,
    date                TEXT    NOT NULL,
    description         TEXT    NOT NULL,
    clean_description   TEXT,
    amount              REAL    NOT NULL,
    transaction_type    TEXT    NOT NULL
                            CHECK(transaction_type IN ('DEBIT','CREDIT','UNKNOWN')),
    balance             REAL,
    merchant_name       TEXT,
    merchant_raw        TEXT,
    category            TEXT    DEFAULT 'Uncategorized',
    subcategory         TEXT,
    location            TEXT,
    confidence_score    REAL    DEFAULT 0.0,
    enrichment_method   TEXT,
    enriched_at         TEXT    DEFAULT (datetime('now')),
    FOREIGN KEY (raw_id)       REFERENCES transactions_raw(id),
    FOREIGN KEY (statement_id) REFERENCES statements(id)
);
CREATE INDEX IF NOT EXISTS idx_txn_date        ON transactions(date);
CREATE INDEX IF NOT EXISTS idx_txn_merchant    ON transactions(merchant_name);
CREATE INDEX IF NOT EXISTS idx_txn_category    ON transactions(category);
CREATE INDEX IF NOT EXISTS idx_txn_subcategory ON transactions(subcategory);
CREATE INDEX IF NOT EXISTS idx_txn_raw_id      ON transactions(raw_id);
"""
