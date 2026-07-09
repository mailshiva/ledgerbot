#!/usr/bin/env python3
"""
Migration: Create bank statement tables (A1)

Creates 4 new tables for bank statement data alongside existing
credit card tables (statements, transactions_raw, transactions).

Tables created:
  1. bank_statements           — statement-level metadata (shared by all account types)
  2. bank_transactions_raw     — raw checking/savings transactions
  3. bank_transactions         — NLP-enriched checking/savings transactions
  4. loan_transactions         — loan payment records

Design decisions:
  - Separate table families for credit cards vs bank accounts vs loans
  - bank_transactions have account_type (checking/savings) and running balance
  - loan_transactions have payment/principal/interest breakdown
  - All tables use TEXT dates (ISO 8601) for SQLite compatibility
  - Foreign keys from transactions → bank_statements via statement_id
  - No merchant/subcategory/location columns — bank descriptions are simpler
    than credit cards and don't need merchant normalization

Usage:
  python migrate_bank_tables.py                          # default: ~/sqlLite_DB/bank_data.db
  python migrate_bank_tables.py --db /path/to/other.db   # custom path
  python migrate_bank_tables.py --dry-run                 # print SQL only

Supabase sync:
  The same DDL works in Supabase/PostgreSQL with minor type adjustments.
  See generate_supabase_sql() for the PostgreSQL version.
"""

import argparse
import sqlite3
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# SQLite DDL
# ---------------------------------------------------------------------------

SQLITE_DDL = """
-- =========================================================================
-- Bank Statements (shared metadata for checking, savings, and loan sections)
-- =========================================================================
CREATE TABLE IF NOT EXISTS bank_statements (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    filename        TEXT NOT NULL,
    original_path   TEXT NOT NULL,
    file_hash       TEXT NOT NULL UNIQUE,
    bank_name       TEXT NOT NULL,
    statement_period_start TEXT,
    statement_period_end   TEXT,
    import_date     TEXT DEFAULT (datetime('now')),
    total_transactions INTEGER DEFAULT 0,
    total_debits    REAL DEFAULT 0.0,
    total_credits   REAL DEFAULT 0.0
);

-- =========================================================================
-- Bank Transactions — Raw (checking + savings)
-- =========================================================================
CREATE TABLE IF NOT EXISTS bank_transactions_raw (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    statement_id    INTEGER NOT NULL,
    bank_name       TEXT NOT NULL,
    account_type    TEXT NOT NULL CHECK(account_type IN ('checking', 'savings')),
    date            TEXT NOT NULL,
    description     TEXT NOT NULL,
    amount          REAL NOT NULL,
    transaction_type TEXT NOT NULL CHECK(transaction_type IN ('DEBIT', 'CREDIT', 'UNKNOWN')),
    balance         REAL,
    category        TEXT DEFAULT 'Uncategorized',
    confidence_score REAL DEFAULT 1.0,
    raw_text        TEXT,
    created_at      TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (statement_id) REFERENCES bank_statements(id)
);

-- =========================================================================
-- Bank Transactions — Enriched (checking + savings)
-- =========================================================================
CREATE TABLE IF NOT EXISTS bank_transactions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    raw_id              INTEGER NOT NULL UNIQUE,
    statement_id        INTEGER,
    bank_name           TEXT NOT NULL,
    account_type        TEXT NOT NULL CHECK(account_type IN ('checking', 'savings')),
    date                TEXT NOT NULL,
    description         TEXT NOT NULL,
    clean_description   TEXT,
    amount              REAL NOT NULL,
    transaction_type    TEXT NOT NULL CHECK(transaction_type IN ('DEBIT', 'CREDIT', 'UNKNOWN')),
    balance             REAL,
    category            TEXT DEFAULT 'Uncategorized',
    confidence_score    REAL DEFAULT 0.0,
    enrichment_method   TEXT,
    enriched_at         TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (raw_id) REFERENCES bank_transactions_raw(id),
    FOREIGN KEY (statement_id) REFERENCES bank_statements(id)
);

-- =========================================================================
-- Loan Transactions
-- =========================================================================
CREATE TABLE IF NOT EXISTS loan_transactions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    statement_id    INTEGER NOT NULL,
    bank_name       TEXT NOT NULL,
    loan_identifier TEXT,
    date            TEXT NOT NULL,
    description     TEXT NOT NULL,
    payment_amount  REAL,
    principal_amount REAL,
    interest_amount REAL,
    balance         REAL,
    raw_text        TEXT,
    created_at      TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (statement_id) REFERENCES bank_statements(id)
);

-- =========================================================================
-- Indexes for common query patterns
-- =========================================================================
CREATE INDEX IF NOT EXISTS idx_bank_txn_raw_date
    ON bank_transactions_raw(date);
CREATE INDEX IF NOT EXISTS idx_bank_txn_raw_account_type
    ON bank_transactions_raw(account_type);
CREATE INDEX IF NOT EXISTS idx_bank_txn_raw_bank_name
    ON bank_transactions_raw(bank_name);
CREATE INDEX IF NOT EXISTS idx_bank_txn_raw_stmt
    ON bank_transactions_raw(statement_id);

CREATE INDEX IF NOT EXISTS idx_bank_txn_date
    ON bank_transactions(date);
CREATE INDEX IF NOT EXISTS idx_bank_txn_account_type
    ON bank_transactions(account_type);
CREATE INDEX IF NOT EXISTS idx_bank_txn_bank_name
    ON bank_transactions(bank_name);
CREATE INDEX IF NOT EXISTS idx_bank_txn_category
    ON bank_transactions(category);

CREATE INDEX IF NOT EXISTS idx_loan_txn_date
    ON loan_transactions(date);
CREATE INDEX IF NOT EXISTS idx_loan_txn_stmt
    ON loan_transactions(statement_id);
"""


# ---------------------------------------------------------------------------
# PostgreSQL DDL (for Supabase)
# ---------------------------------------------------------------------------

SUPABASE_DDL = """
-- =========================================================================
-- Bank Statements (shared metadata)
-- =========================================================================
CREATE TABLE IF NOT EXISTS bank_statements (
    id              BIGSERIAL PRIMARY KEY,
    filename        TEXT NOT NULL,
    original_path   TEXT NOT NULL,
    file_hash       TEXT NOT NULL UNIQUE,
    bank_name       TEXT NOT NULL,
    statement_period_start DATE,
    statement_period_end   DATE,
    import_date     TIMESTAMPTZ DEFAULT now(),
    total_transactions INTEGER DEFAULT 0,
    total_debits    NUMERIC(12,2) DEFAULT 0.0,
    total_credits   NUMERIC(12,2) DEFAULT 0.0
);

-- =========================================================================
-- Bank Transactions — Raw (checking + savings)
-- =========================================================================
CREATE TABLE IF NOT EXISTS bank_transactions_raw (
    id              BIGSERIAL PRIMARY KEY,
    statement_id    BIGINT NOT NULL REFERENCES bank_statements(id),
    bank_name       TEXT NOT NULL,
    account_type    TEXT NOT NULL CHECK(account_type IN ('checking', 'savings')),
    date            DATE NOT NULL,
    description     TEXT NOT NULL,
    amount          NUMERIC(12,2) NOT NULL,
    transaction_type TEXT NOT NULL CHECK(transaction_type IN ('DEBIT', 'CREDIT', 'UNKNOWN')),
    balance         NUMERIC(12,2),
    category        TEXT DEFAULT 'Uncategorized',
    confidence_score NUMERIC(3,2) DEFAULT 1.0,
    raw_text        TEXT,
    created_at      TIMESTAMPTZ DEFAULT now()
);

-- =========================================================================
-- Bank Transactions — Enriched (checking + savings)
-- =========================================================================
CREATE TABLE IF NOT EXISTS bank_transactions (
    id                  BIGSERIAL PRIMARY KEY,
    raw_id              BIGINT NOT NULL UNIQUE REFERENCES bank_transactions_raw(id),
    statement_id        BIGINT REFERENCES bank_statements(id),
    bank_name           TEXT NOT NULL,
    account_type        TEXT NOT NULL CHECK(account_type IN ('checking', 'savings')),
    date                DATE NOT NULL,
    description         TEXT NOT NULL,
    clean_description   TEXT,
    amount              NUMERIC(12,2) NOT NULL,
    transaction_type    TEXT NOT NULL CHECK(transaction_type IN ('DEBIT', 'CREDIT', 'UNKNOWN')),
    balance             NUMERIC(12,2),
    category            TEXT DEFAULT 'Uncategorized',
    confidence_score    NUMERIC(3,2) DEFAULT 0.0,
    enrichment_method   TEXT,
    enriched_at         TIMESTAMPTZ DEFAULT now()
);

-- =========================================================================
-- Loan Transactions
-- =========================================================================
CREATE TABLE IF NOT EXISTS loan_transactions (
    id              BIGSERIAL PRIMARY KEY,
    statement_id    BIGINT NOT NULL REFERENCES bank_statements(id),
    bank_name       TEXT NOT NULL,
    loan_identifier TEXT,
    date            DATE NOT NULL,
    description     TEXT NOT NULL,
    payment_amount  NUMERIC(12,2),
    principal_amount NUMERIC(12,2),
    interest_amount NUMERIC(12,2),
    balance         NUMERIC(12,2),
    raw_text        TEXT,
    created_at      TIMESTAMPTZ DEFAULT now()
);

-- =========================================================================
-- Indexes
-- =========================================================================
CREATE INDEX IF NOT EXISTS idx_bank_txn_raw_date ON bank_transactions_raw(date);
CREATE INDEX IF NOT EXISTS idx_bank_txn_raw_account_type ON bank_transactions_raw(account_type);
CREATE INDEX IF NOT EXISTS idx_bank_txn_raw_bank_name ON bank_transactions_raw(bank_name);
CREATE INDEX IF NOT EXISTS idx_bank_txn_raw_stmt ON bank_transactions_raw(statement_id);

CREATE INDEX IF NOT EXISTS idx_bank_txn_date ON bank_transactions(date);
CREATE INDEX IF NOT EXISTS idx_bank_txn_account_type ON bank_transactions(account_type);
CREATE INDEX IF NOT EXISTS idx_bank_txn_bank_name ON bank_transactions(bank_name);
CREATE INDEX IF NOT EXISTS idx_bank_txn_category ON bank_transactions(category);

CREATE INDEX IF NOT EXISTS idx_loan_txn_date ON loan_transactions(date);
CREATE INDEX IF NOT EXISTS idx_loan_txn_stmt ON loan_transactions(statement_id);

-- =========================================================================
-- Enable Row Level Security (recommended for Supabase)
-- =========================================================================
ALTER TABLE bank_statements ENABLE ROW LEVEL SECURITY;
ALTER TABLE bank_transactions_raw ENABLE ROW LEVEL SECURITY;
ALTER TABLE bank_transactions ENABLE ROW LEVEL SECURITY;
ALTER TABLE loan_transactions ENABLE ROW LEVEL SECURITY;
"""


# ---------------------------------------------------------------------------
# Migration logic
# ---------------------------------------------------------------------------

def run_sqlite_migration(db_path: str, dry_run: bool = False) -> None:
    """Create bank tables in SQLite database."""
    path = Path(db_path)
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"  Created directory: {path.parent}")

    if dry_run:
        print("\n--- SQLite DDL (dry run) ---\n")
        print(SQLITE_DDL)
        return

    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(SQLITE_DDL)
        conn.commit()
        print(f"  ✓ SQLite migration complete: {path}")

        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [row[0] for row in cursor.fetchall()]
        expected = {
            "bank_statements",
            "bank_transactions_raw",
            "bank_transactions",
            "loan_transactions",
        }
        created = expected & set(tables)
        print(f"  ✓ Verified tables: {', '.join(sorted(created))}")

        missing = expected - set(tables)
        if missing:
            print(f"  ⚠ Missing tables: {', '.join(sorted(missing))}")

    finally:
        conn.close()


def generate_supabase_sql(output_path: str | None = None) -> str:
    if output_path:
        Path(output_path).write_text(SUPABASE_DDL)
        print(f"  ✓ Supabase SQL written to: {output_path}")
    return SUPABASE_DDL


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Create bank statement tables (migration A1)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python migrate_bank_tables.py                              # Run migration
  python migrate_bank_tables.py --dry-run                    # Print SQL only
  python migrate_bank_tables.py --db /tmp/test.db            # Custom DB path
  python migrate_bank_tables.py --supabase-sql supabase.sql  # Export Supabase DDL
        """,
    )

    parser.add_argument(
        "--db",
        default=str(Path.home() / "sqlLite_DB" / "bank_data.db"),
        help="Path to SQLite database (default: ~/sqlLite_DB/bank_data.db)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print DDL without executing")
    parser.add_argument("--supabase-sql", metavar="FILE", help="Write PostgreSQL DDL to file")

    args = parser.parse_args()

    print("=" * 60)
    print("MIGRATION A1: Bank Statement Tables")
    print("=" * 60)

    print(f"\n[1/2] SQLite → {args.db}")
    run_sqlite_migration(args.db, dry_run=args.dry_run)

    if args.supabase_sql:
        print(f"\n[2/2] Supabase DDL → {args.supabase_sql}")
        generate_supabase_sql(args.supabase_sql)
    else:
        print(f"\n[2/2] Supabase DDL → skipped (use --supabase-sql FILE to export)")

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()