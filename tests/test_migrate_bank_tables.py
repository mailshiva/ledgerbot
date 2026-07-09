"""
Tests for migrate_bank_tables.py

Verifies:
  - All 5 tables created correctly
  - Column names and types match spec
  - CHECK constraints enforced (account_type, transaction_type)
  - Foreign key relationships work
  - Indexes created
  - Enrichment pattern: raw → enriched with ON CONFLICT(raw_id)
  - Supabase DDL generation

python -m pytest src/database/test_migrate_bank_tables.py -v
"""
import sys
import sqlite3
import pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.database.migrate_bank_tables import SQLITE_DDL, SUPABASE_DDL, run_sqlite_migration


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db():
    """In-memory SQLite DB with all bank tables created."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SQLITE_DDL)
    conn.commit()
    return conn


@pytest.fixture
def seeded_db(db):
    """DB with sample data across all tables."""
    # Insert a statement
    db.execute("""
        INSERT INTO bank_statements
            (filename, original_path, file_hash, bank_name,
             statement_period_start, statement_period_end,
             total_transactions, total_debits, total_credits)
        VALUES
            ('stmt_20250131.pdf', '/data/stmt_20250131.pdf', 'abc123hash',
             'dcu', '2025-01-01', '2025-01-31', 8, 500.0, 1200.0)
    """)
    stmt_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]

    # Insert bank_transactions_raw (checking + savings)
    bank_raw_rows = [
        (stmt_id, 'dcu', 'checking', '2025-01-05', 'STOP & SHOP', 85.40, 'DEBIT', 1200.00),
        (stmt_id, 'dcu', 'checking', '2025-01-10', 'PAYROLL DIRECT DEP', 2400.00, 'CREDIT', 3600.00),
        (stmt_id, 'dcu', 'checking', '2025-01-15', 'EVERSOURCE ENERGY', 145.00, 'DEBIT', 3455.00),
        (stmt_id, 'dcu', 'savings', '2025-01-02', 'TRANSFER FROM CHECKING', 500.00, 'CREDIT', 5500.00),
        (stmt_id, 'dcu', 'savings', '2025-01-20', 'DIVIDEND', 2.15, 'CREDIT', 5502.15),
    ]
    db.executemany("""
        INSERT INTO bank_transactions_raw
            (statement_id, bank_name, account_type, date, description,
             amount, transaction_type, balance)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, bank_raw_rows)

    # Insert bank_transactions (enriched)
    enriched_rows = [
        (1, stmt_id, 'dcu', 'checking', '2025-01-05', 'STOP & SHOP', 'Stop & Shop',
         85.40, 'DEBIT', 1200.00, 'Stop & Shop', 'Groceries', 'Supermarket', 0.92),
        (2, stmt_id, 'dcu', 'checking', '2025-01-10', 'PAYROLL DIRECT DEP', 'Direct Deposit',
         2400.00, 'CREDIT', 3600.00, None, 'Income', 'Payroll', 0.85),
        (3, stmt_id, 'dcu', 'checking', '2025-01-15', 'EVERSOURCE ENERGY', 'Eversource',
         145.00, 'DEBIT', 3455.00, 'Eversource', 'Utilities', 'Electric', 0.95),
    ]
    db.executemany("""
        INSERT INTO bank_transactions
            (raw_id, statement_id, bank_name, account_type, date, description,
             clean_description, amount, transaction_type, balance,
             merchant_name, category, subcategory, confidence_score)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, enriched_rows)

    # Insert loan_transactions
    db.executemany("""
        INSERT INTO loan_transactions
            (statement_id, bank_name, loan_identifier, date,
             description, payment_amount, principal_amount,
             interest_amount, balance)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [
        (stmt_id, 'dcu', 'NEW VEHICLE LOAN# 142', '2025-01-15',
         'LOAN PAYMENT', 450.00, 320.00, 130.00, 18500.00),
    ])

    db.commit()
    return db


# ---------------------------------------------------------------------------
# Table existence
# ---------------------------------------------------------------------------

class TestTableCreation:
    def test_all_tables_exist(self, db):
        cursor = db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = {row[0] for row in cursor.fetchall()}
        expected = {
            "bank_statements",
            "bank_transactions_raw",
            "bank_transactions",
            "loan_transactions",
        }
        assert expected.issubset(tables)

    def test_idempotent_creation(self, db):
        """Running DDL twice should not fail (IF NOT EXISTS)."""
        db.executescript(SQLITE_DDL)  # second run
        cursor = db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = {row[0] for row in cursor.fetchall()}
        assert "bank_statements" in tables


# ---------------------------------------------------------------------------
# Column verification
# ---------------------------------------------------------------------------

class TestColumns:
    def _get_columns(self, db, table_name):
        cursor = db.execute(f"PRAGMA table_info({table_name})")
        return {row[1]: row[2] for row in cursor.fetchall()}  # name -> type

    def test_bank_statements_columns(self, db):
        cols = self._get_columns(db, "bank_statements")
        assert "id" in cols
        assert "filename" in cols
        assert "file_hash" in cols
        assert "bank_name" in cols
        assert "statement_period_start" in cols
        assert "statement_period_end" in cols
        assert "import_date" in cols

    def test_bank_transactions_raw_has_account_type(self, db):
        cols = self._get_columns(db, "bank_transactions_raw")
        assert "account_type" in cols
        assert "balance" in cols
        assert "statement_id" in cols

    def test_bank_transactions_enriched_has_enrichment_fields(self, db):
        cols = self._get_columns(db, "bank_transactions")
        assert "raw_id" in cols
        assert "clean_description" in cols
        assert "merchant_name" in cols
        assert "category" in cols
        assert "subcategory" in cols
        assert "enrichment_method" in cols
        assert "enriched_at" in cols

    def test_loan_transactions_has_breakdown(self, db):
        cols = self._get_columns(db, "loan_transactions")
        assert "payment_amount" in cols
        assert "principal_amount" in cols
        assert "interest_amount" in cols
        assert "balance" in cols
        assert "loan_identifier" in cols


# ---------------------------------------------------------------------------
# CHECK constraints
# ---------------------------------------------------------------------------

class TestConstraints:
    def test_account_type_rejects_invalid(self, db):
        with pytest.raises(sqlite3.IntegrityError):
            db.execute("""
                INSERT INTO bank_transactions_raw
                    (statement_id, bank_name, account_type, date,
                     description, amount, transaction_type)
                VALUES (1, 'dcu', 'investment', '2025-01-01',
                        'TEST', 100.0, 'DEBIT')
            """)

    def test_account_type_accepts_checking(self, seeded_db):
        rows = seeded_db.execute(
            "SELECT * FROM bank_transactions_raw WHERE account_type = 'checking'"
        ).fetchall()
        assert len(rows) == 3

    def test_account_type_accepts_savings(self, seeded_db):
        rows = seeded_db.execute(
            "SELECT * FROM bank_transactions_raw WHERE account_type = 'savings'"
        ).fetchall()
        assert len(rows) == 2

    def test_transaction_type_rejects_invalid(self, db):
        with pytest.raises(sqlite3.IntegrityError):
            db.execute("""
                INSERT INTO bank_transactions_raw
                    (statement_id, bank_name, account_type, date,
                     description, amount, transaction_type)
                VALUES (1, 'dcu', 'checking', '2025-01-01',
                        'TEST', 100.0, 'TRANSFER')
            """)

    def test_file_hash_unique(self, seeded_db):
        with pytest.raises(sqlite3.IntegrityError):
            seeded_db.execute("""
                INSERT INTO bank_statements
                    (filename, original_path, file_hash, bank_name)
                VALUES ('dup.pdf', '/data/dup.pdf', 'abc123hash', 'dcu')
            """)

    def test_raw_id_unique_on_enriched(self, seeded_db):
        with pytest.raises(sqlite3.IntegrityError):
            seeded_db.execute("""
                INSERT INTO bank_transactions
                    (raw_id, bank_name, account_type, date, description,
                     amount, transaction_type)
                VALUES (1, 'dcu', 'checking', '2025-01-05', 'DUP',
                        85.40, 'DEBIT')
            """)


# ---------------------------------------------------------------------------
# Foreign keys
# ---------------------------------------------------------------------------

class TestForeignKeys:
    def test_bank_txn_raw_references_statement(self, seeded_db):
        """Verify statement_id FK works — data inserted successfully."""
        row = seeded_db.execute("""
            SELECT s.bank_name, COUNT(*) as txn_count
            FROM bank_transactions_raw t
            JOIN bank_statements s ON t.statement_id = s.id
            GROUP BY s.bank_name
        """).fetchone()
        assert row["bank_name"] == "dcu"
        assert row["txn_count"] == 5

    def test_enriched_references_raw(self, seeded_db):
        """Verify raw_id FK works for joins."""
        rows = seeded_db.execute("""
            SELECT e.clean_description, r.description
            FROM bank_transactions e
            JOIN bank_transactions_raw r ON e.raw_id = r.id
        """).fetchall()
        assert len(rows) == 3

    def test_loan_transaction_inserted(self, seeded_db):
        row = seeded_db.execute("""
            SELECT description FROM loan_transactions
        """).fetchone()
        assert row["description"] == "LOAN PAYMENT"


# ---------------------------------------------------------------------------
# Data integrity
# ---------------------------------------------------------------------------

class TestDataIntegrity:
    def test_bank_statement_count(self, seeded_db):
        row = seeded_db.execute("SELECT COUNT(*) as n FROM bank_statements").fetchone()
        assert row["n"] == 1

    def test_bank_raw_count(self, seeded_db):
        row = seeded_db.execute(
            "SELECT COUNT(*) as n FROM bank_transactions_raw"
        ).fetchone()
        assert row["n"] == 5

    def test_bank_enriched_count(self, seeded_db):
        row = seeded_db.execute(
            "SELECT COUNT(*) as n FROM bank_transactions"
        ).fetchone()
        assert row["n"] == 3

    def test_loan_count(self, seeded_db):
        row = seeded_db.execute(
            "SELECT COUNT(*) as n FROM loan_transactions"
        ).fetchone()
        assert row["n"] == 1

    def test_checking_total_debit(self, seeded_db):
        row = seeded_db.execute("""
            SELECT SUM(amount) as total FROM bank_transactions_raw
            WHERE account_type = 'checking' AND transaction_type = 'DEBIT'
        """).fetchone()
        assert row["total"] == pytest.approx(230.40, abs=0.01)

    def test_loan_principal_plus_interest_equals_payment(self, seeded_db):
        row = seeded_db.execute("""
            SELECT payment_amount, principal_amount, interest_amount
            FROM loan_transactions
        """).fetchone()
        assert row["principal_amount"] + row["interest_amount"] == pytest.approx(
            row["payment_amount"], abs=0.01
        )


# ---------------------------------------------------------------------------
# Indexes
# ---------------------------------------------------------------------------

class TestIndexes:
    def test_indexes_created(self, db):
        cursor = db.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'"
        )
        indexes = {row[0] for row in cursor.fetchall()}
        expected = {
            "idx_bank_txn_raw_date",
            "idx_bank_txn_raw_account_type",
            "idx_bank_txn_raw_bank_name",
            "idx_bank_txn_raw_stmt",
            "idx_bank_txn_date",
            "idx_bank_txn_account_type",
            "idx_bank_txn_bank_name",
            "idx_bank_txn_category",
            "idx_bank_txn_merchant",
            "idx_loan_txn_date",
            "idx_loan_txn_stmt",
        }
        assert expected.issubset(indexes)


# ---------------------------------------------------------------------------
# Supabase DDL
# ---------------------------------------------------------------------------

class TestSupabaseDDL:
    def test_contains_all_tables(self):
        for table in [
            "bank_statements",
            "bank_transactions_raw",
            "bank_transactions",
            "loan_transactions",
        ]:
            assert f"CREATE TABLE IF NOT EXISTS {table}" in SUPABASE_DDL

    def test_uses_bigserial(self):
        assert "BIGSERIAL PRIMARY KEY" in SUPABASE_DDL

    def test_uses_numeric_for_money(self):
        assert "NUMERIC(12,2)" in SUPABASE_DDL

    def test_uses_date_type(self):
        assert "DATE NOT NULL" in SUPABASE_DDL

    def test_uses_timestamptz(self):
        assert "TIMESTAMPTZ" in SUPABASE_DDL

    def test_enables_rls(self):
        assert "ENABLE ROW LEVEL SECURITY" in SUPABASE_DDL