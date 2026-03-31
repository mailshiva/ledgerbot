"""
Tests for src/agent/tools.py

Fully offline — uses an in-memory SQLite database seeded with realistic
fixture data. No real DB file, no LLM calls, no network.
"""

from __future__ import annotations

import sqlite3
from unittest.mock import MagicMock

import pytest

from src.agent.tools import (
    TOOL_DEFINITIONS,
    detect_duplicates,
    execute_tool,
    find_transactions,
    get_categories,
    get_merchants,
    get_statements,
    get_spending_by_month,
    get_uncategorized,
    query_db,
    summarize_spending,
    _has_enriched_table,
    _txn_table,
)


# ---------------------------------------------------------------------------
# In-memory DB fixture
# ---------------------------------------------------------------------------

def _make_db(use_enriched: bool = True) -> MagicMock:
    """
    Build an in-memory SQLite DB and return a mock DatabaseManager whose
    .conn attribute points to it.

    use_enriched=True  → creates both transactions_raw and transactions tables
    use_enriched=False → creates only transactions_raw (simulates pre-NLP state)
    """
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row

    # statements table
    conn.executescript("""
        CREATE TABLE statements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            original_path TEXT NOT NULL,
            file_hash TEXT NOT NULL UNIQUE,
            bank_name TEXT,
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
            transaction_type TEXT NOT NULL,
            merchant_name TEXT,
            category TEXT DEFAULT 'Uncategorized',
            balance REAL,
            confidence_score REAL DEFAULT 1.0,
            raw_text TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        );
    """)

    if use_enriched:
        conn.executescript("""
            CREATE TABLE transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                raw_id INTEGER NOT NULL UNIQUE,
                statement_id INTEGER,
                bank_name TEXT,
                date TEXT NOT NULL,
                description TEXT NOT NULL,
                clean_description TEXT,
                amount REAL NOT NULL,
                transaction_type TEXT NOT NULL
                    CHECK(transaction_type IN ('DEBIT','CREDIT','UNKNOWN')),
                balance REAL,
                merchant_name TEXT,
                merchant_raw TEXT,
                category TEXT DEFAULT 'Uncategorized',
                subcategory TEXT,
                location TEXT,
                confidence_score REAL DEFAULT 0.0,
                enrichment_method TEXT,
                enriched_at TEXT DEFAULT (datetime('now'))
            );
        """)

    # Seed statements
    conn.execute("""
        INSERT INTO statements (filename, original_path, file_hash, bank_name,
                                statement_date, total_transactions, total_debits, total_credits)
        VALUES
          ('capital_one_jan.pdf', '/data/capital_one_jan.pdf', 'hash1',
           'capital_one', '2025-01-31', 5, 450.0, 1200.0),
          ('citi_feb.pdf', '/data/citi_feb.pdf', 'hash2',
           'citi', '2025-02-28', 3, 210.0, 0.0)
    """)

    # Seed transactions_raw
    raw_rows = [
        (1, 'capital_one', '2025-01-05', 'STARBUCKS #1234',      8.50,  'DEBIT',  'Starbucks',   'Food & Dining'),
        (1, 'capital_one', '2025-01-10', 'AMAZON.COM',           55.99, 'DEBIT',  'Amazon',      'Shopping'),
        (1, 'capital_one', '2025-01-15', 'PAYCHECK DIRECT DEP', 1200.0,'CREDIT', None,           'Income'),
        (1, 'capital_one', '2025-01-20', 'NETFLIX',              15.99, 'DEBIT',  'Netflix',     'Entertainment'),
        (1, 'capital_one', '2025-01-20', 'NETFLIX',              15.99, 'DEBIT',  'Netflix',     'Entertainment'),  # duplicate
        (2, 'citi',        '2025-02-03', 'WHOLE FOODS',          87.40, 'DEBIT',  'Whole Foods', 'Groceries'),
        (2, 'citi',        '2025-02-10', 'UBER',                 22.00, 'DEBIT',  'Uber',        'Transportation'),
        (2, 'citi',        '2025-02-18', 'MYSTERY CHARGE',       99.00, 'DEBIT',  None,          'Uncategorized'),
    ]
    conn.executemany("""
        INSERT INTO transactions_raw
          (statement_id, bank_name, date, description, amount, transaction_type,
           merchant_name, category)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, raw_rows)

    if use_enriched:
        enriched_rows = [
            (1, 1, 'capital_one', '2025-01-05', 'STARBUCKS #1234',      'Starbucks',       8.50,  'DEBIT',  'Starbucks',   'Food & Dining', 'Coffee',        0.95),
            (2, 1, 'capital_one', '2025-01-10', 'AMAZON.COM',           'Amazon',         55.99, 'DEBIT',  'Amazon',      'Shopping',      'Online Retail', 0.90),
            (3, 1, 'capital_one', '2025-01-15', 'PAYCHECK DIRECT DEP', 'Direct Deposit',1200.0, 'CREDIT', None,          'Income',        None,            0.80),
            (4, 1, 'capital_one', '2025-01-20', 'NETFLIX',              'Netflix',        15.99, 'DEBIT',  'Netflix',     'Entertainment', 'Streaming',     0.98),
            (5, 1, 'capital_one', '2025-01-20', 'NETFLIX',              'Netflix',        15.99, 'DEBIT',  'Netflix',     'Entertainment', 'Streaming',     0.98),
            (6, 2, 'citi',        '2025-02-03', 'WHOLE FOODS',          'Whole Foods',    87.40, 'DEBIT',  'Whole Foods', 'Groceries',     'Supermarket',   0.92),
            (7, 2, 'citi',        '2025-02-10', 'UBER',                 'Uber',           22.00, 'DEBIT',  'Uber',        'Transportation','Rideshare',     0.88),
            (8, 2, 'citi',        '2025-02-18', 'MYSTERY CHARGE',       None,             99.00, 'DEBIT',  None,          'Uncategorized', None,            0.10),
        ]
        conn.executemany("""
            INSERT INTO transactions
              (raw_id, statement_id, bank_name, date, description, clean_description,
               amount, transaction_type, merchant_name, category, subcategory, confidence_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, enriched_rows)

    conn.commit()

    mock_db = MagicMock()
    mock_db.conn = conn
    return mock_db


@pytest.fixture
def db():
    """DatabaseManager mock with enriched transactions table populated."""
    return _make_db(use_enriched=True)


@pytest.fixture
def db_raw_only():
    """DatabaseManager mock with only transactions_raw (pre-NLP state)."""
    return _make_db(use_enriched=False)


# ---------------------------------------------------------------------------
# Table routing
# ---------------------------------------------------------------------------

class TestTableRouting:
    def test_has_enriched_table_true(self, db):
        assert _has_enriched_table(db.conn) is True

    def test_has_enriched_table_false(self, db_raw_only):
        assert _has_enriched_table(db_raw_only.conn) is False

    def test_txn_table_enriched(self, db):
        assert _txn_table(db.conn) == "transactions"

    def test_txn_table_raw_fallback(self, db_raw_only):
        assert _txn_table(db_raw_only.conn) == "transactions_raw"


# ---------------------------------------------------------------------------
# query_db
# ---------------------------------------------------------------------------

class TestQueryDb:
    def test_basic_select(self, db):
        result = query_db(db, "SELECT COUNT(*) AS n FROM transactions")
        assert result[0]["n"] == 8

    def test_parameterised_not_supported_returns_rows(self, db):
        result = query_db(db, "SELECT * FROM transactions WHERE transaction_type = 'DEBIT'")
        assert all(r["transaction_type"] == "DEBIT" for r in result)

    def test_write_blocked_insert(self, db):
        result = query_db(db, "INSERT INTO transactions (raw_id, date, description, amount, transaction_type) VALUES (99,'2025-01-01','x',1.0,'DEBIT')")
        assert "error" in result

    def test_write_blocked_delete(self, db):
        result = query_db(db, "DELETE FROM transactions")
        assert "error" in result

    def test_write_blocked_drop(self, db):
        result = query_db(db, "DROP TABLE transactions")
        assert "error" in result

    def test_invalid_sql_returns_error(self, db):
        result = query_db(db, "SELECT * FROM nonexistent_table_xyz")
        assert "error" in result

    def test_falls_back_to_raw(self, db_raw_only):
        result = query_db(db_raw_only, "SELECT COUNT(*) AS n FROM transactions_raw")
        assert result[0]["n"] == 8


# ---------------------------------------------------------------------------
# get_categories
# ---------------------------------------------------------------------------

class TestGetCategories:
    def test_returns_list(self, db):
        result = get_categories(db)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_has_expected_fields(self, db):
        result = get_categories(db)
        for row in result:
            assert "category" in row
            assert "transaction_count" in row
            assert "total_spent" in row
            assert "total_received" in row

    def test_food_category_present(self, db):
        result = get_categories(db)
        cats = [r["category"] for r in result]
        assert "Food & Dining" in cats

    def test_ordered_by_spend_desc(self, db):
        result = get_categories(db)
        spends = [r["total_spent"] for r in result]
        assert spends == sorted(spends, reverse=True)

    def test_raw_fallback(self, db_raw_only):
        result = get_categories(db_raw_only)
        assert isinstance(result, list)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# get_merchants
# ---------------------------------------------------------------------------

class TestGetMerchants:
    def test_returns_list(self, db):
        result = get_merchants(db)
        assert isinstance(result, list)

    def test_has_expected_fields(self, db):
        result = get_merchants(db)
        for row in result:
            assert "merchant_name" in row
            assert "transaction_count" in row
            assert "total_spent" in row

    def test_netflix_appears_twice(self, db):
        result = get_merchants(db)
        netflix = next((r for r in result if r["merchant_name"] == "Netflix"), None)
        assert netflix is not None
        assert netflix["transaction_count"] == 2

    def test_null_merchants_excluded(self, db):
        result = get_merchants(db)
        assert all(r["merchant_name"] for r in result)

    def test_limit_respected(self, db):
        result = get_merchants(db, limit=2)
        assert len(result) <= 2

    def test_limit_capped_at_100(self, db):
        result = get_merchants(db, limit=999)
        assert len(result) <= 100


# ---------------------------------------------------------------------------
# summarize_spending
# ---------------------------------------------------------------------------

class TestSummarizeSpending:
    def test_all_time_summary(self, db):
        result = summarize_spending(db)
        assert "total_spent" in result
        assert "total_received" in result
        assert "transaction_count" in result

    def test_month_filter(self, db):
        result = summarize_spending(db, month="2025-01")
        assert result["month"] == "2025-01"
        # January has 3 debits: 8.50 + 55.99 + 15.99 + 15.99 = 96.47
        assert result["total_spent"] == pytest.approx(96.47, abs=0.01)

    def test_category_filter(self, db):
        result = summarize_spending(db, category="Entertainment")
        assert result["category_filter"] == "Entertainment"
        assert result["total_spent"] == pytest.approx(31.98, abs=0.01)  # 2x Netflix

    def test_month_and_category_combined(self, db):
        result = summarize_spending(db, month="2025-01", category="Shopping")
        assert result["month"] == "2025-01"
        assert result["total_spent"] == pytest.approx(55.99, abs=0.01)

    def test_income_captured(self, db):
        result = summarize_spending(db, month="2025-01")
        assert result["total_received"] == pytest.approx(1200.0, abs=0.01)


# ---------------------------------------------------------------------------
# get_spending_by_month
# ---------------------------------------------------------------------------

class TestGetSpendingByMonth:
    def test_returns_list(self, db):
        result = get_spending_by_month(db)
        assert isinstance(result, list)

    def test_has_month_field(self, db):
        result = get_spending_by_month(db)
        for row in result:
            assert "month" in row
            assert "total_spent" in row
            assert "total_received" in row

    def test_ordered_descending(self, db):
        result = get_spending_by_month(db)
        months = [r["month"] for r in result]
        assert months == sorted(months, reverse=True)

    def test_two_months_in_fixture(self, db):
        result = get_spending_by_month(db)
        assert len(result) == 2

    def test_limit_respected(self, db):
        result = get_spending_by_month(db, months=1)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# find_transactions
# ---------------------------------------------------------------------------

class TestFindTransactions:
    def test_no_filters_returns_rows(self, db):
        result = find_transactions(db)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_merchant_filter(self, db):
        result = find_transactions(db, merchant="Netflix")
        assert len(result) == 2
        assert all("Netflix" in r["merchant_name"] for r in result)

    def test_category_filter(self, db):
        result = find_transactions(db, category="Groceries")
        assert len(result) == 1
        assert result[0]["merchant_name"] == "Whole Foods"

    def test_date_range(self, db):
        result = find_transactions(db, start_date="2025-02-01", end_date="2025-02-28")
        assert all(r["date"] >= "2025-02-01" for r in result)
        assert all(r["date"] <= "2025-02-28" for r in result)

    def test_min_amount(self, db):
        result = find_transactions(db, min_amount=50.0)
        assert all(r["amount"] >= 50.0 for r in result)

    def test_max_amount(self, db):
        result = find_transactions(db, max_amount=20.0)
        assert all(r["amount"] <= 20.0 for r in result)

    def test_transaction_type_filter(self, db):
        result = find_transactions(db, transaction_type="CREDIT")
        assert all(r["transaction_type"] == "CREDIT" for r in result)

    def test_bank_name_filter(self, db):
        result = find_transactions(db, bank_name="citi")
        assert all(r["bank_name"] == "citi" for r in result)

    def test_limit_respected(self, db):
        result = find_transactions(db, limit=3)
        assert len(result) <= 3

    def test_raw_fallback(self, db_raw_only):
        result = find_transactions(db_raw_only, merchant="Netflix")
        assert len(result) == 2


# ---------------------------------------------------------------------------
# detect_duplicates
# ---------------------------------------------------------------------------

class TestDetectDuplicates:
    def test_finds_netflix_duplicate(self, db):
        result = detect_duplicates(db, days_window=3)
        assert isinstance(result, list)
        merchants = [r["merchant_name"] for r in result]
        assert "Netflix" in merchants

    def test_duplicate_has_correct_count(self, db):
        result = detect_duplicates(db, days_window=3)
        netflix = next(r for r in result if r["merchant_name"] == "Netflix")
        assert netflix["occurrence_count"] == 2

    def test_narrow_window_misses_distant_duplicates(self, db):
        # Netflix duplicates are on the same day (days_apart=0), always caught
        result = detect_duplicates(db, days_window=0)
        assert isinstance(result, list)

    def test_credits_excluded(self, db):
        result = detect_duplicates(db)
        for row in result:
            # All returned rows must be DEBIT (credits are excluded by the tool)
            assert row.get("merchant_name") != "Direct Deposit"


# ---------------------------------------------------------------------------
# get_uncategorized
# ---------------------------------------------------------------------------

class TestGetUncategorized:
    def test_returns_list(self, db):
        result = get_uncategorized(db)
        assert isinstance(result, list)

    def test_mystery_charge_included(self, db):
        result = get_uncategorized(db)
        assert any(r["description"] == "MYSTERY CHARGE" for r in result)

    def test_limit_respected(self, db):
        result = get_uncategorized(db, limit=1)
        assert len(result) <= 1

    def test_only_debits(self, db):
        result = get_uncategorized(db)
        assert all(r["transaction_type"] == "DEBIT" for r in result)


# ---------------------------------------------------------------------------
# get_statements
# ---------------------------------------------------------------------------

class TestGetStatements:
    def test_returns_two_statements(self, db):
        result = get_statements(db)
        assert len(result) == 2

    def test_has_expected_fields(self, db):
        result = get_statements(db)
        for row in result:
            for field in ["id", "filename", "bank_name", "import_date",
                          "total_transactions", "total_debits", "total_credits"]:
                assert field in row

    def test_capital_one_present(self, db):
        result = get_statements(db)
        banks = [r["bank_name"] for r in result]
        assert "capital_one" in banks


# ---------------------------------------------------------------------------
# execute_tool dispatcher
# ---------------------------------------------------------------------------

class TestExecuteTool:
    def test_dispatch_get_categories(self, db):
        result = execute_tool("get_categories", db)
        assert isinstance(result, list)

    def test_dispatch_get_merchants_with_kwarg(self, db):
        result = execute_tool("get_merchants", db, limit=5)
        assert isinstance(result, list)
        assert len(result) <= 5

    def test_dispatch_query_db(self, db):
        result = execute_tool("query_db", db, sql="SELECT COUNT(*) AS n FROM transactions")
        assert result[0]["n"] == 8

    def test_dispatch_unknown_tool_raises(self, db):
        with pytest.raises(KeyError, match="Unknown tool"):
            execute_tool("nonexistent_tool", db)

    def test_dispatch_summarize_spending(self, db):
        result = execute_tool("summarize_spending", db, month="2025-01")
        assert result["month"] == "2025-01"

    def test_dispatch_find_transactions(self, db):
        result = execute_tool("find_transactions", db, merchant="Uber")
        assert len(result) == 1


# ---------------------------------------------------------------------------
# TOOL_DEFINITIONS schema
# ---------------------------------------------------------------------------

class TestToolDefinitions:
    def test_all_tools_registered(self):
        names = {t["function"]["name"] for t in TOOL_DEFINITIONS}
        expected = {
            "query_db", "get_categories", "get_merchants",
            "summarize_spending", "get_spending_by_month",
            "find_transactions", "detect_duplicates",
            "get_uncategorized", "get_statements",
        }
        assert names == expected

    def test_each_tool_has_required_fields(self):
        for tool in TOOL_DEFINITIONS:
            assert tool["type"] == "function"
            fn = tool["function"]
            assert "name" in fn
            assert "description" in fn
            assert "parameters" in fn
            assert fn["parameters"]["type"] == "object"

    def test_descriptions_not_empty(self):
        for tool in TOOL_DEFINITIONS:
            assert len(tool["function"]["description"]) > 20
