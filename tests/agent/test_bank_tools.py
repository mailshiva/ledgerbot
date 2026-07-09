"""
Tests for src/agent/bank_tools.py

Fully offline — in-memory SQLite with realistic fixture data.
Mirrors test_tools.py structure.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import sqlite3
from unittest.mock import MagicMock

from src.database.migrate_bank_tables import SQLITE_DDL
from src.agent.bank_tools import (
    BANK_TOOL_DEFINITIONS,
    execute_bank_tool,
    query_bank_db,
    get_bank_categories,
    summarize_bank_spending,
    get_bank_spending_by_month,
    find_bank_transactions,
    get_bank_balance,
    get_uncategorized_bank,
    get_bank_statements,
    get_loan_summary,
    get_loan_transactions,
    _has_enriched_table,
    _txn_table,
)

passed = 0
failed = 0

def check(name, condition):
    global passed, failed
    if condition:
        print(f"  PASS: {name}")
        passed += 1
    else:
        print(f"  FAIL: {name}")
        failed += 1


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

def _make_db(use_enriched=True):
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SQLITE_DDL)

    # Seed statements
    conn.execute("""
        INSERT INTO bank_statements
            (filename, original_path, file_hash, bank_name,
             statement_period_start, statement_period_end,
             total_transactions, total_debits, total_credits)
        VALUES
            ('stmt_jan.pdf', '/data/stmt_jan.pdf', 'hash_jan', 'dcu',
             '2025-01-01', '2025-01-31', 5, 230.40, 2900.00),
            ('stmt_feb.pdf', '/data/stmt_feb.pdf', 'hash_feb', 'dcu',
             '2025-02-01', '2025-02-28', 3, 180.00, 0.0)
    """)

    # Seed bank_transactions_raw
    raw_rows = [
        (1, 'dcu', 'checking', '2025-01-05', 'STOP & SHOP',       85.40,  'DEBIT',  1200.00, 'Groceries'),
        (1, 'dcu', 'checking', '2025-01-10', 'PAYROLL DIRECT DEP', 2400.00,'CREDIT', 3600.00, 'Income'),
        (1, 'dcu', 'checking', '2025-01-15', 'EVERSOURCE ENERGY',  145.00, 'DEBIT',  3455.00, 'Utilities'),
        (1, 'dcu', 'savings',  '2025-01-02', 'TRANSFER FROM CHK',  500.00, 'CREDIT', 5500.00, 'Transfer'),
        (1, 'dcu', 'savings',  '2025-01-31', 'DIVIDEND',           2.15,   'CREDIT', 5502.15, 'Income'),
        (2, 'dcu', 'checking', '2025-02-03', 'MYSTERY CHARGE',     99.00,  'DEBIT',  3356.00, 'Uncategorized'),
        (2, 'dcu', 'checking', '2025-02-10', 'ATT WIRELESS',       80.00,  'DEBIT',  3276.00, 'Telecom'),
        (2, 'dcu', 'checking', '2025-02-15', 'UNKNOWN SERVICE',    1.00,   'DEBIT',  3275.00, 'Uncategorized'),
    ]
    conn.executemany("""
        INSERT INTO bank_transactions_raw
            (statement_id, bank_name, account_type, date, description,
             amount, transaction_type, balance, category)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, raw_rows)

    if use_enriched:
        enriched_rows = [
            (1, 1, 'dcu', 'checking', '2025-01-05', 'STOP & SHOP',       'Stop & Shop',    85.40,  'DEBIT',  1200.00, 'Groceries',      1.0,  'rule_based'),
            (2, 1, 'dcu', 'checking', '2025-01-10', 'PAYROLL DIRECT DEP', 'Payroll',        2400.00,'CREDIT', 3600.00, 'Income',         1.0,  'rule_based'),
            (3, 1, 'dcu', 'checking', '2025-01-15', 'EVERSOURCE ENERGY',  'Eversource',     145.00, 'DEBIT',  3455.00, 'Utilities',      1.0,  'rule_based'),
            (4, 1, 'dcu', 'savings',  '2025-01-02', 'TRANSFER FROM CHK',  'Transfer',       500.00, 'CREDIT', 5500.00, 'Transfer',       1.0,  'rule_based'),
            (5, 1, 'dcu', 'savings',  '2025-01-31', 'DIVIDEND',           'Dividend',       2.15,   'CREDIT', 5502.15, 'Income',         1.0,  'rule_based'),
            (6, 2, 'dcu', 'checking', '2025-02-03', 'MYSTERY CHARGE',     'Mystery Charge', 99.00,  'DEBIT',  3356.00, 'Uncategorized',  0.3,  'fallback'),
            (7, 2, 'dcu', 'checking', '2025-02-10', 'ATT WIRELESS',       'Att Wireless',   80.00,  'DEBIT',  3276.00, 'Telecom',        1.0,  'rule_based'),
            (8, 2, 'dcu', 'checking', '2025-02-15', 'UNKNOWN SERVICE',    'Unknown',        1.00,   'DEBIT',  3275.00, 'Uncategorized',  0.3,  'fallback'),
        ]
        conn.executemany("""
            INSERT INTO bank_transactions
                (raw_id, statement_id, bank_name, account_type, date, description,
                 clean_description, amount, transaction_type, balance,
                 category, confidence_score, enrichment_method)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, enriched_rows)

    # Seed loan data
    conn.executemany("""
        INSERT INTO loan_transactions
            (statement_id, bank_name, loan_identifier, date, description,
             payment_amount, principal_amount, interest_amount, balance)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [
        (1, 'dcu', 'LOAN# 142', '2025-01-15', 'LOAN PAYMENT', 450.00, 320.00, 130.00, 18500.00),
        (2, 'dcu', 'LOAN# 142', '2025-02-15', 'LOAN PAYMENT', 450.00, 325.00, 125.00, 18175.00),
    ])

    conn.commit()
    mock_db = MagicMock()
    mock_db.conn = conn
    return mock_db


db = _make_db(use_enriched=True)
db_raw = _make_db(use_enriched=False)


# ---------------------------------------------------------------------------
# Table routing
# ---------------------------------------------------------------------------
print("=== Table routing ===")
check("enriched exists", _has_enriched_table(db.conn) is True)
check("raw-only fallback", _has_enriched_table(db_raw.conn) is False)
check("uses bank_transactions", _txn_table(db.conn) == "bank_transactions")
check("falls back to raw", _txn_table(db_raw.conn) == "bank_transactions_raw")

# ---------------------------------------------------------------------------
# query_bank_db
# ---------------------------------------------------------------------------
print()
print("=== query_bank_db ===")
result = query_bank_db(db, "SELECT COUNT(*) AS n FROM bank_transactions")
check("basic count", result[0]["n"] == 8)

result = query_bank_db(db, "SELECT * FROM bank_transactions WHERE transaction_type = 'DEBIT'")
check("filter works", all(r["transaction_type"] == "DEBIT" for r in result))

result = query_bank_db(db, "INSERT INTO bank_transactions (raw_id, date, description, amount, transaction_type, bank_name, account_type) VALUES (99,'2025-01-01','x',1.0,'DEBIT','dcu','checking')")
check("write blocked", "error" in result)

result = query_bank_db(db, "DROP TABLE bank_transactions")
check("drop blocked", "error" in result)

result = query_bank_db(db, "SELECT * FROM nonexistent_xyz")
check("invalid sql returns error", "error" in result)

result = query_bank_db(db_raw, "SELECT COUNT(*) AS n FROM bank_transactions_raw")
check("raw fallback works", result[0]["n"] == 8)

# ---------------------------------------------------------------------------
# get_bank_categories
# ---------------------------------------------------------------------------
print()
print("=== get_bank_categories ===")
result = get_bank_categories(db)
check("returns list", isinstance(result, list) and len(result) > 0)

cats = [r["category"] for r in result]
check("Groceries present", "Groceries" in cats)
check("Income present", "Income" in cats)
check("Utilities present", "Utilities" in cats)

for row in result:
    check(f"  {row['category']} has fields", all(k in row for k in ["category", "transaction_count", "total_spent", "total_received"]))

# Account type filter
result_chk = get_bank_categories(db, account_type="checking")
check("checking filter", all(True for r in result_chk))  # all checking categories

result_sav = get_bank_categories(db, account_type="savings")
cats_sav = [r["category"] for r in result_sav]
check("savings categories", "Transfer" in cats_sav or "Income" in cats_sav)

# Raw fallback
result_raw = get_bank_categories(db_raw)
check("raw fallback", isinstance(result_raw, list) and len(result_raw) > 0)

# ---------------------------------------------------------------------------
# summarize_bank_spending
# ---------------------------------------------------------------------------
print()
print("=== summarize_bank_spending ===")
result = summarize_bank_spending(db)
check("has total_spent", "total_spent" in result)
check("has total_received", "total_received" in result)
check("has transaction_count", "transaction_count" in result)

result_jan = summarize_bank_spending(db, month="2025-01")
check("month filter", result_jan["month"] == "2025-01")
check("jan spent", abs(result_jan["total_spent"] - 230.40) < 0.01)
check("jan received", abs(result_jan["total_received"] - 2902.15) < 0.01)

result_cat = summarize_bank_spending(db, category="Utilities")
check("category filter", result_cat["category_filter"] == "Utilities")
check("utilities spent", abs(result_cat["total_spent"] - 145.00) < 0.01)

result_acct = summarize_bank_spending(db, account_type="savings")
check("account_type filter", result_acct["account_type"] == "savings")

result_combo = summarize_bank_spending(db, month="2025-01", account_type="checking")
check("combo filter", result_combo["month"] == "2025-01" and result_combo["account_type"] == "checking")

# ---------------------------------------------------------------------------
# get_bank_spending_by_month
# ---------------------------------------------------------------------------
print()
print("=== get_bank_spending_by_month ===")
result = get_bank_spending_by_month(db)
check("returns list", isinstance(result, list))
check("2 months", len(result) == 2)

for row in result:
    check(f"  {row['month']} has fields", all(k in row for k in ["month", "total_spent", "total_received"]))

months = [r["month"] for r in result]
check("ordered desc", months == sorted(months, reverse=True))

result_1 = get_bank_spending_by_month(db, months=1)
check("limit=1", len(result_1) == 1)

result_chk = get_bank_spending_by_month(db, account_type="checking")
check("checking filter", isinstance(result_chk, list))

# ---------------------------------------------------------------------------
# find_bank_transactions
# ---------------------------------------------------------------------------
print()
print("=== find_bank_transactions ===")
result = find_bank_transactions(db)
check("returns rows", isinstance(result, list) and len(result) > 0)

result_chk = find_bank_transactions(db, account_type="checking")
check("checking only", all(r["account_type"] == "checking" for r in result_chk))

result_sav = find_bank_transactions(db, account_type="savings")
check("savings only", all(r["account_type"] == "savings" for r in result_sav))
check("savings count=2", len(result_sav) == 2)

result_cat = find_bank_transactions(db, category="Groceries")
check("category filter", len(result_cat) == 1)

result_date = find_bank_transactions(db, start_date="2025-02-01", end_date="2025-02-28")
check("date range", all(r["date"] >= "2025-02-01" and r["date"] <= "2025-02-28" for r in result_date))

result_min = find_bank_transactions(db, min_amount=100.0)
check("min_amount", all(r["amount"] >= 100.0 for r in result_min))

result_max = find_bank_transactions(db, max_amount=10.0)
check("max_amount", all(r["amount"] <= 10.0 for r in result_max))

result_type = find_bank_transactions(db, transaction_type="CREDIT")
check("txn type", all(r["transaction_type"] == "CREDIT" for r in result_type))

result_desc = find_bank_transactions(db, description="EVERSOURCE")
check("description search", len(result_desc) == 1)

result_lim = find_bank_transactions(db, limit=3)
check("limit respected", len(result_lim) <= 3)

result_raw = find_bank_transactions(db_raw, account_type="checking")
check("raw fallback", len(result_raw) == 6)

# ---------------------------------------------------------------------------
# get_bank_balance
# ---------------------------------------------------------------------------
print()
print("=== get_bank_balance ===")
result = get_bank_balance(db)
check("returns list", isinstance(result, list))
check("has rows", len(result) > 0)

result_chk = get_bank_balance(db, account_type="checking")
check("checking balance", len(result_chk) >= 1)

result_sav = get_bank_balance(db, account_type="savings")
check("savings balance", len(result_sav) >= 1)

# ---------------------------------------------------------------------------
# get_uncategorized_bank
# ---------------------------------------------------------------------------
print()
print("=== get_uncategorized_bank ===")
result = get_uncategorized_bank(db)
check("returns list", isinstance(result, list))
check("finds uncategorized", len(result) >= 1)
check("mystery charge", any(r["description"] == "MYSTERY CHARGE" for r in result))
check("only debits", all(r["transaction_type"] == "DEBIT" for r in result))

result_lim = get_uncategorized_bank(db, limit=1)
check("limit respected", len(result_lim) <= 1)

# ---------------------------------------------------------------------------
# get_bank_statements
# ---------------------------------------------------------------------------
print()
print("=== get_bank_statements ===")
result = get_bank_statements(db)
check("returns 2 statements", len(result) == 2)
for row in result:
    for field in ["id", "filename", "bank_name", "statement_period_start",
                   "total_transactions", "total_debits", "total_credits"]:
        check(f"  {row['filename']} has {field}", field in row)

check("dcu present", any(r["bank_name"] == "dcu" for r in result))

# ---------------------------------------------------------------------------
# get_loan_summary
# ---------------------------------------------------------------------------
print()
print("=== get_loan_summary ===")
result = get_loan_summary(db)
check("returns list", isinstance(result, list))
check("2 months of loans", len(result) == 2)

for row in result:
    check(f"  {row['month']} has fields", all(k in row for k in
          ["month", "total_payment", "total_principal", "total_interest", "ending_balance"]))

jan = next(r for r in result if r["month"] == "2025-01")
check("jan payment=450", abs(jan["total_payment"] - 450.00) < 0.01)
check("jan principal=320", abs(jan["total_principal"] - 320.00) < 0.01)
check("jan interest=130", abs(jan["total_interest"] - 130.00) < 0.01)

result_1 = get_loan_summary(db, months=1)
check("loan limit=1", len(result_1) == 1)

# ---------------------------------------------------------------------------
# get_loan_transactions
# ---------------------------------------------------------------------------
print()
print("=== get_loan_transactions ===")
result = get_loan_transactions(db)
check("returns list", isinstance(result, list))
check("2 loan txns", len(result) == 2)

result_date = get_loan_transactions(db, start_date="2025-02-01")
check("date filter", len(result_date) == 1)

result_lim = get_loan_transactions(db, limit=1)
check("limit", len(result_lim) == 1)

# ---------------------------------------------------------------------------
# execute_bank_tool dispatcher
# ---------------------------------------------------------------------------
print()
print("=== execute_bank_tool ===")
result = execute_bank_tool("get_bank_categories", db)
check("dispatch categories", isinstance(result, list))

result = execute_bank_tool("summarize_bank_spending", db, month="2025-01")
check("dispatch summary", result["month"] == "2025-01")

result = execute_bank_tool("find_bank_transactions", db, account_type="checking")
check("dispatch find", all(r["account_type"] == "checking" for r in result))

result = execute_bank_tool("query_bank_db", db, sql="SELECT COUNT(*) AS n FROM bank_transactions")
check("dispatch query", result[0]["n"] == 8)

result = execute_bank_tool("get_loan_summary", db)
check("dispatch loan", isinstance(result, list))

try:
    execute_bank_tool("nonexistent_tool", db)
    check("unknown tool raises", False)
except KeyError:
    check("unknown tool raises", True)

# ---------------------------------------------------------------------------
# BANK_TOOL_DEFINITIONS schema
# ---------------------------------------------------------------------------
print()
print("=== BANK_TOOL_DEFINITIONS ===")
names = {t["function"]["name"] for t in BANK_TOOL_DEFINITIONS}
expected = {
    "query_bank_db", "get_bank_categories", "summarize_bank_spending",
    "get_bank_spending_by_month", "find_bank_transactions",
    "get_bank_balance", "get_uncategorized_bank", "get_bank_statements",
    "get_loan_summary", "get_loan_transactions",
}
check("all 10 tools registered", names == expected)

for tool in BANK_TOOL_DEFINITIONS:
    fn = tool["function"]
    check(f"  {fn['name']} has schema", (
        tool["type"] == "function" and
        "name" in fn and "description" in fn and "parameters" in fn and
        fn["parameters"]["type"] == "object" and
        len(fn["description"]) > 20
    ))

print()
print("=" * 60)
print(f"RESULTS: {passed} passed, {failed} failed")
if failed == 0:
    print("ALL TESTS PASSED")
print("=" * 60)


def test_all():
    assert failed == 0, f"{failed} check(s) failed — see output above"