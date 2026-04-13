"""
Tests for updated migrate_bank_tables.py and bank_enrichment_pipeline.py

Verifies:
  - Schema: no merchant_name, merchant_raw, subcategory, location in bank tables
  - Rule-based categorization: keyword matching
  - clean_description: noise stripping
  - Enrichment pipeline: full run, skip logic, reprocess, filters
  - ON CONFLICT(raw_id) DO UPDATE
"""

import sqlite3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.database.migrate_bank_tables import SQLITE_DDL, SUPABASE_DDL
from src.nlp.bank_enrichment_pipeline import (
    BankEnrichmentPipeline, PipelineStats,
    categorize_description, clean_description, _coerce,
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


def make_conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SQLITE_DDL)
    conn.commit()
    return conn


def seed(conn):
    conn.execute("""
        INSERT INTO bank_statements
            (filename, original_path, file_hash, bank_name,
             statement_period_start, statement_period_end)
        VALUES ('stmt.pdf', '/data/stmt.pdf', 'hash1', 'dcu', '2025-01-01', '2025-01-31')
    """)
    rows = [
        (1, "dcu", "checking", "2025-01-05", "STOP & SHOP", 85.40, "DEBIT", 1200.00),
        (1, "dcu", "checking", "2025-01-10", "PAYROLL DIRECT DEP", 2400.00, "CREDIT", 3600.00),
        (1, "dcu", "checking", "2025-01-15", "ACH DEBIT EVERSOURCE ENERGY", 145.00, "DEBIT", 3455.00),
        (1, "dcu", "savings", "2025-01-02", "TRANSFER FROM CHECKING", 500.00, "CREDIT", 5500.00),
        (1, "dcu", "savings", "2025-01-31", "DIVIDEND", 2.15, "CREDIT", 5502.15),
    ]
    conn.executemany("""
        INSERT INTO bank_transactions_raw
            (statement_id, bank_name, account_type, date, description,
             amount, transaction_type, balance)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, rows)
    conn.commit()


# =====================================================================
print("=== Schema: removed columns ===")
conn = make_conn()

def get_cols(conn, table):
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}

raw_cols = get_cols(conn, "bank_transactions_raw")
check("raw: no merchant_name", "merchant_name" not in raw_cols)
check("raw: no subcategory", "subcategory" not in raw_cols)
check("raw: no location", "location" not in raw_cols)
check("raw: has description", "description" in raw_cols)
check("raw: has category", "category" in raw_cols)
check("raw: has balance", "balance" in raw_cols)
check("raw: has account_type", "account_type" in raw_cols)

enr_cols = get_cols(conn, "bank_transactions")
check("enriched: no merchant_name", "merchant_name" not in enr_cols)
check("enriched: no merchant_raw", "merchant_raw" not in enr_cols)
check("enriched: no subcategory", "subcategory" not in enr_cols)
check("enriched: no location", "location" not in enr_cols)
check("enriched: has clean_description", "clean_description" in enr_cols)
check("enriched: has category", "category" in enr_cols)
check("enriched: has confidence_score", "confidence_score" in enr_cols)
check("enriched: has enrichment_method", "enrichment_method" in enr_cols)

loan_cols = get_cols(conn, "loan_transactions")
check("loan: no subcategory", "subcategory" not in loan_cols)
check("loan: has category", "category" in loan_cols)
conn.close()

print()
print("=== Schema: Supabase DDL ===")
check("supa: no merchant_name in bank_transactions", "merchant_name" not in
      SUPABASE_DDL.split("Bank Transactions — Enriched")[1].split("Loan Transactions")[0])
check("supa: no subcategory in bank_transactions", "subcategory" not in
      SUPABASE_DDL.split("Bank Transactions — Enriched")[1].split("Loan Transactions")[0])
check("supa: no location in bank_transactions", "location    " not in
      SUPABASE_DDL.split("Bank Transactions — Enriched")[1].split("Loan Transactions")[0])

print()
print("=== Schema: no merchant index ===")
indexes = {r[0] for r in make_conn().execute(
    "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'"
).fetchall()}
check("no merchant index", "idx_bank_txn_merchant" not in indexes)
check("category index exists", "idx_bank_txn_category" in indexes)

print()
print("=== categorize_description ===")
check("payroll→Income", categorize_description("PAYROLL DIRECT DEP") == ("Income", 1.0))
check("dividend→Income", categorize_description("DIVIDEND") == ("Income", 1.0))
check("direct deposit→Income", categorize_description("DIRECT DEPOSIT FROM EMPLOYER") == ("Income", 1.0))
check("eversource→Utilities", categorize_description("ACH DEBIT EVERSOURCE ENERGY")[0] == "Utilities")
check("transfer→Transfer", categorize_description("TRANSFER FROM CHECKING")[0] == "Transfer")
check("atm→ATM", categorize_description("ATM WITHDRAWAL CHASE")[0] == "ATM")
check("stop&shop→Groceries", categorize_description("STOP & SHOP")[0] == "Groceries")
check("walmart→Groceries", categorize_description("WALMART SUPERCENTER")[0] == "Groceries")
check("amazon→Shopping", categorize_description("AMZN MKTP US")[0] == "Shopping")
check("venmo→Venmo/P2P", categorize_description("VENMO PAYMENT")[0] == "Venmo / P2P")
check("check→Check", categorize_description("CHECK # 1234")[0] == "Check")
check("geico→Insurance", categorize_description("GEICO AUTO INS")[0] == "Insurance")
check("unknown→Uncategorized", categorize_description("RANDOM THING XYZ")[0] == "Uncategorized")
check("unknown confidence=0.3", categorize_description("RANDOM THING XYZ")[1] == 0.3)
check("case insensitive", categorize_description("payroll direct dep")[0] == "Income")

print()
print("=== clean_description ===")
check("strips digits", "12345" not in clean_description("STOP & SHOP 12345"))
check("strips phone", "555-123-4567" not in clean_description("CALL 555-123-4567 NOW"))
check("strips punctuation", "*" not in clean_description("AMZN*MKTP"))
check("strips extra space", "  " not in clean_description("STOP   &   SHOP"))
check("title case", clean_description("PAYROLL DIRECT DEP") == "Payroll Direct Dep")
check("preserves mixed case", clean_description("Stop & Shop") == "Stop & Shop")

print()
print("=== Pipeline: enrich_row ===")
conn = make_conn()
seed(conn)
pipeline = BankEnrichmentPipeline.__new__(BankEnrichmentPipeline)
pipeline._conn = conn
pipeline._supa = None
pipeline.batch_size = 200
pipeline.reprocess_all = False
pipeline.sqlite_only = True

row = conn.execute("SELECT * FROM bank_transactions_raw WHERE id = 1").fetchone()
enriched = pipeline._enrich_row(row)
check("enrich: raw_id", enriched["raw_id"] == 1)
check("enrich: account_type", enriched["account_type"] == "checking")
check("enrich: category=Groceries", enriched["category"] == "Groceries")
check("enrich: confidence=1.0", enriched["confidence_score"] == 1.0)
check("enrich: method=rule_based", enriched["enrichment_method"] == "rule_based")
check("enrich: no merchant_name key", "merchant_name" not in enriched)
check("enrich: no subcategory key", "subcategory" not in enriched)
check("enrich: no location key", "location" not in enriched)
check("enrich: clean_description set", enriched["clean_description"] is not None)

row2 = conn.execute("SELECT * FROM bank_transactions_raw WHERE id = 2").fetchone()
enriched2 = pipeline._enrich_row(row2)
check("enrich payroll: category=Income", enriched2["category"] == "Income")

row3 = conn.execute("SELECT * FROM bank_transactions_raw WHERE id = 3").fetchone()
enriched3 = pipeline._enrich_row(row3)
check("enrich eversource: category=Utilities", enriched3["category"] == "Utilities")

row4 = conn.execute("SELECT * FROM bank_transactions_raw WHERE id = 4").fetchone()
enriched4 = pipeline._enrich_row(row4)
check("enrich transfer: category=Transfer", enriched4["category"] == "Transfer")
check("enrich transfer: account_type=savings", enriched4["account_type"] == "savings")

print()
print("=== Pipeline: bulk_upsert ===")
pipeline._bulk_upsert([enriched])
result = conn.execute("SELECT * FROM bank_transactions WHERE raw_id = 1").fetchone()
check("upsert: row exists", result is not None)
check("upsert: category", result["category"] == "Groceries")
check("upsert: no merchant_name col", "merchant_name" not in dict(result).keys()
      if isinstance(result, sqlite3.Row) else True)

# Update and re-upsert
enriched["category"] = "Updated"
pipeline._bulk_upsert([enriched])
count = conn.execute("SELECT COUNT(*) FROM bank_transactions WHERE raw_id = 1").fetchone()[0]
check("upsert: no duplicate", count == 1)
updated = conn.execute("SELECT category FROM bank_transactions WHERE raw_id = 1").fetchone()
check("upsert: value updated", updated["category"] == "Updated")

print()
print("=== Pipeline: full run ===")
conn.execute("DELETE FROM bank_transactions")
conn.commit()
stats = pipeline.run()
check("run: total_raw=5", stats.total_raw == 5)
check("run: processed=5", stats.processed == 5)
check("run: errors=0", stats.errors == 0)
check("run: skipped=0", stats.skipped == 0)

enriched_count = conn.execute("SELECT COUNT(*) FROM bank_transactions").fetchone()[0]
check("run: 5 enriched rows", enriched_count == 5)

checking_ct = conn.execute(
    "SELECT COUNT(*) FROM bank_transactions WHERE account_type='checking'"
).fetchone()[0]
savings_ct = conn.execute(
    "SELECT COUNT(*) FROM bank_transactions WHERE account_type='savings'"
).fetchone()[0]
check("run: 3 checking", checking_ct == 3)
check("run: 2 savings", savings_ct == 2)

# Verify categories assigned
income_ct = conn.execute(
    "SELECT COUNT(*) FROM bank_transactions WHERE category='Income'"
).fetchone()[0]
check("run: income transactions found", income_ct >= 1)

print()
print("=== Pipeline: skip already enriched ===")
stats2 = pipeline.run()
check("re-run: skipped=5", stats2.skipped == 5)
check("re-run: processed=0", stats2.processed == 0)

print()
print("=== Pipeline: reprocess ===")
pipeline.reprocess_all = True
stats3 = pipeline.run()
check("reprocess: processed=5", stats3.processed == 5)
check("reprocess: skipped=0", stats3.skipped == 0)
pipeline.reprocess_all = False

print()
print("=== Pipeline: account_type filter ===")
conn.execute("DELETE FROM bank_transactions")
conn.commit()
stats4 = pipeline.run(account_type="savings")
check("filter savings: total_raw=2", stats4.total_raw == 2)
check("filter savings: processed=2", stats4.processed == 2)

print()
print("=== Pipeline: statement_id filter ===")
conn.execute("DELETE FROM bank_transactions")
conn.commit()
stats5 = pipeline.run(statement_id=1)
check("filter stmt: total_raw=5", stats5.total_raw == 5)
check("filter stmt: processed=5", stats5.processed == 5)

print()
print("=== _coerce ===")
out = _coerce({"raw_id": "1", "amount": "85.40", "description": "TEST"})
check("coerce int", isinstance(out["raw_id"], int))
check("coerce float", isinstance(out["amount"], float))
check("coerce str", isinstance(out["description"], str))

print()
print("=" * 60)
print(f"RESULTS: {passed} passed, {failed} failed")
if failed == 0:
    print("ALL TESTS PASSED")
print("=" * 60)