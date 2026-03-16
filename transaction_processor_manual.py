#!/usr/bin/env python3
"""
Week 1 CLI Tool - Bank Statement Parser
Usage: python transaction_processor_manual.py <command> [args]

Commands:
  test              Run a self-test with the sample CSV
  process <file>    Parse a CSV or PDF statement and store in DB
  stats             Show summary statistics from the database
  recent [n]        Show the n most recent transactions (default 10)
  merchant <name>   Show all transactions for a merchant
  monthly           Show monthly spending breakdown
  statements        List all imported statement files
  config            Show current config path and settings
  clear             Clear all data from the database
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import load_config, CONFIG_FILE
from src.database.db_manager import DatabaseManager, DuplicateStatementError


# ------------------------------------------------------------------ #
#  Shared helpers                                                       #
# ------------------------------------------------------------------ #

def get_db() -> DatabaseManager:
    """Return a DatabaseManager using the path from config."""
    return DatabaseManager()   # reads config internally


def print_header(title: str):
    print(f"\n{'='*58}")
    print(f"  {title}")
    print(f"{'='*58}")


def print_transaction(t: dict, index: int = None):
    prefix = f"[{index:>2}] " if index is not None else ""
    sign = "+" if t.get("transaction_type") == "CREDIT" else "-"
    merchant = t.get("merchant_name") or t.get("description", "")
    print(f"{prefix}{t['date']}  {sign}${t['amount']:>8.2f}"
          f"  {t.get('transaction_type','?'):<6}"
          f"  [{t.get('category',''):<14}]"
          f"  {merchant[:35]}")


# ------------------------------------------------------------------ #
#  Commands                                                            #
# ------------------------------------------------------------------ #

def cmd_config():
    """Show config file location and active settings."""
    print_header("Configuration")
    cfg = load_config()
    print(f"  Config file : {CONFIG_FILE}")
    print(f"  DB path     : {cfg.db_path}")
    print(f"  Default year: {cfg.default_year}")
    print(f"  Confidence  : {cfg.confidence_threshold}")

    db_exists = os.path.exists(cfg.db_path)
    print(f"  DB exists   : {'✅  yes' if db_exists else '⚠️   no (will be created on first import)'}")


def cmd_test():
    """Process the bundled sample statement as a smoke test."""
    print_header("Running Self-Test")

    sample = os.path.join(os.path.dirname(__file__),
                          "data", "samples", "sample_bank_statement.csv")
    if not os.path.exists(sample):
        print(f"❌  Sample file not found: {sample}")
        sys.exit(1)

    from src.parsers.csv_parser import CSVStatementParser
    print(f"📄  Parsing : {sample}")
    result = CSVStatementParser(sample).parse()

    summary = result["summary"]
    print(f"✅  Parsed {summary['total_transactions']} transactions")
    print(f"    Debits : ${summary['total_debits']:.2f}")
    print(f"    Credits: ${summary['total_credits']:.2f}")
    print(f"    Net    : ${summary['net']:.2f}")

    db = get_db()
    print(f"💾  DB path : {db.db_path}")

    try:
        file_hash = db.check_duplicate(sample)
    except DuplicateStatementError:
        print("⚠️   Sample already imported — skipping DB write (duplicate check works ✅)")
        return

    stmt_id = db.save_statement(sample, statement_date="2024-01", file_hash=file_hash)
    saved   = db.save_transactions(result["transactions"], statement_id=stmt_id)
    print(f"💾  Saved {saved} transactions (statement id={stmt_id})")

    print("\n  First 5 transactions:")
    for i, t in enumerate(result["transactions"][:5], 1):
        print_transaction(t, i)
    print("\n✅  Self-test passed!\n")


def cmd_process(filepath: str):
    """Parse a CSV or PDF and save to DB — aborts if already imported."""
    print_header(f"Processing: {os.path.basename(filepath)}")

    if not os.path.exists(filepath):
        print(f"❌  File not found: {filepath}")
        sys.exit(1)

    # ── Duplicate check BEFORE parsing (fast, no wasted work) ─────────
    db = get_db()
    print(f"💾  DB path : {db.db_path}")
    try:
        file_hash = db.check_duplicate(filepath)
    except DuplicateStatementError as e:
        print(f"\n❌  Duplicate file detected — aborting.\n")
        print(f"    {e}")
        sys.exit(1)

    # ── Parse ──────────────────────────────────────────────────────────
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".csv":
        from src.parsers.csv_parser import CSVStatementParser
        result = CSVStatementParser(filepath).parse()
    elif ext == ".pdf":
        from src.parsers.pdf_parser import PDFStatementParser
        result = PDFStatementParser(filepath).parse()
        bank = result.get("metadata", {}).get("bank_name", "unknown")
        print(f"🏦  Detected bank : {bank}")
    else:
        print(f"❌  Unsupported file type: {ext}")
        sys.exit(1)

    summary = result["summary"]
    print(f"📊  Transactions  : {summary['total_transactions']}")
    print(f"    Debits        : ${summary['total_debits']:.2f}")
    print(f"    Credits       : ${summary['total_credits']:.2f}")
    print(f"    Net           : ${summary['net']:.2f}")

    # ── Save ───────────────────────────────────────────────────────────
    bank_name = result.get("metadata", {}).get("bank_name")
    stmt_id   = db.save_statement(filepath, bank_name=bank_name, file_hash=file_hash)
    saved     = db.save_transactions(result["transactions"], statement_id=stmt_id, bank_name=bank_name)
    print(f"💾  Saved {saved} transactions (statement id={stmt_id})")


def cmd_statements():
    """List all statement files that have been imported."""
    print_header("Imported Statements")
    db = get_db()
    rows = db.get_statements()
    if not rows:
        print("  No statements imported yet.")
        return
    print(f"  {'ID':<4} {'Imported':<20} {'Bank':<12} {'Txns':>5}  {'File'}")
    print(f"  {'-'*70}")
    for r in rows:
        print(f"  {r['id']:<4} {r['import_date'][:19]:<20} "
              f"{(r['bank_name'] or 'unknown'):<12} {r['total_transactions']:>5}  "
              f"{r['filename']}")


def cmd_stats():
    print_header("Database Statistics")
    db = get_db()
    print(f"  DB path            : {db.db_path}")
    stats = db.get_summary_stats()
    print(f"  Total transactions : {stats['total']}")
    print(f"  Total debits       : ${stats['total_debits']:.2f}")
    print(f"  Total credits      : ${stats['total_credits']:.2f}")
    print(f"  Net balance        : ${stats['net']:.2f}")
    print(f"  Average debit      : ${stats['avg_debit']:.2f}")
    print(f"  Largest transaction: ${stats['largest_transaction']:.2f}")
    largest = db.get_largest_transaction()
    if largest:
        print(f"\n  📌 Largest: {largest['date']}  ${largest['amount']:.2f}"
              f"  {largest.get('merchant_name', largest['description'])}")


def cmd_recent(n: int = 10):
    print_header(f"Recent {n} Transactions")
    db = get_db()
    rows = db.get_transactions(limit=n)
    if not rows:
        print("  No transactions found. Run 'test' or 'process' first.")
        return
    for i, t in enumerate(rows, 1):
        print_transaction(t, i)


def cmd_merchant(name: str):
    print_header(f"Transactions: {name}")
    db = get_db()
    rows = db.get_transactions_by_merchant(name)
    if not rows:
        print(f"  No transactions found for '{name}'")
        return
    for i, t in enumerate(rows, 1):
        print_transaction(t, i)
    total = sum(t["amount"] for t in rows)
    print(f"\n  Total: ${total:.2f}  ({len(rows)} transactions)")


def cmd_monthly():
    print_header("Monthly Spending")
    db = get_db()
    rows = db.get_monthly_spending()
    if not rows:
        print("  No data yet.")
        return
    print(f"  {'Month':<12} {'Spending':>10} {'Income':>10} {'Count':>6}")
    print(f"  {'-'*44}")
    for r in rows:
        print(f"  {r['month']:<12} ${r['spending']:>9.2f}"
              f" ${r['income']:>9.2f} {r['transaction_count']:>6}")


def cmd_clear():
    confirm = input("⚠️  This will delete ALL data. Type 'yes' to confirm: ")
    if confirm.strip().lower() == "yes":
        get_db().clear_all()
        print("✅  Database cleared.")
    else:
        print("Aborted.")


# ------------------------------------------------------------------ #
#  Entry point                                                         #
# ------------------------------------------------------------------ #

COMMANDS = {
    "config":     lambda args: cmd_config(),
    "test":       lambda args: cmd_test(),
    "process":    lambda args: cmd_process(args[0]) if args else print("Usage: process <file>"),
    "stats":      lambda args: cmd_stats(),
    "recent":     lambda args: cmd_recent(int(args[0])) if args else cmd_recent(),
    "merchant":   lambda args: cmd_merchant(args[0]) if args else print("Usage: merchant <name>"),
    "monthly":    lambda args: cmd_monthly(),
    "statements": lambda args: cmd_statements(),
    "clear":      lambda args: cmd_clear(),
}


def print_usage():
    print(__doc__)


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print_usage()
        sys.exit(0)

    command = sys.argv[1].lower()
    extra_args = sys.argv[2:]

    if command not in COMMANDS:
        print(f"❌  Unknown command: '{command}'")
        print_usage()
        sys.exit(1)

    try:
        COMMANDS[command](extra_args)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as e:
        print(f"❌  Error: {e}")
        raise
