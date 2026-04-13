#!/usr/bin/env python3
"""
STEP 3: Import CSVs into Supabase.

Run after export_to_csv.py:
    python import_to_superbase.py

Reads credentials from macOS Keychain (service: supabase_bank).
If you haven't stored them yet, the script will prompt you.
"""

import csv
import subprocess
import sys
from pathlib import Path

from supabase import create_client, Client
import re
from datetime import datetime

NUMERIC_COLS = {
    "total_debits", "total_credits", "amount", "balance", "confidence_score"
}
INTEGER_COLS = {
    "id", "statement_id", "raw_id", "total_transactions"
}
DATE_COLS = {
    "date", "statement_date"
}
TIMESTAMP_COLS = {
    "import_date", "created_at", "enriched_at"
}

# Matches any run of non-digit, non-colon, non-space, non-dash chars
# that sneak into timestamp strings (e.g. "03uman:27:58" → "03:27:58")
_TS_GARBAGE = re.compile(r"[^0-9\s:\-\.+Z]")
# ----------------------------------------------------------------
# Config
# ----------------------------------------------------------------
CSV_DIR    = Path("/tmp/bank_export")
BATCH_SIZE = 500  # rows per API call — stay under Supabase's 1MB limit

# FK order — parent tables must be imported first
TABLES = [
    "statements",
    "transactions_raw",
    "transactions",
]

# ----------------------------------------------------------------
# Credentials
# ----------------------------------------------------------------

def _keychain_get(service: str, account: str) -> str | None:
    try:
        result = subprocess.run(
            ["security", "find-generic-password", "-s", service, "-a", account, "-w"],
            capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return None


def _keychain_set(service: str, account: str, value: str):
    subprocess.run(
        ["security", "add-generic-password", "-s", service, "-a", account, "-w", value],
        check=True
    )


def get_supabase_client() -> Client:
    url = _keychain_get("supabase_bank", "url")
    key = _keychain_get("supabase_bank", "anon_key")

    if not url or not key:
        print("Supabase credentials not found in Keychain.")
        print("Enter them now — stored once, never prompted again.\n")
        url = input("  Supabase project URL (https://xxxx.supabase.co): ").strip()
        key = input("  Supabase anon key: ").strip()
        _keychain_set("supabase_bank", "url", url)
        _keychain_set("supabase_bank", "anon_key", key)
        print("  ✓  Credentials saved to Keychain\n")

    return create_client(url, key)


# ----------------------------------------------------------------
# Type coercion — CSV is all strings, Postgres needs proper types
# ----------------------------------------------------------------

NUMERIC_COLS = {
    "total_debits", "total_credits", "amount", "balance", "confidence_score"
}
INTEGER_COLS = {
    "id", "statement_id", "raw_id", "total_transactions"
}
DATE_COLS = {
    "date", "statement_date"
}

def _clean_timestamp(val: str) -> str | None:
    """
    Scrub corrupted timestamp strings and parse to ISO format.
    '2026-04-03 03uman:27:58' → '2026-04-03 03:27:58'
    Returns None if the value is unrecoverable.
    """
    if not val:
        return None

    # Strip any alphabetic garbage that crept into the time portion
    cleaned = _TS_GARBAGE.sub("", val).strip()

    # Try common formats after cleaning
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(cleaned, fmt).strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue

    # Unrecoverable — null it out rather than failing the whole batch
    print(f"    ⚠️   Could not parse timestamp '{val}' (cleaned: '{cleaned}') — setting NULL")
    return None


def coerce_row(row: dict) -> dict:
    """Convert CSV string values to correct Python types for Supabase."""
    out = {}
    for k, v in row.items():
        if v == "" or v is None:
            out[k] = None
            continue
        if k in INTEGER_COLS:
            out[k] = int(v)
        elif k in NUMERIC_COLS:
            out[k] = float(v)
        elif k in DATE_COLS:
            out[k] = v  # already YYYY-MM-DD from export script
        elif k in TIMESTAMP_COLS:
            out[k] = _clean_timestamp(v)
        else:
            out[k] = v
    return out



# ----------------------------------------------------------------
# Import one table
# ----------------------------------------------------------------

def load_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def import_table(client: Client, table: str, csv_path: Path) -> int:
    if not csv_path.exists():
        print(f"  ⚠️   {csv_path} not found — skipping {table}")
        return 0

    raw_rows = load_csv(csv_path)
    if not raw_rows:
        print(f"  ⚠️   {table}: CSV is empty — skipping")
        return 0

    rows = [coerce_row(r) for r in raw_rows]
    total  = len(rows)
    imported = 0

    print(f"  Importing {table}: {total:,} rows in batches of {BATCH_SIZE}...")

    for i in range(0, total, BATCH_SIZE):
        batch = rows[i : i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE

        try:
            result = (
                client.table(table)
                .upsert(batch, on_conflict="id")  # safe to re-run
                .execute()
            )
            imported += len(batch)
            print(f"    batch {batch_num}/{total_batches} — {imported:,}/{total:,} rows ✓")

        except Exception as e:
            print(f"    ❌  batch {batch_num} failed: {e}")
            print(f"    First row of failed batch: {batch[0]}")
            print(f"    Fix the error above and re-run — upsert is safe to retry.")
            sys.exit(1)

    return imported


# ----------------------------------------------------------------
# Post-import: reset sequences so new inserts don't collide
# ----------------------------------------------------------------

RESET_SEQUENCES_SQL = """
SELECT setval('statements_id_seq',       (SELECT MAX(id) FROM statements));
SELECT setval('transactions_raw_id_seq', (SELECT MAX(id) FROM transactions_raw));
SELECT setval('transactions_id_seq',     (SELECT MAX(id) FROM transactions));
"""


# ----------------------------------------------------------------
# Verification query
# ----------------------------------------------------------------

def verify(client: Client):
    print("\n  Verifying row counts...")
    for table in TABLES:
        result = client.table(table).select("id", count="exact").execute()
        count = result.count
        print(f"    {table:<25} {count:>6,} rows")


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------

def main():
    print("=" * 60)
    print("STEP 3: CSV → Supabase import")
    print("=" * 60)
    print()

    client = get_supabase_client()
    print("✓  Connected to Supabase\n")

    total_imported = 0

    for table in TABLES:
        csv_path = CSV_DIR / f"{table}.csv"
        count = import_table(client, table, csv_path)
        total_imported += count
        print(f"  ✓  {table}: {count:,} rows imported\n")

    verify(client)

    print()
    print("=" * 60)
    print(f"  Total imported: {total_imported:,} rows")
    print()
    print("  IMPORTANT — run this in Supabase SQL Editor now:")
    print("  (resets auto-increment so new rows don't conflict)")
    print()
    print(RESET_SEQUENCES_SQL)
    print("=" * 60)


if __name__ == "__main__":
    main()