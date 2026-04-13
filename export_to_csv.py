#!/usr/bin/env python3
"""
STEP 2: Export SQLite → CSV for Supabase import.

Run from your project root:
    python export_to_csv.py

Exports 3 tables in correct FK order:
    1. statements
    2. transactions_raw
    3. transactions

Output: /tmp/bank_export/
    statements.csv
    transactions_raw.csv
    transactions.csv
"""

import sqlite3
import csv
import sys
from pathlib import Path
from datetime import datetime

# ----------------------------------------------------------------
# Config
# ----------------------------------------------------------------
DB_PATH  = Path.home() / "sqlLite_DB" / "bank_data.db"
OUT_DIR  = Path("/tmp/bank_export")

# FK order matters — parent tables first
TABLES = [
    "statements",
    "transactions_raw",
    "transactions",
]

# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------

def check_db_exists():
    if not DB_PATH.exists():
        print(f"❌  Database not found: {DB_PATH}")
        sys.exit(1)
    print(f"✓  Database: {DB_PATH}")
    print(f"   Size: {DB_PATH.stat().st_size / 1024:.1f} KB\n")


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    return row is not None


def export_table(conn: sqlite3.Connection, table: str, out_path: Path) -> int:
    """
    Export one table to CSV.
    Returns row count exported.

    Date fields (TEXT in SQLite) are normalized to ISO format YYYY-MM-DD
    so Postgres DATE columns accept them without complaint.
    """
    rows = conn.execute(f"SELECT * FROM {table}").fetchall()

    if not rows:
        print(f"  ⚠️   {table}: 0 rows — writing empty CSV (headers only)")
        # Still write headers so Supabase import doesn't fail
        cols = [desc[0] for desc in conn.execute(f"PRAGMA table_info({table})").fetchall()]
        # PRAGMA table_info returns: cid, name, type, notnull, dflt_value, pk
        col_names = [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(col_names)
        return 0

    col_names = list(rows[0].keys())

    # Identify date-like columns to normalize
    date_cols = {c for c in col_names if c in ("date", "statement_date")}

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=col_names)
        writer.writeheader()

        for row in rows:
            d = dict(row)

            # Normalize date TEXT → YYYY-MM-DD (Postgres DATE format)
            for col in date_cols:
                val = d.get(col)
                if val and isinstance(val, str):
                    # Handle common SQLite date formats
                    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y%m%d"):
                        try:
                            d[col] = datetime.strptime(val, fmt).strftime("%Y-%m-%d")
                            break
                        except ValueError:
                            continue

            # Replace None with empty string (CSV-friendly)
            d = {k: ("" if v is None else v) for k, v in d.items()}
            writer.writerow(d)

    return len(rows)


def print_preview(path: Path, n: int = 3):
    """Print first n rows of a CSV for sanity check."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = [r for _, r in zip(range(n + 1), reader)]  # header + n rows
    for i, row in enumerate(rows):
        prefix = "  HDR" if i == 0 else f"  [{i}]"
        # Truncate long values for display
        display = [v[:30] + "…" if len(v) > 30 else v for v in row]
        print(f"  {prefix}  {', '.join(display)}")


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------

def main():
    print("=" * 60)
    print("STEP 2: SQLite → CSV export for Supabase")
    print("=" * 60)
    print()

    check_db_exists()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓  Output dir: {OUT_DIR}\n")

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    total_rows = 0
    results = []

    for table in TABLES:
        out_path = OUT_DIR / f"{table}.csv"

        if not table_exists(conn, table):
            print(f"  ⚠️   {table}: table not found in SQLite — skipping")
            results.append((table, 0, "SKIPPED"))
            continue

        print(f"  Exporting {table}...")
        count = export_table(conn, table, out_path)
        total_rows += count

        print(f"  ✓  {count:,} rows → {out_path}")
        if count > 0:
            print_preview(out_path)
        print()

        results.append((table, count, "OK"))

    conn.close()

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for table, count, status in results:
        icon = "✓" if status == "OK" else "⚠️"
        print(f"  {icon}  {table:<25} {count:>6,} rows   {out_path.parent / (table + '.csv')}")

    print(f"\n  Total rows exported: {total_rows:,}")
    print()
    print("NEXT STEP — import these 3 files into Supabase:")
    print("  1. Supabase dashboard → Table Editor → statements")
    print("  2. Click 'Import data' → upload /tmp/bank_export/statements.csv")
    print("  3. Repeat for transactions_raw.csv, then transactions.csv")
    print("  (Import in that order — foreign keys require parent rows first)")
    print()
    print(f"  Files are in: {OUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
