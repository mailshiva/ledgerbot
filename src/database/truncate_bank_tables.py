#!/usr/bin/env python3
"""
src/database/truncate_bank_tables.py

Delete all rows from the five bank statement tables in SQLite and/or Supabase.

Tables cleared (in dependency order — children before parent):
  1. bank_transactions        (enriched checking/savings; FK → bank_transactions_raw)
  2. bank_transactions_raw    (raw checking/savings;      FK → bank_statements)
  3. loan_transactions        (loan payments;             FK → bank_statements)
  4. bank_statements          (statement metadata; parent of all above)

AUTOINCREMENT sequences are also reset so the next ingest starts from id=1.

Usage:
  python truncate_bank_tables.py                  # dual truncate (SQLite + Supabase)
  python truncate_bank_tables.py --dry-run        # show row counts, make no changes
  python truncate_bank_tables.py --sqlite-only    # skip Supabase
  python truncate_bank_tables.py --supabase-only  # skip SQLite
  python truncate_bank_tables.py --force          # skip confirmation prompt
  python truncate_bank_tables.py --db /path/to.db # custom SQLite path

WARNING: This is a destructive, irreversible operation on live data.
         Always verify with --dry-run first.
"""

from __future__ import annotations

import argparse
import sqlite3
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ---------------------------------------------------------------------------
# Tables — ordered so FK children are deleted before FK parents
# ---------------------------------------------------------------------------

TABLES = [
    "bank_transactions",      # enriched; FK → bank_transactions_raw
    "bank_transactions_raw",  # raw;      FK → bank_statements
    "loan_transactions",      # loan payments; FK → bank_statements
    "bank_statements",        # parent
]

DEFAULT_DB = Path.home() / "sqlLite_DB" / "bank_data.db"


# ---------------------------------------------------------------------------
# Keychain / Supabase helpers (same pattern as ingest_bank_statements.py)
# ---------------------------------------------------------------------------

def _keychain_get(service: str, account: str) -> str:
    result = subprocess.run(
        ["security", "find-generic-password", "-s", service, "-a", account, "-w"],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def _get_supabase_client():
    """Create Supabase client using Keychain credentials."""
    from supabase import create_client
    url = _keychain_get("supabase_bank", "url")
    key = _keychain_get("supabase_bank", "anon_key")
    return create_client(url, key)


# ---------------------------------------------------------------------------
# Row-count helpers
# ---------------------------------------------------------------------------

def _sqlite_counts(conn: sqlite3.Connection) -> dict[str, int]:
    counts = {}
    for table in TABLES:
        try:
            row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
            counts[table] = row[0]
        except sqlite3.OperationalError:
            counts[table] = -1  # table doesn't exist yet
    return counts


def _supabase_counts(supa) -> dict[str, int]:
    counts = {}
    for table in TABLES:
        try:
            result = supa.table(table).select("id", count="exact").limit(0).execute()
            counts[table] = result.count or 0
        except Exception as e:
            counts[table] = -1
    return counts


# ---------------------------------------------------------------------------
# Truncate helpers
# ---------------------------------------------------------------------------

def _truncate_sqlite(conn: sqlite3.Connection) -> dict[str, int]:
    """Delete all rows from each table and reset AUTOINCREMENT sequences."""
    deleted = {}
    conn.execute("PRAGMA foreign_keys = OFF")
    try:
        for table in TABLES:
            try:
                cursor = conn.execute(f"DELETE FROM {table}")
                deleted[table] = cursor.rowcount
                # Reset SQLite AUTOINCREMENT counter
                conn.execute(
                    "DELETE FROM sqlite_sequence WHERE name = ?", (table,)
                )
            except sqlite3.OperationalError as e:
                print(f"  ⚠ SQLite: could not clear {table}: {e}")
                deleted[table] = 0
        conn.commit()
    finally:
        conn.execute("PRAGMA foreign_keys = ON")
    return deleted


def _truncate_supabase(supa) -> dict[str, int]:
    """
    Truncate all Supabase tables with RESTART IDENTITY CASCADE so that
    BIGSERIAL sequences reset to 1, keeping IDs in sync with SQLite.

    Uses Supabase's rpc() to execute raw SQL — requires the anon role to
    have EXECUTE on the helper function, or service_role_key.

    Table order (children first) is enforced by CASCADE, but we list them
    explicitly so the counts are reported per-table.
    """
    # Single TRUNCATE with RESTART IDENTITY resets all sequences atomically.
    truncate_sql = (
        "TRUNCATE "
        + ", ".join(TABLES)
        + " RESTART IDENTITY CASCADE;"
    )
    counts_before = {}
    for table in TABLES:
        try:
            result = supa.table(table).select("id", count="exact").limit(0).execute()
            counts_before[table] = result.count or 0
        except Exception:
            counts_before[table] = 0

    try:
        supa.rpc("exec_sql", {"query": truncate_sql}).execute()
    except Exception:
        # rpc exec_sql may not exist — fall back to per-table DELETE
        # (sequences will NOT reset; warn the user)
        print(
            "  ⚠ Supabase: rpc exec_sql unavailable — falling back to DELETE.\n"
            "    Sequences will NOT reset. Run this in the Supabase SQL editor\n"
            "    to reset sequences:\n\n"
            f"    TRUNCATE {', '.join(TABLES)} RESTART IDENTITY CASCADE;\n"
        )
        for table in TABLES:
            try:
                supa.table(table).delete().gt("id", 0).execute()
            except Exception as e:
                print(f"  ⚠ Supabase: could not clear {table}: {e}")

    return counts_before


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _print_counts(label: str, counts: dict[str, int]) -> None:
    print(f"\n  {label}:")
    for table, n in counts.items():
        if n == -1:
            print(f"    {table:<30}  (table not found)")
        else:
            print(f"    {table:<30}  {n:>6} rows")


def _print_deleted(label: str, deleted: dict[str, int]) -> None:
    print(f"\n  {label}:")
    for table, n in deleted.items():
        print(f"    {table:<30}  {n:>6} rows deleted")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def truncate(
    db_path: Path,
    dry_run: bool = False,
    sqlite_only: bool = False,
    supabase_only: bool = False,
    force: bool = False,
) -> None:
    print("=" * 60)
    print("TRUNCATE: Bank Statement Tables")
    print("=" * 60)

    if not supabase_only:
        print(f"  SQLite:   {db_path}")
    if not sqlite_only:
        print(f"  Supabase: enabled")

    # ── Connect ──────────────────────────────────────────────────────────────
    conn = None
    supa = None

    if not supabase_only:
        if not db_path.exists():
            print(f"\n  ⚠ SQLite DB not found: {db_path}")
            if not sqlite_only:
                pass  # still try Supabase
        else:
            conn = sqlite3.connect(str(db_path))

    if not sqlite_only:
        try:
            supa = _get_supabase_client()
            print("  ✓ Supabase connected")
        except Exception as e:
            print(f"  ⚠ Supabase unavailable: {e}")
            if supabase_only:
                sys.exit(1)

    # ── Show current counts ───────────────────────────────────────────────────
    print("\n--- Current row counts ---")
    if conn:
        _print_counts("SQLite", _sqlite_counts(conn))
    if supa:
        _print_counts("Supabase", _supabase_counts(supa))

    if dry_run:
        print("\n  DRY RUN — no changes made.")
        if conn:
            conn.close()
        return

    # ── Confirm ───────────────────────────────────────────────────────────────
    if not force:
        print()
        answer = input(
            "  This will permanently delete ALL rows. Type YES to confirm: "
        ).strip()
        if answer != "YES":
            print("  Aborted.")
            if conn:
                conn.close()
            sys.exit(0)

    # ── Execute ───────────────────────────────────────────────────────────────
    print("\n--- Deleting rows ---")

    if conn:
        sqlite_deleted = _truncate_sqlite(conn)
        _print_deleted("SQLite", sqlite_deleted)
        conn.close()

    if supa:
        supa_counts = _truncate_supabase(supa)
        _print_deleted("Supabase (rows before truncate)", supa_counts)

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Delete all rows from bank statement tables (SQLite + Supabase)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python truncate_bank_tables.py --dry-run          # preview row counts, no changes
  python truncate_bank_tables.py                    # dual truncate with confirmation
  python truncate_bank_tables.py --force            # skip confirmation prompt
  python truncate_bank_tables.py --sqlite-only      # SQLite only
  python truncate_bank_tables.py --supabase-only    # Supabase only
  python truncate_bank_tables.py --db /tmp/test.db  # custom SQLite path
        """,
    )
    ap.add_argument("--db", type=Path, default=DEFAULT_DB,
                    help=f"SQLite database path (default: {DEFAULT_DB})")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show row counts only, make no changes")
    ap.add_argument("--sqlite-only", action="store_true",
                    help="Clear SQLite only, skip Supabase")
    ap.add_argument("--supabase-only", action="store_true",
                    help="Clear Supabase only, skip SQLite")
    ap.add_argument("--force", action="store_true",
                    help="Skip the confirmation prompt")

    args = ap.parse_args()

    if args.sqlite_only and args.supabase_only:
        print("Error: --sqlite-only and --supabase-only are mutually exclusive.")
        sys.exit(1)

    truncate(
        db_path=args.db,
        dry_run=args.dry_run,
        sqlite_only=args.sqlite_only,
        supabase_only=args.supabase_only,
        force=args.force,
    )


if __name__ == "__main__":
    main()