#!/usr/bin/env python3
"""
Ingest DCU bank statement PDFs via dual-write (SQLite + Supabase).

Follows the same dual-write contract as DualWriteManager:
  - WRITES → SQLite first (savepoint), then Supabase. Rolls back SQLite
    if Supabase fails.
  - Duplicate detection via SHA-256 file_hash (UNIQUE on bank_statements).
  - Re-running is safe (idempotent via upsert + conflict detection).

Tables written:
  - bank_statements           (one row per PDF, conflict on file_hash)
  - bank_transactions_raw     (checking + savings, bulk upsert)
  - loan_transactions     (loan payments, bulk upsert)

Usage:
  python ingest_bank_statements.py                            # dual-write (default)
  python ingest_bank_statements.py --dry-run                  # parse only, no DB writes
  python ingest_bank_statements.py --sqlite-only              # skip Supabase
  python ingest_bank_statements.py --verbose                  # per-file details
  python ingest_bank_statements.py --source /path/to/pdfs     # custom source
  python ingest_bank_statements.py --db /path/to/db           # custom DB
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.parsers.dcu_parser import DCUParser
from src.database.migrate_bank_tables import SQLITE_DDL


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_SOURCE = Path("/Users/sivakumarprabhakaran/checking_savings")
DEFAULT_DB = Path.home() / "sqlLite_DB" / "bank_data.db"
CHUNK_SIZE = 500  # Supabase bulk upsert limit


# ---------------------------------------------------------------------------
# Credentials — same Keychain pattern as DualWriteManager
# ---------------------------------------------------------------------------

def _keychain_get(service: str, account: str) -> str:
    """Read a credential from macOS Keychain."""
    result = subprocess.run(
        ["security", "find-generic-password",
         "-s", service, "-a", account, "-w"],
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
# Type coercion — matches DualWriteManager._coerce()
# ---------------------------------------------------------------------------

_NUMERIC_COLS = {
    "total_debits", "total_credits", "amount", "balance",
    "confidence_score", "payment_amount", "principal_amount",
    "interest_amount",
}
_INTEGER_COLS = {
    "id", "statement_id", "raw_id", "total_transactions",
}


def _coerce(row: dict) -> dict:
    """Ensure Python types are correct for Supabase REST API."""
    out = {}
    for k, v in row.items():
        if v is None:
            out[k] = None
        elif k in _INTEGER_COLS:
            out[k] = int(v)
        elif k in _NUMERIC_COLS:
            out[k] = float(v)
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------

def ensure_tables(conn: sqlite3.Connection) -> None:
    """Create bank tables if they don't exist (idempotent)."""
    conn.executescript(SQLITE_DDL)
    conn.commit()


@contextmanager
def _sqlite_savepoint(conn: sqlite3.Connection, name: str = "sp_bank_ingest"):
    """
    SQLite savepoint = nested transaction.
    Rolls back only this operation if Supabase fails,
    leaving any outer transaction intact.
    Same pattern as DualWriteManager._sqlite_savepoint().
    """
    conn.execute(f"SAVEPOINT {name}")
    try:
        yield name
        conn.execute(f"RELEASE {name}")
        conn.commit()
    except Exception:
        conn.execute(f"ROLLBACK TO {name}")
        conn.execute(f"RELEASE {name}")
        raise


def _sqlite_upsert(conn: sqlite3.Connection, table: str, row: dict) -> None:
    """INSERT OR REPLACE a single row into SQLite."""
    cols = list(row.keys())
    placeholders = ", ".join("?" * len(cols))
    col_names = ", ".join(cols)
    values = [row[c] for c in cols]
    conn.execute(
        f"INSERT OR REPLACE INTO {table} ({col_names}) VALUES ({placeholders})",
        values,
    )


def _sqlite_insert(conn: sqlite3.Connection, table: str, row: dict) -> int:
    """INSERT a single row. Returns lastrowid."""
    cols = list(row.keys())
    placeholders = ", ".join("?" * len(cols))
    col_names = ", ".join(cols)
    values = [row[c] for c in cols]
    cursor = conn.execute(
        f"INSERT INTO {table} ({col_names}) VALUES ({placeholders})",
        values,
    )
    return cursor.lastrowid


# ---------------------------------------------------------------------------
# Dual-write: bank_statements
# ---------------------------------------------------------------------------

def upsert_bank_statement(
    conn: sqlite3.Connection,
    supa,
    metadata: dict,
    sqlite_only: bool = False,
) -> tuple[int, int | None] | None:
    """
    Dual-write a bank_statements row.

    Returns:
        (sqlite_id, supa_id) on success, None if duplicate (file_hash conflict).
        supa_id is None when sqlite_only=True.

    SQLite and Supabase use independent AUTOINCREMENT / BIGSERIAL IDs.
    Transaction rows must use the correct ID for each target.
    """
    existing = conn.execute(
        "SELECT id FROM bank_statements WHERE file_hash = ?",
        (metadata["file_hash"],),
    ).fetchone()
    if existing:
        return None

    supa_id = None

    with _sqlite_savepoint(conn):
        sqlite_id = _sqlite_insert(conn, "bank_statements", metadata)

        if not sqlite_only and supa:
            try:
                result = (
                    supa.table("bank_statements")
                    .upsert(_coerce(metadata), on_conflict="file_hash")
                    .execute()
                )
                if not result.data:
                    raise RuntimeError(
                        "Supabase upsert returned no data for bank_statements"
                    )
                supa_id = result.data[0].get("id")
                log.debug(
                    "Supabase bank_statements OK: %s (sqlite_id=%d, supa_id=%s)",
                    metadata["filename"], sqlite_id, supa_id,
                )
            except Exception as e:
                log.error(
                    "Supabase failed for bank_statements — "
                    "rolling back SQLite: %s", e
                )
                raise

    return (sqlite_id, supa_id)


# ---------------------------------------------------------------------------
# Dual-write: bank_transactions_raw (bulk)
# ---------------------------------------------------------------------------

def upsert_bank_transactions_bulk(
    conn: sqlite3.Connection,
    supa,
    sqlite_stmt_id: int,
    supa_stmt_id: int | None,
    rows: list[dict],
    sqlite_only: bool = False,
) -> int:
    """
    Bulk dual-write bank_transactions_raw rows.
    Uses sqlite_stmt_id for SQLite and supa_stmt_id for Supabase.
    Returns count inserted.
    """
    if not rows:
        return 0

    sqlite_rows = [{**row, "statement_id": sqlite_stmt_id} for row in rows]

    with _sqlite_savepoint(conn):
        for row in sqlite_rows:
            _sqlite_upsert(conn, "bank_transactions_raw", row)

        if not sqlite_only and supa and supa_stmt_id is not None:
            try:
                supa_rows = [{**row, "statement_id": supa_stmt_id} for row in rows]
                total = 0
                for i in range(0, len(supa_rows), CHUNK_SIZE):
                    chunk = [_coerce(r) for r in supa_rows[i : i + CHUNK_SIZE]]
                    result = (
                        supa.table("bank_transactions_raw")
                        .insert(chunk)
                        .execute()
                    )
                    total += len(result.data or [])
                log.debug("Supabase bank_transactions_raw: %d rows", total)
            except Exception as e:
                log.error(
                    "Supabase bulk failed for bank_transactions_raw — "
                    "rolling back SQLite: %s", e
                )
                raise

    return len(sqlite_rows)


# ---------------------------------------------------------------------------
# Dual-write: loan_transactions (bulk)
# ---------------------------------------------------------------------------

def upsert_loan_transactions_bulk(
    conn: sqlite3.Connection,
    supa,
    sqlite_stmt_id: int,
    supa_stmt_id: int | None,
    rows: list[dict],
    sqlite_only: bool = False,
) -> int:
    """
    Bulk dual-write loan_transactions rows.
    Uses sqlite_stmt_id for SQLite and supa_stmt_id for Supabase.
    Returns count inserted.
    """
    if not rows:
        return 0

    sqlite_rows = [{**row, "statement_id": sqlite_stmt_id} for row in rows]

    with _sqlite_savepoint(conn):
        for row in sqlite_rows:
            _sqlite_upsert(conn, "loan_transactions", row)

        if not sqlite_only and supa and supa_stmt_id is not None:
            try:
                supa_rows = [{**row, "statement_id": supa_stmt_id} for row in rows]
                total = 0
                for i in range(0, len(supa_rows), CHUNK_SIZE):
                    chunk = [_coerce(r) for r in supa_rows[i : i + CHUNK_SIZE]]
                    result = (
                        supa.table("loan_transactions")
                        .insert(chunk)
                        .execute()
                    )
                    total += len(result.data or [])
                log.debug("Supabase loan_transactions: %d rows", total)
            except Exception as e:
                log.error(
                    "Supabase bulk failed for loan_transactions — "
                    "rolling back SQLite: %s", e
                )
                raise

    return len(sqlite_rows)


# ---------------------------------------------------------------------------
# Main ingestion loop
# ---------------------------------------------------------------------------

def ingest_directory(
    source_dir: Path,
    db_path: Path,
    dry_run: bool = False,
    sqlite_only: bool = False,
    verbose: bool = False,
) -> dict:
    """
    Parse all PDFs in source_dir and dual-write to SQLite + Supabase.
    Returns a summary dict.
    """
    if not source_dir.exists():
        print(f"❌ Source directory not found: {source_dir}")
        sys.exit(1)

    pdfs = sorted(source_dir.glob("*.pdf"))
    if not pdfs:
        print(f"❌ No PDF files found in: {source_dir}")
        sys.exit(1)

    parser = DCUParser()
    stats = {
        "total_files": len(pdfs),
        "imported": 0,
        "skipped_duplicate": 0,
        "skipped_error": 0,
        "bank_transactions": 0,
        "loan_transactions": 0,
    }

    conn = None
    supa = None

    if not dry_run:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        ensure_tables(conn)

        if not sqlite_only:
            try:
                supa = _get_supabase_client()
                print("  ✓ Supabase connected")
            except Exception as e:
                print(f"  ⚠ Supabase unavailable ({e}), falling back to SQLite-only")
                sqlite_only = True

    print(f"\nScanning: {source_dir}")
    print(f"Found:    {len(pdfs)} PDF files")
    if dry_run:
        print("Mode:     DRY RUN (parse only)")
    elif sqlite_only:
        print("Mode:     SQLite only")
    else:
        print("Mode:     Dual write (SQLite + Supabase)")
    print()

    t0 = time.time()

    for i, pdf_path in enumerate(pdfs, 1):
        prefix = f"[{i}/{len(pdfs)}]"

        try:
            result = parser.parse(pdf_path)
            period = (
                f"{result.metadata.get('statement_period_start', '?')} → "
                f"{result.metadata.get('statement_period_end', '?')}"
            )

            if dry_run:
                print(
                    f"  {prefix} {pdf_path.name}: "
                    f"{result.total_bank_transactions} bank + "
                    f"{result.total_loan_transactions} loan | "
                    f"{period}"
                )
                stats["imported"] += 1
                stats["bank_transactions"] += result.total_bank_transactions
                stats["loan_transactions"] += result.total_loan_transactions
                continue

            # Dual-write statement
            ids = upsert_bank_statement(
                conn, supa, result.metadata, sqlite_only=sqlite_only,
            )
            if ids is None:
                if verbose:
                    print(f"  {prefix} {pdf_path.name}: skipped (duplicate)")
                stats["skipped_duplicate"] += 1
                continue

            sqlite_stmt_id, supa_stmt_id = ids

            # Dual-write bank transactions
            bank_count = upsert_bank_transactions_bulk(
                conn, supa, sqlite_stmt_id, supa_stmt_id, result.bank_rows,
                sqlite_only=sqlite_only,
            )

            # Dual-write loan transactions
            loan_count = upsert_loan_transactions_bulk(
                conn, supa, sqlite_stmt_id, supa_stmt_id, result.loan_rows,
                sqlite_only=sqlite_only,
            )

            stats["imported"] += 1
            stats["bank_transactions"] += bank_count
            stats["loan_transactions"] += loan_count

            if verbose:
                print(
                    f"  {prefix} {pdf_path.name}: "
                    f"{bank_count} bank + {loan_count} loan | "
                    f"{period}"
                )

        except Exception as e:
            print(f"  {prefix} ❌ {pdf_path.name}: {e}")
            stats["skipped_error"] += 1

    elapsed = time.time() - t0
    if conn:
        conn.close()

    stats["elapsed_seconds"] = round(elapsed, 2)
    return stats


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(stats: dict) -> None:
    """Print ingestion summary."""
    print()
    print("=" * 60)
    print("INGESTION SUMMARY")
    print("=" * 60)
    print(f"  Files scanned:       {stats['total_files']}")
    print(f"  Files imported:      {stats['imported']}")
    print(f"  Skipped (duplicate): {stats['skipped_duplicate']}")
    print(f"  Skipped (error):     {stats['skipped_error']}")
    print(f"  Bank transactions:   {stats['bank_transactions']}")
    print(f"  Loan transactions:   {stats['loan_transactions']}")
    print(f"  Elapsed:             {stats['elapsed_seconds']}s")
    print("=" * 60)


def print_db_summary(db_path: Path) -> None:
    """Print DB contents after ingestion."""
    if not db_path.exists():
        return

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        print(f"\n  DB totals ({db_path.name}):")
        for table in [
            "bank_statements", "bank_transactions_raw", "loan_transactions",
        ]:
            n = conn.execute(f"SELECT COUNT(*) as n FROM {table}").fetchone()["n"]
            print(f"    {table}: {n} rows")

        # Account type breakdown
        rows = conn.execute("""
            SELECT account_type, COUNT(*) as cnt,
                   ROUND(SUM(amount), 2) as total
            FROM bank_transactions_raw GROUP BY account_type
        """).fetchall()
        if rows:
            print(f"\n  By account type:")
            for r in rows:
                print(f"    {r['account_type']}: {r['cnt']} txns, ${r['total']:,.2f}")

        # Bank breakdown
        rows = conn.execute("""
            SELECT bank_name, COUNT(*) as cnt
            FROM bank_statements GROUP BY bank_name
        """).fetchall()
        if rows:
            print(f"\n  By bank:")
            for r in rows:
                print(f"    {r['bank_name']}: {r['cnt']} statements")

        # Date range
        r = conn.execute("""
            SELECT MIN(date) as earliest, MAX(date) as latest
            FROM bank_transactions_raw
        """).fetchone()
        if r and r["earliest"]:
            print(f"\n  Date range: {r['earliest']} → {r['latest']}")

    finally:
        conn.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Ingest DCU bank statements (dual-write: SQLite + Supabase)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ingest_bank_statements.py                          # dual-write
  python ingest_bank_statements.py --dry-run                # parse only
  python ingest_bank_statements.py --sqlite-only            # skip Supabase
  python ingest_bank_statements.py --verbose                # per-file details
  python ingest_bank_statements.py --source /other/pdfs     # custom source
        """,
    )
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                     help=f"PDF source directory (default: {DEFAULT_SOURCE})")
    ap.add_argument("--db", type=Path, default=DEFAULT_DB,
                     help=f"SQLite database path (default: {DEFAULT_DB})")
    ap.add_argument("--dry-run", action="store_true",
                     help="Parse only, no DB writes")
    ap.add_argument("--sqlite-only", action="store_true",
                     help="Write to SQLite only, skip Supabase")
    ap.add_argument("--verbose", "-v", action="store_true",
                     help="Show per-file details")

    args = ap.parse_args()

    print("=" * 60)
    print("INGEST: DCU Bank Statements (Dual Write)")
    print("=" * 60)
    print(f"  Source: {args.source}")
    print(f"  DB:     {args.db}")

    stats = ingest_directory(
        source_dir=args.source,
        db_path=args.db,
        dry_run=args.dry_run,
        sqlite_only=args.sqlite_only,
        verbose=args.verbose,
    )

    print_summary(stats)
    if not args.dry_run:
        print_db_summary(args.db)
    print()


if __name__ == "__main__":
    main()