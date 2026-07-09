#!/usr/bin/env python3
"""
scripts/reingest_boa.py

Delete all BOA bank statement data from both SQLite and Supabase, then
re-run ingestion and enrichment so the fixed balance column is populated.

Steps:
  1. Connect to SQLite and Supabase.
  2. Collect all BOA statement IDs (SQLite) and matching Supabase IDs (by file_hash).
  3. Delete from Supabase:  bank_transactions → bank_transactions_raw → bank_statements
  4. Delete from SQLite:    bank_transactions → bank_transactions_raw → bank_statements
  5. Re-run: python src/database/ingest_bank_statements.py
  6. Re-run: python -m src.nlp.bank_enrichment_pipeline

Usage:
  python scripts/reingest_boa.py              # full dual-write cleanup + re-ingest
  python scripts/reingest_boa.py --dry-run    # show what would be deleted, no changes
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DEFAULT_DB = Path.home() / "sqlLite_DB" / "bank_data.db"


# ---------------------------------------------------------------------------
# Supabase client
# ---------------------------------------------------------------------------

def _keychain_get(service: str, account: str) -> str:
    result = subprocess.run(
        ["security", "find-generic-password", "-s", service, "-a", account, "-w"],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def _get_supabase_client():
    from supabase import create_client
    url = _keychain_get("supabase_bank", "url")
    key = _keychain_get("supabase_bank", "anon_key")
    return create_client(url, key)


# ---------------------------------------------------------------------------
# Deletion helpers
# ---------------------------------------------------------------------------

def _supa_delete_in_chunks(supa, table: str, column: str, ids: list[int], dry_run: bool) -> int:
    """Delete rows from Supabase where column IN ids (chunked to avoid URL limits)."""
    if not ids:
        return 0
    CHUNK = 100
    total = 0
    for i in range(0, len(ids), CHUNK):
        chunk = ids[i:i + CHUNK]
        if dry_run:
            log.info("[DRY-RUN] Would DELETE from supa.%s WHERE %s IN %s", table, column, chunk)
            total += len(chunk)
        else:
            result = supa.table(table).delete().in_(column, chunk).execute()
            deleted = len(result.data or [])
            total += deleted
            log.info("Supabase DELETE %s WHERE %s IN chunk[%d]: %d rows", table, column, i // CHUNK, deleted)
    return total


def _sqlite_delete(conn: sqlite3.Connection, table: str, column: str, ids: list[int], dry_run: bool) -> int:
    """Delete rows from SQLite where column IN ids."""
    if not ids:
        return 0
    placeholders = ",".join("?" * len(ids))
    sql = f"DELETE FROM {table} WHERE {column} IN ({placeholders})"
    if dry_run:
        log.info("[DRY-RUN] Would: %s  params=%s", sql, ids)
        return len(ids)
    cursor = conn.execute(sql, ids)
    log.info("SQLite DELETE %s WHERE %s IN (%d ids): %d rows affected", table, column, len(ids), cursor.rowcount)
    return cursor.rowcount


# ---------------------------------------------------------------------------
# Main cleanup
# ---------------------------------------------------------------------------

def cleanup(conn: sqlite3.Connection, supa, dry_run: bool) -> None:
    # ── Collect SQLite IDs ──────────────────────────────────────────────────
    stmt_rows = conn.execute(
        "SELECT id, file_hash, filename FROM bank_statements WHERE bank_name = 'boa'"
    ).fetchall()

    if not stmt_rows:
        log.info("No BOA statements found in SQLite — nothing to delete.")
        return

    sqlite_stmt_ids = [r[0] for r in stmt_rows]
    file_hashes     = [r[1] for r in stmt_rows]

    log.info("Found %d BOA statements in SQLite: IDs %s..%s",
             len(sqlite_stmt_ids), sqlite_stmt_ids[0], sqlite_stmt_ids[-1])

    # Raw transaction IDs in SQLite
    sqlite_raw_ids: list[int] = []
    for sid in sqlite_stmt_ids:
        rows = conn.execute(
            "SELECT id FROM bank_transactions_raw WHERE statement_id = ?", (sid,)
        ).fetchall()
        sqlite_raw_ids.extend(r[0] for r in rows)

    log.info("Found %d BOA raw transactions in SQLite", len(sqlite_raw_ids))

    # Enriched transaction IDs in SQLite (raw_id FK)
    sqlite_enriched_ids: list[int] = []
    if sqlite_raw_ids:
        placeholders = ",".join("?" * len(sqlite_raw_ids))
        rows = conn.execute(
            f"SELECT id FROM bank_transactions WHERE raw_id IN ({placeholders})",
            sqlite_raw_ids,
        ).fetchall()
        sqlite_enriched_ids = [r[0] for r in rows]

    log.info("Found %d BOA enriched transactions in SQLite", len(sqlite_enriched_ids))

    # ── Collect Supabase IDs ────────────────────────────────────────────────
    log.info("Fetching matching statement IDs from Supabase by file_hash ...")
    supa_stmt_ids: list[int] = []
    CHUNK = 50
    for i in range(0, len(file_hashes), CHUNK):
        chunk = file_hashes[i:i + CHUNK]
        result = supa.table("bank_statements").select("id").in_("file_hash", chunk).execute()
        supa_stmt_ids.extend(r["id"] for r in (result.data or []))

    log.info("Found %d BOA statements in Supabase", len(supa_stmt_ids))

    supa_raw_ids: list[int] = []
    for i in range(0, len(supa_stmt_ids), CHUNK):
        chunk = supa_stmt_ids[i:i + CHUNK]
        result = supa.table("bank_transactions_raw").select("id").in_("statement_id", chunk).execute()
        supa_raw_ids.extend(r["id"] for r in (result.data or []))

    log.info("Found %d BOA raw transactions in Supabase", len(supa_raw_ids))

    # ── Delete from Supabase (child → parent order) ─────────────────────────
    log.info("--- Deleting from Supabase ---")
    _supa_delete_in_chunks(supa, "bank_transactions",     "raw_id",       supa_raw_ids,  dry_run)
    _supa_delete_in_chunks(supa, "bank_transactions_raw", "statement_id", supa_stmt_ids, dry_run)
    _supa_delete_in_chunks(supa, "bank_statements",       "id",           supa_stmt_ids, dry_run)

    # ── Delete from SQLite (child → parent order) ────────────────────────────
    log.info("--- Deleting from SQLite ---")
    _sqlite_delete(conn, "bank_transactions",     "raw_id",       sqlite_raw_ids,  dry_run)
    _sqlite_delete(conn, "bank_transactions_raw", "statement_id", sqlite_stmt_ids, dry_run)
    _sqlite_delete(conn, "bank_statements",       "id",           sqlite_stmt_ids, dry_run)

    if not dry_run:
        conn.commit()
        log.info("SQLite changes committed.")


# ---------------------------------------------------------------------------
# Re-ingest + re-enrich
# ---------------------------------------------------------------------------

def run_step(cmd: list[str], label: str) -> None:
    log.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        log.error("%s failed with exit code %d", label, result.returncode)
        sys.exit(result.returncode)
    log.info("%s completed successfully.", label)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Delete BOA data and re-ingest with fixed balance parsing.")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted without making changes.")
    parser.add_argument("--db", default=str(DEFAULT_DB), help="Path to SQLite database.")
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)

    log.info("Connecting to Supabase ...")
    supa = _get_supabase_client()

    log.info("=== Step 1: Cleanup BOA data %s===", "(DRY-RUN) " if args.dry_run else "")
    cleanup(conn, supa, dry_run=args.dry_run)
    conn.close()

    if args.dry_run:
        log.info("Dry-run complete — no data was changed. Re-run without --dry-run to apply.")
        return

    log.info("=== Step 2: Re-ingest BOA PDFs (dual-write) ===")
    run_step(
        [sys.executable, "src/database/ingest_bank_statements.py", "--verbose"],
        "Ingestion",
    )

    log.info("=== Step 3: Re-enrich BOA transactions (dual-write) ===")
    run_step(
        [sys.executable, "-m", "src.nlp.bank_enrichment_pipeline", "--reprocess"],
        "Enrichment",
    )

    log.info("=== All done! BOA data re-ingested with balance column populated. ===")


if __name__ == "__main__":
    main()
