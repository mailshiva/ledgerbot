#!/usr/bin/env python3
"""
src/nlp/bank_enrichment_pipeline.py
────────────────────────────────────
Enriches bank_transactions_raw → bank_transactions using simple
rule-based categorization. No spaCy or RapidFuzz needed — bank
statement descriptions are predictable enough for keyword matching.

Enrichment produces:
  - clean_description:  Noise-stripped, title-cased description
  - category:           Rule-based category (Income, Utilities, Transfer, etc.)
  - confidence_score:   1.0 for keyword match, 0.3 for fallback
  - enrichment_method:  "rule_based" or "fallback"

Tables:
  - Reads from:  bank_transactions_raw (checking + savings)
  - Writes to:   bank_transactions (enriched)

Usage:
  python -m src.nlp.bank_enrichment_pipeline                    # all rows
  python -m src.nlp.bank_enrichment_pipeline --statement-id 3   # single statement
  python -m src.nlp.bank_enrichment_pipeline --reprocess        # force re-enrich
  python -m src.nlp.bank_enrichment_pipeline --sqlite-only      # skip Supabase
  python -m src.nlp.bank_enrichment_pipeline --account-type checking
"""

from __future__ import annotations

import argparse
import logging
import re
import sqlite3
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

try:
    from migrate_bank_tables import SQLITE_DDL
except ImportError:
    SQLITE_DDL = ""  # tests will inject DDL directly

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BATCH_SIZE = 200
CONFIDENCE_THRESHOLD = 0.0
DEFAULT_DB = Path.home() / "sqlLite_DB" / "bank_data.db"


# ---------------------------------------------------------------------------
# Rule-based categorization
# ---------------------------------------------------------------------------

# category → list of keyword patterns (matched against uppercased description)
BANK_CATEGORY_RULES: dict[str, list[str]] = {
    "Income": [
        "PAYROLL", "DIRECT DEP", "DIRECT DEPOSIT",
        "DIVIDEND", "INTEREST PAYMENT", "TAX REFUND",
    ],
    "Salary": ["TATA CONSULTANCYDIRECT","TATA CONSULTANCY DES:DIRECT", "TATA CONS SERVS", "SALARY"],
    "School Fee Reimbursement": ["TATA CONSULTANCYCORP PMT"],
    "ARHIPP": ["BKOFAMERICA MOBILE", "ARKANSAS HIPP"],
    "Investments": ["ROBINHOOD DES", "JM BULLION",],
    "India Transfers": ["WESTERN UNION DES", "CONTINENTAL EXC", "CROBO"],
    "Transfer": [
        "TRANSFER FROM", "TRANSFER TO", "XFER", "TFR",
        "ONLINE TRANSFER", "INTERNAL TRANSFER", "ZELLE",
        "REALTIME TRANSFER",        # Chase internal transfers
        "DIGITAL FEDERAL A2A",      # DCU → Chase A2A transfers
        "BANK OF AMERICA P2P",      # BOA → Chase P2P transfers
        "A2A XFER",
        "TRANSFER", "TRNSFR"
    ],
    "Utilities": [
        "EVERSOURCE", "ELECTRIC", "GAS BILL", "WATER BILL",
        "SEWER", "NATIONAL GRID", "UTILITY",
        "CITY OF", "BENTONVILLE",
    ],
    "Rent & Mortgage": [
        "RENT", "MORTGAGE", "HOUSING", "LEASE", "WALTONCROSSINGAP",
    ],
    "Insurance": [
        "INSURANCE", "GEICO", "STATEFARM", "STATE FARM",
        "ALLSTATE", "PROGRESSIVE", "LIBERTY MUTUAL", "TEFRA",
    ],
    "Telecom": [
        "ATT", "AT&T", "VERIZON", "T-MOBILE", "TMOBILE",
        "COMCAST", "XFINITY", "SPECTRUM", "INTERNET",
        "WIRELESS", "ULTRA MOBILE",
    ],
    "Loan Payment": [
        "LOAN PAYMENT", "LOAN PMT", "AUTO LOAN",
        "STUDENT LOAN", "CAR PAYMENT",
    ],
    "Credit Card Payment": ["CITI CARD", "ROBINHOOD CARD"],
    "ATM": [
        "ATM WITHDRAWAL", "ATM DEPOSIT", "ATM", "WITHDRWL"
    ],
    "Fee": [
        "SERVICE FEE", "MONTHLY FEE", "OVERDRAFT", "NSF FEE",
        "MAINTENANCE FEE", "ATM FEE", "BRIGHTWHEEL",
    ],
    "Groceries": [
        "STOP & SHOP", "STOP&SHOP", "WHOLE FOODS", "WALMART",
        "MARKET BASKET", "TRADER JOE", "ALDI", "COSTCO",
        "SAMS CLUB", "GROCERY", "TARGET DEBIT",
    ],
    "Shopping": [
        "AMAZON", "AMZN", "TARGET", "EBAY", "PAYPAL",
    ],
    "Food & Dining": [
        "RESTAURANT", "DUNKIN", "STARBUCKS", "MCDONALD",
        "CHIPOTLE", "GRUBHUB", "DOORDASH", "UBER EATS",
    ],
    "Healthcare": [
        "PHARMACY", "CVS", "WALGREENS", "DOCTOR", "MEDICAL",
        "HOSPITAL", "DENTAL", "HEALTH", "MANA"
    ],
    "Venmo / P2P": [
        "VENMO", "ZELLE", "CASH APP", "CASHAPP",
    ],
    "Check": [
        "CHECK #", "CHECK NO", "CHK",
    ],
    "Verida reimbursement": ["SOUTHEASTRANS"],
    "Tax": ["USATAX", "ARSTTAX"],
}


def categorize_description(description: str) -> tuple[str, float]:
    """
    Categorize a bank transaction description using keyword rules.

    Returns:
        (category, confidence)
    """
    desc_upper = description.upper()

    for category, keywords in BANK_CATEGORY_RULES.items():
        for keyword in keywords:
            if keyword in desc_upper:
                return category, 1.0

    return "Uncategorized", 0.3


def clean_description(raw: str) -> str:
    """
    Strip noise from bank description and title-case it.

    Removes:
      - Long digit strings (transaction IDs)
      - Phone numbers
      - Trailing state codes (2-letter)
      - Punctuation noise (*, #, @)
      - Extra whitespace
    """
    text = raw
    text = re.sub(r"\b\d{5,}\b", " ", text)           # long digit strings
    text = re.sub(r"\b\d{3}-\d{3}-\d{4}\b", " ", text)  # phone numbers
    text = re.sub(r"[*#@]", " ", text)                 # punctuation noise
    text = re.sub(r"\s{2,}", " ", text)                # extra whitespace
    text = text.strip()
    # Title-case but preserve common abbreviations
    if text.isupper() or text.islower():
        text = text.title()
    return text


# ---------------------------------------------------------------------------
# Supabase helpers
# ---------------------------------------------------------------------------

def _keychain_get(service: str, account: str) -> str:
    result = subprocess.run(
        ["security", "find-generic-password",
         "-s", service, "-a", account, "-w"],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def _get_supabase_client():
    from supabase import create_client
    url = _keychain_get("supabase_bank", "url")
    key = _keychain_get("supabase_bank", "anon_key")
    return create_client(url, key)


_NUMERIC_COLS = {"amount", "balance", "confidence_score"}
_INTEGER_COLS = {"id", "raw_id", "statement_id"}


def _coerce(row: dict) -> dict:
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
# Stats
# ---------------------------------------------------------------------------

@dataclass
class PipelineStats:
    total_raw: int = 0
    processed: int = 0
    skipped: int = 0
    errors: int = 0
    elapsed_seconds: float = 0.0

    def __str__(self) -> str:
        return (
            f"BankEnrichment: {self.elapsed_seconds:.1f}s | "
            f"raw={self.total_raw} processed={self.processed} "
            f"skipped={self.skipped} errors={self.errors}"
        )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class BankEnrichmentPipeline:
    """
    Enriches bank_transactions_raw → bank_transactions using
    rule-based keyword categorization.
    """

    def __init__(
        self,
        db_path: Path = DEFAULT_DB,
        batch_size: int = BATCH_SIZE,
        reprocess_all: bool = False,
        sqlite_only: bool = False,
        db: Optional[object] = None,
    ) -> None:
        self.batch_size = batch_size
        self.reprocess_all = reprocess_all
        self.sqlite_only = sqlite_only

        if db is not None:
            self._conn = db.sqlite.conn if hasattr(db, "sqlite") else db.conn
            self._supa = (
                db.supabase if hasattr(db, "supabase") and not sqlite_only else None
            )
            log.info("Using injected DB manager")
        else:
            self._conn = sqlite3.connect(str(db_path))
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._conn.execute("PRAGMA journal_mode = WAL")

            if not sqlite_only:
                try:
                    self._supa = _get_supabase_client()
                    log.info("Supabase connected for enrichment")
                except Exception as e:
                    log.warning("Supabase unavailable (%s), SQLite only", e)
                    self._supa = None
            else:
                self._supa = None

        # Ensure tables exist
        if SQLITE_DDL:
            self._conn.executescript(SQLITE_DDL)
            self._conn.commit()

        # Build SQLite→Supabase statement_id map (matched by file_hash).
        # SQLite and Supabase use independent AUTOINCREMENT/BIGSERIAL sequences
        # that can diverge after truncate/re-ingest cycles.  This map ensures
        # the correct Supabase FK is used when writing bank_transactions.
        self._stmt_id_map: dict[int, int] = {}   # {sqlite_id: supa_id}
        self._raw_id_map: dict[int, int] = {}     # {sqlite_raw_id: supa_raw_id}
        if self._supa:
            self._stmt_id_map = self._build_stmt_id_map()
            self._raw_id_map = self._build_raw_id_map()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(
        self,
        statement_id: Optional[int] = None,
        account_type: Optional[str] = None,
    ) -> PipelineStats:
        stats = PipelineStats()
        t0 = time.time()

        stats.total_raw = self._count_raw(statement_id, account_type)
        log.info(
            "bank_transactions_raw to enrich: %d (stmt=%s, acct=%s)",
            stats.total_raw, statement_id, account_type,
        )

        offset = 0
        while True:
            rows = self._fetch_batch(statement_id, account_type, offset)
            if not rows:
                break

            batch: list[dict] = []
            for row in rows:
                if self._should_skip(row["id"]):
                    stats.skipped += 1
                    continue
                try:
                    batch.append(self._enrich_row(row))
                    stats.processed += 1
                except Exception as exc:
                    log.warning("Enrichment failed raw_id=%s: %s", row["id"], exc)
                    stats.errors += 1

            if batch:
                self._bulk_upsert(batch)
                log.debug("Upserted %d rows (offset=%d)", len(batch), offset)

            offset += self.batch_size

        stats.elapsed_seconds = time.time() - t0
        log.info(str(stats))
        return stats

    # ------------------------------------------------------------------
    # Internal: statement_id mapping (SQLite ↔ Supabase)
    # ------------------------------------------------------------------

    def _build_stmt_id_map(self) -> dict[int, int]:
        """
        Map SQLite bank_statements.id → Supabase bank_statements.id
        by matching on file_hash (the natural business key).
        Logs a warning for any SQLite statements with no Supabase match.
        """
        sqlite_rows = self._conn.execute(
            "SELECT id, file_hash FROM bank_statements"
        ).fetchall()

        supa_page = self._supa.table("bank_statements").select("id,file_hash").execute()
        supa_by_hash: dict[str, int] = {
            r["file_hash"]: r["id"] for r in (supa_page.data or [])
        }

        mapping: dict[int, int] = {}
        for row in sqlite_rows:
            supa_id = supa_by_hash.get(row["file_hash"])
            if supa_id is not None:
                mapping[row["id"]] = supa_id
            else:
                log.warning(
                    "No Supabase match for SQLite statement_id=%d file_hash=%s",
                    row["id"], row["file_hash"],
                )

        log.info("statement_id map built: %d/%d matched", len(mapping), len(sqlite_rows))
        return mapping

    def _build_raw_id_map(self) -> dict[int, int]:
        """
        Map SQLite bank_transactions_raw.id → Supabase bank_transactions_raw.id.

        Matches rows by (supa_statement_id, date, description, amount).
        Handles genuine duplicates (same natural key appearing N times in both
        stores) by collecting all IDs per key and assigning them 1:1 in
        ascending ID order, so every Supabase raw row gets a distinct mapping.
        """
        # Fetch all Supabase raw rows (paginated — default limit is 1000)
        supa_raw: list[dict] = []
        page_size = 1000
        offset = 0
        while True:
            page = (
                self._supa.table("bank_transactions_raw")
                .select("id,statement_id,date,description,amount")
                .range(offset, offset + page_size - 1)
                .execute()
            )
            if not page.data:
                break
            supa_raw.extend(page.data)
            if len(page.data) < page_size:
                break
            offset += page_size

        # Build lookup: natural_key → sorted list of supa_raw_ids.
        # Using a list handles genuine duplicates (same transaction ingested
        # twice) where the same natural key appears more than once.
        supa_by_key: dict[tuple, list[int]] = {}
        for r in supa_raw:
            key = (
                r["statement_id"],
                r["date"],
                r["description"],
                round(float(r["amount"]), 4),
            )
            supa_by_key.setdefault(key, []).append(r["id"])
        for ids in supa_by_key.values():
            ids.sort()  # deterministic assignment

        # Group SQLite raw rows by their translated natural key.
        sqlite_rows = self._conn.execute(
            "SELECT id, statement_id, date, description, amount FROM bank_transactions_raw"
        ).fetchall()

        sqlite_by_key: dict[tuple, list[int]] = {}
        unmapped_stmt = 0
        for row in sqlite_rows:
            supa_stmt_id = self._stmt_id_map.get(row["statement_id"])
            if supa_stmt_id is None:
                unmapped_stmt += 1
                continue
            key = (
                supa_stmt_id,
                row["date"],
                row["description"],
                round(float(row["amount"]), 4),
            )
            sqlite_by_key.setdefault(key, []).append(row["id"])
        for ids in sqlite_by_key.values():
            ids.sort()  # match assignment order on both sides

        # 1:1 ordered assignment: sqlite_ids[i] → supa_ids[i]
        mapping: dict[int, int] = {}
        unmatched = 0
        for key, sqlite_ids in sqlite_by_key.items():
            supa_ids = supa_by_key.get(key, [])
            for i, sqlite_id in enumerate(sqlite_ids):
                if i < len(supa_ids):
                    mapping[sqlite_id] = supa_ids[i]
                else:
                    unmatched += 1
                    log.debug(
                        "No Supabase match for SQLite raw_id=%d — duplicate count mismatch",
                        sqlite_id,
                    )

        if unmapped_stmt:
            log.warning("raw_id map: %d SQLite rows skipped (no stmt mapping)", unmapped_stmt)
        if unmatched:
            log.warning("raw_id map: %d SQLite rows could not be matched to Supabase", unmatched)
        log.info("raw_id map built: %d/%d matched", len(mapping), len(sqlite_rows))
        return mapping

    # ------------------------------------------------------------------
    # Internal: query helpers
    # ------------------------------------------------------------------

    def _count_raw(self, stmt_id: Optional[int], acct: Optional[str]) -> int:
        sql = "SELECT COUNT(*) FROM bank_transactions_raw WHERE 1=1"
        params: list = []
        if stmt_id is not None:
            sql += " AND statement_id = ?"
            params.append(stmt_id)
        if acct:
            sql += " AND account_type = ?"
            params.append(acct)
        return self._conn.execute(sql, params).fetchone()[0]

    def _fetch_batch(
        self, stmt_id: Optional[int], acct: Optional[str], offset: int,
    ) -> list[sqlite3.Row]:
        sql = """
            SELECT id, statement_id, bank_name, account_type,
                   date, description, amount, transaction_type, balance
            FROM   bank_transactions_raw WHERE 1=1
        """
        params: list = []
        if stmt_id is not None:
            sql += " AND statement_id = ?"
            params.append(stmt_id)
        if acct:
            sql += " AND account_type = ?"
            params.append(acct)
        sql += " ORDER BY id LIMIT ? OFFSET ?"
        params.extend([self.batch_size, offset])
        return self._conn.execute(sql, params).fetchall()

    def _should_skip(self, raw_id: int) -> bool:
        if self.reprocess_all:
            return False
        row = self._conn.execute(
            "SELECT confidence_score FROM bank_transactions WHERE raw_id = ?",
            (raw_id,),
        ).fetchone()
        if row is None:
            return False
        return row["confidence_score"] >= CONFIDENCE_THRESHOLD

    # ------------------------------------------------------------------
    # Internal: enrich
    # ------------------------------------------------------------------

    def _enrich_row(self, row: sqlite3.Row) -> dict:
        description = row["description"] or ""
        amount = float(row["amount"] or 0.0)
        transaction_type = row["transaction_type"] or "UNKNOWN"

        cleaned = clean_description(description)
        category, confidence = categorize_description(description)

        return {
            "raw_id": row["id"],
            "statement_id": row["statement_id"],
            "bank_name": row["bank_name"],
            "account_type": row["account_type"],
            "date": row["date"],
            "description": description,
            "clean_description": cleaned,
            "amount": amount,
            "transaction_type": transaction_type,
            "balance": row["balance"],
            "category": category,
            "confidence_score": confidence,
            "enrichment_method": "rule_based" if confidence >= 1.0 else "fallback",
        }

    # ------------------------------------------------------------------
    # Internal: upsert
    # ------------------------------------------------------------------

    def _bulk_upsert(self, rows: list[dict]) -> None:
        for row in rows:
            cols = list(row.keys())
            col_names = ", ".join(cols)
            placeholders = ", ".join("?" * len(cols))
            update_clause = ", ".join(
                f"{c} = excluded.{c}" for c in cols if c != "raw_id"
            )
            self._conn.execute(
                f"""
                INSERT INTO bank_transactions ({col_names})
                VALUES ({placeholders})
                ON CONFLICT(raw_id) DO UPDATE SET {update_clause}
                """,
                [row[c] for c in cols],
            )
        self._conn.commit()

        if self._supa:
            try:
                # Remap SQLite IDs → Supabase IDs before writing.
                # Both statement_id and raw_id use independent sequences in
                # SQLite vs Supabase and must be translated before the FK check.
                supa_rows = []
                skipped_no_raw_id = 0
                for r in rows:
                    supa_r = dict(r)
                    supa_r["statement_id"] = self._stmt_id_map.get(
                        r["statement_id"], r["statement_id"]
                    )
                    supa_raw_id = self._raw_id_map.get(r["raw_id"])
                    if supa_raw_id is None:
                        skipped_no_raw_id += 1
                        log.debug(
                            "Skipping Supabase write: no raw_id mapping for sqlite_raw_id=%d",
                            r["raw_id"],
                        )
                        continue
                    supa_r["raw_id"] = supa_raw_id
                    supa_rows.append(_coerce(supa_r))
                if skipped_no_raw_id:
                    log.warning(
                        "%d enriched rows skipped for Supabase (no raw_id mapping)",
                        skipped_no_raw_id,
                    )

                chunk_size = 500
                for i in range(0, len(supa_rows), chunk_size):
                    chunk = supa_rows[i : i + chunk_size]
                    self._supa.table("bank_transactions").upsert(
                        chunk, on_conflict="raw_id"
                    ).execute()
                log.debug("Supabase bank_transactions: %d rows", len(supa_rows))
            except Exception as e:
                log.warning("Supabase enrichment upsert failed: %s", e)

    def close(self):
        pass  # caller owns connection lifecycle


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Enrich bank_transactions_raw → bank_transactions (rule-based)"
    )
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--statement-id", type=int, default=None)
    ap.add_argument("--account-type", choices=["checking", "savings"], default=None)
    ap.add_argument("--reprocess", action="store_true")
    ap.add_argument("--sqlite-only", action="store_true")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--log-level", default="INFO",
                     choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = ap.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    pipeline = BankEnrichmentPipeline(
        db_path=args.db,
        batch_size=args.batch_size,
        reprocess_all=args.reprocess,
        sqlite_only=args.sqlite_only,
    )
    stats = pipeline.run(
        statement_id=args.statement_id,
        account_type=args.account_type,
    )

    print()
    print("=" * 60)
    print("BANK ENRICHMENT SUMMARY")
    print("=" * 60)
    print(f"  Total raw:    {stats.total_raw}")
    print(f"  Processed:    {stats.processed}")
    print(f"  Skipped:      {stats.skipped}")
    print(f"  Errors:       {stats.errors}")
    print(f"  Elapsed:      {stats.elapsed_seconds:.1f}s")
    print("=" * 60)

    pipeline.close()
    raise SystemExit(0 if stats.errors == 0 else 1)


if __name__ == "__main__":
    main()