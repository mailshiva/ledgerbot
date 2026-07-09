"""
src/nlp/enrichment_pipeline.py
───────────────────────────────
Reads every row from `transactions_raw`, runs your Week 2 NLP stack
(TransactionProcessor → six layers), and upserts enriched rows into
the `transactions` table.

db_path is loaded from ~/.bank_parser/config.yaml exactly the way
Week 1 code did it – via src.config.Config.load().

Write behaviour (dual-write):
    All upserts go to SQLite AND Supabase atomically.
    If Supabase fails, SQLite is rolled back — no partial state.

Read behaviour:
    Pipeline-internal reads (count, fetch batch, skip-check) use SQLite
    directly for speed. Agent queries go to Supabase via DualWriteManager.

Usage
-----
    # run with default config (~/.bank_parser/config.yaml)
    python -m src.nlp.enrichment_pipeline

    # override config file location
    python -m src.nlp.enrichment_pipeline --config /path/to/config.yaml

    # process a single statement
    python -m src.nlp.enrichment_pipeline --statement-id 3

    # force re-enrich rows that already have results
    python -m src.nlp.enrichment_pipeline --reprocess

    # or import and call directly:
    from src.nlp.enrichment_pipeline import EnrichmentPipeline
    pipeline = EnrichmentPipeline()      # reads ~/.bank_parser/config.yaml
    stats    = pipeline.run()

    # inject an existing DualWriteManager (e.g. from main.py or tests):
    from src.database.dual_write_manager import DualWriteManager
    db = DualWriteManager(str(Path.home() / "sqlLite_DB" / "bank_data.db"))
    pipeline = EnrichmentPipeline(db=db)
    stats = pipeline.run()
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Config  (Week 1 pattern) ──────────────────────────────────────────────────
from src.config import Config

# ── Week 2 NLP module ─────────────────────────────────────────────────────────
from src.nlp.transaction_processor import TransactionProcessor, Transaction

# ── DB helpers ────────────────────────────────────────────────────────────────
from src.database.enriched_transaction import (
    EnrichedTransaction,
    apply_migration,
    bulk_upsert,        # kept for backward-compat (tests pass raw sqlite3.Connection)
    bulk_upsert_dual,   # dual-write aware version used in production
)
from src.database.dual_write_manager import DualWriteManager

log = logging.getLogger(__name__)


def _get(result, *keys, default=None):
    """Get a value from a dict or object, trying multiple key/attr names."""
    for key in keys:
        if isinstance(result, dict):
            if key in result:
                return result[key]
        else:
            val = getattr(result, key, None)
            if val is not None:
                return val
    return default

BATCH_SIZE           = 200
CONFIDENCE_THRESHOLD = 0.0   # skip re-enrichment if score already >= this


# ---------------------------------------------------------------------------
# Result summary
# ---------------------------------------------------------------------------

@dataclass
class PipelineStats:
    total_raw: int         = 0
    processed: int         = 0
    skipped: int           = 0
    errors: int            = 0
    elapsed_seconds: float = 0.0

    def __str__(self) -> str:
        return (
            f"EnrichmentPipeline finished in {self.elapsed_seconds:.1f}s | "
            f"raw={self.total_raw}  processed={self.processed}  "
            f"skipped={self.skipped}  errors={self.errors}"
        )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class EnrichmentPipeline:
    """
    Orchestrates the full raw → enriched journey.

    Parameters
    ----------
    config_path:
        Optional path to a config.yaml.  When None, uses the default
        ~/.bank_parser/config.yaml – identical to the Week 1 behaviour.
    batch_size:
        How many transactions_raw rows to fetch per loop iteration.
    reprocess_all:
        If True, re-enrich rows that already have a `transactions` entry.
        If False (default), skip raw_ids that already have a result with
        confidence_score >= CONFIDENCE_THRESHOLD.
    sql_migration_path:
        Optional path to migrate_schema.sql.  When None, the inline DDL
        inside enriched_transaction.py is used.
    db:
        Optional DualWriteManager instance.  When provided, it is used
        directly and no new connection is created.  When None (default),
        a DualWriteManager is constructed from config_path so that all
        existing CLI usage continues to work unchanged.
    """

    def __init__(
        self,
        config_path: Optional[Path] = None,
        batch_size: int = BATCH_SIZE,
        reprocess_all: bool = False,
        sql_migration_path: Optional[Path] = None,
        db: Optional[DualWriteManager] = None,  # ← injected or auto-created
    ) -> None:
        # ── Load config the Week 1 way ────────────────────────────────
        self.cfg = Config.load(config_path)
        self.cfg.ensure_db_dir()
        log.info("Using database: %s", self.cfg.db_path)

        self.batch_size         = batch_size
        self.reprocess_all      = reprocess_all
        self.sql_migration_path = sql_migration_path

        # ── Set up dual-write manager ─────────────────────────────────
        # Accept an injected DualWriteManager (e.g. from main.py so the
        # same connection is shared) or create one from config so that
        # plain CLI usage requires zero changes.
        if db is not None:
            self._db = db
            log.info("Using injected DualWriteManager")
        else:
            self._db = DualWriteManager(str(self.cfg.db_path))
            log.info("Created DualWriteManager from config")

        # ── Load spaCy / NLP model once per pipeline instance ────────
        log.info("Loading NLP model (%s)…", self.cfg.spacy_model)
        self._processor = TransactionProcessor()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, statement_id: Optional[int] = None) -> PipelineStats:
        """
        Run the enrichment pipeline.

        Parameters
        ----------
        statement_id:
            When set, only enrich transactions belonging to this statement.
            When None (default), processes all rows in transactions_raw.
        """
        stats = PipelineStats()
        t0    = time.time()

        # Pipeline-internal reads (count, fetch, skip-check) always use
        # SQLite directly — fast local reads, no network round-trips.
        # DualWriteManager owns the connection lifecycle; we never close it here.
        conn = self._db.sqlite.conn
        apply_migration(conn, self.sql_migration_path)

        stats.total_raw = self._count_raw(conn, statement_id)
        log.info(
            "transactions_raw rows to process: %d (statement_id=%s)",
            stats.total_raw, statement_id,
        )

        offset = 0
        while True:
            raw_rows = self._fetch_batch(conn, statement_id, offset)
            if not raw_rows:
                break

            enriched_batch: List[EnrichedTransaction] = []
            for row in raw_rows:
                if self._should_skip(conn, row["id"]):
                    stats.skipped += 1
                    continue
                try:
                    enriched = self._enrich_row(row)
                    enriched_batch.append(enriched)
                    stats.processed += 1
                except Exception as exc:
                    log.warning(
                        "Enrichment failed for raw_id=%s: %s", row["id"], exc
                    )
                    stats.errors += 1

            if enriched_batch:
                # Dual-write: SQLite first, then Supabase.
                # Rolls back SQLite automatically if Supabase fails.
                bulk_upsert_dual(self._db, enriched_batch)
                log.debug(
                    "Dual-upserted %d rows (offset=%d)", len(enriched_batch), offset
                )

            offset += self.batch_size

        stats.elapsed_seconds = time.time() - t0
        log.info(str(stats))
        return stats

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    # NOTE: _open_db() is intentionally removed.
    # The pipeline no longer owns connection lifecycle — DualWriteManager does.
    # All internal reads use self._db.sqlite.conn directly.

    def _count_raw(
        self, conn: sqlite3.Connection, statement_id: Optional[int]
    ) -> int:
        if statement_id is not None:
            row = conn.execute(
                "SELECT COUNT(*) FROM transactions_raw WHERE statement_id = ?",
                (statement_id,),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT COUNT(*) FROM transactions_raw"
            ).fetchone()
        return row[0]

    def _fetch_batch(
        self,
        conn: sqlite3.Connection,
        statement_id: Optional[int],
        offset: int,
    ) -> List[sqlite3.Row]:
        if statement_id is not None:
            return conn.execute(
                """
                SELECT id, statement_id, bank_name,
                       date, description, amount, transaction_type,
                       balance, raw_text
                FROM   transactions_raw
                WHERE  statement_id = ?
                ORDER  BY id
                LIMIT  ? OFFSET ?
                """,
                (statement_id, self.batch_size, offset),
            ).fetchall()
        return conn.execute(
            """
            SELECT id, statement_id, bank_name,
                   date, description, amount, transaction_type,
                   balance, raw_text
            FROM   transactions_raw
            ORDER  BY id
            LIMIT  ? OFFSET ?
            """,
            (self.batch_size, offset),
        ).fetchall()

    def _should_skip(self, conn: sqlite3.Connection, raw_id: int) -> bool:
        """Return True if this raw_id already has a good enrichment result."""
        if self.reprocess_all:
            return False
        row = conn.execute(
            "SELECT confidence_score FROM transactions WHERE raw_id = ?",
            (raw_id,),
        ).fetchone()
        if row is None:
            return False
        return row["confidence_score"] >= CONFIDENCE_THRESHOLD

    def _enrich_row(self, row: sqlite3.Row) -> EnrichedTransaction:
        """Run the NLP stack on one raw row and return an EnrichedTransaction."""
        description      = row["description"] or ""
        amount           = float(row["amount"] or 0.0)
        transaction_type = row["transaction_type"] or "UNKNOWN"

        raw_dict = {
            "id": str(row["id"]),
            "raw_description": description,
            "amount": amount,
            "date": row["date"],
        }
        result: Transaction = self._processor.process(raw_dict)

        return EnrichedTransaction(
            raw_id=row["id"],
            statement_id=row["statement_id"],
            bank_name=row["bank_name"],
            date=row["date"],
            description=description,
            amount=amount,
            transaction_type=transaction_type,
            balance=row["balance"],
            clean_description=_get(result, "raw_description"),
            merchant_name=_get(result, "merchant"),
            merchant_raw=description,
            category=_get(result, "category") or "Uncategorized",
            subcategory=_get(result, "subcategory"),
            location=_get(result, "location"),
            confidence_score=float(_get(result, "confidence") or 0.0),
            enrichment_method="nlp",
        )


# ---------------------------------------------------------------------------
# CLI  – note: no --db flag. db_path always comes from config.yaml
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enrich transactions_raw → transactions via the NLP pipeline."
    )
    parser.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help="Path to config.yaml (default: ~/.bank_parser/config.yaml)",
    )
    parser.add_argument(
        "--statement-id",
        type=int,
        default=None,
        metavar="ID",
        help="Only process rows for this statement_id (default: all rows)",
    )
    parser.add_argument(
        "--reprocess",
        action="store_true",
        default=False,
        help="Re-enrich rows that already have results (default: skip)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        metavar="N",
        help=f"DB fetch batch size (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--migration-sql",
        default=None,
        metavar="PATH",
        help="Path to migrate_schema.sql (default: use inline DDL)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level   = getattr(logging, args.log_level),
        format  = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt = "%H:%M:%S",
    )

    # DualWriteManager is created inside __init__ from config.
    # CLI usage is completely unchanged — no new flags needed.
    pipeline = EnrichmentPipeline(
        config_path        = Path(args.config) if args.config else None,
        batch_size         = args.batch_size,
        reprocess_all      = args.reprocess,
        sql_migration_path = Path(args.migration_sql) if args.migration_sql else None,
    )
    stats = pipeline.run(statement_id=args.statement_id)
    raise SystemExit(0 if stats.errors == 0 else 1)


if __name__ == "__main__":
    main()
