"""
Statement Watcher  — scan-once mode
====================================
Scans a folder for PDF files that have NOT yet been imported into the
database (detected via SHA-256 hash), processes each new file, and exits.

Designed to be called from cron or a shell script:

    # Every 30 minutes
    */30 * * * * /path/to/venv/bin/python /path/to/bank_statement_parser_regex/watcher.py

Usage:
    python watcher.py                        # scan folder from config
    python watcher.py /path/to/statements    # override folder for this run
    python watcher.py --dry-run              # scan but do not write to DB

Exit codes:
    0  — completed (even if some files failed; see log for details)
    1  — configuration or folder error
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# ------------------------------------------------------------------ #
#  Ensure project root is on path when run directly                   #
# ------------------------------------------------------------------ #
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.database.db_manager import DatabaseManager, DuplicateStatementError


# ------------------------------------------------------------------ #
#  Logging setup                                                       #
# ------------------------------------------------------------------ #

def _setup_logging(log_file: str) -> logging.Logger:
    """Configure a logger that writes to both file and stdout."""
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    fmt = "%(asctime)s  %(levelname)-8s  %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger("watcher")


# ------------------------------------------------------------------ #
#  Core scan logic                                                     #
# ------------------------------------------------------------------ #

def scan_folder(folder: str, db: DatabaseManager,
                dry_run: bool = False, logger: logging.Logger = None) -> dict:
    """
    Scan *folder* for PDF files not yet in the database.
    Returns a summary dict with keys: found, new, imported, skipped, failed.
    """
    log = logger or logging.getLogger("watcher")
    folder_path = Path(folder)

    if not folder_path.exists():
        log.error(f"Watch folder does not exist: {folder}")
        return {"found": 0, "new": 0, "imported": 0, "skipped": 0, "failed": 0}

    pdfs = sorted(folder_path.glob("*.pdf"))
    log.info(f"Scanning: {folder}  ({len(pdfs)} PDF(s) found)")

    stats = {"found": len(pdfs), "new": 0, "imported": 0, "skipped": 0, "failed": 0}

    for pdf_path in pdfs:
        filepath = str(pdf_path)

        # ── Duplicate check ──────────────────────────────────────────
        try:
            file_hash = db.check_duplicate(filepath)
        except DuplicateStatementError as e:
            log.info(f"  SKIP  {pdf_path.name}  (already imported on {e.import_date})")
            stats["skipped"] += 1
            continue

        stats["new"] += 1
        log.info(f"  NEW   {pdf_path.name}")

        if dry_run:
            log.info(f"        [dry-run] would import {pdf_path.name}")
            continue

        # ── Parse ────────────────────────────────────────────────────
        try:
            from src.parsers.pdf_parser import PDFStatementParser
            result     = PDFStatementParser(filepath).parse()
            bank_name  = result.get("metadata", {}).get("bank_name", "unknown")
            summary    = result["summary"]

            stmt_id = db.save_statement(
                filepath,
                bank_name=bank_name,
                file_hash=file_hash,
            )
            saved = db.save_transactions(
                result["transactions"],
                statement_id=stmt_id,
                bank_name=bank_name,
            )

            log.info(
                f"        imported  bank={bank_name}  "
                f"txns={saved}  "
                f"debits=${summary['total_debits']:.2f}  "
                f"credits=${summary['total_credits']:.2f}"
            )
            stats["imported"] += 1

        except Exception as exc:
            log.error(f"        FAILED  {pdf_path.name}: {exc}", exc_info=True)
            stats["failed"] += 1

    return stats


# ------------------------------------------------------------------ #
#  Entry point                                                         #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description="Scan a folder for new PDF statements and import them."
    )
    parser.add_argument(
        "folder",
        nargs="?",
        help="Folder to scan (overrides config). Defaults to watcher.folder in config.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Detect new files but do not write to the database.",
    )
    args = parser.parse_args()

    # ── Load config ──────────────────────────────────────────────────
    cfg    = load_config()
    folder = args.folder or cfg.watch_folder
    logger = _setup_logging(cfg.watch_log_file)

    logger.info("=" * 60)
    logger.info(f"Bank Statement Watcher  —  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"DB     : {cfg.db_path}")
    logger.info(f"Folder : {folder}")
    if args.dry_run:
        logger.info("Mode   : DRY RUN (nothing will be written)")
    logger.info("=" * 60)

    # ── Validate folder ──────────────────────────────────────────────
    if not Path(folder).exists():
        logger.error(f"Folder not found: {folder}")
        logger.error("Update 'watcher.folder' in ~/.bank_parser/config.yaml")
        sys.exit(1)

    # ── Run scan ─────────────────────────────────────────────────────
    db    = DatabaseManager()
    stats = scan_folder(folder, db, dry_run=args.dry_run, logger=logger)

    # ── Summary ──────────────────────────────────────────────────────
    logger.info("-" * 60)
    logger.info(
        f"Done.  found={stats['found']}  new={stats['new']}  "
        f"imported={stats['imported']}  skipped={stats['skipped']}  "
        f"failed={stats['failed']}"
    )

    if stats["failed"]:
        sys.exit(1)   # non-zero so cron can alert on failures


if __name__ == "__main__":
    main()
