"""
Shared test helpers and constants.
Works with both pytest (fixtures) and plain unittest (direct imports).
"""
import os
import sys
import shutil
import tempfile
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

SAMPLES_DIR = Path(__file__).parent.parent / "data" / "samples"

SAMPLE_PDFS = {
    "robinhood":        SAMPLES_DIR / "Credit_Statement_March_2026.pdf",
    "citi_jan":         SAMPLES_DIR / "January2026.pdf",
    "citi_feb":         SAMPLES_DIR / "February2026.pdf",
    "bofa":             SAMPLES_DIR / "eStmt_2025-09-05.pdf",
    "capital_one_sep":  SAMPLES_DIR / "20250901-Venture_card_statement-3205.pdf",
    "capital_one_feb":  SAMPLES_DIR / "20250201-Venture_card_statement-3205.pdf",
    "capital_one_dec":  SAMPLES_DIR / "20251201-Venture_card_statement-3205.pdf",
    "citi_dec":         SAMPLES_DIR / "December_16.pdf",
    "citi_feb18":       SAMPLES_DIR / "February_18.pdf",
    "robinhood_feb":    SAMPLES_DIR / "Credit_Statement_February_2026.pdf",
}
SAMPLE_CSV = SAMPLES_DIR / "sample_bank_statement.csv"

# Only include files that actually exist on disk
AVAILABLE_PDFS = {k: v for k, v in SAMPLE_PDFS.items() if v.exists()}


def make_tmp_db(db_path=None):
    """Return a fresh DatabaseManager at db_path (or a temp file)."""
    from src.database.db_manager import DatabaseManager
    if db_path is None:
        fd, db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
    return DatabaseManager(db_path=db_path), db_path


def parse_and_save(filepath, db):
    """Parse a PDF and save it to db. Returns (result, stmt_id)."""
    from src.parsers.pdf_parser import PDFStatementParser
    result    = PDFStatementParser(str(filepath)).parse()
    bank_name = result["metadata"]["bank_name"]
    file_hash = db.check_duplicate(str(filepath))
    stmt_id   = db.save_statement(str(filepath), bank_name=bank_name,
                                   file_hash=file_hash)
    db.save_transactions(result["transactions"], statement_id=stmt_id,
                          bank_name=bank_name)
    return result, stmt_id


# ── pytest fixtures (ignored when running with unittest) ──────────
try:
    import pytest

    @pytest.fixture()
    def tmp_db(tmp_path):
        from src.database.db_manager import DatabaseManager
        db = DatabaseManager(db_path=str(tmp_path / "test.db"))
        yield db
        db.close()

    @pytest.fixture()
    def sample_pdfs():
        return AVAILABLE_PDFS

    @pytest.fixture()
    def sample_csv():
        return SAMPLE_CSV

    @pytest.fixture()
    def tmp_watch_folder(tmp_path):
        folder = tmp_path / "statements"
        folder.mkdir()
        return folder

    @pytest.fixture()
    def watch_folder_with_pdfs(tmp_path):
        folder = tmp_path / "statements"
        folder.mkdir()
        copied = []
        for name, src in AVAILABLE_PDFS.items():
            dst = folder / src.name
            shutil.copy(src, dst)
            copied.append(dst)
        return folder, copied

except ImportError:
    pass  # pytest not installed — fixtures unused, unittest path works fine
