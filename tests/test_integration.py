"""
Integration Tests  —  run with:
    python -m unittest tests.test_integration -v     # no pytest needed
    pytest tests/test_integration.py -v              # if pytest is installed
"""
import os
import re
import sys
import shutil
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.conftest import AVAILABLE_PDFS, SAMPLE_CSV, parse_and_save, make_tmp_db


# ================================================================== #
#  Base class with DB lifecycle                                        #
# ================================================================== #

class DBTestCase(unittest.TestCase):
    """Provides self.db (fresh temp DB) for each test."""

    def setUp(self):
        self.db, self._db_path = make_tmp_db()

    def tearDown(self):
        self.db.close()
        if os.path.exists(self._db_path):
            os.unlink(self._db_path)


# ================================================================== #
#  Parse accuracy — one test per supported bank                       #
# ================================================================== #

class TestParseAccuracy(unittest.TestCase):
    """
    Verify each sample PDF parses to exactly the amounts shown on the
    paper statement.
    """

    EXPECTATIONS = {
        # key             bank            txns  debits    credits
        "robinhood":       ("robinhood",   53, 1361.13, 1868.67),
        "citi_jan":        ("citi",         3,  139.24,    0.00),
        "citi_feb":        ("citi",         1,   84.47,    0.00),
        "bofa":            ("bofa",         6,   88.69,   99.84),
        "capital_one_sep": ("capital_one",  2,  142.30,   65.60),
        "capital_one_feb": ("capital_one",  1,   95.00,    0.00),
        "capital_one_dec": ("capital_one",  0,    0.00,    0.00),
        "citi_dec":         ("citi",          2,    0.00,  243.66),
        "citi_feb18":       ("citi",         24,  750.49, 1915.93),
        "robinhood_feb":  ("robinhood",  38, 1857.74, 1584.32),
    }

    def _parse(self, key):
        from src.parsers.pdf_parser import PDFStatementParser
        return PDFStatementParser(str(AVAILABLE_PDFS[key])).parse()

    def _check(self, key):
        if key not in AVAILABLE_PDFS:
            self.skipTest(f"Sample file for {key!r} not found")
        exp_bank, exp_txns, exp_deb, exp_cred = self.EXPECTATIONS[key]
        result  = self._parse(key)
        summary = result["summary"]
        meta    = result["metadata"]
        self.assertEqual(meta["bank_name"], exp_bank,
                         f"Wrong bank: got {meta['bank_name']!r}")
        self.assertEqual(summary["total_transactions"], exp_txns,
                         f"Txn count: got {summary['total_transactions']}")
        self.assertAlmostEqual(summary["total_debits"], exp_deb, places=2,
                               msg=f"Debits: got {summary['total_debits']}")
        self.assertAlmostEqual(summary["total_credits"], exp_cred, places=2,
                               msg=f"Credits: got {summary['total_credits']}")

    def test_robinhood(self):   self._check("robinhood")
    def test_citi_jan(self):    self._check("citi_jan")
    def test_citi_feb(self):    self._check("citi_feb")
    def test_bofa(self):        self._check("bofa")
    def test_capital_one_sep(self): self._check("capital_one_sep")
    def test_capital_one_feb(self): self._check("capital_one_feb")
    def test_capital_one_dec(self): self._check("capital_one_dec")
    def test_citi_dec(self):         self._check("citi_dec")
    def test_citi_feb18(self):        self._check("citi_feb18")
    def test_robinhood_feb(self):    self._check("robinhood_feb")

    def test_all_transactions_have_required_fields(self):
        required = {"date", "description", "amount",
                    "transaction_type", "merchant_name", "category"}
        for key, path in AVAILABLE_PDFS.items():
            from src.parsers.pdf_parser import PDFStatementParser
            result = PDFStatementParser(str(path)).parse()
            for t in result["transactions"]:
                missing = required - set(t.keys())
                self.assertFalse(missing,
                    f"[{key}] Transaction missing fields: {missing}")

    def test_all_dates_iso_format(self):
        iso_re = re.compile(r"^\d{4}-\d{2}-\d{2}$")
        for key, path in AVAILABLE_PDFS.items():
            from src.parsers.pdf_parser import PDFStatementParser
            for t in PDFStatementParser(str(path)).parse()["transactions"]:
                self.assertRegex(t["date"], iso_re,
                    f"[{key}] Bad date: {t['date']!r}")

    def test_all_amounts_positive(self):
        for key, path in AVAILABLE_PDFS.items():
            from src.parsers.pdf_parser import PDFStatementParser
            for t in PDFStatementParser(str(path)).parse()["transactions"]:
                self.assertGreaterEqual(t["amount"], 0,
                    f"[{key}] Negative amount: {t['amount']}")

    def test_transaction_type_valid(self):
        valid = {"DEBIT", "CREDIT", "UNKNOWN"}
        for key, path in AVAILABLE_PDFS.items():
            from src.parsers.pdf_parser import PDFStatementParser
            for t in PDFStatementParser(str(path)).parse()["transactions"]:
                self.assertIn(t["transaction_type"], valid,
                    f"[{key}] Invalid type: {t['transaction_type']!r}")


# ================================================================== #
#  Database round-trip                                                 #
# ================================================================== #

class TestDatabaseRoundTrip(DBTestCase):

    def test_parse_and_retrieve_bofa(self):
        if "bofa" not in AVAILABLE_PDFS:
            self.skipTest("BofA sample not found")
        result, _ = parse_and_save(AVAILABLE_PDFS["bofa"], self.db)
        rows = self.db.get_transactions(limit=100)
        self.assertEqual(len(rows), result["summary"]["total_transactions"])

    def test_debit_filter(self):
        if "bofa" not in AVAILABLE_PDFS:
            self.skipTest("BofA sample not found")
        parse_and_save(AVAILABLE_PDFS["bofa"], self.db)
        debit_rows = self.db.get_transactions(transaction_type="DEBIT")
        self.assertTrue(all(r["transaction_type"] == "DEBIT" for r in debit_rows))

    def test_bank_name_persisted(self):
        if "robinhood" not in AVAILABLE_PDFS:
            self.skipTest("Robinhood sample not found")
        parse_and_save(AVAILABLE_PDFS["robinhood"], self.db)
        rows = self.db.get_transactions(bank_name="robinhood", limit=200)
        self.assertGreater(len(rows), 0)
        self.assertTrue(all(r["bank_name"] == "robinhood" for r in rows))

    def test_filter_by_bank_isolated(self):
        if not {"bofa", "citi_jan"}.issubset(AVAILABLE_PDFS):
            self.skipTest("Need bofa and citi_jan samples")
        parse_and_save(AVAILABLE_PDFS["bofa"],     self.db)
        parse_and_save(AVAILABLE_PDFS["citi_jan"], self.db)
        bofa_rows = self.db.get_transactions(bank_name="bofa",  limit=200)
        citi_rows = self.db.get_transactions(bank_name="citi",  limit=200)
        self.assertEqual(len(bofa_rows), 6)
        self.assertEqual(len(citi_rows), 3)
        self.assertTrue(all(r["bank_name"] == "bofa" for r in bofa_rows))
        self.assertTrue(all(r["bank_name"] == "citi" for r in citi_rows))

    def test_statements_table_populated(self):
        if "capital_one" not in AVAILABLE_PDFS:
            self.skipTest("Capital One sample not found")
        parse_and_save(AVAILABLE_PDFS["capital_one"], self.db)
        stmts = self.db.get_statements()
        self.assertEqual(len(stmts), 1)
        self.assertEqual(stmts[0]["bank_name"], "capital_one")
        self.assertEqual(stmts[0]["total_transactions"], 2)

    def test_summary_stats_match_parse(self):
        if "bofa" not in AVAILABLE_PDFS:
            self.skipTest("BofA sample not found")
        result, _ = parse_and_save(AVAILABLE_PDFS["bofa"], self.db)
        stats = self.db.get_summary_stats()
        self.assertEqual(stats["total"], result["summary"]["total_transactions"])
        self.assertAlmostEqual(
            stats["total_debits"], result["summary"]["total_debits"], places=2)
        self.assertAlmostEqual(
            stats["total_credits"], result["summary"]["total_credits"], places=2)


# ================================================================== #
#  Duplicate detection                                                 #
# ================================================================== #

class TestDuplicateDetection(DBTestCase):

    def test_second_import_raises(self):
        if "citi_feb" not in AVAILABLE_PDFS:
            self.skipTest("Citi Feb sample not found")
        from src.database.db_manager import DuplicateStatementError
        parse_and_save(AVAILABLE_PDFS["citi_feb"], self.db)
        with self.assertRaises(DuplicateStatementError):
            self.db.check_duplicate(str(AVAILABLE_PDFS["citi_feb"]))

    def test_different_files_not_duplicates(self):
        keys = list(AVAILABLE_PDFS.keys())
        if len(keys) < 2:
            self.skipTest("Need at least 2 sample files")
        parse_and_save(AVAILABLE_PDFS[keys[0]], self.db)
        result = self.db.check_duplicate(str(AVAILABLE_PDFS[keys[1]]))
        self.assertIsInstance(result, str)

    def test_renamed_duplicate_detected(self):
        if "citi_jan" not in AVAILABLE_PDFS:
            self.skipTest("Citi Jan sample not found")
        from src.database.db_manager import DuplicateStatementError
        original = AVAILABLE_PDFS["citi_jan"]
        tmp = tempfile.mktemp(suffix=".pdf")
        shutil.copy(original, tmp)
        try:
            parse_and_save(original, self.db)
            with self.assertRaises(DuplicateStatementError):
                self.db.check_duplicate(tmp)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)


# ================================================================== #
#  Watcher integration                                                 #
# ================================================================== #

class TestWatcher(DBTestCase):

    def _make_watch_folder(self):
        folder = tempfile.mkdtemp()
        copied = []
        for key, src in AVAILABLE_PDFS.items():
            dst = os.path.join(folder, src.name)
            shutil.copy(src, dst)
            copied.append(dst)
        return folder, copied

    def test_scan_imports_all_new_files(self):
        from src.watcher import scan_folder
        folder, copied = self._make_watch_folder()
        try:
            stats = scan_folder(folder, self.db)
            self.assertEqual(stats["found"],    len(copied))
            self.assertEqual(stats["new"],      len(copied))
            self.assertEqual(stats["imported"], len(copied))
            self.assertEqual(stats["skipped"],  0)
            self.assertEqual(stats["failed"],   0)
        finally:
            shutil.rmtree(folder)

    def test_scan_skips_already_imported(self):
        from src.watcher import scan_folder
        folder, _ = self._make_watch_folder()
        try:
            scan_folder(folder, self.db)            # first pass
            stats = scan_folder(folder, self.db)    # second pass
            self.assertEqual(stats["new"],      0)
            self.assertEqual(stats["imported"], 0)
            self.assertEqual(stats["skipped"],  stats["found"])
        finally:
            shutil.rmtree(folder)

    def test_dry_run_does_not_write(self):
        if not AVAILABLE_PDFS:
            self.skipTest(
                "No sample PDFs in data/samples/ — copy your statements there to run watcher tests"
            )
        from src.watcher import scan_folder
        # Own fresh DB — immune to test ordering and OS temp path reuse.
        fresh_db, fresh_db_path = make_tmp_db()
        folder, copied = self._make_watch_folder()
        try:
            stats = scan_folder(folder, fresh_db, dry_run=True)
            self.assertGreater(stats["new"], 0,
                msg="Expected new files — is AVAILABLE_PDFS empty?")
            self.assertEqual(stats["imported"], 0,
                msg="dry_run=True should not import anything")
            rows = fresh_db.get_transactions(limit=1)
            self.assertEqual(len(rows), 0,
                msg="DB should be empty after a dry run")
        finally:
            fresh_db.close()
            if os.path.exists(fresh_db_path):
                os.unlink(fresh_db_path)
            shutil.rmtree(folder)

    def test_empty_folder_returns_zero(self):
        from src.watcher import scan_folder
        folder = tempfile.mkdtemp()
        try:
            stats = scan_folder(folder, self.db)
            self.assertEqual(stats["found"],    0)
            self.assertEqual(stats["imported"], 0)
        finally:
            shutil.rmtree(folder)

    def test_missing_folder_returns_zero(self):
        from src.watcher import scan_folder
        stats = scan_folder("/nonexistent/path/statements", self.db)
        self.assertEqual(stats["found"], 0)

    def test_incremental_scan(self):
        """Second scan picks up only the file added between runs."""
        from src.watcher import scan_folder
        folder = tempfile.mkdtemp()
        try:
            keys = list(AVAILABLE_PDFS.keys())
            if len(keys) < 2:
                self.skipTest("Need at least 2 sample files")

            # First scan — one file
            src1 = AVAILABLE_PDFS[keys[0]]
            shutil.copy(src1, os.path.join(folder, src1.name))
            stats1 = scan_folder(folder, self.db)
            self.assertEqual(stats1["imported"], 1)

            # Add second file then rescan
            src2 = AVAILABLE_PDFS[keys[1]]
            shutil.copy(src2, os.path.join(folder, src2.name))
            stats2 = scan_folder(folder, self.db)
            self.assertEqual(stats2["imported"], 1)   # only the new one
            self.assertEqual(stats2["skipped"],  1)   # first is duplicate
        finally:
            shutil.rmtree(folder)


# ================================================================== #
#  CSV parser round-trip                                               #
# ================================================================== #

def _pandas_available():
    try:
        import pandas  # noqa
        return True
    except ImportError:
        return False


@unittest.skipUnless(_pandas_available(), "pandas not installed — pip install pandas")
class TestCSVParser(DBTestCase):

    def test_sample_csv_parses(self):
        from src.parsers.csv_parser import CSVStatementParser
        result = CSVStatementParser(str(SAMPLE_CSV)).parse()
        self.assertEqual(result["summary"]["total_transactions"], 23)
        self.assertAlmostEqual(
            result["summary"]["total_debits"], 1085.71, places=2)

    def test_csv_round_trip(self):
        from src.parsers.csv_parser import CSVStatementParser
        result    = CSVStatementParser(str(SAMPLE_CSV)).parse()
        file_hash = self.db.check_duplicate(str(SAMPLE_CSV))
        stmt_id   = self.db.save_statement(str(SAMPLE_CSV),
                                            bank_name="unknown",
                                            file_hash=file_hash)
        saved = self.db.save_transactions(
            result["transactions"], statement_id=stmt_id, bank_name="unknown"
        )
        self.assertEqual(saved, result["summary"]["total_transactions"])
        rows = self.db.get_transactions(limit=100)
        self.assertEqual(len(rows), saved)


if __name__ == "__main__":
    unittest.main(verbosity=2)
