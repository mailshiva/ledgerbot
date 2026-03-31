"""
Unit Tests  —  run with:
    python -m unittest tests.test_unit -v        # no pytest needed
    pytest tests/test_unit.py -v                 # if pytest is installed
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path
import sqlite3

sys.path.insert(0, str(Path(__file__).parent.parent))
MIGRATION_PATH = 'src/database/migrate_schema.sql'

# ================================================================== #
#  RuleBasedExtractor                                                  #
# ================================================================== #

class TestDateExtraction(unittest.TestCase):

    def setUp(self):
        from src.extractors.rule_based import RuleBasedExtractor
        self.ex = RuleBasedExtractor()

    def test_iso_date(self):
        r = self.ex.extract_transactions("2024-01-15 STARBUCKS $5.67")
        self.assertEqual(len(r), 1)
        self.assertEqual(r[0]["date"], "2024-01-15")

    def test_slash_date(self):
        r = self.ex.extract_transactions("01/15/2024 NETFLIX $15.99")
        self.assertEqual(r[0]["date"], "2024-01-15")

    def test_short_slash_date(self):
        r = self.ex.extract_transactions("1/5/24 AMAZON $9.99")
        self.assertEqual(r[0]["date"], "2024-01-05")

    def test_header_line_skipped(self):
        text = "Date Description Amount\n2024-01-01 STARBUCKS $5.00"
        r = self.ex.extract_transactions(text)
        self.assertEqual(len(r), 1)

    def test_empty_lines_skipped(self):
        text = "\n\n2024-01-01 NETFLIX $15.99\n\n"
        self.assertEqual(len(self.ex.extract_transactions(text)), 1)


class TestAmountExtraction(unittest.TestCase):

    def setUp(self):
        from src.extractors.rule_based import RuleBasedExtractor
        self.ex = RuleBasedExtractor()

    def test_plain_amount(self):
        r = self.ex.extract_transactions("2024-03-01 WALMART $45.67")
        self.assertAlmostEqual(r[0]["amount"], 45.67)

    def test_comma_amount(self):
        r = self.ex.extract_transactions("2024-03-01 PAYROLL $3,000.00")
        self.assertAlmostEqual(r[0]["amount"], 3000.00)

    def test_negative_amount_is_debit(self):
        r = self.ex.extract_transactions("2024-03-01 REFUND -$12.50")
        self.assertAlmostEqual(r[0]["amount"], 12.50)

    def test_parentheses_amount_is_debit(self):
        r = self.ex.extract_transactions("2024-03-01 CHARGE (25.00)")
        self.assertEqual(r[0]["transaction_type"], "DEBIT")


class TestTransactionType(unittest.TestCase):

    def setUp(self):
        from src.extractors.rule_based import RuleBasedExtractor
        self.ex = RuleBasedExtractor()

    def test_credit_keyword(self):
        r = self.ex.extract_transactions("2024-03-01 PAYROLL DEPOSIT $3000.00")
        self.assertEqual(r[0]["transaction_type"], "CREDIT")

    def test_debit_keyword(self):
        r = self.ex.extract_transactions("2024-03-01 ATM WITHDRAWAL $200.00")
        self.assertEqual(r[0]["transaction_type"], "DEBIT")


class TestMerchantExtraction(unittest.TestCase):

    def setUp(self):
        from src.extractors.rule_based import RuleBasedExtractor
        self.ex = RuleBasedExtractor()

    CASES = [
        ("STARBUCKS #1234",         "Starbucks"),
        ("WHOLE FOODS MARKET",      "Whole Foods"),
        ("AMAZON.COM PURCHASE",     "Amazon"),
        ("NETFLIX SUBSCRIPTION",    "Netflix"),
        ("WAL-MART #4686",          "Walmart"),
        ("WM SUPERCENTER #100",     "Walmart"),
        ("SAMS CLUB #4969",         "Sam's Club"),
        ("CHICK-FIL-A #01604",      "Chick-fil-A"),
        ("MCDONALD'S F33300",       "McDonald's"),
        ("SHELL OIL 10089570005",   "Shell"),
        ("ANTHROPIC ANTHROPIC.COM", "Anthropic"),
    ]

    def test_merchant_standardisation(self):
        for desc, expected in self.CASES:
            with self.subTest(desc=desc):
                self.assertEqual(self.ex.extract_merchant_name(desc), expected)


class TestCategorisation(unittest.TestCase):

    def setUp(self):
        from src.extractors.rule_based import RuleBasedExtractor
        self.ex = RuleBasedExtractor()

    CASES = [
        ("Walmart",   "WM SUPERCENTER",       "Groceries"),
        ("Amazon",    "AMAZON.COM",           "Shopping"),
        ("Shell",     "SHELL OIL",            "Transportation"),
        ("Netflix",   "NETFLIX SUBSCRIPTION", "Entertainment"),
        ("Walgreens", "WALGREENS #16021",      "Health"),
        ("Anthropic", "ANTHROPIC.COM",         "Technology"),
    ]

    def test_categories(self):
        for merchant, desc, expected in self.CASES:
            with self.subTest(merchant=merchant):
                self.assertEqual(self.ex._categorise(merchant, desc), expected)


# ================================================================== #
#  PDFStatementParser — no file I/O                                   #
# ================================================================== #

class TestBankDetection(unittest.TestCase):

    def setUp(self):
        from src.parsers.pdf_parser import PDFStatementParser
        # Bypass __init__ (which checks file existence) for unit tests
        self.p = PDFStatementParser.__new__(PDFStatementParser)
        self.p.filepath = "/fake/path.pdf"
        self.p.filename = "path.pdf"

    CASES = [
        ("creditcards@robinhood.com payment",       "robinhood"),
        ("citicards.com Customer Service",          "citi"),
        ("Citi Double Cash Card statement",         "citi"),
        ("bankofamerica.com payment options",       "bofa"),
        ("Bank of America Visa Signature",          "bofa"),
        ("capitalone.com Venture Credit Card",      "capital_one"),
        ("Capital One mobile payment",              "capital_one"),
        ("chase.com account summary",               "chase"),
        ("wells fargo statement",                   "wells_fargo"),
        ("random text with no bank name",           "unknown"),
    ]

    def test_bank_detection(self):
        for text, expected in self.CASES:
            with self.subTest(expected=expected):
                self.assertEqual(self.p._detect_bank(text), expected)


class TestYearDetection(unittest.TestCase):

    def setUp(self):
        from src.parsers.pdf_parser import PDFStatementParser
        self.p = PDFStatementParser.__new__(PDFStatementParser)
        self.p.filepath = "/fake/path.pdf"
        self.p.filename = "path.pdf"

    # (text, filename, expected_year)
    CASES = [
        ("Billing Period: 01/17/26-02/17/26",         "",                          2026),
        ("August 6 - September 5, 2025",              "",                          2025),
        ("Statement Closing Date 09/05/2025",          "",                          2025),
        # Zip codes must NOT trigger year detection
        ("P.O. Box 75267-2050 Dallas TX 75285-1001",  "",                          2025),
        # Filename takes priority over text content
        ("no year in text at all",                    "Credit_Statement_2026.pdf", 2026),
        ("no year in text at all",                    "January2026.pdf",           2026),
        ("no year in text at all",                    "eStmt_2025-09-05.pdf",      2025),
        # Citi 2-digit year in text (no filename year)
        ("New balance as of 01/16/26: -$50.22",       "",                          2026),
        # Robinhood pdfplumber space-stripped date
        ("StatementClosingDate February04,2026",       "",                          2026),
    ]

    def test_year_detection(self):
        for text, filename, expected in self.CASES:
            with self.subTest(text=text[:40], filename=filename):
                self.assertEqual(self.p._detect_year(text, filename=filename), expected)

    def test_zip_code_not_detected_as_year(self):
        """Zip like 75267-2050 must not produce year 2050."""
        text = "P.O. Box 75267-2050 no real year here"
        year = self.p._detect_year(text)
        self.assertNotEqual(year, 2050)

    def test_to_iso_date(self):
        self.assertEqual(self.p._to_iso("08/06", 2025), "2025-08-06")
        self.assertEqual(self.p._to_iso("01/31", 2026), "2026-01-31")


# ================================================================== #
#  DatabaseManager                                                     #
# ================================================================== #

class TestDatabaseHash(unittest.TestCase):

    def test_hash_is_stable(self):
        from src.database.db_manager import DatabaseManager
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            f.write(b"fake pdf content")
            path = f.name
        try:
            h1 = DatabaseManager.compute_hash(path)
            h2 = DatabaseManager.compute_hash(path)
            self.assertEqual(h1, h2)
            self.assertEqual(len(h1), 64)  # SHA-256 hex
        finally:
            os.unlink(path)

    def test_hash_differs_for_different_content(self):
        from src.database.db_manager import DatabaseManager
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            f.write(b"content A"); path_a = f.name
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            f.write(b"content B"); path_b = f.name
        try:
            self.assertNotEqual(
                DatabaseManager.compute_hash(path_a),
                DatabaseManager.compute_hash(path_b),
            )
        finally:
            os.unlink(path_a); os.unlink(path_b)


class TestDatabaseSchema(unittest.TestCase):

    def setUp(self):
        from src.database.db_manager import DatabaseManager
        fd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.db = DatabaseManager(db_path=self.db_path)


    def tearDown(self):
        self.db.close()
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)

    def test_tables_created(self):
        tables = {
            r[0] for r in
            self.db.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        self.assertIn("transactions", tables)
        self.assertIn("statements", tables)

    def test_bank_name_column_in_transactions(self):
        cols = {
            r[1] for r in
            self.db.conn.execute("PRAGMA table_info(transactions)").fetchall()
        }
        self.assertIn("bank_name", cols)

    def test_file_hash_column_in_statements(self):
        cols = {
            r[1] for r in
            self.db.conn.execute("PRAGMA table_info(statements)").fetchall()
        }
        self.assertIn("file_hash", cols)
        self.assertIn("original_path", cols)


class TestDuplicateDetection(unittest.TestCase):

    def setUp(self):
        from src.database.db_manager import DatabaseManager
        fd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.db = DatabaseManager(db_path=self.db_path)
        fd2, self.pdf_path = tempfile.mkstemp(suffix=".pdf")
        os.close(fd2)
        with open(self.pdf_path, "wb") as f:
            f.write(b"statement content")

    def tearDown(self):
        self.db.close()
        for p in [self.db_path, self.pdf_path]:
            if os.path.exists(p):
                os.unlink(p)

    def test_first_check_returns_hash(self):
        from src.database.db_manager import DatabaseManager
        result = self.db.check_duplicate(self.pdf_path)
        self.assertIsInstance(result, str)
        self.assertEqual(len(result), 64)

    def test_second_import_raises(self):
        from src.database.db_manager import DuplicateStatementError
        fh = self.db.check_duplicate(self.pdf_path)
        self.db.save_statement(self.pdf_path, bank_name="test", file_hash=fh)
        with self.assertRaises(DuplicateStatementError):
            self.db.check_duplicate(self.pdf_path)

    def test_duplicate_error_message_informative(self):
        from src.database.db_manager import DuplicateStatementError
        fh = self.db.check_duplicate(self.pdf_path)
        self.db.save_statement(self.pdf_path, bank_name="test", file_hash=fh)
        try:
            self.db.check_duplicate(self.pdf_path)
        except DuplicateStatementError as e:
            self.assertIn("already imported", str(e).lower())


class TestTransactionPersistence(unittest.TestCase):

    def setUp(self):
        from src.database.db_manager import DatabaseManager
        fd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.db = DatabaseManager(db_path=self.db_path)
        fd2, self.pdf_path = tempfile.mkstemp(suffix=".pdf")
        os.close(fd2)
        with open(self.pdf_path, "wb") as f:
            f.write(b"content")
        fh = self.db.check_duplicate(self.pdf_path)
        self.stmt_id = self.db.save_statement(
            self.pdf_path, bank_name="bofa", file_hash=fh
        )

    def tearDown(self):
        self.db.close()
        for p in [self.db_path, self.pdf_path]:
            if os.path.exists(p):
                os.unlink(p)

    def _make_txns(self, bank="bofa"):
        return [
            {"date": "2025-08-06", "description": "AMAZON", "amount": 7.65,
             "transaction_type": "DEBIT", "merchant_name": "Amazon",
             "category": "Shopping", "confidence_score": 0.9},
            {"date": "2025-08-19", "description": "PAYMENT", "amount": 49.92,
             "transaction_type": "CREDIT", "merchant_name": "Payment",
             "category": "Income", "confidence_score": 1.0},
        ]

    def test_save_and_count(self):
        saved = self.db.save_transactions(
            self._make_txns(), statement_id=self.stmt_id, bank_name="bofa"
        )
        self.assertEqual(saved, 2)

    def test_bank_name_stored(self):
        self.db.save_transactions(
            self._make_txns(), statement_id=self.stmt_id, bank_name="bofa"
        )
        rows = self.db.get_transactions(limit=10)
        self.assertTrue(all(r["bank_name"] == "bofa" for r in rows))


    def test_monthly_spending(self):
        txns = [
            {"date": "2025-08-01", "description": "A", "amount": 10.0,
             "transaction_type": "DEBIT", "merchant_name": "A",
             "category": "Other", "confidence_score": 1.0},
            {"date": "2025-08-15", "description": "B", "amount": 20.0,
             "transaction_type": "DEBIT", "merchant_name": "B",
             "category": "Other", "confidence_score": 1.0},
            {"date": "2025-09-01", "description": "C", "amount": 5.0,
             "transaction_type": "CREDIT", "merchant_name": "C",
             "category": "Other", "confidence_score": 1.0},
        ]
        self.db.save_transactions(txns, statement_id=self.stmt_id)
        monthly = {r["month"]: r for r in self.db.get_monthly_spending()}
        self.assertIn("2025-08", monthly)
        self.assertAlmostEqual(monthly["2025-08"]["spending"], 30.0)
        self.assertIn("2025-09", monthly)
        self.assertAlmostEqual(monthly["2025-09"]["income"], 5.0)

    def test_summary_stats(self):
        self.db.save_transactions(
            self._make_txns(), statement_id=self.stmt_id, bank_name="bofa"
        )
        stats = self.db.get_summary_stats()
        self.assertEqual(stats["total"], 2)
        self.assertAlmostEqual(stats["total_debits"], 7.65, places=2)
        self.assertAlmostEqual(stats["total_credits"], 49.92, places=2)


# ================================================================== #
#  Config                                                              #
# ================================================================== #

class TestConfig(unittest.TestCase):

    def test_defaults_with_empty_dict(self):
        from src.config import Config, DEFAULTS
        cfg = Config({})
        self.assertEqual(cfg.db_path, DEFAULTS["database"]["path"])
        self.assertEqual(cfg.watch_folder, DEFAULTS["watcher"]["folder"])
        self.assertEqual(cfg.default_year, DEFAULTS["parser"]["default_year"])

    def test_override_db_path(self):
        from src.config import Config
        cfg = Config({"database": {"path": "/tmp/custom.db"}})
        self.assertEqual(cfg.db_path, "/tmp/custom.db")

    def test_override_watch_folder(self):
        from src.config import Config
        cfg = Config({"watcher": {"folder": "/tmp/my_statements"}})
        self.assertEqual(cfg.watch_folder, "/tmp/my_statements")

    def test_partial_override_keeps_other_defaults(self):
        from src.config import Config, DEFAULTS
        cfg = Config({"database": {"path": "/tmp/x.db"}})
        self.assertEqual(cfg.watch_folder, DEFAULTS["watcher"]["folder"])

    def test_watch_log_file_default(self):
        from src.config import Config, DEFAULTS
        cfg = Config({})
        self.assertEqual(cfg.watch_log_file, DEFAULTS["watcher"]["log_file"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
