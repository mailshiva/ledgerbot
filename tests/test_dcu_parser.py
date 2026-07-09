"""
Tests for DCU parser (dcu_parser.py) and base bank parser (base_bank_parser.py)

Tests:
  - Date normalisation (JAN05 → 2025-01-05, cross-year handling)
  - Period date normalisation (01-31-25 → 2025-01-31)
  - Transaction type inference (DEBIT/CREDIT)
  - Savings section parsing
  - Checking section parsing (including multi-line continuation)
  - Loan section parsing
  - Output row shapes match bank_transactions_raw / loan_transactions schema
  - Section splitting logic
  - Multi-date line rejection (inline summary rows)
  - Stop-word filtering

Uses synthetic line data — no real PDFs needed.




"""

from __future__ import annotations

import re
import sqlite3
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.parsers.base_bank_parser import BaseBankParser, ParseResult, MONTH_MAP, RAW_DATE_PATTERN
from src.parsers.dcu_parser import DCUParser


# ---------------------------------------------------------------------------
# Date normalisation
# ---------------------------------------------------------------------------

class TestDateNormalisation:
    def test_basic_date(self):
        assert BaseBankParser.normalise_date("JAN05", "01-01-25", "01-31-25") == "2025-01-05"

    def test_february(self):
        assert BaseBankParser.normalise_date("FEB14", "02-01-25", "02-28-25") == "2025-02-14"

    def test_december(self):
        assert BaseBankParser.normalise_date("DEC25", "12-01-24", "12-31-24") == "2024-12-25"

    def test_cross_year_dec_belongs_to_start_year(self):
        # Period: Dec 2024 → Jan 2025
        # DEC31 should be 2024
        assert BaseBankParser.normalise_date("DEC31", "12-01-24", "01-31-25") == "2024-12-31"

    def test_cross_year_jan_belongs_to_end_year(self):
        # Period: Dec 2024 → Jan 2025
        # JAN03 should be 2025
        assert BaseBankParser.normalise_date("JAN03", "12-01-24", "01-31-25") == "2025-01-03"

    def test_no_period_uses_default_year(self):
        result = BaseBankParser.normalise_date("MAR15", None, None)
        assert result == "2025-03-15"

    def test_only_end_period(self):
        assert BaseBankParser.normalise_date("APR10", None, "04-30-25") == "2025-04-10"

    def test_already_normalised_passes_through(self):
        assert BaseBankParser.normalise_date("2025-01-05", "01-01-25", "01-31-25") == "2025-01-05"

    def test_lowercase_rejected(self):
        # normalise_date uppercases, but "jan05" should still work
        assert BaseBankParser.normalise_date("jan05", "01-01-25", "01-31-25") == "2025-01-05"

    def test_quarterly_statement(self):
        # Period: Apr → Jun 2022
        assert BaseBankParser.normalise_date("MAY15", "04-01-22", "06-30-22") == "2022-05-15"


class TestPeriodDateNormalisation:
    def test_basic(self):
        assert BaseBankParser.normalise_period_date("01-31-25") == "2025-01-31"

    def test_december(self):
        assert BaseBankParser.normalise_period_date("12-01-24") == "2024-12-01"

    def test_empty(self):
        assert BaseBankParser.normalise_period_date("") == ""

    def test_y2k_range(self):
        assert BaseBankParser.normalise_period_date("06-30-22") == "2022-06-30"


# ---------------------------------------------------------------------------
# Transaction type inference
# ---------------------------------------------------------------------------

class TestTransactionTypeInference:
    def test_payroll_is_credit(self):
        assert DCUParser._infer_transaction_type("PAYROLL DIRECT DEP", 2400.0) == "CREDIT"

    def test_deposit_is_credit(self):
        assert DCUParser._infer_transaction_type("MOBILE DEPOSIT", 500.0) == "CREDIT"

    def test_dividend_is_credit(self):
        assert DCUParser._infer_transaction_type("DIVIDEND", 2.15) == "CREDIT"

    def test_transfer_from_is_credit(self):
        assert DCUParser._infer_transaction_type("TRANSFER FROM SAVINGS", 200.0) == "CREDIT"

    def test_purchase_is_debit(self):
        # DCU debits always carry negative amounts in the PDF
        assert DCUParser._infer_transaction_type("DEBIT CARD PURCHASE", -45.00) == "DEBIT"

    def test_atm_is_debit(self):
        assert DCUParser._infer_transaction_type("ATM WITHDRAWAL", -200.0) == "DEBIT"

    def test_fee_is_debit(self):
        assert DCUParser._infer_transaction_type("MONTHLY FEE", -12.0) == "DEBIT"

    def test_transfer_to_is_debit(self):
        assert DCUParser._infer_transaction_type("TRANSFER TO SAVINGS", -500.0) == "DEBIT"

    def test_negative_amount_is_debit(self):
        # DCU: negative = DEBIT regardless of description
        assert DCUParser._infer_transaction_type("UNKNOWN ENTRY", -50.0) == "DEBIT"

    def test_unknown_positive_is_credit(self):
        # DCU: positive = CREDIT regardless of description
        assert DCUParser._infer_transaction_type("STOP & SHOP", 85.40) == "CREDIT"

    def test_check_is_debit(self):
        assert DCUParser._infer_transaction_type("CHECK # 1234", -100.0) == "DEBIT"


# ---------------------------------------------------------------------------
# Savings parsing
# ---------------------------------------------------------------------------

class TestParseSavings:
    def setup_method(self):
        self.parser = DCUParser()

    def test_basic_savings_line(self):
        lines = [
            "JAN02 TRANSFER FROM CHECKING 500.00 5,500.00",
        ]
        rows = self.parser._parse_savings(lines, "01-01-25", "01-31-25")
        assert len(rows) == 1
        row = rows[0]
        assert row["date"] == "2025-01-02"
        assert row["account_type"] == "savings"
        assert row["amount"] == 500.00
        assert row["balance"] == 5500.00
        assert row["transaction_type"] == "CREDIT"
        assert row["bank_name"] == "dcu"

    def test_dividend_entry(self):
        lines = [
            "JAN31 DIVIDEND 2.15 5,502.15",
        ]
        rows = self.parser._parse_savings(lines, "01-01-25", "01-31-25")
        assert len(rows) == 1
        assert rows[0]["transaction_type"] == "CREDIT"
        assert rows[0]["amount"] == 2.15

    def test_stop_words_filtered(self):
        lines = [
            "PREVIOUS BALANCE 5,000.00",
            "JAN02 TRANSFER FROM CHECKING 500.00 5,500.00",
            "NEW BALANCE 5,502.15",
        ]
        rows = self.parser._parse_savings(lines, "01-01-25", "01-31-25")
        assert len(rows) == 1

    def test_empty_section(self):
        rows = self.parser._parse_savings([], "01-01-25", "01-31-25")
        assert rows == []

    def test_raw_text_preserved(self):
        lines = ["JAN05 SOME DEPOSIT 100.00 5,100.00"]
        rows = self.parser._parse_savings(lines, "01-01-25", "01-31-25")
        assert rows[0]["raw_text"] == "JAN05 SOME DEPOSIT 100.00 5,100.00"


# ---------------------------------------------------------------------------
# Checking parsing
# ---------------------------------------------------------------------------

class TestParseChecking:
    def setup_method(self):
        self.parser = DCUParser()

    def test_basic_checking_line(self):
        # DCU: purchases carry negative amounts
        lines = [
            "JAN05 STOP & SHOP -85.40 1,200.00",
        ]
        rows = self.parser._parse_checking(lines, "01-01-25", "01-31-25")
        assert len(rows) == 1
        row = rows[0]
        assert row["date"] == "2025-01-05"
        assert row["account_type"] == "checking"
        assert row["amount"] == 85.40
        assert row["balance"] == 1200.00
        assert row["transaction_type"] == "DEBIT"

    def test_multiline_continuation(self):
        lines = [
            "JAN10 ACH DEBIT EVERSOURCE -145.00 1,055.00",
            "ENERGY ELECTRIC BILL",
        ]
        rows = self.parser._parse_checking(lines, "01-01-25", "01-31-25")
        assert len(rows) == 1
        assert "ENERGY ELECTRIC BILL" in rows[0]["description"]
        assert " | " in rows[0]["description"]

    def test_two_transactions_back_to_back(self):
        lines = [
            "JAN05 STOP & SHOP -85.40 1,200.00",
            "JAN10 PAYROLL DIRECT DEP 2,400.00 3,600.00",
        ]
        rows = self.parser._parse_checking(lines, "01-01-25", "01-31-25")
        assert len(rows) == 2
        assert rows[0]["transaction_type"] == "DEBIT"
        assert rows[1]["transaction_type"] == "CREDIT"

    def test_multi_date_line_rejected(self):
        """Lines like 'JAN03 2,644.58 JAN17 2,692.55' are inline summaries."""
        lines = [
            "JAN03 2,644.58 JAN17 2,692.55",
        ]
        rows = self.parser._parse_checking(lines, "01-01-25", "01-31-25")
        assert len(rows) == 0

    def test_stop_words_flush_pending(self):
        lines = [
            "JAN05 STOP & SHOP -85.40 1,200.00",
            "DEPOSITS, DIVIDENDS",
            "JAN10 PAYROLL DIRECT DEP 2,400.00 3,600.00",
        ]
        rows = self.parser._parse_checking(lines, "01-01-25", "01-31-25")
        assert len(rows) == 2

    def test_empty_section(self):
        rows = self.parser._parse_checking([], "01-01-25", "01-31-25")
        assert rows == []


# ---------------------------------------------------------------------------
# Loan parsing
# ---------------------------------------------------------------------------

class TestParseLoan:
    def setup_method(self):
        self.parser = DCUParser()

    def test_basic_loan_line(self):
        all_lines = ["NEW VEHICLE LOAN# 142 PREVIOUS BALANCE 19,000.00"]
        lines = [
            "JAN15 LOAN PAYMENT 450.00 -320.00 18,680.00",
        ]
        rows = self.parser._parse_loan(lines, "01-01-25", "01-31-25", all_lines)
        assert len(rows) == 1
        row = rows[0]
        assert row["date"] == "2025-01-15"
        assert row["payment_amount"] == 450.00
        assert row["principal_amount"] == 320.00  # abs value
        assert row["balance"] == 18680.00
        assert row["bank_name"] == "dcu"
        assert "LOAN# 142" in row["loan_identifier"]

    def test_interest_calculated(self):
        all_lines = ["NEW VEHICLE LOAN# 142 PREVIOUS BALANCE 19,000.00"]
        lines = [
            "JAN15 PAYMENT 450.00 -320.00 18,680.00",
        ]
        rows = self.parser._parse_loan(lines, "01-01-25", "01-31-25", all_lines)
        # interest = 450 - 320 = 130
        assert rows[0]["interest_amount"] == 130.00

    def test_stop_words_filtered(self):
        all_lines = ["NEW VEHICLE LOAN# 142"]
        lines = [
            "INTEREST RATE DETAIL",
            "JAN15 PAYMENT 450.00 -320.00 18,680.00",
            "TOTALS YEAR TO DATE",
        ]
        rows = self.parser._parse_loan(lines, "01-01-25", "01-31-25", all_lines)
        assert len(rows) == 1

    def test_empty_section(self):
        rows = self.parser._parse_loan([], "01-01-25", "01-31-25", [])
        assert rows == []

    def test_raw_text_preserved(self):
        all_lines = ["NEW VEHICLE LOAN# 142"]
        lines = ["JAN15 PAYMENT 450.00 -320.00 18,680.00"]
        rows = self.parser._parse_loan(lines, "01-01-25", "01-31-25", all_lines)
        assert rows[0]["raw_text"] == "JAN15 PAYMENT 450.00 -320.00 18,680.00"


# ---------------------------------------------------------------------------
# Section splitting
# ---------------------------------------------------------------------------

class TestSplitSections:
    def setup_method(self):
        self.parser = DCUParser()

    def test_routes_to_correct_buckets(self):
        lines = [
            "PRIMARY SAVINGS ACCT# 1",
            "JAN02 DEPOSIT 500.00 5,500.00",
            "FREE CHECKING ACCT# 2",
            "JAN05 STOP & SHOP 85.40 1,200.00",
            "NEW VEHICLE LOAN# 142 PREVIOUS BALANCE 19,000.00",
            "JAN15 PAYMENT 450.00 -320.00 18,680.00",
        ]
        buckets = self.parser._split_sections(lines)
        assert len(buckets["savings"]) == 1
        assert len(buckets["checking"]) == 1
        assert len(buckets["loan"]) == 2  # header line + transaction
        assert len(buckets["summary"]) == 0

    def test_boilerplate_resets_current(self):
        lines = [
            "FREE CHECKING ACCT# 2",
            "JAN05 STOP & SHOP 85.40 1,200.00",
            "BILLING RIGHTS SUMMARY",
            "Some boilerplate text that should be ignored",
        ]
        buckets = self.parser._split_sections(lines)
        assert len(buckets["checking"]) == 1

    def test_summary_captured(self):
        lines = [
            "S T A T E M E N T  S U M M A R Y",
            "Account summary line 1",
            "Account summary line 2",
        ]
        buckets = self.parser._split_sections(lines)
        assert len(buckets["summary"]) == 2


# ---------------------------------------------------------------------------
# Output row schema validation
# ---------------------------------------------------------------------------

class TestOutputSchema:
    """Verify that output rows have the right keys for DB insertion."""

    def setup_method(self):
        self.parser = DCUParser()

    def test_bank_row_has_required_keys(self):
        lines = ["JAN05 STOP & SHOP 85.40 1,200.00"]
        rows = self.parser._parse_savings(lines, "01-01-25", "01-31-25")
        row = rows[0]

        required_keys = {
            "bank_name", "account_type", "date", "description",
            "amount", "transaction_type", "balance", "raw_text",
        }
        assert required_keys.issubset(set(row.keys()))

    def test_loan_row_has_required_keys(self):
        all_lines = ["NEW VEHICLE LOAN# 142"]
        lines = ["JAN15 PAYMENT 450.00 -320.00 18,680.00"]
        rows = self.parser._parse_loan(lines, "01-01-25", "01-31-25", all_lines)
        row = rows[0]

        required_keys = {
            "bank_name", "loan_identifier", "date", "description",
            "payment_amount", "principal_amount", "interest_amount",
            "balance", "raw_text",
        }
        assert required_keys.issubset(set(row.keys()))

    def test_bank_row_inserts_into_sqlite(self):
        """Verify rows actually insert into the bank_transactions_raw schema."""
        from src.database.migrate_bank_tables import SQLITE_DDL

        conn = sqlite3.connect(":memory:")
        conn.executescript(SQLITE_DDL)

        # Insert a statement first (FK target)
        conn.execute("""
            INSERT INTO bank_statements
                (filename, original_path, file_hash, bank_name)
            VALUES ('test.pdf', '/test.pdf', 'testhash', 'dcu')
        """)

        lines = ["JAN05 STOP & SHOP 85.40 1,200.00"]
        rows = self.parser._parse_savings(lines, "01-01-25", "01-31-25")
        row = rows[0]

        conn.execute("""
            INSERT INTO bank_transactions_raw
                (statement_id, bank_name, account_type, date, description,
                 amount, transaction_type, balance, raw_text)
            VALUES (1, :bank_name, :account_type, :date, :description,
                    :amount, :transaction_type, :balance, :raw_text)
        """, row)
        conn.commit()

        result = conn.execute("SELECT * FROM bank_transactions_raw").fetchone()
        assert result is not None

    def test_loan_row_inserts_into_sqlite(self):
        """Verify loan rows actually insert into loan_transactions."""
        from src.database.migrate_bank_tables import SQLITE_DDL

        conn = sqlite3.connect(":memory:")
        conn.executescript(SQLITE_DDL)

        conn.execute("""
            INSERT INTO bank_statements
                (filename, original_path, file_hash, bank_name)
            VALUES ('test.pdf', '/test.pdf', 'testhash', 'dcu')
        """)

        all_lines = ["NEW VEHICLE LOAN# 142"]
        lines = ["JAN15 PAYMENT 450.00 -320.00 18,680.00"]
        rows = self.parser._parse_loan(lines, "01-01-25", "01-31-25", all_lines)
        row = rows[0]

        conn.execute("""
            INSERT INTO loan_transactions
                (statement_id, bank_name, loan_identifier, date, description,
                 payment_amount, principal_amount, interest_amount, balance, raw_text)
            VALUES (1, :bank_name, :loan_identifier, :date, :description,
                    :payment_amount, :principal_amount, :interest_amount,
                    :balance, :raw_text)
        """, row)
        conn.commit()

        result = conn.execute("SELECT * FROM loan_transactions").fetchone()
        assert result is not None


# ---------------------------------------------------------------------------
# ParseResult
# ---------------------------------------------------------------------------

class TestParseResult:
    def test_totals(self):
        result = ParseResult(
            metadata={"filename": "test.pdf"},
            bank_rows=[
                {"amount": 85.40, "transaction_type": "DEBIT"},
                {"amount": 2400.00, "transaction_type": "CREDIT"},
                {"amount": 145.00, "transaction_type": "DEBIT"},
            ],
            loan_rows=[
                {"payment_amount": 450.00},
            ],
        )
        assert result.total_bank_transactions == 3
        assert result.total_loan_transactions == 1
        assert result.total_transactions == 4
        assert result.total_debits == 230.40
        assert result.total_credits == 2400.00


# ---------------------------------------------------------------------------
# File hash
# ---------------------------------------------------------------------------

class TestFileHash:
    def test_hash_is_deterministic(self, tmp_path):
        f = tmp_path / "test.pdf"
        f.write_bytes(b"fake pdf content")
        h1 = BaseBankParser.file_hash(f)
        h2 = BaseBankParser.file_hash(f)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex length

    def test_different_content_different_hash(self, tmp_path):
        f1 = tmp_path / "a.pdf"
        f2 = tmp_path / "b.pdf"
        f1.write_bytes(b"content A")
        f2.write_bytes(b"content B")
        assert BaseBankParser.file_hash(f1) != BaseBankParser.file_hash(f2)