"""
src/parsers/base_bank_parser.py

Abstract base class for bank statement PDF parsers.

Each bank-specific parser (DCU, BOA, etc.) extends this class and
implements the parse() method, which returns:
  - A StatementMetadata dict for the bank_statements table
  - A list of BankTransactionRow dicts for bank_transactions_raw
  - A list of LoanTransactionRow dicts for loan_transactions

This separation keeps the parsing logic bank-specific while the
DB insertion logic stays generic in the ingest pipeline.

Usage:
    from src.parsers.dcu_parser import DCUParser

    parser = DCUParser()
    result = parser.parse("data/raw/stmt_20250131.pdf")

    result.metadata       # dict → bank_statements row
    result.bank_rows      # list[dict] → bank_transactions_raw rows
    result.loan_rows      # list[dict] → loan_transactions rows
"""

from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Month mapping for date normalisation
# ---------------------------------------------------------------------------

MONTH_MAP = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}

# Pattern for raw dates like "JAN05", "DEC31"
RAW_DATE_PATTERN = re.compile(r"^([A-Z]{3})(\d{2})$")


# ---------------------------------------------------------------------------
# Output data classes
# ---------------------------------------------------------------------------

@dataclass
class ParseResult:
    """
    Structured output from a bank statement parser.

    metadata:   dict matching bank_statements columns
    bank_rows:  list of dicts matching bank_transactions_raw columns
    loan_rows:  list of dicts matching loan_transactions columns
    """
    metadata: dict
    bank_rows: list[dict] = field(default_factory=list)
    loan_rows: list[dict] = field(default_factory=list)

    @property
    def total_bank_transactions(self) -> int:
        return len(self.bank_rows)

    @property
    def total_loan_transactions(self) -> int:
        return len(self.loan_rows)

    @property
    def total_transactions(self) -> int:
        return self.total_bank_transactions + self.total_loan_transactions

    @property
    def total_debits(self) -> float:
        return sum(
            r["amount"] for r in self.bank_rows
            if r.get("transaction_type") == "DEBIT"
        )

    @property
    def total_credits(self) -> float:
        return sum(
            r["amount"] for r in self.bank_rows
            if r.get("transaction_type") == "CREDIT"
        )


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseBankParser(ABC):
    """
    Abstract base class for bank statement parsers.

    Subclasses must implement:
      - parse(path) → ParseResult
      - bank_name  (class attribute or property)
    """

    bank_name: str = ""  # e.g. "dcu", "boa"

    @abstractmethod
    def parse(self, path: str | Path) -> ParseResult:
        """
        Parse a single bank statement PDF and return structured data.

        Args:
            path: Path to the PDF file.

        Returns:
            ParseResult with metadata, bank_rows, and loan_rows.
        """
        ...

    # ── Shared utilities ──────────────────────────────────────────────────

    @staticmethod
    def file_hash(path: str | Path) -> str:
        """Compute SHA-256 hash of a file for idempotent imports."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def normalise_date(
        raw_date: str,
        period_start: str | None = None,
        period_end: str | None = None,
    ) -> str:
        """
        Convert raw bank date like "JAN05" to ISO "2025-01-05".

        Uses the statement period (e.g. "01-01-25" to "01-31-25") to
        determine the correct year. Handles cross-year statements where
        e.g. period is 12-01-24 to 01-31-25 and a DEC transaction
        belongs to 2024 while a JAN transaction belongs to 2025.

        Args:
            raw_date:     e.g. "JAN05", "DEC31"
            period_start: e.g. "01-01-25" or "12-01-24"
            period_end:   e.g. "01-31-25"

        Returns:
            ISO date string like "2025-01-05"
        """
        m = RAW_DATE_PATTERN.match(raw_date.upper())
        if not m:
            return raw_date  # already normalised or unrecognised

        month_name = m.group(1)
        day = int(m.group(2))
        month = MONTH_MAP.get(month_name)
        if not month:
            return raw_date

        # Determine year from statement period
        if period_end:
            # period_end like "01-31-25" → parse end month and 2-digit year
            parts = period_end.split("-")
            end_month = int(parts[0])
            end_year_2d = int(parts[2])
            # Convert 2-digit year: 00-69 → 2000s, 70-99 → 1900s
            end_year = 2000 + end_year_2d if end_year_2d < 70 else 1900 + end_year_2d

            if period_start:
                start_parts = period_start.split("-")
                start_month = int(start_parts[0])
                start_year_2d = int(start_parts[2])
                start_year = (
                    2000 + start_year_2d
                    if start_year_2d < 70
                    else 1900 + start_year_2d
                )

                # Cross-year detection:
                # If start is in a later month than end (e.g. Dec→Jan),
                # months >= start_month belong to start_year,
                # months <= end_month belong to end_year
                if start_month > end_month:
                    year = start_year if month >= start_month else end_year
                else:
                    year = end_year
            else:
                year = end_year
        else:
            # No period info — fall back to filename or current year
            year = 2025  # safe default

        return f"{year}-{month:02d}-{day:02d}"

    @staticmethod
    def normalise_period_date(raw: str) -> str:
        """
        Convert period date like "01-31-25" to ISO "2025-01-31".

        Args:
            raw: e.g. "01-31-25"

        Returns:
            ISO date string like "2025-01-31"
        """
        if not raw:
            return ""
        parts = raw.split("-")
        if len(parts) != 3:
            return raw
        mm, dd, yy = parts
        year = 2000 + int(yy) if int(yy) < 70 else 1900 + int(yy)
        return f"{year}-{int(mm):02d}-{int(dd):02d}"