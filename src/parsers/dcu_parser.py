"""
src/parsers/dcu_parser.py

DCU (Digital Federal Credit Union) bank statement PDF parser.

Parses three account sections from each DCU statement:
  1. PRIMARY SAVINGS — single-line transactions
  2. FREE CHECKING — multi-line transactions (description continuation)
  3. NEW VEHICLE LOAN — 3-column rows (payment / principal / balance)

Outputs rows shaped for:
  - bank_transactions_raw  (checking + savings)
  - loan_transactions  (loan payments)
  - bank_statements        (statement metadata)

Adapted from bank-statement-rag/src/ingest/parser.py with:
  - Date normalisation: JAN05 → 2025-01-05 (cross-year safe)
  - Transaction type detection: DEBIT/CREDIT from amount context
  - Output as dicts matching the A1 migration schema
  - Base class for future BOA/other bank parsers

Usage:
    from src.parsers.dcu_parser import DCUParser

    parser = DCUParser()
    result = parser.parse("data/raw/stmt_20250131.pdf")

    print(result.metadata)           # bank_statements row
    print(result.bank_rows[0])       # bank_transactions_raw row
    print(result.loan_rows[0])       # loan_transactions row
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional


import pdfplumber

from src.parsers.base_bank_parser import BaseBankParser, ParseResult


# ---------------------------------------------------------------------------
# DCU-specific patterns (from bank-statement-rag parser)
# ---------------------------------------------------------------------------

_MONTH = r"(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\d{2}"
_DATE = re.compile(rf"^{_MONTH}\s+")
_AMT = r"-?[\d,]+\.\d{2}"

SECTION = {
    "savings": re.compile(r"PRIMARY SAVINGS ACCT#\s*\d+"),
    "checking": re.compile(r"FREE CHECKING ACCT#\s*\d+"),
    "loan": re.compile(r"NEW VEHICLE LOAN#\s*(\d+)"),
    "summary": re.compile(
        r"S\s*T\s*A\s*T\s*E\s*M\s*E\s*N\s*T\s+S\s*U\s*M\s*M\s*A\s*R\s*Y"
    ),
    "boilerplate": re.compile(r"BILLING RIGHTS|Direct general inquiries|Rev:\s*\d"),
}

PERIOD = re.compile(r"(\d{2}-\d{2}-\d{2})\s+to\s+(\d{2}-\d{2}-\d{2})")

# Stop words for each section — lines containing these are metadata, not transactions
STOP_SAVINGS = {
    "NEW BALANCE", "PREVIOUS BALANCE", "ANNUAL PERCENTAGE YIELD",
}
STOP_CHECKING = {
    "DEPOSITS, DIVIDENDS", "WITHDRAWALS, FEES",
    "TOTAL DIVIDENDS", "TOTAL DEPOSITS", "TOTAL FEES",
    "TOTAL WITHDRAWALS", "NEW BALANCE", "PREVIOUS BALANCE",
    "DATE AMOUNT", "DATE TRANSACTION",
}
STOP_LOAN = {
    "INTEREST RATE DETAIL", "FEES CHARGED", "INTEREST CHARGED",
    "TOTALS YEAR", "EFFECTIVE DATES", "TRANSACTIONS",
    "TOTAL FEES", "TOTAL INTEREST",
}


# ---------------------------------------------------------------------------
# DCU Parser
# ---------------------------------------------------------------------------

class DCUParser(BaseBankParser):
    """
    Parses DCU bank statement PDFs into rows for bank_transactions_raw
    and loan_transactions tables.
    """

    bank_name = "dcu"

    # ── Public entry point ────────────────────────────────────────────────

    def parse(self, path: str | Path) -> ParseResult:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")

        lines = self._extract_lines(path)
        period_start, period_end = self._detect_period(lines)
        raw_sections = self._split_sections(lines)

        # Parse each section into raw transaction rows
        bank_rows = []
        bank_rows.extend(
            self._parse_savings(raw_sections["savings"], period_start, period_end)
        )
        bank_rows.extend(
            self._parse_checking(raw_sections["checking"], period_start, period_end)
        )
        loan_rows = self._parse_loan(
            raw_sections["loan"], period_start, period_end, lines
        )

        # Compute totals
        total_debits = sum(r["amount"] for r in bank_rows if r["transaction_type"] == "DEBIT")
        total_credits = sum(r["amount"] for r in bank_rows if r["transaction_type"] == "CREDIT")

        # Build statement metadata
        metadata = {
            "filename": path.name,
            "original_path": str(path),
            "file_hash": self.file_hash(path),
            "bank_name": self.bank_name,
            "statement_period_start": (
                self.normalise_period_date(period_start) if period_start else None
            ),
            "statement_period_end": (
                self.normalise_period_date(period_end) if period_end else None
            ),
            "total_transactions": len(bank_rows) + len(loan_rows),
            "total_debits": round(total_debits, 2),
            "total_credits": round(total_credits, 2),
        }

        return ParseResult(
            metadata=metadata,
            bank_rows=bank_rows,
            loan_rows=loan_rows,
        )

    # ── Text extraction ───────────────────────────────────────────────────

    def _extract_lines(self, path: Path) -> list[str]:
        """Extract all text lines, skipping boilerplate pages."""
        lines: list[str] = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                if SECTION["boilerplate"].search(text):
                    continue
                lines.extend(text.splitlines())
        return lines

    # ── Section splitting ─────────────────────────────────────────────────

    def _split_sections(self, lines: list[str]) -> dict[str, list[str]]:
        """Route lines into savings/checking/loan/summary buckets."""
        buckets: dict[str, list[str]] = {
            "savings": [], "checking": [], "loan": [], "summary": [],
        }
        current: Optional[str] = None

        for line in lines:
            s = line.strip()
            if not s:
                continue

            if SECTION["boilerplate"].search(s):
                current = None
                continue
            if SECTION["summary"].search(s):
                current = "summary"
                continue
            if SECTION["savings"].search(s):
                current = "savings"
                continue
            if SECTION["checking"].search(s):
                current = "checking"
                continue
            if SECTION["loan"].search(s):
                current = "loan"
                # Previous balance lives on this same header line
                buckets["loan"].append(s)
                continue

            if current:
                buckets[current].append(s)

        return buckets

    # ── Section parsers ───────────────────────────────────────────────────

    def _parse_savings(
        self,
        lines: list[str],
        period_start: str | None,
        period_end: str | None,
    ) -> list[dict]:
        """Parse savings transactions into bank_transactions_raw row dicts."""
        rows: list[dict] = []

        for line in lines:
            if any(kw in line for kw in STOP_SAVINGS):
                continue
            if not _DATE.match(line):
                continue

            nums = re.findall(_AMT, line)
            parts = line.split()
            if len(nums) >= 2:
                raw_date = parts[0]
                description = self._extract_description(line, raw_date, nums)
                amount = float(nums[-2].replace(",", ""))
                balance = float(nums[-1].replace(",", ""))
                txn_type = self._infer_transaction_type(description, amount)

                rows.append({
                    "bank_name": self.bank_name,
                    "account_type": "savings",
                    "date": self.normalise_date(raw_date, period_start, period_end),
                    "description": description,
                    "amount": abs(amount),
                    "transaction_type": txn_type,
                    "balance": balance,
                    "raw_text": line.strip(),
                })

        return rows

    def _parse_checking(
        self,
        lines: list[str],
        period_start: str | None,
        period_end: str | None,
    ) -> list[dict]:
        """
        Parse checking transactions into bank_transactions_raw row dicts.
        Handles multi-line transactions where description continues on next line(s).
        """
        rows: list[dict] = []
        pending: Optional[dict] = None

        for line in lines:
            if any(kw in line for kw in STOP_CHECKING):
                if pending:
                    rows.append(self._finalise_checking_row(pending))
                    pending = None
                continue

            if _DATE.match(line):
                # Reject inline summary rows with 2+ month abbreviations
                month_hits = re.findall(
                    r"\b(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\d{2}\b",
                    line,
                )
                if len(month_hits) > 1:
                    continue

                if pending:
                    rows.append(self._finalise_checking_row(pending))

                nums = re.findall(_AMT, line)
                parts = line.split()
                raw_date = parts[0]

                if len(nums) >= 2:
                    description = self._extract_description(line, raw_date, nums)
                    amount = float(nums[-2].replace(",", ""))
                    balance = float(nums[-1].replace(",", ""))

                    pending = {
                        "bank_name": self.bank_name,
                        "account_type": "checking",
                        "date": self.normalise_date(
                            raw_date, period_start, period_end
                        ),
                        "description": description,
                        "amount": abs(amount),
                        "_raw_amount": amount,  # signed; used for type inference after join
                        "transaction_type": None,  # filled by _finalise_checking_row
                        "balance": balance,
                        "raw_text": line.strip(),
                        "_continuation": [],
                    }
                else:
                    pending = None
            else:
                # Continuation line — belongs to previous transaction
                if pending:
                    pending["_continuation"].append(line.strip())

        if pending:
            rows.append(self._finalise_checking_row(pending))

        return rows

    def _parse_loan(
        self,
        lines: list[str],
        period_start: str | None,
        period_end: str | None,
        all_lines: list[str],
    ) -> list[dict]:
        """Parse loan transactions into loan_transactions row dicts."""
        rows: list[dict] = []
        loan_identifier = self._extract_loan_identifier(all_lines)

        for line in lines:
            if any(kw in line for kw in STOP_LOAN):
                continue
            if not _DATE.match(line):
                continue

            nums = re.findall(_AMT, line)
            parts = line.split()
            raw_date = parts[0]

            # Loan rows have 3 amounts: payment, principal, balance
            if len(nums) >= 3:
                description = self._extract_description(
                    line, raw_date, nums, n_amounts=3
                )
                payment = float(nums[-3].replace(",", ""))
                principal = float(nums[-2].replace(",", ""))
                balance = float(nums[-1].replace(",", ""))

                # Interest = payment - |principal|
                # Principal is typically negative (reduction), so use abs
                interest = round(payment - abs(principal), 2)

                rows.append({
                    "bank_name": self.bank_name,
                    "loan_identifier": loan_identifier,
                    "date": self.normalise_date(
                        raw_date, period_start, period_end
                    ),
                    "description": description,
                    "payment_amount": payment,
                    "principal_amount": abs(principal),
                    "interest_amount": max(interest, 0.0),
                    "balance": balance,
                    "raw_text": line.strip(),
                })

        return rows

    # ── Metadata extractors ───────────────────────────────────────────────

    def _detect_period(self, lines: list[str]) -> tuple[str | None, str | None]:
        """Extract statement period start/end from header lines."""
        for line in lines:
            m = PERIOD.search(line)
            if m:
                return m.group(1), m.group(2)
        return None, None

    def _extract_loan_identifier(self, lines: list[str]) -> str:
        """Extract loan identifier like 'NEW VEHICLE LOAN# 142'."""
        for line in lines:
            m = SECTION["loan"].search(line)
            if m:
                # Return the full section header as identifier
                return line.strip().split("PREVIOUS")[0].strip()
        return ""

    # ── Transaction type inference ────────────────────────────────────────

    @staticmethod
    def _infer_transaction_type(description: str, amount: float) -> str:
        """
        Infer DEBIT/CREDIT from the transaction amount sign.

        DCU statements use sign consistently:
          - Positive amount → CREDIT (salary, deposit, dividend, transfer in)
          - Negative amount → DEBIT  (withdrawal, loan payment, fee, transfer out)

        Keyword matching is only used as a last resort when amount is exactly 0.
        """
        if amount > 0:
            return "CREDIT"
        if amount < 0:
            return "DEBIT"

        # amount == 0 edge case: fall back to keywords
        desc = description.split(" | ")[0].upper()
        credit_keywords = ["DEPOSIT", "PAYROLL", "DIVIDEND", "REFUND", "ACH CREDIT", "DIRECT DEP"]
        if any(kw in desc for kw in credit_keywords):
            return "CREDIT"
        return "DEBIT"

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _extract_description(
        line: str,
        date: str,
        nums: list[str],
        n_amounts: int = 2,
    ) -> str:
        """
        Extract description text between the date and the trailing amounts.
        Works by finding where the last N amounts begin and slicing.
        """
        last_amounts = nums[-n_amounts:]
        search_from = line.index(date) + len(date)
        cutoff = len(line)

        for amt in last_amounts:
            idx = line.rfind(amt)
            if idx > search_from:
                cutoff = min(cutoff, idx)

        return line[search_from:cutoff].strip()

    def _finalise_checking_row(self, pending: dict) -> dict:
        """Join continuation lines into description, then infer transaction type."""
        description = pending["description"]
        continuations = pending.pop("_continuation", [])
        if continuations:
            description += " | " + " ".join(continuations)

        pending["description"] = description
        raw_amount = pending.pop("_raw_amount", pending["amount"])
        pending["transaction_type"] = self._infer_transaction_type(description, raw_amount)
        return pending