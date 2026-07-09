"""
src/parsers/chase_parser.py

JPMorgan Chase bank statement PDF parser.

Parses the "Transaction Detail" section from Chase Total Checking statements.
Section is bounded by literal markers:  *start*transaction detail
                                        *end*transaction detail

Date format:  MM/DD  (year inferred from statement period)
Period line:  "May 14, 2026throughJune 11, 2026"  (no spaces around "through")

Multi-line transactions (continuation lines) are joined with " | ".

Outputs rows shaped for:
  - bank_transactions_raw  (account_type = "checking")
  - bank_statements        (statement metadata)

Usage:
    from src.parsers.chase_parser import ChaseParser

    parser = ChaseParser()
    result = parser.parse("data/raw/ED02167E-...-list.pdf")

    print(result.metadata)       # bank_statements row
    print(result.bank_rows[0])   # bank_transactions_raw row
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import pdfplumber

from src.parsers.base_bank_parser import BaseBankParser, ParseResult


# ---------------------------------------------------------------------------
# Chase-specific patterns
# ---------------------------------------------------------------------------

# Period: "May 14, 2026throughJune 11, 2026" (no space around "through")
_PERIOD = re.compile(
    r"(\w+\s+\d{1,2},\s*\d{4})\s*through\s*(\w+\s+\d{1,2},\s*\d{4})",
    re.IGNORECASE,
)

_MONTH_NAMES = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12,
}

# Transaction line: starts with MM/DD, optionally a second MM/DD for posted date
_TXN_DATE = re.compile(r"^(\d{2}/\d{2})(?:\s+\d{2}/\d{2})?\s+")

# Monetary values that appear on transaction lines (must have exactly 2 decimal places)
_AMT = r"-?[\d,]+\.\d{2}"

# Lines inside the transaction section that are noise / headers
_SKIP_LINE = re.compile(
    r"^(?:"
    r"TRANSACTION DETAIL"
    r"|DATE\s+DESCRIPTION"
    r"|Beginning Balance"
    r"|Ending Balance"
    r"|Page\s+\d"
    r")",
    re.IGNORECASE,
)

# Repeated page headers (period header + account number line repeat on every page)
_PAGE_PERIOD = re.compile(
    r"\w+\s+\d{1,2},\s*\d{4}\s*through\s*\w+\s+\d{1,2},\s*\d{4}",
    re.IGNORECASE,
)
_ACCOUNT_NUMBER_LINE = re.compile(r"^Account\s+Number:", re.IGNORECASE)

# Trailing label noise appended to description before the amount column
_TRAILING_LABEL = re.compile(r"\s*Transaction#:\s*$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Chase Parser
# ---------------------------------------------------------------------------

class ChaseParser(BaseBankParser):
    """
    Parses JPMorgan Chase bank statement PDFs into rows for bank_transactions_raw.
    Handles Chase Total Checking statements with *start*/*end* section markers.
    """

    bank_name = "chase"

    # ── Public entry point ────────────────────────────────────────────────

    def parse(self, path: str | Path) -> ParseResult:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")

        lines = self._extract_lines(path)
        period_start, period_end = self._detect_period(lines)
        bank_rows = self._parse_transactions(lines, period_start, period_end)

        total_debits = sum(
            r["amount"] for r in bank_rows if r["transaction_type"] == "DEBIT"
        )
        total_credits = sum(
            r["amount"] for r in bank_rows if r["transaction_type"] == "CREDIT"
        )

        metadata = {
            "filename": path.name,
            "original_path": str(path),
            "file_hash": self.file_hash(path),
            "bank_name": self.bank_name,
            "statement_period_start": period_start,
            "statement_period_end": period_end,
            "total_transactions": len(bank_rows),
            "total_debits": round(total_debits, 2),
            "total_credits": round(total_credits, 2),
        }

        return ParseResult(metadata=metadata, bank_rows=bank_rows, loan_rows=[])

    # ── Text extraction ───────────────────────────────────────────────────

    def _extract_lines(self, path: Path) -> list[str]:
        """Extract text lines from every page of the PDF."""
        lines: list[str] = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                lines.extend(text.splitlines())
        return lines

    # ── Period detection ──────────────────────────────────────────────────

    def _detect_period(self, lines: list[str]) -> tuple[Optional[str], Optional[str]]:
        """
        Extract ISO start/end from 'May 14, 2026throughJune 11, 2026'.
        Returns (period_start, period_end) as 'YYYY-MM-DD' strings.
        """
        for line in lines:
            m = _PERIOD.search(line)
            if m:
                return (
                    self._month_date_to_iso(m.group(1)),
                    self._month_date_to_iso(m.group(2)),
                )
        return None, None

    @staticmethod
    def _month_date_to_iso(raw: str) -> Optional[str]:
        """Convert 'May 14, 2026' → '2026-05-14'."""
        m = re.match(r"(\w+)\s+(\d{1,2}),\s*(\d{4})", raw.strip())
        if not m:
            return None
        month = _MONTH_NAMES.get(m.group(1))
        if not month:
            return None
        return f"{int(m.group(3))}-{month:02d}-{int(m.group(2)):02d}"

    # ── Date normalisation ────────────────────────────────────────────────

    def _normalise_txn_date(
        self,
        raw: str,
        period_start: Optional[str],
        period_end: Optional[str],
    ) -> str:
        """
        Convert 'MM/DD' → 'YYYY-MM-DD'.

        Year is taken from the statement period. Cross-year statements
        (e.g. Dec → Jan) are handled: months in the start-year half
        use start_year; months in the end-year half use end_year.
        """
        m = re.match(r"(\d{2})/(\d{2})", raw)
        if not m:
            return raw
        mm, dd = int(m.group(1)), int(m.group(2))

        if period_end:
            end_year = int(period_end[:4])
            if period_start:
                start_year = int(period_start[:4])
                start_month = int(period_start[5:7])
                if start_year != end_year:
                    # Cross-year: txn month >= period start month → start year
                    year = start_year if mm >= start_month else end_year
                else:
                    year = end_year
            else:
                year = end_year
        else:
            year = 2026  # safe fallback

        return f"{year}-{mm:02d}-{dd:02d}"

    # ── Transaction parsing ───────────────────────────────────────────────

    def _parse_transactions(
        self,
        lines: list[str],
        period_start: Optional[str],
        period_end: Optional[str],
    ) -> list[dict]:
        """
        Parse lines inside *start*transaction detail … *end*transaction detail.

        State machine:
          in_section = True  → inside the TRANSACTION DETAIL block
          pending            → last seen transaction row (may get continuations)

        Transaction line format:
          MM/DD [MM/DD] <description> <amount> <balance>

        Continuation lines (no leading date) are appended to the pending
        transaction description with ' | ' separator.

        Amount sign determines type:
          negative → DEBIT   (outgoing payment / transfer)
          positive → CREDIT  (deposit / incoming transfer)
        """
        rows: list[dict] = []
        in_section = False
        pending: Optional[dict] = None

        def finalise() -> None:
            """Flush pending row: join continuations then append."""
            if pending is None:
                return
            conts = pending.pop("_continuation", [])
            if conts:
                pending["description"] += " | " + " | ".join(conts)
            rows.append(pending)

        for line in lines:
            s = line.strip()
            if not s:
                continue

            # ── Section boundary markers ───────────────────────────────
            if s.lower() == "*start*transaction detail":
                in_section = True
                continue
            if s.lower() == "*end*transaction detail":
                finalise()
                pending = None
                in_section = False
                continue

            if not in_section:
                continue

            # ── Skip noise / header lines ──────────────────────────────
            if _SKIP_LINE.match(s):
                finalise()
                pending = None
                continue
            # Period header and account number repeat on every page — ignore
            if _PAGE_PERIOD.search(s) or _ACCOUNT_NUMBER_LINE.match(s):
                continue

            # ── Transaction line ───────────────────────────────────────
            date_m = _TXN_DATE.match(s)
            if date_m:
                finalise()
                raw_date = date_m.group(1)   # first MM/DD token
                desc_start = date_m.end()    # char index after date(s)

                # Find all decimal-format numbers on this line
                amt_matches = list(re.finditer(_AMT, s))
                if len(amt_matches) < 2:
                    # Can't determine both amount and balance — skip
                    pending = None
                    continue

                amount_str = amt_matches[-2].group()
                balance_str = amt_matches[-1].group()
                amount_raw = float(amount_str.replace(",", ""))
                balance = abs(float(balance_str.replace(",", "")))
                txn_type = "DEBIT" if amount_raw < 0 else "CREDIT"

                # Description sits between date end and the amount column
                desc_end = amt_matches[-2].start()
                description = s[desc_start:desc_end].strip()
                # Strip trailing label noise ("Transaction#:")
                description = _TRAILING_LABEL.sub("", description).strip()

                pending = {
                    "bank_name": self.bank_name,
                    "account_type": "checking",
                    "date": self._normalise_txn_date(
                        raw_date, period_start, period_end
                    ),
                    "description": description,
                    "amount": abs(amount_raw),
                    "transaction_type": txn_type,
                    "balance": balance,
                    "raw_text": s,
                    "_continuation": [],
                }
                continue

            # ── Continuation line ──────────────────────────────────────
            if pending is not None:
                pending["_continuation"].append(s)

        finalise()
        return rows
