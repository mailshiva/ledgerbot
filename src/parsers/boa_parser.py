"""
src/parsers/boa_parser.py

Bank of America (BOA) bank statement PDF parser.

Parses three transaction sections from each BOA statement:
  1. Deposits and other additions  → CREDIT
  2. Other subtractions            → DEBIT
  3. Checks                        → DEBIT

Date format:  MM/DD/YY  (e.g. 03/19/26)
Period line:  "for March 7, 2026 to April 7, 2026"

Multi-line transactions (continuation ID lines) are joined with " | ".

Outputs rows shaped for:
  - bank_transactions_raw  (account_type = "checking")
  - bank_statements        (statement metadata)

Usage:
    from src.parsers.boa_parser import BOAParser

    parser = BOAParser()
    result = parser.parse("data/raw/eStmt_2026-04-07.pdf")

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
# BOA-specific patterns
# ---------------------------------------------------------------------------

# Transaction line starts with MM/DD/YY followed by a space
_DATE = re.compile(r"^\d{2}/\d{2}/\d{2} ")
_AMT = r"-?[\d,]+\.\d{2}"

# Month name → int for period parsing
_MONTH_NAMES = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12,
}

# Statement period: "for March 7, 2026 to April 7, 2026"
#                or "March 7, 2026 to April 7, 2026"
_PERIOD = re.compile(
    r"(?:for\s+)?(\w+\s+\d{1,2},\s+\d{4})\s+to\s+(\w+\s+\d{1,2},\s+\d{4})"
)

# Account number: "Account number: 4870 0377 9414" or "Account # 4870 0377 9414"
_ACCT = re.compile(r"Account\s+(?:number|#)[:\s]+(\d[\d\s]+\d)")

# Section headers
_SEC_DEPOSITS = re.compile(r"Deposits and other additions", re.IGNORECASE)
_SEC_SUBTRACTIONS = re.compile(
    r"(?:Other subtractions|Withdrawals and other subtractions)", re.IGNORECASE
)
_SEC_CHECKS = re.compile(r"^Checks$")

# Page / repetitive header line that contains the account number
_PAGE_HEADER = re.compile(r"Account\s+#\s*\d")

# Lines that mark the end of a section or are pure noise
_STOP = re.compile(
    r"^(?:"
    r"Total"
    r"|Date\s+(?:Description|Check)"
    r"|continued on"
    r"|Pause and verify"
    r"|Braille"
    r"|Page \d+"
    r"|PULL:"
    r"|When you use the QRC"
    r"|Scan the code"
    r"|Learn more"
    r"|- Scammers"
    r"|- Hang up"
    r"|If you want to verify"
    r"|for business purposes"
    r")"
)

# Boilerplate pages we can skip entirely
_BOILERPLATE_PAGE = re.compile(
    r"BILLING RIGHTS|PLEASE READ THIS DOCUMENT|Arbitration Agreement",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# BOA Parser
# ---------------------------------------------------------------------------

class BOAParser(BaseBankParser):
    """
    Parses Bank of America bank statement PDFs into rows for bank_transactions_raw.
    """

    bank_name = "boa"

    # ── Public entry point ────────────────────────────────────────────────

    def parse(self, path: str | Path) -> ParseResult:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")

        lines = self._extract_lines(path)
        period_start, period_end = self._detect_period(lines)
        bank_rows = self._parse_transactions(lines)

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

        return ParseResult(
            metadata=metadata,
            bank_rows=bank_rows,
            loan_rows=[],
        )

    # ── Text extraction ───────────────────────────────────────────────────

    def _extract_lines(self, path: Path) -> list[str]:
        """Extract text lines, skipping pure boilerplate pages."""
        lines: list[str] = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                # Skip pages that are purely legal/arbitration boilerplate
                # but keep pages that also contain transaction data
                if _BOILERPLATE_PAGE.search(text):
                    if "Deposits" not in text and "subtractions" not in text:
                        continue
                lines.extend(text.splitlines())
        return lines

    # ── Period detection ──────────────────────────────────────────────────

    def _detect_period(self, lines: list[str]) -> tuple[str | None, str | None]:
        """Extract ISO start/end dates from 'for March 7, 2026 to April 7, 2026'."""
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
        """Convert 'March 7, 2026' → '2026-03-07'."""
        m = re.match(r"(\w+)\s+(\d{1,2}),\s+(\d{4})", raw.strip())
        if not m:
            return None
        month = _MONTH_NAMES.get(m.group(1))
        if not month:
            return None
        return f"{int(m.group(3))}-{month:02d}-{int(m.group(2)):02d}"

    # ── Date normalisation ────────────────────────────────────────────────

    @staticmethod
    def _normalise_date(raw: str) -> str:
        """Convert 'MM/DD/YY' → 'YYYY-MM-DD'."""
        m = re.match(r"(\d{2})/(\d{2})/(\d{2})", raw)
        if not m:
            return raw
        mm, dd, yy = int(m.group(1)), int(m.group(2)), int(m.group(3))
        year = 2000 + yy if yy < 70 else 1900 + yy
        return f"{year}-{mm:02d}-{dd:02d}"

    # ── Transaction parsing ───────────────────────────────────────────────

    def _parse_transactions(self, lines: list[str]) -> list[dict]:
        """
        Walk all lines and parse deposits, subtractions, and checks.

        Section state machine:
          current_type = "CREDIT" → inside deposits section
          current_type = "DEBIT"  → inside subtractions or checks section
          in_checks = True        → inside the Checks section (no continuations)
        """
        rows: list[dict] = []
        current_type: Optional[str] = None
        in_checks: bool = False
        pending: Optional[dict] = None

        def finalise() -> None:
            """Flush pending transaction: join continuations and append."""
            if pending is None:
                return
            conts = pending.pop("_continuation", [])
            if conts:
                pending["description"] = (
                    pending["description"] + " | " + " ".join(conts)
                )
            rows.append(pending)

        for line in lines:
            s = line.strip()
            if not s:
                continue

            # Skip repetitive page header lines ("... Account # XXXX ...")
            if _PAGE_HEADER.search(s):
                continue

            # Stop / noise lines: finalise any open transaction but keep section state
            if _STOP.match(s):
                finalise()
                pending = None
                continue

            # ── Section detection ──────────────────────────────────────

            if _SEC_DEPOSITS.search(s) and "continued" not in s.lower():
                finalise()
                pending = None
                current_type = "CREDIT"
                in_checks = False
                continue

            if _SEC_SUBTRACTIONS.search(s):
                finalise()
                pending = None
                current_type = "DEBIT"
                in_checks = False
                continue

            if _SEC_CHECKS.match(s):
                finalise()
                pending = None
                current_type = "DEBIT"
                in_checks = True
                continue

            # ── Transaction line ───────────────────────────────────────

            if _DATE.match(s):
                finalise()

                nums = re.findall(_AMT, s)
                if not nums or current_type is None:
                    pending = None
                    continue

                raw_date = s[:8]          # "MM/DD/YY"

                # If two or more numbers on the line: last is running balance,
                # second-to-last is the transaction amount.
                # If only one number: it is the transaction amount, no balance.
                if len(nums) >= 2:
                    amount_str = nums[-2]
                    balance = abs(float(nums[-1].replace(",", "")))
                else:
                    amount_str = nums[-1]
                    balance = None

                amount = abs(float(amount_str.replace(",", "")))

                if in_checks:
                    # Format: "MM/DD/YY  <check#>  <amount>  [<balance>]"
                    parts = s.split()
                    check_num = parts[1] if len(parts) >= 3 else ""
                    description = f"Check #{check_num}" if check_num else "Check"
                else:
                    # Description sits between the date and the transaction amount
                    idx = s.rfind(amount_str)
                    description = s[9:idx].strip()   # 9 = len("MM/DD/YY ")

                pending = {
                    "bank_name": self.bank_name,
                    "account_type": "checking",
                    "date": self._normalise_date(raw_date),
                    "description": description,
                    "amount": amount,
                    "transaction_type": current_type,
                    "balance": balance,
                    "raw_text": s,
                    "_continuation": [],
                }
                continue

            # ── Continuation line ──────────────────────────────────────

            if pending is not None and not in_checks:
                pending["_continuation"].append(s)

        finalise()
        return rows
