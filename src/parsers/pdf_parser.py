"""
PDF Statement Parser - Week 1
Extracts transactions from PDF bank statements using pdfplumber.

Supported banks: Robinhood, Citi, BofA, Capital One
"""

import os
import re
from typing import Dict, List, Optional

try:
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

from dateutil import parser as dp

# ------------------------------------------------------------------ #
#  Bank signatures  (checked in order — most specific first)          #
# ------------------------------------------------------------------ #

BANK_SIGNATURES = {
    "robinhood":   ["robinhood credit", "robinhoodcredit", "robinhood.com",
                    "creditcards@robinhood", "robinhood gold", "robinhood card"],
    "citi":        ["citicards.com", "citi double cash", "citi bank", "citibank",
                    "citi cards", "citicards", "citi®"],
    "bofa":        ["bank of america", "bankofamerica.com", "bofa"],
    "capital_one": ["capital one", "capitalone.com", "venture credit card"],
    "chase":       ["chase.com", "jpmorgan", "chase bank"],
    "wells_fargo": ["wells fargo"],
    "amex":        ["american express", "amex"],
    "discover":    ["discover bank", "discover card"],
    "usbank":      ["u.s. bank", "us bank", "usbank"],
    "barclays":    ["barclays"],
}

# ------------------------------------------------------------------ #
#  Section boundary markers                                            #
# ------------------------------------------------------------------ #

TRANSACTION_START_MARKERS = [
    r"^TRANSACTIONS\s*$",
    r"^TRANSACTIONS\s*\(continued\)",
    r"^Transactions\s*$",                   # BofA / Capital One mixed-case
    r"^TRANSACTION\s+DETAIL",
    r"^ACCOUNT\s+ACTIVITY",
    r"^TRANSACTION\s+HISTORY",
    r"^Tran\s+Post\s*$",                    # Robinhood header
    r"Date\s+Date\s+ReferenceNumber",
    r"^Trans\.\s+Post\s*$",                 # Citi
    r"^date\s+date\s+Description\s+Amount", # Citi column header
]

TRANSACTION_END_MARKERS = [
    r"^TOTAL\s+FEES",
    r"^Total\s+Fees\s+for\s+This\s+Period",  # Capital One fees sub-total = real end
    r"^INTEREST\s+CHARGED",
    r"^Interest\s+Charged\s*$",
    r"^FEES\s+CHARGED",
    r"^Fees\s+charged",
    # NOTE: bare "Fees" is NOT an end marker — it is a Capital One sub-section
    # header inside the transactions page. Use "Total Fees for This Period" instead.
    r"^INTEREST\s+CHARGE\s+CALCULATION",
    r"^Interest\s+charge\s+calculation",
    r"^Year.to.Date",
    r"^Totals\s+Year-to-Date",
    r"^2\d{3}\s+Totals\s+Year",
    r"^2\d{3}\s+totals\s+year",
    r"^ACCOUNT\s+MESSAGES",
    r"^Account\s+messages",
    r"^Important\s+Messages",
    r"^IMPORTANT\s+INFORMATION",
    r"^BILLING\s+RIGHTS",
    r"^PAYMENT\s+OPTIONS",
    r"^1200\s+NZJ",
]

TRANSACTION_NOISE_PATTERNS = [
    r"^Tran\s+Post\s*$",
    r"^Trans\.\s+Post\s*$",
    r"^date\s+date\s+Description",
    r"^Date\s+Date\s+Reference",
    r"^Date\s+Description",
    r"^Transaction\s+Posting\s+Reference",       # BofA column header line 1
    r"^Date\s+Date\s+Description\s+Number",      # BofA column header line 2
    r"^Number\s+Number\s+Amount\s+Total",        # BofA column header line 3
    r"^Trans\s+Date\s+Post\s+Date",
    r"^Trans Date\s+Post Date",
    r"^Standard\s+Purchases\s*$",
    r"^Standard\s+Credits\s*$",
    r"^Payments\s+and\s+Other\s+Credits\s*$",
    r"^Purchases\s+and\s+Adjustments\s*$",
    r"^TOTAL\s+PAYMENTS\s+AND\s+OTHER\s+CREDITS",  # BofA sub-total line
    r"^TOTAL\s+PURCHASES\s+AND\s+ADJUSTMENTS",     # BofA sub-total line
    r"^TOTAL\s+INTEREST\s+CHARGED\s+FOR",          # BofA sub-total line
    r"^SIVAKUMAR\s+PRABHAKARAN\s+#",
    r"^DEEPA\s+RAMPRASAD\s+#",
    r"Visit\s+capitalone\.com",
    r"TransactionDescription\s+Amount",
    r"Transactionscontinued",
    r"^Available\s+Credit",            # Capital One account summary bleed
    r"^Cash\s+Advance\s+Credit",      # Capital One account summary bleed
    r"^Fees\s*$",                      # Capital One fees sub-section label (noise inside section)
    r"^Interest\s+Charged\s*$",       # Capital One interest sub-section label
    r"^Trans\s+Date\s+Post\s+Date\s+Description\s+Amount",  # Capital One column header
    r"^\s*$",
]


class PDFStatementParser:
    """Parses PDF bank statements using pdfplumber."""

    def __init__(self, filepath: str):
        if not PDF_AVAILABLE:
            raise ImportError("pdfplumber is not installed. Run: pip install pdfplumber")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        self.filepath = filepath
        self.filename = os.path.basename(filepath)

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def parse(self) -> Dict:
        all_text = []
        page_count = 0

        with pdfplumber.open(self.filepath) as pdf:
            page_count = len(pdf.pages)
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    all_text.append(text)

        full_text = "\n".join(all_text)
        bank_name  = self._detect_bank(full_text)
        closing_month, stmt_year = self._detect_billing_period(full_text, filename=self.filename)
        txn_text   = self._extract_transactions_section(full_text)

        from src.extractors.rule_based import RuleBasedExtractor
        extractor = RuleBasedExtractor()

        # Banks with dedicated parsers: use section text if available, else return
        # empty — never fall back to full-text scraping, which picks up account
        # summary numbers (credit limits, available credit) as fake transactions.
        DEDICATED_BANKS = {"robinhood", "citi", "bofa", "capital_one"}

        if txn_text.strip():
            transactions = self._parse_transaction_lines(txn_text, extractor, bank_name, stmt_year, closing_month)
        elif bank_name in DEDICATED_BANKS:
            transactions = []   # zero-transaction statement — don't scrape full text
        else:
            transactions = extractor.extract_transactions(full_text)

        return {
            "transactions": transactions,
            "summary": self._summarise(transactions),
            "metadata": {
                "filename":  self.filename,
                "bank_name": bank_name,
                "page_count": page_count,
                "statement_year": stmt_year,
                "transactions_text_length": len(txn_text),
            },
        }

    # ------------------------------------------------------------------ #
    #  Bank & year detection                                               #
    # ------------------------------------------------------------------ #

    def _detect_bank(self, text: str) -> str:
        lower = text.lower()
        for bank, keywords in BANK_SIGNATURES.items():
            if any(k in lower for k in keywords):
                return bank
        return "unknown"

    def _detect_billing_period(self, text: str, filename: str = ""):
        """
        Return (closing_month, closing_year) for the statement.
        Used to assign the correct year to each MM/DD transaction:
          txn_month <= closing_month  =>  closing_year
          txn_month >  closing_month  =>  closing_year - 1  (prior year)
        e.g. 'Dec 6 - Jan 5, 2026': closing=(1,2026); Dec txns get 2025.
        """
        MONTH_MAP = {
            'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,
            'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12,
        }

        # Billing period line: 'Month D - Month D, YYYY'
        m = re.search(
            r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*'
            r'[\s\w,\-]+'
            r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*'
            r'\s*[\s,\-]?\s*(\d{1,2}),?\s*(20\d{2})',
            text, re.IGNORECASE
        )
        if m:
            em = MONTH_MAP.get(m.group(1).lower()[:3], 0)
            if em:
                return em, int(m.group(3))

        # Closing date keyword with month name (handles pdfplumber space-stripping)
        m = re.search(
            r'(?:closing\s*date|statement\s*closing)'
            r'[^\n]{0,20}?'
            r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*'
            r'\s*(\d{1,2}),?\s*(20\d{2})',
            text, re.IGNORECASE
        )
        if m:
            em = MONTH_MAP.get(m.group(1).lower()[:3], 0)
            if em:
                return em, int(m.group(3))

        # Closing date as MM/DD/YYYY
        m = re.search(
            r'(?:closing\s*date|statement\s*closing)[^\n]*'
            r'(\d{1,2})/(\d{1,2})/(20\d{2})',
            text, re.IGNORECASE
        )
        if m:
            return int(m.group(2)), int(m.group(3))

        # Filename: YYYY-MM-DD  (e.g. eStmt_2026-01-05.pdf)
        fn = re.sub(r'\.pdf$', '', filename, flags=re.IGNORECASE)
        m = re.search(r'(20\d{2})-(\d{2})-\d{2}', fn)
        if m:
            return int(m.group(2)), int(m.group(1))

        # Filename: MonthYYYY  (e.g. January2026.pdf, Credit_Statement_February_2026.pdf)
        m = re.search(
            r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*(20\d{2})',
            fn, re.IGNORECASE
        )
        if m:
            em = MONTH_MAP.get(m.group(1).lower()[:3], 0)
            if em:
                return em, int(m.group(2))

        # Any 4-digit year in filename
        m = re.search(r'(?<!\d)(20\d{2})(?!\d)', fn)
        if m:
            return 12, int(m.group(1))  # conservative: assume Dec closing

        # Citi compact billing period: 'Billing Period: 11/19/25-12/16/25'
        # End date is the closing date
        m = re.search(
            r'(?:billing\s*period)[^\n]*'
            r'(\d{1,2})/(\d{1,2})/(\d{2})\s*[-\u2013]\s*'
            r'(\d{1,2})/(\d{1,2})/(\d{2})',
            text, re.IGNORECASE
        )
        if m:
            return int(m.group(4)), 2000 + int(m.group(6))

        # Text: any MM/DD/YY (Citi bare 2-digit years)
        m = re.search(r'\b\d{1,2}/\d{1,2}/(\d{2})\b', text)
        if m:
            return None, 2000 + int(m.group(1))

        return None, 2025

    def _detect_year(self, text: str, filename: str = "") -> int:
        """Convenience wrapper returning just the closing year."""
        _, year = self._detect_billing_period(text, filename=filename)
        return year

    def _year_for_month(self, txn_month: int, closing_month, closing_year: int) -> int:
        """
        Given a txn month and closing month/year, return the correct year.
        Txn month > closing month means it belongs to the previous year.
        """
        if closing_month is None:
            return closing_year
        if txn_month <= closing_month:
            return closing_year
        return closing_year - 1

    def _extract_transactions_section(self, full_text: str) -> str:
        lines = full_text.splitlines()
        result_lines = []
        inside = False

        start_re = [re.compile(p, re.IGNORECASE) for p in TRANSACTION_START_MARKERS]
        end_re   = [re.compile(p, re.IGNORECASE) for p in TRANSACTION_END_MARKERS]
        noise_re = [re.compile(p, re.IGNORECASE) for p in TRANSACTION_NOISE_PATTERNS]

        for line in lines:
            stripped = line.strip()

            if not inside:
                if re.match(r"^date\s+date\s+Description\s+Amount", stripped, re.IGNORECASE):
                    inside = True
                    continue
                if any(r.search(stripped) for r in start_re):
                    inside = True
                continue

            if any(r.search(stripped) for r in end_re):
                inside = False
                continue

            if any(r.search(stripped) for r in noise_re):
                continue

            # Drop Citi rewards-column bleed (no leading date)
            if not re.match(r"^\d{2}/\d{2}", stripped) and not re.match(r"^[A-Z][a-z]{2}\s+\d", stripped):
                if re.match(r"^\d{1,4}$", stripped):
                    continue
                if re.search(r"thankyou\.com|Purchase Tracker|Bonus from Citi|ThankYou Points|IT'S EASY TO REDEEM", stripped, re.IGNORECASE):
                    continue

            result_lines.append(stripped)

        return "\n".join(result_lines)

    # ------------------------------------------------------------------ #
    #  Bank-aware dispatch                                                 #
    # ------------------------------------------------------------------ #

    def _parse_transaction_lines(self, text: str, extractor, bank_name: str, year: int, closing_month=None) -> List[Dict]:
        if bank_name == "robinhood":
            return self._parse_robinhood(text, extractor, year, closing_month)
        if bank_name == "citi":
            return self._parse_citi(text, extractor, year, closing_month)
        if bank_name == "bofa":
            return self._parse_bofa(text, extractor, year, closing_month)
        if bank_name == "capital_one":
            return self._parse_capital_one(text, extractor, year, closing_month)
        return extractor.extract_transactions(text)

    # ------------------------------------------------------------------ #
    #  Shared date helper                                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _to_iso(raw: str, year: int) -> str:
        """Parse MM/DD or 'Aug 6' style dates, attach the given year."""
        try:
            parsed = dp.parse(raw, dayfirst=False)
            return f"{year}-{parsed.month:02d}-{parsed.day:02d}"
        except Exception:
            return raw

    def _to_iso_smart(self, raw: str, closing_year: int, closing_month) -> str:
        """
        Parse MM/DD date and assign the correct year based on billing period.
        If txn_month > closing_month the transaction is from the prior year.
        """
        try:
            parsed = dp.parse(raw, dayfirst=False)
            txn_year = self._year_for_month(parsed.month, closing_month, closing_year)
            return f"{txn_year}-{parsed.month:02d}-{parsed.day:02d}"
        except Exception:
            return raw

    # ------------------------------------------------------------------ #
    #  Robinhood                                                           #
    # ------------------------------------------------------------------ #

    def _parse_robinhood(self, text: str, extractor, year: int, closing_month=None) -> List[Dict]:
        """MM/DD  MM/DD  REF(10+)  description  amount[-]"""
        transactions = []
        line_re = re.compile(
            r"^(\d{2}/\d{2})\s+\d{2}/\d{2}\s+[A-Z0-9]{10,}\s+(.+?)\s+([\d,]+\.\d{2}-?)$",
            re.IGNORECASE
        )
        for line in text.splitlines():
            line = line.strip()
            m = line_re.match(line)
            if not m:
                continue
            date_raw, desc, amount_raw = m.group(1), m.group(2), m.group(3)
            is_credit   = amount_raw.endswith("-")
            clean_amt   = float(amount_raw.rstrip("-").replace(",", ""))
            trans_type  = "CREDIT" if is_credit else "DEBIT"
            desc        = desc.strip()
            merchant    = extractor.extract_merchant_name(desc)
            transactions.append({
                "date": self._to_iso_smart(date_raw, year, closing_month),
                "description": desc,
                "amount": round(clean_amt, 2),
                "transaction_type": trans_type,
                "merchant_name": merchant,
                "category": extractor._categorise(merchant, desc),
                "balance": None,
                "raw_text": line,
                "confidence_score": 0.95,
            })
        return transactions

    # ------------------------------------------------------------------ #
    #  Citi                                                                #
    # ------------------------------------------------------------------ #

    def _parse_citi(self, text: str, extractor, year: int, closing_month=None) -> List[Dict]:
        """
        Citi formats:
          MM/DD  MM/DD  description  [-]$amount   (two dates — purchases)
          MM/DD  description  [-]$amount           (one date  — payments)
        Negative amounts (-$) are credits (payments/refunds).

        The right-side rewards column bleeds into some lines:
          01/26 01/26 SAMSCLUB #4969 ... $53.69 Eligible Purchases: $750.49
        We must capture the FIRST $amount on the line, not the last.
        Strategy: match description as [^$]+ (no dollar signs) so it stops
        at the first $ which is the real transaction amount.
        """
        transactions = []
        # Match the FIRST $amount after the description (description has no $)
        line_re = re.compile(
            r"^(\d{2}/\d{2})\s+"          # trans date
            r"(?:\d{2}/\d{2}\s+)?"         # optional post date
            r"([^$]+?)\s+"                    # description — stops before first $
            r"(-?\$[\d,]+\.\d{2})",       # first amount (no end anchor)
            re.IGNORECASE
        )
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            m = line_re.match(line)
            if not m:
                continue
            date_raw, desc, amount_str = m.group(1), m.group(2).strip(), m.group(3)
            # Skip header/noise lines that have no real description
            if not desc or re.match(r"^(Trans\.?|Post|date|Description)\b", desc, re.I):
                continue
            is_credit  = amount_str.startswith("-")
            clean_amt  = float(amount_str.replace("-", "").replace("$", "").replace(",", ""))
            trans_type = "CREDIT" if is_credit else "DEBIT"
            merchant   = extractor.extract_merchant_name(desc)
            transactions.append({
                "date": self._to_iso_smart(date_raw, year, closing_month),
                "description": desc,
                "amount": round(clean_amt, 2),
                "transaction_type": trans_type,
                "merchant_name": merchant,
                "category": extractor._categorise(merchant, desc),
                "balance": None,
                "raw_text": line,
                "confidence_score": 0.97,
            })
        return transactions

    # ------------------------------------------------------------------ #
    #  Bank of America                                                     #
    # ------------------------------------------------------------------ #

    def _parse_bofa(self, text: str, extractor, year: int, closing_month=None) -> List[Dict]:
        """
        BofA format:
          MM/DD  MM/DD  description  refnum  acctnum  amount
        Payments have negative amounts (e.g. -49.92).
        The refnum and acctnum are 4-digit numbers — we strip them.
        """
        transactions = []

        # MM/DD  MM/DD  <desc>  <4-digit-ref>  <4-digit-acct>  <amount>
        # Amount may be negative (credits/payments)
        line_re = re.compile(
            r"^(\d{2}/\d{2})\s+"       # trans date
            r"\d{2}/\d{2}\s+"          # post date
            r"(.+?)\s+"                # description
            r"\d{4}\s+"                # reference number (4 digits)
            r"\d{4}\s+"                # account suffix (4 digits)
            r"(-?[\d,]+\.\d{2})$",     # amount (negative = credit)
        )

        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue

            m = line_re.match(line)
            if not m:
                continue

            date_raw, desc, amount_raw = m.group(1), m.group(2), m.group(3)
            clean_amt  = float(amount_raw.replace(",", ""))
            trans_type = "CREDIT" if clean_amt < 0 else "DEBIT"
            clean_amt  = abs(clean_amt)
            desc       = desc.strip()
            merchant   = extractor.extract_merchant_name(desc)

            transactions.append({
                "date": self._to_iso_smart(date_raw, year, closing_month),
                "description": desc,
                "amount": round(clean_amt, 2),
                "transaction_type": trans_type,
                "merchant_name": merchant,
                "category": extractor._categorise(merchant, desc),
                "balance": None,
                "raw_text": line,
                "confidence_score": 0.96,
            })

        return transactions

    # ------------------------------------------------------------------ #
    #  Capital One                                                         #
    # ------------------------------------------------------------------ #

    def _parse_capital_one(self, text: str, extractor, year: int, closing_month=None) -> List[Dict]:
        """
        Capital One Venture format:
          Aug 6   Aug 6   DESCRIPTION   $142.30     (debit — positive)
          Aug 21  Aug 22  DESCRIPTION   - $65.60    (credit — negative with space)
        Dates use full month abbreviations.
        """
        transactions = []

        # Month name  day  Month name  day  description  [-]  $amount
        line_re = re.compile(
            r"^([A-Z][a-z]{2}\s+\d{1,2})\s+"    # trans date  e.g. "Aug 6"
            r"[A-Z][a-z]{2}\s+\d{1,2}\s+"        # post date (ignore)
            r"(.+?)\s+"                           # description
            r"(-\s*)?\$([\d,]+\.\d{2})$",        # optional "- " then $amount
        )

        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue

            m = line_re.match(line)
            if not m:
                continue

            date_raw   = m.group(1)   # e.g. "Aug 6"
            desc       = m.group(2).strip()
            is_credit  = m.group(3) is not None   # the "- " prefix
            amount_raw = m.group(4)
            clean_amt  = float(amount_raw.replace(",", ""))
            trans_type = "CREDIT" if is_credit else "DEBIT"
            merchant   = extractor.extract_merchant_name(desc)

            # Capital One uses "Aug 6" — append year for full parse
            try:
                parsed = dp.parse(f"{date_raw} {year}", dayfirst=False)
                txn_year = self._year_for_month(parsed.month, closing_month, year)
                iso_date = f"{txn_year}-{parsed.month:02d}-{parsed.day:02d}"
            except Exception:
                iso_date = date_raw

            transactions.append({
                "date": iso_date,
                "description": desc,
                "amount": round(clean_amt, 2),
                "transaction_type": trans_type,
                "merchant_name": merchant,
                "category": extractor._categorise(merchant, desc),
                "balance": None,
                "raw_text": line,
                "confidence_score": 0.96,
            })

        return transactions

    # ------------------------------------------------------------------ #
    #  Summary                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _summarise(transactions: List[Dict]) -> Dict:
        if not transactions:
            return {"total_transactions": 0, "total_debits": 0, "total_credits": 0, "net": 0}
        debits  = sum(t["amount"] for t in transactions if t.get("transaction_type") == "DEBIT")
        credits = sum(t["amount"] for t in transactions if t.get("transaction_type") == "CREDIT")
        return {
            "total_transactions": len(transactions),
            "total_debits":  round(debits, 2),
            "total_credits": round(credits, 2),
            "net": round(credits - debits, 2),
        }
