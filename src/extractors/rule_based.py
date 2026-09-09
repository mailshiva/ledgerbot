"""
Rule-Based Transaction Extractor - Week 1
Uses regex patterns to extract structured transaction data from raw text.
"""

import re
from datetime import datetime
from typing import List, Dict, Optional
from dateutil import parser as date_parser


class RuleBasedExtractor:
    """Extracts transactions from raw text using regex patterns."""

    # ------------------------------------------------------------------ #
    #  Date patterns                                                       #
    # ------------------------------------------------------------------ #
    DATE_PATTERNS = [
        r'\d{4}-\d{2}-\d{2}',           # 2024-01-15
        r'\d{2}/\d{2}/\d{4}',           # 01/15/2024
        r'\d{1,2}/\d{1,2}/\d{2,4}',     # 1/15/24
        r'\d{2}-\d{2}-\d{4}',           # 01-15-2024
        r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}',
    ]

    # ------------------------------------------------------------------ #
    #  Amount patterns                                                     #
    # ------------------------------------------------------------------ #
    AMOUNT_PATTERNS = [
        r'-?\$\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?',   # -$1,234.56
        r'\(\$?\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?\)', # (1,234.56)  <- parentheses = negative
        r'-?\d{1,3}(?:,\d{3})*\.\d{2}',              # 1234.56
    ]

    # ------------------------------------------------------------------ #
    #  Merchant standardisation map                                        #
    # ------------------------------------------------------------------ #
    MERCHANT_MAP = {
        r'STARBUCKS': 'Starbucks',
        r'WHOLE\s*FOODS': 'Whole Foods',
        r'AMAZON|AMZN': 'Amazon',
        r'NETFLIX': 'Netflix',
        r'CLAUDE': 'Claude',
        r'OPENAI': 'Openai',
        r'BOYSANDGIRLS': 'boysandgirls',
        r'KOHL': 'Kohl',
        r'JC PENNY': 'JC Penny',
        r'BELK': 'Belk',
        r'OLD NAVY': 'old navy',
        r'OLIVE GARDEN': 'Olive Garden',
        r'SPOTIFY': 'Spotify',
        r'UBER\s*(TECHNOLOGIES|EATS)?': 'Uber',
        r'LYFT': 'Lyft',
        r'CHIPOTLE': 'Chipotle',
        r'MCDONALDS|MC\s*DONALD': "McDonald's",
        r'TRADER\s*JOE': "Trader Joe's",
        r'COSTCO': 'Costco',
        r'CHEVRON': 'Chevron',
        r'SHELL': 'Shell',
        r'APPLE\.COM|ITUNES': 'Apple',
        r'VENMO': 'Venmo',
        r'ZELLE': 'Zelle',
        r'PAYPAL': 'PayPal',
        r'WALMART|WAL-MART|WM\s*SUPERCENTER': 'Walmart',
        r'TARGET': 'Target',
        r'SAFEWAY': 'Safeway',
        r'SAMS\s*CLUB|SAMSCLUB': "Sam's Club",
        r'WALGREENS': 'Walgreens',
        r'CVS': 'CVS',
        r'KROGER': 'Kroger',
        r'CHICK.FIL.A': 'Chick-fil-A',
        r'PANDA EXPRESS': 'Panda Express',
        r'POPEYES': 'Popeyes',
        r'BURGER\s*KING': 'Burger King',
        r'BRAUMS': "Braum's",
        r'KRISPY\s*KREME': 'Krispy Kreme',
        r'TEMU': 'Temu',
        r'VIOC|VALVOLINE': 'Valvoline',
        r'ANTHROPIC': 'Anthropic',
        r'ULTRA\s*MOBILE': 'Ultra Mobile',
    }

    # ------------------------------------------------------------------ #
    #  Debit / credit indicators                                          #
    # ------------------------------------------------------------------ #
    DEBIT_KEYWORDS = [
        'purchase', 'payment', 'withdrawal', 'debit', 'pos',
        'atm', 'fee', 'charge', 'paid', 'buy',
    ]
    CREDIT_KEYWORDS = [
        'deposit', 'credit', 'payroll', 'salary', 'refund',
        'transfer in', 'zelle in', 'direct deposit', 'return',
    ]

    def __init__(self):
        self._compile_patterns()

    def _compile_patterns(self):
        """Pre-compile regex patterns for performance."""
        self.date_regex = re.compile(
            '|'.join(self.DATE_PATTERNS), re.IGNORECASE
        )
        self.amount_regex = re.compile(
            '|'.join(self.AMOUNT_PATTERNS)
        )

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def extract_transactions(self, text: str) -> List[Dict]:
        """
        Main entry point. Accepts multi-line text and returns a list of
        extracted transaction dicts.
        """
        transactions = []
        lines = text.strip().splitlines()

        for line in lines:
            line = line.strip()
            if not line or self._is_header_line(line):
                continue

            result = self._parse_line(line)
            if result:
                transactions.append(result)

        return transactions

    def extract_from_structured(self, rows: List[Dict]) -> List[Dict]:
        """
        Accept already-structured rows (e.g. from CSV parser) and
        normalise / enrich them.
        """
        transactions = []
        for row in rows:
            t = {
                "date": self._normalise_date(str(row.get("date", ""))),
                "description": str(row.get("description", "")).strip(),
                "amount": self._clean_amount(str(row.get("amount", "0"))),
                "transaction_type": self._infer_type_from_row(row),
                "balance": self._clean_amount(str(row.get("balance", ""))) if row.get("balance") else None,
                "raw_text": str(row),
                "confidence_score": 1.0,
            }
            t["merchant_name"] = self.extract_merchant_name(t["description"])
            t["category"] = self._categorise(t["merchant_name"], t["description"])
            if t["date"] and t["amount"] is not None:
                transactions.append(t)
        return transactions

    # ------------------------------------------------------------------ #
    #  Line parsing                                                        #
    # ------------------------------------------------------------------ #

    def _parse_line(self, line: str) -> Optional[Dict]:
        """Try to extract a transaction from a single text line."""
        date_str = self._find_date(line)
        amount = self._find_amount(line)

        if not date_str or amount is None:
            return None

        normalised_date = self._normalise_date(date_str)
        if not normalised_date:
            return None

        description = self._extract_description(line, date_str, str(amount))
        transaction_type = self._infer_type(line, amount)

        # Parentheses always mean negative / debit
        if re.search(r'\([\$\d,. ]+\)', line):
            transaction_type = 'DEBIT'
            amount = abs(amount)
        elif amount < 0:
            transaction_type = 'DEBIT'
            amount = abs(amount)

        confidence = self._score_confidence(date_str, amount, description)

        return {
            "date": normalised_date,
            "description": description,
            "amount": round(amount, 2),
            "transaction_type": transaction_type,
            "merchant_name": self.extract_merchant_name(description),
            "category": self._categorise(self.extract_merchant_name(description), description),
            "raw_text": line,
            "confidence_score": confidence,
        }

    # ------------------------------------------------------------------ #
    #  Field extractors                                                    #
    # ------------------------------------------------------------------ #

    def _find_date(self, text: str) -> Optional[str]:
        m = self.date_regex.search(text)
        return m.group() if m else None

    def _find_amount(self, text: str) -> Optional[float]:
        m = self.amount_regex.search(text)
        if not m:
            return None
        return self._clean_amount(m.group())

    def _clean_amount(self, raw: str) -> Optional[float]:
        """Convert a raw amount string to a float."""
        if not raw:
            return None
        negative = raw.startswith('-') or ('(' in raw)
        cleaned = re.sub(r'[^\d.]', '', raw)
        if not cleaned:
            return None
        try:
            value = float(cleaned)
            return -value if negative else value
        except ValueError:
            return None

    def _normalise_date(self, raw: str) -> Optional[str]:
        """Parse a date string and return ISO format YYYY-MM-DD."""
        if not raw:
            return None
        try:
            return date_parser.parse(raw, dayfirst=False).strftime('%Y-%m-%d')
        except Exception:
            return None

    def _extract_description(self, line: str, date_str: str, amount_str: str) -> str:
        """Remove date and amount tokens to get the description."""
        desc = line
        desc = desc.replace(date_str, '')
        # Remove the raw amount token
        desc = re.sub(re.escape(amount_str), '', desc)
        # Remove leftover $, (, )
        desc = re.sub(r'[\$\(\)]', '', desc)
        return ' '.join(desc.split()).strip()

    def extract_merchant_name(self, description: str) -> str:
        """
        Attempt to standardise a merchant name.
        First tries known merchant patterns, then cleans the raw description.
        """
        desc_upper = description.upper()

        for pattern, standard_name in self.MERCHANT_MAP.items():
            if re.search(pattern, desc_upper):
                return standard_name

        # Remove trailing reference numbers, store numbers, etc.
        cleaned = re.sub(r'#\d+', '', description)
        cleaned = re.sub(r'\s+\d{4,}', '', cleaned)  # strip long numeric codes
        cleaned = re.sub(r'\bDBA\b.*', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\s{2,}', ' ', cleaned).strip()

        # Title-case for readability
        return cleaned.title() if cleaned else description.title()

    # ------------------------------------------------------------------ #
    #  Classification helpers                                              #
    # ------------------------------------------------------------------ #

    def _infer_type(self, line: str, amount: float) -> str:
        """Determine DEBIT / CREDIT from keywords and amount sign."""
        lower = line.lower()
        if any(k in lower for k in self.CREDIT_KEYWORDS):
            return 'CREDIT'
        if any(k in lower for k in self.DEBIT_KEYWORDS):
            return 'DEBIT'
        # Negative amounts are debits
        if amount < 0:
            return 'DEBIT'
        return 'UNKNOWN'

    def _infer_type_from_row(self, row: Dict) -> str:
        """Infer transaction type from a structured row."""
        explicit = str(row.get("transaction_type", row.get("type", ""))).upper()
        if explicit in ('DEBIT', 'CREDIT'):
            return explicit

        amount_raw = str(row.get("amount", ""))
        amount = self._clean_amount(amount_raw)

        desc = str(row.get("description", "")).lower()
        if any(k in desc for k in self.CREDIT_KEYWORDS):
            return 'CREDIT'
        if any(k in desc for k in self.DEBIT_KEYWORDS):
            return 'DEBIT'
        if amount is not None and amount < 0:
            return 'DEBIT'
        return 'DEBIT'  # default assumption

    def _categorise(self, merchant: str, description: str) -> str:
        """Simple rule-based categorisation."""
        CATEGORY_RULES = {
            'Groceries': ['whole foods', 'trader joe', 'safeway', 'costco', 'walmart',
                          'sam\'s club', 'kroger', 'wm supercenter'],
            'Dining':    ['starbucks', 'chipotle', "mcdonald", 'restaurant', 'cafe',
                          'pizza', 'chick-fil-a', 'popeyes', 'burger king', 'braum',
                          'krispy kreme', 'a chau', 'olive garden', 'panda express'],
            'Transportation': ['uber', 'lyft', 'chevron', 'shell', 'gas', 'parking',
                               'valvoline', 'vioc'],
            'Entertainment':  ['netflix', 'spotify', 'apple', 'hulu', 'disney', 'itunes'],
            'Shopping':       ['amazon', 'target', 'ebay', 'etsy', 'temu', 'kohl', 'old navy', 'belk', 'jcpenny'],
            'Health':         ['walgreens', 'cvs', 'pharmacy', 'medical', 'doctor',
                               'dental', 'vision', 'eye care'],
            'Utilities':      ['city of ', 'electric', 'water', 'internet', 'phone',
                               'ultra mobile', 'at&t', 'verizon', 't-mobile'],
            'Education':      ['school', 'university', 'college', 'tuition', 'zeffy', 'claude', 'openai', 'boysandgirls'],
            'Technology':     ['anthropic', 'google', 'microsoft', 'adobe'],
            'Income':         ['payroll', 'salary', 'deposit', 'zelle', 'venmo',
                               'direct deposit'],
        }
        text = (merchant + ' ' + description).lower()
        for category, keywords in CATEGORY_RULES.items():
            if any(k in text for k in keywords):
                return category
        return 'Other'

    # ------------------------------------------------------------------ #
    #  Confidence scoring                                                  #
    # ------------------------------------------------------------------ #

    def _score_confidence(self, date_str: str, amount: float, description: str) -> float:
        """Return a 0–1 confidence score for the extraction."""
        score = 0.0
        if date_str:
            score += 0.4
        if amount is not None:
            score += 0.4
        if description and len(description) > 3:
            score += 0.2
        return round(score, 2)

    # ------------------------------------------------------------------ #
    #  Utility                                                             #
    # ------------------------------------------------------------------ #

    def _is_header_line(self, line: str) -> bool:
        """Skip obvious header / footer lines."""
        header_words = ['date', 'description', 'amount', 'balance', 'transaction',
                        'type', '---', '===', 'opening', 'closing', 'total']
        lower = line.lower()
        return any(w in lower for w in header_words) and not self._find_date(line)
