"""
CSV Statement Parser - Week 1
Parses bank statement CSV files with auto-detection of column layouts.
"""

import os
import re
import pandas as pd
from typing import Dict, List, Optional


class CSVStatementParser:
    """Parses CSV bank statements, auto-detecting column mappings."""

    # Possible column name variations for each field
    COLUMN_ALIASES = {
        "date": ["date", "trans date", "transaction date", "posted date", "value date"],
        "description": ["description", "memo", "narrative", "details", "payee", "merchant"],
        "amount": ["amount", "transaction amount", "debit/credit", "value"],
        "debit": ["debit", "withdrawal", "debit amount", "dr"],
        "credit": ["credit", "deposit", "credit amount", "cr"],
        "balance": ["balance", "running balance", "available balance"],
        "type": ["type", "transaction type", "dr/cr", "debit/credit"],
        "category": ["category", "merchant category", "spending category"],
    }

    def __init__(self, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        self.filepath = filepath
        self.filename = os.path.basename(filepath)
        self.column_mapping: Dict[str, str] = {}
        self.df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def parse(self) -> Dict:
        """
        Parse the CSV file. Returns a dict with:
          - transactions: list of normalised transaction dicts
          - summary: basic statistics
          - metadata: file info
        """
        self.df = self._load_csv()
        self.column_mapping = self._detect_columns()
        raw_rows = self._extract_rows()
        transactions = self._normalise(raw_rows)

        return {
            "transactions": transactions,
            "summary": self._summarise(transactions),
            "metadata": {
                "filename": self.filename,
                "rows_read": len(self.df),
                "column_mapping": self.column_mapping,
            },
        }

    # ------------------------------------------------------------------ #
    #  Loading                                                             #
    # ------------------------------------------------------------------ #

    def _load_csv(self) -> pd.DataFrame:
        """Try multiple encodings to load the CSV."""
        for encoding in ("utf-8", "latin-1", "cp1252"):
            try:
                df = pd.read_csv(self.filepath, encoding=encoding)
                df.columns = df.columns.str.strip()
                return df
            except UnicodeDecodeError:
                continue
        raise ValueError(f"Could not decode CSV: {self.filepath}")

    # ------------------------------------------------------------------ #
    #  Column detection                                                    #
    # ------------------------------------------------------------------ #

    def _detect_columns(self) -> Dict[str, str]:
        """Map logical field names to actual CSV column names."""
        cols_lower = {c.lower().strip(): c for c in self.df.columns}
        mapping = {}

        for field, aliases in self.COLUMN_ALIASES.items():
            for alias in aliases:
                if alias in cols_lower:
                    mapping[field] = cols_lower[alias]
                    break

        # Validation: must have at least date + (amount or debit)
        if "date" not in mapping:
            raise ValueError("Could not find a date column in CSV.")
        if "amount" not in mapping and "debit" not in mapping:
            raise ValueError("Could not find an amount/debit column in CSV.")

        return mapping

    # ------------------------------------------------------------------ #
    #  Row extraction                                                      #
    # ------------------------------------------------------------------ #

    def _extract_rows(self) -> List[Dict]:
        """Pull relevant fields from each DataFrame row."""
        rows = []
        for _, row in self.df.iterrows():
            entry = {}
            for field, col in self.column_mapping.items():
                entry[field] = row.get(col, "")
            rows.append(entry)
        return rows

    def _normalise(self, rows: List[Dict]) -> List[Dict]:
        """Convert raw rows to standardised transaction dicts."""
        normalised = []
        for row in rows:
            t = self._normalise_row(row)
            if t:
                normalised.append(t)
        return normalised

    def _normalise_row(self, row: Dict) -> Optional[Dict]:
        """Normalise a single row. Returns None if it can't be parsed."""
        date_str = self._clean_str(row.get("date", ""))
        if not date_str:
            return None

        # Resolve amount (may come from separate debit/credit columns)
        amount, trans_type = self._resolve_amount(row)
        if amount is None:
            return None

        description = self._clean_str(row.get("description", ""))
        balance_raw = self._clean_str(row.get("balance", ""))
        balance = self._parse_amount(balance_raw)

        # Normalise date
        normalised_date = self._parse_date(date_str)
        if not normalised_date:
            return None

        from src.extractors.rule_based import RuleBasedExtractor
        extractor = RuleBasedExtractor()
        merchant = extractor.extract_merchant_name(description)
        category = extractor._categorise(merchant, description)

        return {
            "date": normalised_date,
            "description": description,
            "amount": round(abs(amount), 2),
            "transaction_type": trans_type,
            "merchant_name": merchant,
            "category": category,
            "balance": balance,
            "raw_text": str(row),
            "confidence_score": 1.0,
        }

    # ------------------------------------------------------------------ #
    #  Amount resolution                                                   #
    # ------------------------------------------------------------------ #

    def _resolve_amount(self, row: Dict):
        """Return (amount_float, transaction_type) from various column layouts."""
        # Case 1: explicit debit + credit columns
        if "debit" in row and "credit" in row:
            debit = self._parse_amount(str(row.get("debit", "")))
            credit = self._parse_amount(str(row.get("credit", "")))
            if debit:
                return debit, "DEBIT"
            if credit:
                return credit, "CREDIT"

        # Case 2: single amount column
        raw = self._clean_str(str(row.get("amount", "")))
        if not raw:
            return None, "UNKNOWN"

        amount = self._parse_amount(raw)
        if amount is None:
            return None, "UNKNOWN"

        # Determine type from explicit type column or sign
        explicit_type = self._clean_str(str(row.get("type", ""))).upper()
        if explicit_type in ("DEBIT", "CREDIT"):
            return amount, explicit_type
        if amount < 0:
            return amount, "DEBIT"
        # Positive + no explicit type => default to DEBIT (expense)
        return amount, "DEBIT" if "CREDIT" not in explicit_type else "CREDIT"

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _clean_str(value) -> str:
        if pd.isna(value) if hasattr(pd, "isna") else value != value:
            return ""
        return str(value).strip()

    @staticmethod
    def _parse_amount(raw: str) -> Optional[float]:
        if not raw or raw.lower() in ("nan", "none", "", "-"):
            return None
        negative = raw.startswith("-") or ("(" in raw)
        cleaned = re.sub(r"[^\d.]", "", raw)
        if not cleaned:
            return None
        try:
            value = float(cleaned)
            return -value if negative else value
        except ValueError:
            return None

    @staticmethod
    def _parse_date(raw: str) -> Optional[str]:
        from dateutil import parser as dp
        try:
            return dp.parse(raw, dayfirst=False).strftime("%Y-%m-%d")
        except Exception:
            return None

    @staticmethod
    def _summarise(transactions: List[Dict]) -> Dict:
        if not transactions:
            return {"total_transactions": 0, "total_debits": 0, "total_credits": 0, "net": 0}
        debits = sum(t["amount"] for t in transactions if t["transaction_type"] == "DEBIT")
        credits = sum(t["amount"] for t in transactions if t["transaction_type"] == "CREDIT")
        return {
            "total_transactions": len(transactions),
            "total_debits": round(debits, 2),
            "total_credits": round(credits, 2),
            "net": round(credits - debits, 2),
        }
