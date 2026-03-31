"""
src/nlp/transaction_processor_contract.py
────────────────────────────────────────────
Documents the dict contract that EnrichmentPipeline expects from
TransactionProcessor.process().

You don't need to change your existing TransactionProcessor class –
just ensure its process() method (or whatever you call the main entry
point) returns a dict with the keys below.  Any missing key falls back
to None / 'Uncategorized' / 0.0.

Expected return shape
---------------------
{
    # --- cleaning ---
    "clean_description": str | None,   # description after regex cleaning
                                       # e.g. "AMAZON PRIME*AB1234" → "Amazon Prime"

    # --- merchant ---
    "merchant_name": str | None,       # normalised merchant name
                                       # e.g. "amazon" → "Amazon"
    "merchant_raw": str | None,        # the raw token before normalisation
                                       # e.g. "AMAZON PRIME*AB1234"

    # --- taxonomy ---
    "category":    str,                # top-level, e.g. "Shopping"
    "subcategory": str | None,         # second-level, e.g. "Online Retail"

    # --- location ---
    "location": str | None,            # city/state from spaCy GPE entity
                                       # e.g. "San Francisco, CA"

    # --- confidence ---
    "confidence_score": float,          # composite 0.0 – 1.0
                                        # suggested weights:
                                        #   substring match → 0.95
                                        #   fuzzy match     → match_score / 100
                                        #   spaCy only      → 0.60
                                        #   rules only      → 0.50
                                        #   fallback        → 0.20

    # --- provenance ---
    "enrichment_method": str | None,    # 'substring' | 'fuzzy' | 'spacy'
                                        # | 'rules'   | 'fallback'
}

Minimal implementation stub
---------------------------
If your TransactionProcessor doesn't yet return subcategory / location /
enrichment_method, add them incrementally.  The pipeline handles None.

Example
-------
class TransactionProcessor:
    def process(self, description: str, amount: float,
                transaction_type: str) -> dict:
        clean = self._clean_description(description)
        merchant, method, score = self._resolve_merchant(clean)
        category, subcategory  = self._categorize(merchant or clean)
        location               = self._extract_location(clean)
        return {
            "clean_description": clean,
            "merchant_name":     merchant,
            "merchant_raw":      clean,     # or raw token before norm
            "category":          category,
            "subcategory":       subcategory,
            "location":          location,
            "confidence_score":  score,
            "enrichment_method": method,
        }
"""

# This file is documentation only – no runnable code.
# Import it in tests if you want to validate your TransactionProcessor
# against the contract:
#
#   from src.nlp.transaction_processor import TransactionProcessor
#   from src.nlp.transaction_processor_contract import EXPECTED_KEYS
#
#   result = TransactionProcessor().process("STARBUCKS #1234", 5.50, "DEBIT")
#   for key in EXPECTED_KEYS:
#       assert key in result, f"Missing key: {key}"

EXPECTED_KEYS = [
    "clean_description",
    "merchant_name",
    "merchant_raw",
    "category",
    "subcategory",
    "location",
    "confidence_score",
    "enrichment_method",
]
