"""
Week 2: NLP Techniques for Financial Transaction Parsing
=========================================================
Topics covered:
  1. spaCy pipeline setup & custom components
  2. Merchant name normalization (regex + fuzzy matching)
  3. Rule-based & ML transaction categorization
  4. Putting it all together in a TransactionProcessor

For reprocessing incase categories are updated:

python3 -m src.nlp.enrichment_pipeline --reprocess

Requirements:
    pip install spacy rapidfuzz
    python -m spacy download en_core_web_sm
"""

import re
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import spacy
from spacy.language import Language
from spacy.tokens import Doc, Token
from rapidfuzz import fuzz, process


# ─────────────────────────────────────────────
# 1. DATA MODEL
# ─────────────────────────────────────────────

@dataclass
class Transaction:
    id: str
    raw_description: str
    amount: float
    date: str
    merchant: Optional[str] = None
    category: Optional[str] = None
    subcategory: Optional[str] = None
    location: Optional[str] = None
    confidence: float = 0.0
    flags: list = field(default_factory=list)

    @property
    def is_debit(self) -> bool:
        return self.amount < 0

    def __repr__(self):
        return (
            f"Transaction({self.id} | {self.merchant or self.raw_description[:20]!r} | "
            f"${self.amount:.2f} | {self.category or 'uncategorized'})"
        )


# ─────────────────────────────────────────────
# 2. MERCHANT NORMALIZATION
# ─────────────────────────────────────────────

# Known merchant aliases: canonical name → list of patterns/fragments
MERCHANT_ALIASES: dict[str, list[str]] = {
    "Amazon":         ["amzn", "amazon", "amzn*mktp"],
    "Amazon Prime":   ["amazon prime"],
    "Disney Plus":     ["disney plus"],
    "Temu":           ["temu.com", "temuus"],
    "Shein":          ["shein.com", "sheinus"],
    "Uber":           ["uber *trip", "uber trip"],
    "Uber Eats":      ["uber eats", "ubereats"],
    "Whole Foods":    ["wholefds", "whole foods", "whole foods market"],
    "Netflix":        ["netflix"],
    "Spotify":        ["spotify"],
    "Community Butcher": ["community butcher", "communitybutcher"],
    "Shell":          ["shell oil", "shell"],
    "Public Transportation": ["ozark regional transit"],
    "Great Clips":     ["great clips", "greatclips"],
    "Chevron":        ["chevron"],
    "Maverik":        ["maverik"],
    "Phillips":        ["phillips 66"],
    "CVS Pharmacy":   ["cvs", "cvs/pharmacy"],
    "Walgreens":      ["walgreens"],
    "Delta Airlines": ["delta air", "delta airlines"],
    "United Airlines":["united airlines", "united air"],
    "Jcpenny":        ["jcpenny"],
    "Khols":          ["khols"],
    "Macys":          ["Macys"],
    "Marshalls":      ["marshalls"],
    "Starbucks":      ["starbucks"],
    "Taco Bell":      ["taco bell"],
    "Chipotle":       ["chipotle"],
    "Chick-Fil-A":    ["chick-fil-a"],
    "Dunkin":         ["dunkin", "dd *dunkin"],
    "McDonald's":     ["mcdonalds", "mcdonald's"],
    "Popeyes":        ["popeyes"],
    "Krispy Kreme":    ["krispy kreme"],
    "Burger King":    ["burgerking"],
    "Blue Bottle Coffee": ["blue bottle", "bluebottle coffee"],
    "Target":         ["target"],
    "Braums":         ["braums", "Braums"],
    "Landers":        ["landers"],
    "Walmart":        ["walmart", "wm supercenter", "wm neighborhood"],
    "Sams Club":      ["sams club"],
    "Aldi":           ["aldi"],
    "Dollar-General": ["dollar-general", "dollar general"],
    "Dollar-Tree":     ["dollartree"],
    "Malco":          ["malco"],
    "Infiniti":      ["infiniti"],
    "India Mart":     ["india mart", "indiamart", "namaste"],
    "Asian Amigo":    ["asian amigo", "achau"],
    "Bentonville Utilities": ["city of "],
    "Bentonville Community Center": ["act*bentonville"],
    "Planet Fitness": ["planet fitness"],
    "Bawarchi":      ["bawarchi"],
    "PayPal":         ["paypal"],
    "Capital One":    ["Capital One"],
    "Credit Re-Payment": ["online payment", "Payment Thankyou"],
    "Credit Cashback": ["credit reward", "credit travel"],
    "Venmo":          ["venmo"],
    "eBay":           ["ebay"],
    "Chase ATM":      ["atm withdrawal chase", "chase bank"],
    "ATT":           ["att", "att wireless","att*bill"],
    "Ultra Wireless": ["ultra wireless", "ultra"],
    "Geico":     ["geico"],
    "State Farm": ["statefarm", "state farm"],
    "Discount Tires": ["discount tires"],
    "Guess Who": ["guesswho"],
    "Bentonville Eye Care": ["eyecare"]
}

# Noise patterns to strip before matching
_NOISE_PATTERNS = [
    r"\b\d{5,}\b",                        # long digit strings (transaction IDs)
    r"\b[A-Z]{2}\b$",                     # trailing state codes
    r"\b(pending|help\.\S+|www\.\S+)\b",  # status/URL fragments
    r"[*#@]",                             # punctuation noise
    r"\b\d{3}-\d{3}-\d{4}\b",            # phone numbers
    r"\s{2,}",                            # extra whitespace
]


def clean_description(raw: str) -> str:
    """Strip noise tokens from a raw bank description."""
    text = raw.upper()
    for pat in _NOISE_PATTERNS:
        text = re.sub(pat, " ", text, flags=re.IGNORECASE)
    return text.strip()


def normalize_merchant(
    raw: str,
    threshold: int = 75,
    spacy_hint: Optional[str] = None,
) -> tuple[str, float]:
    """
    Normalize a raw transaction description to a canonical merchant name.

    Four-layer strategy (stops at first hit):
      1. Exact / substring match against known aliases         (fast path, confidence 1.0)
      2. Fuzzy match with RapidFuzz against known aliases      (slow path, confidence 0.75–1.0)
      3. Fuzzy match using spaCy NER hint against aliases      (NLP path, confidence 0.55–0.85)
      4. Fallback: first meaningful token from cleaned string  (last resort, confidence 0.3)

    Args:
        raw:         Raw bank description string.
        threshold:   Minimum RapidFuzz score (0–100) to accept a fuzzy match.
        spacy_hint:  doc._.merchant extracted by the spaCy merchant_extractor
                     component. Used as an additional signal when regex/fuzzy
                     matching on the raw string fails.

    Returns:
        (merchant_name, confidence_0_to_1)
    """
    cleaned = clean_description(raw)

    # Build alias lookup for fuzzy matching — exclude short aliases (< 6 chars)
    # because partial_ratio on short strings produces too many false positives
    # (e.g. "GEICO" matching "RIDGECO" in "BreckenridgeCO").
    # Layer 1 substring match handles short aliases reliably when they truly appear.
    all_aliases = [
        (alias, canonical)
        for canonical, aliases in MERCHANT_ALIASES.items()
        for alias in aliases
        if len(alias) >= 6
    ]
    alias_strings = [a[0].upper() for a in all_aliases]

    # ── Layer 1: fast substring match ────────────────────────────────────
    for canonical, aliases in MERCHANT_ALIASES.items():
        for alias in aliases:
            if alias.upper() in cleaned:
                return canonical, 1.0

    # ── Layer 2: fuzzy match on cleaned raw string ────────────────────────
    match = process.extractOne(cleaned, alias_strings, scorer=fuzz.partial_ratio)
    if match and match[1] >= threshold:
        idx = alias_strings.index(match[0])
        return all_aliases[idx][1], match[1] / 100.0

    # ── Layer 3: fuzzy match on spaCy NER hint ────────────────────────────
    # spaCy's NER often extracts a cleaner merchant token (e.g. "Starbucks"
    # from "STARBUCKS STORE 12345 SEATTLE WA") than our regex cleaner does.
    # We run the same fuzzy pass on that hint, but cap confidence at 0.85
    # to signal it came from an indirect source.
    if spacy_hint:
        hint_cleaned = spacy_hint.upper().strip()
        hint_match = process.extractOne(
            hint_cleaned, alias_strings, scorer=fuzz.partial_ratio
        )
        if hint_match and hint_match[1] >= threshold:
            idx = alias_strings.index(hint_match[0])
            # Cap at 0.85 — NLP hint is less reliable than direct string match
            confidence = min(hint_match[1] / 100.0, 0.85)
            return all_aliases[idx][1], confidence

    # ── Layer 4: fallback — first meaningful token ────────────────────────
    # Prefer the spaCy hint as a human-readable fallback over a raw token,
    # since NER has already stripped noise (IDs, state codes, etc.)
    if spacy_hint and len(spacy_hint.strip()) > 2:
        return spacy_hint.strip().title(), 0.3

    tokens = [t for t in cleaned.split() if len(t) > 2]
    fallback = tokens[0].title() if tokens else raw[:20].strip()
    return fallback, 0.3


# ─────────────────────────────────────────────
# 3. CATEGORY RULES
# ─────────────────────────────────────────────

# Category taxonomy: category → (subcategory → merchant set)
CATEGORY_RULES: dict[str, dict[str, set[str]]] = {
    "Shopping": {
        "Online":    {"Amazon", "eBay", "PayPal","Michael Kors", "Coach", "Hoka", "Shein", "Temu"},
        "Retail":    {"Target", "Jcpenny", "Khols", "Macys", "Marshalls", "Dollar-General",
                      "Dollar-Tree", "Aldi", "GuessWho",},
    },
    "Utilities": {
        "Water and Electricity": {"Bentonville Utilities"},
        "Wireless & Internet": {"ATT", "Ultra Wireless"},
    },
    "Entertainment": {
        "Movies & TV": {"Malco", "Hulu", "Netflix", "Amazon Prime", "Disney Plus"}
    },
    "Car Related": {
        "Car Service": {"Landers"},
        "Car Tires": {"Discount Tires"},
    },
    "Dining": {
        "Coffee":    {"Starbucks", "Dunkin", "Blue Bottle Coffee", "Krispy Kreme"},
        "Fast Food": {"McDonald's", "Popeyes", "Burger King", "Chipotle", "Chick-Fil-A", "Taco Bell"},
        "Restaurant":{"Chipotle", "Olive Garden", "Bawarchi", "Cuisine", "Panera", "Taj"},
        "Delivery":  {"Uber Eats"},
    },
    "Transportation": {
        "Rideshare": {"Uber", "Public Transportation"},
        "Airlines":  {"Delta Airlines", "United Airlines"},
        "Gas":       {"Shell", "Chevron", "Maverik", "Phillips"},
    },
    "Kids Related": {
        "Gymnastics":   {"Planet Fitness","Infiniti", "Fastlane", "Altitude", },
        "BCC Lessons": {"Bentonville Community Center"}
    },
    "Groceries": {
        "Supermarket": {"Whole Foods", "Walmart", "Sams Club","Aldi"},
        "Desi Groceries": {"India Mart", "Asian Amigo", "Achau", "Namaste", "Community Butcher"},
        "Dairy": {"Braums"}
    },
    "Banking": {
        "Credit Card": {"Capital One", "Citi", "Bofa", "Robinhood", "Card Payment", "Credit Re-Payment", "Credit Cashback",},
    },
    "Medical & Beauty": {
        "Pharmacy":  {"CVS Pharmacy", "Walgreens"},
        "Beauty":   {"Great Clips"},
        "Hospitals": {"Mercy", "Washington Regional", "North West Medical", "North West Health", "Bentonville Eye Care"}
    },
    "Transfers & ATM": {
        "ATM":       {"Chase ATM"},
        "P2P":       {"Venmo", "zelle"},
    },
    "Insurance": {
        "Insurance": {"Geico", "State Farm"},
    },
}

# Flat reverse lookup: merchant → (category, subcategory)
_MERCHANT_TO_CATEGORY: dict[str, tuple[str, str]] = {}
for _cat, _subs in CATEGORY_RULES.items():
    for _sub, _merchants in _subs.items():
        for _m in _merchants:
            _MERCHANT_TO_CATEGORY[_m] = (_cat, _sub)


def categorize(merchant: str, confidence: float) -> tuple[str, str, float]:
    """
    Map a normalized merchant name to (category, subcategory, adjusted_confidence).
    If the merchant is unknown, return a low-confidence fallback.
    """
    if merchant in _MERCHANT_TO_CATEGORY:
        cat, sub = _MERCHANT_TO_CATEGORY[merchant]
        return cat, sub, confidence
    return "Uncategorized", "Other", 0.2


# ─────────────────────────────────────────────
# 4. spaCy CUSTOM COMPONENTS
# ─────────────────────────────────────────────

# Register custom extension attributes on Token and Doc
if not Token.has_extension("is_noise"):
    Token.set_extension("is_noise", default=False)
if not Token.has_extension("is_amount"):
    Token.set_extension("is_amount", default=False)
if not Doc.has_extension("merchant"):
    Doc.set_extension("merchant", default=None)
if not Doc.has_extension("location"):
    Doc.set_extension("location", default=None)


@Language.component("transaction_tokenizer")
def transaction_tokenizer(doc: Doc) -> Doc:
    """
    Tag tokens as noise (IDs, phone numbers, state codes) or monetary amounts.
    This runs AFTER the default tokenizer and tagger.
    """
    noise_re = re.compile(r"^\d{4,}$|^\d{3}-\d{3}-\d{4}$|^[A-Z]{2}$")
    amount_re = re.compile(r"^\$?\d+\.\d{2}$")

    for token in doc:
        if noise_re.match(token.text):
            token._.is_noise = True
        if amount_re.match(token.text):
            token._.is_amount = True
    return doc


@Language.component("merchant_extractor")
def merchant_extractor(doc: Doc) -> Doc:
    """
    Extract a best-guess merchant span from the doc using NER + heuristics.
    Stores result in doc._.merchant.
    """
    # Prefer spaCy NER ORG entities
    org_ents = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
    if org_ents:
        doc._.merchant = org_ents[0]
        return doc

    # Fallback: first meaningful non-noise token(s)
    meaningful = [
        t.text for t in doc
        if not t._.is_noise and not t.is_punct and not t.is_space and len(t.text) > 1
    ]
    doc._.merchant = " ".join(meaningful[:3]) if meaningful else doc.text[:20]
    return doc


@Language.component("location_extractor")
def location_extractor(doc: Doc) -> Doc:
    """Extract GPE (city/state) entities as the transaction location."""
    gpe_ents = [ent.text for ent in doc.ents if ent.label_ == "GPE"]
    if gpe_ents:
        doc._.location = ", ".join(gpe_ents[:2])
    return doc


def build_spacy_pipeline() -> Language:
    """
    Build and return a spaCy NLP pipeline with custom financial components.
    Pipeline order:
        tok2vec → tagger → parser → ner
        → transaction_tokenizer → merchant_extractor → location_extractor
    """
    nlp = spacy.load("en_core_web_sm", disable=["lemmatizer"])
    nlp.add_pipe("transaction_tokenizer", last=True)
    nlp.add_pipe("merchant_extractor", last=True)
    nlp.add_pipe("location_extractor", last=True)
    return nlp


# ─────────────────────────────────────────────
# 5. TRANSACTION PROCESSOR (puts it all together)
# ─────────────────────────────────────────────

class TransactionProcessor:
    """
    End-to-end pipeline: raw transaction dict → enriched Transaction object.

    Usage:
        processor = TransactionProcessor()
        txn = processor.process({"id": "t001",
                                  "raw_description": "AMZN*MKTP US*2K4F9 SEATTLE WA",
                                  "amount": -45.99,
                                  "date": "2024-01-15"})
        print(txn)
    """

    def __init__(self):
        print("Loading spaCy pipeline…")
        self.nlp = build_spacy_pipeline()
        print(f"Pipeline components: {self.nlp.pipe_names}")

    def process(self, raw: dict) -> Transaction:
        txn = Transaction(**raw)

        # Step 1: Run spaCy pipeline — extracts merchant hint + location
        doc = self.nlp(txn.raw_description)
        spacy_hint = doc._.merchant    # e.g. "Starbucks" from NER
        spacy_location = doc._.location  # e.g. "Seattle, WA" from GPE

        # Step 2: Merchant normalization — all 4 layers, spaCy hint feeds layer 3
        merchant, confidence = normalize_merchant(
            txn.raw_description,
            spacy_hint=spacy_hint,
        )
        txn.merchant = merchant
        txn.confidence = confidence

        # Step 3: Categorization
        txn.category, txn.subcategory, txn.confidence = categorize(merchant, confidence)

        # Step 4: Store location extracted by spaCy
        if spacy_location:
            txn.location = spacy_location

        # Step 5: Flag anomalies
        if abs(txn.amount) > 300:
            txn.flags.append("large_transaction")
        if txn.category == "Uncategorized":
            txn.flags.append("needs_review")

        return txn

    def process_batch(self, records: list[dict]) -> list[Transaction]:
        return [self.process(r) for r in records]

    def summary(self, transactions: list[Transaction]) -> dict:
        """Return spending breakdown by category."""
        from collections import defaultdict
        totals: dict[str, float] = defaultdict(float)
        for t in transactions:
            if t.is_debit:
                totals[t.category] += abs(t.amount)
        return dict(sorted(totals.items(), key=lambda x: x[1], reverse=True))


# ─────────────────────────────────────────────
# 6. DEMO / QUICK TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    data_path = Path(__file__).parent / "data" / "sample_transactions.json"
    with open(data_path) as f:
        records = json.load(f)

    processor = TransactionProcessor()
    transactions = processor.process_batch(records)

    print("\n── Processed Transactions ──────────────────────────────")
    for t in transactions:
        flags = f" ⚑ {', '.join(t.flags)}" if t.flags else ""
        loc = f" [{t.location}]" if t.location else ""
        print(f"  {t.id}  {t.merchant:<22}  {t.category:<20}  ${abs(t.amount):>7.2f}{loc}{flags}")

    print("\n── Spending by Category ────────────────────────────────")
    for cat, total in processor.summary(transactions).items():
        bar = "█" * int(total / 15)
        print(f"  {cat:<22} ${total:>7.2f}  {bar}")

    print("\n── Transactions Needing Review ─────────────────────────")
    flagged = [t for t in transactions if "needs_review" in t.flags]
    for t in flagged:
        print(f"  {t.id}  {t.raw_description[:40]!r}")
