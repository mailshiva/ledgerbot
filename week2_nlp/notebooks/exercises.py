
"""
Week 2 Exercises — NLP Techniques for Financial Parsing
========================================================
Work through these exercises to reinforce each concept.
Run each section independently; answers follow each exercise.
"""

# ── Setup ────────────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, "../..")  # allow importing transaction_processor from parent dir

from week2_nlp.transaction_processor import (
    clean_description,
    normalize_merchant,
    TransactionProcessor,
    CATEGORY_RULES,
)


# ═══════════════════════════════════════════════════════════════════════════
# EXERCISE 1 — Understanding clean_description()
# ═══════════════════════════════════════════════════════════════════════════
"""
Task: Run clean_description() on the three inputs below and observe
what noise gets stripped. Then answer the questions.

Questions:
  a) What 4 types of noise does clean_description() remove?
  b) Why do we uppercase the string before cleaning?
  c) What would happen if we ran categorization BEFORE cleaning?
"""

ex1_inputs = [
    "AMZN*MKTP US*2K4F9 SEATTLE WA",
    "SHELL OIL 57444780203 HOUSTON TX",
    "DD *DUNKIN #342512 Q35 BOSTON MA",
]

print("── Exercise 1: clean_description() ────────────────────────")
for raw in ex1_inputs:
    cleaned = clean_description(raw)
    print(f"  RAW:     {raw}")
    print(f"  CLEANED: {cleaned}\n")

# ANSWER KEY:
# a) Noise types: long digit strings (IDs), state codes, phone numbers, special chars (* # @)
# b) Uppercase normalizes case so alias matching is case-insensitive
# c) We'd get false negatives — "amzn*mktp" wouldn't match "AMZN" in the alias list


# ═══════════════════════════════════════════════════════════════════════════
# EXERCISE 2 — Merchant normalization + fuzzy matching
# ═══════════════════════════════════════════════════════════════════════════
"""
Task: Normalize the merchants below and observe confidence scores.

Questions:
  a) Which ones use the fast path (confidence == 1.0) vs fuzzy path?
  b) "WHOLEFDS" hits the fast path — why? Which alias matches?
  c) Lower the fuzzy threshold to 50. What happens to "CIRCUIT CITY 2341"?
  d) Add a new alias "COSTCO" under a canonical "Costco" entry and test it.
"""

ex2_inputs = [
    "WHOLEFDS #10234 AUSTIN TX",
    "STARBUCKS STORE 12345 SEATTLE WA",
    "CIRCUIT CITY 2341 RICHMOND VA",      # unknown merchant
    "AMZN*MKTP US*ABC123 PORTLAND OR",
]

print("── Exercise 2: normalize_merchant() ───────────────────────")
for raw in ex2_inputs:
    merchant, conf = normalize_merchant(raw)
    path = "fast" if conf == 1.0 else f"fuzzy ({conf:.0%})"
    print(f"  {raw[:40]:<42} → {merchant:<22} [{path}]")

# ANSWER KEY:
# a) Fast path (1.0): WHOLEFDS, STARBUCKS, AMZN  |  Fuzzy: CIRCUIT CITY (no match → fallback)
# b) "wholefds" is listed as an alias for "Whole Foods" — direct substring hit
# c) At threshold=50, Circuit City might fuzzy-match a weak alias; still low confidence
# d) Add to MERCHANT_ALIASES: "Costco": ["costco", "costco wholesale"]


# ═══════════════════════════════════════════════════════════════════════════
# EXERCISE 3 — Category taxonomy & rule extension
# ═══════════════════════════════════════════════════════════════════════════
"""
Task: Study CATEGORY_RULES and answer these questions, then extend the taxonomy.

Questions:
  a) How many total merchants are covered in the current taxonomy?
  b) Which category has the most subcategories?
  c) What category would you add for "AIRBNB" or "VRBO"?

Challenge: Add a new top-level category "Travel" with subcategories
  "Lodging" (Airbnb, Marriott, Hilton) and "Car Rental" (Hertz, Enterprise).
  Then call categorize("Airbnb", 1.0) and confirm it works.
"""

print("\n── Exercise 3: Category taxonomy ──────────────────────────")
total_merchants = sum(
    len(merchants)
    for subs in CATEGORY_RULES.values()
    for merchants in subs.values()
)
print(f"  Total covered merchants: {total_merchants}")
for cat, subs in CATEGORY_RULES.items():
    print(f"  {cat:<22} → {len(subs)} subcategories, {sum(len(m) for m in subs.values())} merchants")

# ANSWER KEY:
# a) Count by running the code above
# b) Shopping has the most (Online, Retail)
# c) Add "Travel > Lodging"


# ═══════════════════════════════════════════════════════════════════════════
# EXERCISE 4 — Full pipeline + batch processing
# ═══════════════════════════════════════════════════════════════════════════
"""
Task: Run the full TransactionProcessor on a custom set of transactions.

Questions:
  a) Which transactions get flagged as "large_transaction"?
  b) What's the total spend in "Food & Dining"?
  c) Why might "VENMO PAYMENT" be tricky to categorize?
  d) How would you improve confidence for P2P payments like Venmo/Zelle?
"""

import json
from pathlib import Path

print("\n── Exercise 4: Full pipeline ───────────────────────────────")
data_path = Path(__file__).parent.parent / "data" / "sample_transactions.json"

with open(data_path) as f:
    records = json.load(f)

processor = TransactionProcessor()
transactions = processor.process_batch(records)

# Category summary
summary = processor.summary(transactions)
print("\n  Spending summary:")
for cat, total in summary.items():
    print(f"    {cat:<22} ${total:>7.2f}")

# Flagged transactions
print("\n  Flagged transactions:")
for t in transactions:
    if t.flags:
        print(f"    {t.id}  {t.merchant:<22}  {t.flags}")

# ANSWER KEY:
# a) Delta Airlines ($342) and United Airlines ($289.50) exceed $300
# b) Sum all Food & Dining transactions
# c) Venmo descriptions don't contain the recipient name — ambiguous
# d) Use memo/note fields, contact lists, or ask users to label recurring payees


# ═══════════════════════════════════════════════════════════════════════════
# EXERCISE 5 — spaCy pipeline inspection (advanced)
# ═══════════════════════════════════════════════════════════════════════════
"""
Task: Directly use the spaCy pipeline to inspect NER and token extensions.

Questions:
  a) What entities does spaCy find in "DELTA AIR ATLANTA GA"?
  b) Which tokens are tagged as is_noise=True?
  c) When would spaCy's NER be more useful than our regex approach?
"""

print("\n── Exercise 5: spaCy pipeline inspection ───────────────────")
nlp = processor.nlp

test_descriptions = [
    "DELTA AIR 0062134500012 ATLANTA GA",
    "STARBUCKS STORE 12345 SEATTLE WA",
    "NETFLIX.COM 866-579-7172 CA",
]

for desc in test_descriptions:
    doc = nlp(desc)
    ents = [(e.text, e.label_) for e in doc.ents]
    noise = [t.text for t in doc if t._.is_noise]
    print(f"\n  Input:    {desc}")
    print(f"  Entities: {ents}")
    print(f"  Noise:    {noise}")
    print(f"  Merchant: {doc._.merchant}")
    print(f"  Location: {doc._.location}")

# ANSWER KEY:
# a) spaCy may find "DELTA AIR" as ORG and "ATLANTA" as GPE
# b) Long digit strings like "0062134500012" and phone numbers
# c) For longer, messier descriptions or non-English text; NER handles novel merchants
