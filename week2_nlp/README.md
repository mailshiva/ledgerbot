# Week 2: NLP Techniques — spaCy, Merchant Normalization & Categorization

## Learning Objectives

By the end of this week you'll be able to:

- Build a **custom spaCy pipeline** with financial-domain components
- **Normalize messy merchant names** using regex + fuzzy matching (RapidFuzz)
- **Categorize transactions** with a rule-based taxonomy
- Wire everything into a **reusable `TransactionProcessor`** class
- Understand where each technique breaks down and how to improve it

---

## Setup

```bash
pip install spacy rapidfuzz
python -m spacy download en_core_web_sm
```

---

## File Structure

```
week2_nlp/
├── transaction_processor.py   ← Main module (read this first)
├── data/
│   └── sample_transactions.json
└── notebooks/
    └── exercises.py           ← Five guided exercises with answers
```

---

## Core Concepts

### 1. Why NLP for Transactions?

Bank descriptions like `AMZN*MKTP US*2K4F9 SEATTLE WA` are designed for
internal routing, not human reading. Before we can analyze spending, we need to:

1. **Clean** — strip IDs, phone numbers, state codes, special characters
2. **Normalize** — map `WHOLEFDS #10234` → `Whole Foods`
3. **Categorize** — map `Whole Foods` → `Groceries > Supermarket`

### 2. Merchant Normalization: Two-Pass Strategy

```
Raw description
     │
     ▼
clean_description()       ← strips noise with regex
     │
     ▼
Substring match           ← O(n) scan of known aliases  [fast path]
     │ no match
     ▼
Fuzzy match (RapidFuzz)   ← partial_ratio scoring       [slow path]
     │ below threshold
     ▼
Fallback: first token     ← best-effort, low confidence
```

**Key insight**: fuzzy matching is powerful but slow. Always try exact/substring
first. For production, pre-build an inverted index.

### 3. spaCy Pipeline Architecture

```
Input text
    │
    ▼  (built-in)
 tok2vec → tagger → parser → ner
    │
    ▼  (our custom components)
 transaction_tokenizer    ← tags noise tokens & amounts
    │
    ▼
 merchant_extractor       ← sets doc._.merchant via NER or heuristics
    │
    ▼
 location_extractor       ← sets doc._.location from GPE entities
```

Custom components use spaCy's **extension attributes** (`Token.set_extension`,
`Doc.set_extension`) to attach domain-specific metadata without breaking the
standard API.

### 4. Category Taxonomy Design

Our taxonomy is a two-level hierarchy:

```
Category (e.g. "Food & Dining")
  └── Subcategory (e.g. "Coffee")
        └── Merchant set (e.g. {"Starbucks", "Dunkin"})
```

We pre-compile a **flat reverse lookup** `_MERCHANT_TO_CATEGORY` at module load
time so categorization is O(1) at runtime.

---

## Key Design Decisions & Trade-offs

| Decision | Why | Trade-off |
|---|---|---|
| Regex before fuzzy | Speed | Misses novel spellings |
| Flat reverse lookup | O(1) categorization | Must rebuild on taxonomy change |
| spaCy `en_core_web_sm` | Lightweight | Lower NER accuracy vs `lg` model |
| Hardcoded aliases | Transparent, debuggable | Doesn't scale to 10k+ merchants |
| `confidence` field | Enables downstream filtering | Needs calibration |

---

## What's Next (Week 3 Preview)

- Replace hardcoded aliases with a **vector similarity model** (sentence-transformers)
- Add **user feedback loop**: confirmed labels retrain the categorizer
- Store enriched transactions in **SQLite** for querying
- Build a **REST API** around the processor (FastAPI)

---

## Quick Reference

```python
from transaction_processor import TransactionProcessor

processor = TransactionProcessor()

# Single transaction
txn = processor.process({
    "id": "t001",
    "raw_description": "AMZN*MKTP US*2K4F9 SEATTLE WA",
    "amount": -45.99,
    "date": "2024-01-15"
})
print(txn.merchant)    # "Amazon"
print(txn.category)   # "Shopping"
print(txn.confidence) # 1.0

# Batch + summary
import json
with open("data/sample_transactions.json") as f:
    records = json.load(f)
transactions = processor.process_batch(records)
print(processor.summary(transactions))
```
