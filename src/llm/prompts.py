"""
Prompt templates for the bank statement agent.

All templates are plain strings with {placeholders} for runtime values.
Keep business logic out of here — this file is pure text.

Schema source of truth: src/database/schema.sql  (transactions table)
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Schema description (single source of truth used in all prompts)
# ---------------------------------------------------------------------------

DB_SCHEMA_DESCRIPTION = """\
Table: transactions
  id                INTEGER   PRIMARY KEY AUTOINCREMENT
  raw_id            INTEGER   FK → transactions_raw(id)
  statement_id      INTEGER   FK → statements(id)
  bank_name         TEXT      e.g. "capital_one", "citi", "bofa", "robinhood"
  date              TEXT      ISO-8601 date (YYYY-MM-DD)
  description       TEXT      original description from the statement
  clean_description TEXT      normalized/cleaned description (may be NULL)
  amount            REAL      always positive; use transaction_type to determine direction
  transaction_type  TEXT      "DEBIT" = money out, "CREDIT" = money in, "UNKNOWN"
  balance           REAL      running balance after this transaction (may be NULL)
  merchant_name     TEXT      normalized merchant name (may be NULL)
  merchant_raw      TEXT      raw merchant string before normalization (may be NULL)
  category          TEXT      spending category, default "Uncategorized"
  subcategory       TEXT      optional sub-category (may be NULL)
  location          TEXT      merchant location if detected (may be NULL)
  confidence_score  REAL      NLP enrichment confidence 0.0-1.0, default 0.0
  enrichment_method TEXT      how the record was enriched, e.g. "spacy", "regex"
  enriched_at       TEXT      ISO-8601 datetime of last enrichment"""

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

AGENT_SYSTEM_PROMPT = (
    "You are a helpful personal finance assistant with read-only access to a SQLite "
    "database containing parsed bank transactions.\n\n"
    "DATABASE SCHEMA\n"
    "---------------\n"
    + DB_SCHEMA_DESCRIPTION
    + "\n\n"
    "KEY CONVENTIONS\n"
    "---------------\n"
    "- Expenses (money out) have transaction_type = 'DEBIT'  and amount > 0\n"
    "- Income / refunds   have transaction_type = 'CREDIT' and amount > 0\n"
    "- To sum spending, always filter WHERE transaction_type = 'DEBIT'\n"
    "- Merchant lookups should use merchant_name (normalized); fall back to\n"
    "  clean_description or description if merchant_name IS NULL\n"
    "- category defaults to 'Uncategorized' when not yet enriched\n\n"
    "AVAILABLE TOOLS\n"
    "---------------\n"
    "- query_db(sql)                          Execute a read-only SQL query; returns JSON list\n"
    "- get_categories()                       All distinct category values\n"
    "- get_merchants(limit)                   Top merchants by transaction count\n"
    "- summarize_spending(month, category)    Pre-computed aggregation for a month/category\n\n"
    "RULES\n"
    "-----\n"
    "1. Always use tools to fetch real data - never invent numbers.\n"
    "2. Use transaction_type = 'DEBIT' when calculating expenses/spending.\n"
    "3. Prefer concise answers; round dollar amounts to 2 decimal places.\n"
    "4. If a question is ambiguous, ask one clarifying question before querying.\n"
    "5. Never modify data (INSERT / UPDATE / DELETE / DROP are forbidden).\n"
    "6. When showing a list of transactions, limit to 10 rows unless asked for more.\n"
)

# ---------------------------------------------------------------------------
# Few-shot NL -> SQL examples
# ---------------------------------------------------------------------------

FEW_SHOT_NL_TO_SQL = """\
Examples of natural-language questions and the SQL they map to:

Q: How much did I spend in total last month?
SQL: SELECT ROUND(SUM(amount), 2) AS total_spent
     FROM transactions
     WHERE transaction_type = 'DEBIT'
       AND date >= date('now', 'start of month', '-1 month')
       AND date <  date('now', 'start of month');

Q: What are my top 5 spending categories this year?
SQL: SELECT category, ROUND(SUM(amount), 2) AS total
     FROM transactions
     WHERE transaction_type = 'DEBIT'
       AND strftime('%Y', date) = strftime('%Y', 'now')
     GROUP BY category
     ORDER BY total DESC
     LIMIT 5;

Q: Show me all Starbucks charges in January 2025.
SQL: SELECT date, clean_description, amount
     FROM transactions
     WHERE transaction_type = 'DEBIT'
       AND (merchant_name LIKE '%starbucks%'
            OR clean_description LIKE '%starbucks%')
       AND date BETWEEN '2025-01-01' AND '2025-01-31'
     ORDER BY date;

Q: Did I have any duplicate charges last month?
SQL: SELECT merchant_name, amount, COUNT(*) AS cnt
     FROM transactions
     WHERE transaction_type = 'DEBIT'
       AND date >= date('now', 'start of month', '-1 month')
       AND date <  date('now', 'start of month')
     GROUP BY merchant_name, amount
     HAVING cnt > 1
     ORDER BY cnt DESC;

Q: How much did I earn (credits) this month?
SQL: SELECT ROUND(SUM(amount), 2) AS total_credits
     FROM transactions
     WHERE transaction_type = 'CREDIT'
       AND date >= date('now', 'start of month');

Q: Which bank account had the most transactions last month?
SQL: SELECT bank_name, COUNT(*) AS txn_count
     FROM transactions
     WHERE date >= date('now', 'start of month', '-1 month')
       AND date <  date('now', 'start of month')
     GROUP BY bank_name
     ORDER BY txn_count DESC
     LIMIT 1;

Q: Show me transactions with low confidence enrichment.
SQL: SELECT date, description, merchant_name, category, confidence_score
     FROM transactions
     WHERE confidence_score < 0.5
     ORDER BY confidence_score ASC
     LIMIT 10;
"""

# ---------------------------------------------------------------------------
# SQL generation prompt
# ---------------------------------------------------------------------------

SQL_GENERATION_PROMPT = """\
You are a SQL expert for a personal finance SQLite database.

{schema}

IMPORTANT CONVENTIONS:
- Expenses = transaction_type = 'DEBIT'
- Income/refunds = transaction_type = 'CREDIT'
- Use merchant_name for merchant lookups (fall back to clean_description if NULL)
- amount is always positive -- direction is determined by transaction_type

{few_shot}

Convert the following natural-language question into a single, read-only SQL query.
Return ONLY the SQL statement -- no explanation, no markdown fences.

Question: {question}
SQL:"""

# ---------------------------------------------------------------------------
# Intent classification prompt
# ---------------------------------------------------------------------------

INTENT_CLASSIFICATION_PROMPT = """\
Classify the user's financial question into exactly one of these intents:

  spending_summary   - asks about total or category-level spending (DEBITs)
  income_summary     - asks about credits, refunds, or income received
  merchant_lookup    - asks about a specific merchant or store
  anomaly_detection  - asks about unusual, duplicate, or suspicious charges
  time_comparison    - asks to compare two time periods
  transaction_list   - asks to list or search individual transactions
  enrichment_quality - asks about confidence scores or unenriched transactions
  general_question   - anything else

Respond with ONLY the intent label, nothing else.

Question: {question}
Intent:"""

# ---------------------------------------------------------------------------
# Answer synthesis prompt (after SQL results are available)
# ---------------------------------------------------------------------------

ANSWER_SYNTHESIS_PROMPT = """\
You are a helpful personal finance assistant.

The user asked: "{question}"

The database returned the following data:
{data}

Write a clear, concise answer in 1-3 sentences.
Round dollar amounts to 2 decimal places.
Do not mention SQL, column names, or technical details.
"""

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def sql_generation_prompt(question: str) -> str:
    return SQL_GENERATION_PROMPT.format(
        schema=DB_SCHEMA_DESCRIPTION,
        few_shot=FEW_SHOT_NL_TO_SQL,
        question=question,
    )


def intent_classification_prompt(question: str) -> str:
    return INTENT_CLASSIFICATION_PROMPT.format(question=question)


def answer_synthesis_prompt(question: str, data: str) -> str:
    return ANSWER_SYNTHESIS_PROMPT.format(question=question, data=data)