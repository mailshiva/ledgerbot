"""
src/agent/tools.py
──────────────────
Database tool functions for the bank statement agent.

Each tool is a plain Python function that accepts a DatabaseManager and
returns a JSON-serialisable value (list of dicts, dict, or scalar).

Design principles
-----------------
- Read-only: every function issues SELECT statements only.
- Dual-table aware: prefer `transactions` (enriched, Week-2 NLP) but fall
  back to `transactions_raw` (always populated by the Week-1 parser).
- Defensive: catches sqlite3 errors and returns a structured error dict so
  the agent loop can surface a friendly message instead of crashing.
- No LLM imports: this module has zero dependency on src.llm.

Tool registry
-------------
TOOL_DEFINITIONS  — OpenAI-style JSON schema list ready to pass to any
                    LLM provider (Gemini, Anthropic, OpenAI, Ollama).
execute_tool()    — dispatcher: call by name with a DatabaseManager +
                    kwargs dict from the LLM's tool-call response.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Table routing helper
# ---------------------------------------------------------------------------

def _has_enriched_table(conn: sqlite3.Connection) -> bool:
    """Return True if the `transactions` (enriched) table exists and has rows."""
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='transactions'"
    ).fetchone()
    if not row:
        return False
    count = conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
    return count > 0


def _txn_table(conn: sqlite3.Connection) -> str:
    """Return 'transactions' if populated, else 'transactions_raw'."""
    return "transactions" if _has_enriched_table(conn) else "transactions_raw"


# ---------------------------------------------------------------------------
# Safe query runner
# ---------------------------------------------------------------------------

def _run(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> list[dict]:
    """Execute a read-only SQL statement and return list of dicts."""
    # Block any write operations
    upper = sql.strip().upper()
    for keyword in ("INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "REPLACE"):
        if upper.startswith(keyword):
            raise PermissionError(f"Write operation '{keyword}' is not permitted in agent tools.")
    logger.info("SQL: %s | params: %s", sql.strip(), params)
    cursor = conn.execute(sql, params)
    cols = [d[0] for d in cursor.description]
    return [dict(zip(cols, row)) for row in cursor.fetchall()]


# ---------------------------------------------------------------------------
# Tool 1 — query_db
# ---------------------------------------------------------------------------
#def query_db(db, sql: str) -> list[dict] | dict:
def query_db(db, sql: str, params: list = None) -> list[dict] | dict:
    """
    Execute a read-only SQL query.
    Routes to Supabase via DualWriteManager shim, or falls back to
    direct SQLite if db is a plain DatabaseManager.
    """
    # Block writes — safety net regardless of which db is in use
    first = sql.strip().split()[0].upper()
    if first in ("INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER", "TRUNCATE"):
        return {"error": f"Write operation '{first}' not permitted via query_db"}

    # DualWriteManager path → Supabase
    if hasattr(db.conn, "execute_sql"):
        return db.conn.execute_sql(sql)

    # Plain DatabaseManager path → SQLite (used in tests with in-memory DB)
    try:
        rows = db.conn.execute(sql, params or []).fetchall()
        return [dict(r) for r in rows]
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Tool 2 — get_categories
# ---------------------------------------------------------------------------

def get_categories(db) -> list[dict] | dict:
    """
    Return all distinct spending categories together with transaction counts
    and total amounts, ordered by total spend descending.

    Queries the enriched `transactions` table if available, otherwise falls
    back to `transactions_raw`.
    """
    try:
        table = _txn_table(db.conn)
        sql = f"""
            SELECT
                category,
                COUNT(*)                                    AS transaction_count,
                ROUND(SUM(CASE WHEN transaction_type = 'DEBIT'
                               THEN amount ELSE 0 END), 2) AS total_spent,
                ROUND(SUM(CASE WHEN transaction_type = 'CREDIT'
                               THEN amount ELSE 0 END), 2) AS total_received
            FROM {table}
            WHERE category IS NOT NULL
            GROUP BY category
            ORDER BY total_spent DESC
        """
        return _run(db.conn, sql)
    except sqlite3.Error as exc:
        return {"error": f"get_categories failed: {exc}"}


# ---------------------------------------------------------------------------
# Tool 3 — get_category_subcategories
# ---------------------------------------------------------------------------

def get_category_subcategories(db, category: str) -> list[dict] | dict:
    """
    Return all distinct subcategories within a given category, with
    transaction counts and total amounts, ordered by total spend descending.

    Parameters
    ----------
    db       : DatabaseManager
    category : The category to drill into (case-insensitive).
    """
    try:
        table = _txn_table(db.conn)
        sql = f"""
            SELECT
                subcategory,
                COUNT(*)                                    AS transaction_count,
                ROUND(SUM(CASE WHEN transaction_type = 'DEBIT'
                               THEN amount ELSE 0 END), 2) AS total_spent,
                ROUND(SUM(CASE WHEN transaction_type = 'CREDIT'
                               THEN amount ELSE 0 END), 2) AS total_received
            FROM {table}
            WHERE LOWER(category) = LOWER(?)
              AND subcategory IS NOT NULL
            GROUP BY subcategory
            ORDER BY total_spent DESC
        """
        return _run(db.conn, sql, (category,))
    except sqlite3.Error as exc:
        return {"error": f"get_category_subcategories failed: {exc}"}


# ---------------------------------------------------------------------------
# Tool 4 — get_merchants
# ---------------------------------------------------------------------------

def get_merchants(db, limit: int = 20) -> list[dict] | dict:
    """
    Return the top merchants by transaction count.

    Uses `merchant_name` from whichever table is available.  Merchants with
    a NULL name are excluded — those are surfaced separately via
    get_uncategorized().

    Parameters
    ----------
    db    : DatabaseManager
    limit : How many merchants to return (default 20, max 100)
    """
    try:
        limit = min(int(limit), 100)
        table = _txn_table(db.conn)
        sql = f"""
            SELECT
                merchant_name,
                COUNT(*)                                    AS transaction_count,
                ROUND(SUM(CASE WHEN transaction_type = 'DEBIT'
                               THEN amount ELSE 0 END), 2) AS total_spent,
                ROUND(AVG(CASE WHEN transaction_type = 'DEBIT'
                               THEN amount END), 2)        AS avg_amount,
                MIN(date)                                   AS first_seen,
                MAX(date)                                   AS last_seen
            FROM {table}
            WHERE merchant_name IS NOT NULL
              AND merchant_name != ''
            GROUP BY merchant_name
            ORDER BY transaction_count DESC
            LIMIT ?
        """
        return _run(db.conn, sql, (limit,))
    except sqlite3.Error as exc:
        return {"error": f"get_merchants failed: {exc}"}


# ---------------------------------------------------------------------------
# Tool 5 — summarize_spending
# ---------------------------------------------------------------------------

def summarize_spending(db, month: str | None = None,
                       category: str | None = None) -> dict:
    """
    Pre-computed spending summary for a given month and/or category.

    Parameters
    ----------
    db       : DatabaseManager
    month    : ISO month string 'YYYY-MM', or None for all time
    category : Category name (partial match, case-insensitive), or None for all

    Returns a dict with total_spent, total_received, transaction_count,
    avg_transaction, largest_transaction, and (if month given) the month.
    """
    try:
        table = _txn_table(db.conn)
        conditions = []
        params: list[Any] = []

        if month:
            # Accept both 'YYYY-MM' and 'YYYY-M'
            conditions.append("strftime('%Y-%m', date) = ?")
            params.append(month[:7])   # safe truncation

        if category:
            conditions.append("category LIKE ?")
            params.append(f"%{category}%")

        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

        sql = f"""
            SELECT
                COUNT(*)                                         AS transaction_count,
                ROUND(SUM(CASE WHEN transaction_type = 'DEBIT'
                               THEN amount ELSE 0 END), 2)      AS total_spent,
                ROUND(SUM(CASE WHEN transaction_type = 'CREDIT'
                               THEN amount ELSE 0 END), 2)      AS total_received,
                ROUND(AVG(CASE WHEN transaction_type = 'DEBIT'
                               THEN amount END), 2)             AS avg_debit,
                ROUND(MAX(amount), 2)                           AS largest_transaction,
                MIN(date)                                       AS period_start,
                MAX(date)                                       AS period_end
            FROM {table}
            {where}
        """
        rows = _run(db.conn, sql, tuple(params))
        result = rows[0] if rows else {}
        if month:
            result["month"] = month[:7]
        if category:
            result["category_filter"] = category
        return result
    except sqlite3.Error as exc:
        return {"error": f"summarize_spending failed: {exc}"}


# ---------------------------------------------------------------------------
# Tool 6 — get_spending_by_month
# ---------------------------------------------------------------------------

def get_spending_by_month(db, months: int = 6) -> list[dict] | dict:
    """
    Return month-by-month spending and income for the last N months.

    Parameters
    ----------
    db     : DatabaseManager
    months : Number of recent months to include (default 6, max 24)
    """
    try:
        months = min(int(months), 24)
        table = _txn_table(db.conn)
        sql = f"""
            SELECT
                strftime('%Y-%m', date)                          AS month,
                COUNT(*)                                         AS transaction_count,
                ROUND(SUM(CASE WHEN transaction_type = 'DEBIT'
                               THEN amount ELSE 0 END), 2)      AS total_spent,
                ROUND(SUM(CASE WHEN transaction_type = 'CREDIT'
                               THEN amount ELSE 0 END), 2)      AS total_received,
                ROUND(AVG(CASE WHEN transaction_type = 'DEBIT'
                               THEN amount END), 2)             AS avg_debit
            FROM {table}
            GROUP BY month
            ORDER BY month DESC
            LIMIT ?
        """
        return _run(db.conn, sql, (months,))
    except sqlite3.Error as exc:
        return {"error": f"get_spending_by_month failed: {exc}"}


# ---------------------------------------------------------------------------
# Tool 7 — find_transactions
# ---------------------------------------------------------------------------

def find_transactions(db,
                      merchant: str | None = None,
                      category: str | None = None,
                      start_date: str | None = None,
                      end_date: str | None = None,
                      min_amount: float | None = None,
                      max_amount: float | None = None,
                      transaction_type: str | None = None,
                      bank_name: str | None = None,
                      limit: int = 20) -> list[dict] | dict:
    """
    Flexible transaction search with multiple optional filters.

    Parameters
    ----------
    db               : DatabaseManager
    merchant         : Partial merchant name match (case-insensitive)
    category         : Partial category match
    start_date       : ISO date 'YYYY-MM-DD'
    end_date         : ISO date 'YYYY-MM-DD'
    min_amount       : Minimum transaction amount
    max_amount       : Maximum transaction amount
    transaction_type : 'DEBIT', 'CREDIT', or 'UNKNOWN'
    bank_name        : Exact bank name (e.g. 'capital_one')
    limit            : Max rows to return (default 20, max 100)
    """
    try:
        limit = min(int(limit), 100)
        table = _txn_table(db.conn)

        # Choose columns based on which table we're querying
        if table == "transactions":
            select_cols = (
                "id, date, description, clean_description, amount, "
                "transaction_type, merchant_name, category, subcategory, "
                "location, bank_name, balance, confidence_score"
            )
        else:
            select_cols = (
                "id, date, description, amount, transaction_type, "
                "merchant_name, category, bank_name, balance"
            )

        conditions = []
        params: list[Any] = []

        if merchant:
            conditions.append("merchant_name LIKE ?")
            params.append(f"%{merchant}%")
        if category:
            conditions.append("category LIKE ?")
            params.append(f"%{category}%")
        if start_date:
            conditions.append("date >= ?")
            params.append(start_date)
        if end_date:
            conditions.append("date <= ?")
            params.append(end_date)
        if min_amount is not None:
            conditions.append("amount >= ?")
            params.append(float(min_amount))
        if max_amount is not None:
            conditions.append("amount <= ?")
            params.append(float(max_amount))
        if transaction_type:
            conditions.append("transaction_type = ?")
            params.append(transaction_type.upper())
        if bank_name:
            conditions.append("bank_name = ?")
            params.append(bank_name.lower())

        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        sql = f"""
            SELECT {select_cols}
            FROM {table}
            {where}
            ORDER BY date DESC
            LIMIT ?
        """
        params.append(limit)
        return _run(db.conn, sql, tuple(params))
    except sqlite3.Error as exc:
        return {"error": f"find_transactions failed: {exc}"}


# ---------------------------------------------------------------------------
# Tool 8 — detect_duplicates
# ---------------------------------------------------------------------------

def detect_duplicates(db, days_window: int = 3) -> list[dict] | dict:
    """
    Find potential duplicate charges: same merchant + same amount appearing
    more than once within a rolling N-day window.

    Parameters
    ----------
    db          : DatabaseManager
    days_window : How many days apart two charges can be to count as potential
                  duplicates (default 3)
    """
    try:
        table = _txn_table(db.conn)
        sql = f"""
            SELECT
                merchant_name,
                amount,
                COUNT(*)  AS occurrence_count,
                MIN(date) AS first_date,
                MAX(date) AS last_date,
                CAST(
                    julianday(MAX(date)) - julianday(MIN(date))
                AS INTEGER)           AS days_apart
            FROM {table}
            WHERE transaction_type = 'DEBIT'
              AND merchant_name IS NOT NULL
              AND merchant_name != ''
            GROUP BY merchant_name, amount
            HAVING occurrence_count > 1
               AND days_apart <= ?
            ORDER BY occurrence_count DESC, amount DESC
        """
        return _run(db.conn, sql, (days_window,))
    except sqlite3.Error as exc:
        return {"error": f"detect_duplicates failed: {exc}"}


# ---------------------------------------------------------------------------
# Tool 9 — get_uncategorized
# ---------------------------------------------------------------------------

def get_uncategorized(db, limit: int = 20) -> list[dict] | dict:
    """
    Return transactions that are still 'Uncategorized' or have no merchant
    name — useful for spotting enrichment gaps.

    Parameters
    ----------
    db    : DatabaseManager
    limit : Max rows to return (default 20, max 100)
    """
    try:
        limit = min(int(limit), 100)
        table = _txn_table(db.conn)
        sql = f"""
            SELECT
                id, date, description, amount, transaction_type,
                merchant_name, category, bank_name
            FROM {table}
            WHERE (category = 'Uncategorized' OR category IS NULL
                   OR merchant_name IS NULL OR merchant_name = '')
              AND transaction_type = 'DEBIT'
            ORDER BY amount DESC
            LIMIT ?
        """
        return _run(db.conn, sql, (limit,))
    except sqlite3.Error as exc:
        return {"error": f"get_uncategorized failed: {exc}"}


# ---------------------------------------------------------------------------
# Tool 10 — get_statements
# ---------------------------------------------------------------------------

def get_statements(db) -> list[dict] | dict:
    """
    List all imported statements with their metadata and totals.
    Useful for the agent to understand what data is available.
    """
    try:
        sql = """
            SELECT
                id, filename, bank_name, statement_date, import_date,
                total_transactions, total_debits, total_credits
            FROM statements
            ORDER BY import_date DESC
        """
        return _run(db.conn, sql)
    except sqlite3.Error as exc:
        return {"error": f"get_statements failed: {exc}"}


def get_subcategory_summary(
    db,
    subcategory: str = None,
    month: str = None,
    bank_name: str = None,
) -> list[dict] | dict:
    """
    Summarise spending grouped by subcategory, with optional filters.

    Parameters
    ----------
    subcategory : filter to a single subcategory (partial match, case-insensitive)
    month       : filter to YYYY-MM  e.g. '2025-01'
    bank_name   : filter to a specific bank  e.g. 'capital_one'
    """
    table = _txn_table(db.conn)
    if table == "transactions_raw":
        return []

    conditions = ["transaction_type = 'DEBIT'", "subcategory IS NOT NULL"]
    params: list = []

    if subcategory:
        conditions.append("LOWER(subcategory) LIKE LOWER(?)")
        params.append(f"%{subcategory}%")
    if month:
        conditions.append("strftime('%Y-%m', date) = ?")
        params.append(month)
    if bank_name:
        conditions.append("LOWER(bank_name) = LOWER(?)")
        params.append(bank_name)

    where = " AND ".join(conditions)

    sql = f"""
        SELECT
            subcategory,
            category,
            COUNT(*)        AS transaction_count,
            SUM(amount)     AS total_spent,
            AVG(amount)     AS avg_transaction,
            MIN(amount)     AS min_amount,
            MAX(amount)     AS max_amount
        FROM {table}
        WHERE {where}
        GROUP BY subcategory, category
        ORDER BY total_spent DESC
    """
    return query_db(db, sql, params)

# ---------------------------------------------------------------------------
# Tool registry — OpenAI-compatible JSON schema
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "query_db",
            "description": (
                "Execute a read-only SQL SELECT query against the bank transaction "
                "database. Use this for any question that requires custom SQL not "
                "covered by the other tools. Never use INSERT, UPDATE, DELETE, DROP."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "A valid read-only SQL SELECT statement."
                    }
                },
                "required": ["sql"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_categories",
            "description": (
                "Return all distinct spending categories with transaction counts "
                "and total amounts. Use this to answer questions about spending "
                "breakdown or to list available categories."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_category_subcategories",
            "description": (
                "Return all distinct subcategories within a given category, with "
                "transaction counts and total amounts. Use this to drill down into "
                "a specific category (e.g. all subcategories under 'Entertainment')."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "The category to drill into, e.g. 'Entertainment' or 'Food'."
                    }
                },
                "required": ["category"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_merchants",
            "description": (
                "Return the top merchants by transaction count with total spend "
                "and average amount. Use this to answer questions about where "
                "the user shops most frequently."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Number of merchants to return (default 20, max 100)."
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_spending",
            "description": (
                "Return a spending summary (totals, averages, counts) optionally "
                "filtered by month and/or category. Use this for high-level "
                "'how much did I spend' questions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "month": {
                        "type": "string",
                        "description": "Month in 'YYYY-MM' format, e.g. '2025-01'. Omit for all-time."
                    },
                    "category": {
                        "type": "string",
                        "description": "Category name or partial match, e.g. 'Food'. Omit for all categories."
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_spending_by_month",
            "description": (
                "Return month-by-month spending and income totals for the last N "
                "months. Use this for trend questions or month-over-month comparisons."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "months": {
                        "type": "integer",
                        "description": "Number of recent months to include (default 6, max 24)."
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "find_transactions",
            "description": (
                "Search for individual transactions using flexible filters: merchant, "
                "category, date range, amount range, transaction type, or bank. "
                "Use this to list or look up specific transactions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "merchant":         {"type": "string",  "description": "Partial merchant name (case-insensitive)."},
                    "category":         {"type": "string",  "description": "Partial category name."},
                    "start_date":       {"type": "string",  "description": "Start date 'YYYY-MM-DD'."},
                    "end_date":         {"type": "string",  "description": "End date 'YYYY-MM-DD'."},
                    "min_amount":       {"type": "number",  "description": "Minimum transaction amount."},
                    "max_amount":       {"type": "number",  "description": "Maximum transaction amount."},
                    "transaction_type": {"type": "string",  "description": "'DEBIT', 'CREDIT', or 'UNKNOWN'."},
                    "bank_name":        {"type": "string",  "description": "Bank identifier e.g. 'capital_one'."},
                    "limit":            {"type": "integer", "description": "Max rows (default 20, max 100)."}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "detect_duplicates",
            "description": (
                "Find potential duplicate charges: same merchant and amount "
                "appearing more than once within a short time window. Use this "
                "when the user asks about suspicious or duplicate charges."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "days_window": {
                        "type": "integer",
                        "description": "Days window for duplicate detection (default 3)."
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_uncategorized",
            "description": (
                "Return transactions that are still 'Uncategorized' or have no "
                "merchant name — useful for spotting enrichment gaps or answering "
                "questions about unrecognized charges."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Max rows to return (default 20, max 100)."
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_statements",
            "description": (
                "List all imported bank statements with metadata (bank, date, "
                "totals). Use this to answer 'what data do you have' questions "
                "or to show the user which statements have been parsed."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
    "type": "function",
    "function": {
        "name": "get_subcategory_summary",
        "description": (
            "Summarise spending grouped by subcategory (e.g. 'Streaming', "
            "'Coffee', 'Rideshare'). Supports filtering by a specific subcategory "
            "name (partial match), a calendar month (YYYY-MM), and/or bank. "
            "Returns subcategory, parent category, transaction count, total spent, "
            "average, min and max transaction amounts."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "subcategory": {
                    "type": "string",
                    "description": (
                        "Filter to a specific subcategory (partial, case-insensitive). "
                        "e.g. 'streaming', 'coffee', 'rideshare'. "
                        "Omit to return all subcategories."
                        ),
                    },
                "month": {
                    "type": "string",
                    "description": "Filter to a calendar month in YYYY-MM format. e.g. '2025-01'.",
                    },
                "bank_name": {
                    "type": "string",
                    "description": "Filter to a specific bank. e.g. 'capital_one', 'citi'.",
                    },
                },
            "required": [],
            },
        },
    },
]

# Quick lookup: tool name → function
_TOOL_FUNCTIONS = {
    "query_db":            query_db,
    "get_categories":               get_categories,
    "get_category_subcategories":   get_category_subcategories,
    "get_merchants":                get_merchants,
    "summarize_spending":  summarize_spending,
    "get_spending_by_month": get_spending_by_month,
    "find_transactions":   find_transactions,
    "detect_duplicates":   detect_duplicates,
    "get_uncategorized":   get_uncategorized,
    "get_statements":      get_statements,
    "get_subcategory_summary": get_subcategory_summary,
}


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def execute_tool(name: str, db, **kwargs) -> Any:
    """
    Call a tool by name, passing the DatabaseManager and any keyword args
    returned by the LLM's tool-call response.

    Parameters
    ----------
    name : Tool name matching one of TOOL_DEFINITIONS
    db   : DatabaseManager instance
    **kwargs : Arguments from the LLM's tool-call JSON

    Returns the tool's result (list of dicts, dict, or scalar).
    Raises KeyError if the tool name is unknown.
    """
    fn = _TOOL_FUNCTIONS.get(name)
    if fn is None:
        raise KeyError(f"Unknown tool: {name!r}. Available: {list(_TOOL_FUNCTIONS)}")
    return fn(db, **kwargs)
