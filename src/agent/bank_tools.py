"""
src/agent/bank_tools.py

Agent tools for querying bank statement data (checking, savings, loans).

Mirrors the credit card tools.py pattern:
  - Read-only SQL via db.conn
  - Table routing: bank_transactions (enriched) → bank_transactions_raw (fallback)
  - TOOL_DEFINITIONS for LLM function calling
  - execute_tool() dispatcher

Key differences from credit card tools:
  - account_type filter (checking/savings) on all tools
  - No merchant-related tools (bank descriptions don't have merchants)
  - get_bank_balance: running balance queries
  - get_loan_summary: loan payment/principal/interest breakdown
  - query_bank_db: free-form SQL against bank tables only
"""

from __future__ import annotations

import re
import sqlite3
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Table routing (enriched → raw fallback)
# ---------------------------------------------------------------------------

def _has_enriched_table(conn) -> bool:
    """Check if the enriched bank_transactions table exists and has data."""
    try:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='bank_transactions'"
        ).fetchone()
        if not row:
            return False
        count = conn.execute("SELECT COUNT(*) FROM bank_transactions").fetchone()
        return count[0] > 0 if count else False
    except Exception:
        return False


def _txn_table(conn) -> str:
    """Return the best available bank transaction table name."""
    return "bank_transactions" if _has_enriched_table(conn) else "bank_transactions_raw"


def _run(conn, sql: str) -> list[dict]:
    """Execute a read-only SQL query and return list of dicts."""
    cursor = conn.execute(sql)
    cols = [d[0] for d in cursor.description]
    return [dict(zip(cols, row)) for row in cursor.fetchall()]


# ---------------------------------------------------------------------------
# query_bank_db
# ---------------------------------------------------------------------------

def query_bank_db(db, sql: str) -> list[dict] | dict:
    """
    Run a free-form read-only SQL query against bank tables.
    Blocks INSERT/UPDATE/DELETE/DROP/ALTER for safety.
    """
    blocked = re.compile(
        r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE)\b", re.IGNORECASE
    )
    if blocked.search(sql):
        return {"error": "Write operations are not allowed. Use SELECT queries only."}
    try:
        return _run(db.conn, sql)
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# get_bank_categories
# ---------------------------------------------------------------------------

def get_bank_categories(db, account_type: str = None) -> list[dict]:
    """
    Get spending breakdown by category for bank transactions.
    Optionally filter by account_type ('checking' or 'savings').
    """
    table = _txn_table(db.conn)
    where = "WHERE 1=1"
    if account_type:
        where += f" AND account_type = '{account_type}'"

    sql = f"""
        SELECT category,
               COUNT(*) AS transaction_count,
               ROUND(SUM(CASE WHEN transaction_type='DEBIT' THEN amount ELSE 0 END), 2) AS total_spent,
               ROUND(SUM(CASE WHEN transaction_type='CREDIT' THEN amount ELSE 0 END), 2) AS total_received
        FROM {table}
        {where}
        GROUP BY category
        ORDER BY total_spent DESC
    """
    return _run(db.conn, sql)


# ---------------------------------------------------------------------------
# summarize_bank_spending
# ---------------------------------------------------------------------------

def summarize_bank_spending(
    db,
    month: str = None,
    category: str = None,
    account_type: str = None,
) -> dict:
    """
    Summarize bank spending with optional filters.
    month: e.g. '2025-01'
    category: e.g. 'Utilities'
    account_type: 'checking' or 'savings'
    """
    table = _txn_table(db.conn)
    where = "WHERE 1=1"
    if month:
        where += f" AND strftime('%Y-%m', date) = '{month}'"
    if category:
        where += f" AND category = '{category}'"
    if account_type:
        where += f" AND account_type = '{account_type}'"

    sql = f"""
        SELECT COUNT(*) AS transaction_count,
               ROUND(SUM(CASE WHEN transaction_type='DEBIT' THEN amount ELSE 0 END), 2) AS total_spent,
               ROUND(SUM(CASE WHEN transaction_type='CREDIT' THEN amount ELSE 0 END), 2) AS total_received
        FROM {table}
        {where}
    """
    row = _run(db.conn, sql)
    result = row[0] if row else {"transaction_count": 0, "total_spent": 0, "total_received": 0}
    if month:
        result["month"] = month
    if category:
        result["category_filter"] = category
    if account_type:
        result["account_type"] = account_type
    return result


# ---------------------------------------------------------------------------
# get_bank_spending_by_month
# ---------------------------------------------------------------------------

def get_bank_spending_by_month(
    db,
    account_type: str = None,
    months: int = 12,
) -> list[dict]:
    """
    Monthly spending/income trend for bank transactions.
    """
    table = _txn_table(db.conn)
    where = "WHERE 1=1"
    if account_type:
        where += f" AND account_type = '{account_type}'"

    months = min(months, 60)
    sql = f"""
        SELECT strftime('%Y-%m', date) AS month,
               COUNT(*) AS transaction_count,
               ROUND(SUM(CASE WHEN transaction_type='DEBIT' THEN amount ELSE 0 END), 2) AS total_spent,
               ROUND(SUM(CASE WHEN transaction_type='CREDIT' THEN amount ELSE 0 END), 2) AS total_received
        FROM {table}
        {where}
        GROUP BY month
        ORDER BY month DESC
        LIMIT {months}
    """
    return _run(db.conn, sql)


# ---------------------------------------------------------------------------
# find_bank_transactions
# ---------------------------------------------------------------------------

def find_bank_transactions(
    db,
    account_type: str = None,
    category: str = None,
    start_date: str = None,
    end_date: str = None,
    min_amount: float = None,
    max_amount: float = None,
    transaction_type: str = None,
    description: str = None,
    bank_name: str = None,
    limit: int = 50,
) -> list[dict]:
    """
    Search bank transactions with flexible filters.
    """
    table = _txn_table(db.conn)
    where = "WHERE 1=1"
    if account_type:
        where += f" AND account_type = '{account_type}'"
    if category:
        where += f" AND category = '{category}'"
    if start_date:
        where += f" AND date >= '{start_date}'"
    if end_date:
        where += f" AND date <= '{end_date}'"
    if min_amount is not None:
        where += f" AND amount >= {min_amount}"
    if max_amount is not None:
        where += f" AND amount <= {max_amount}"
    if transaction_type:
        where += f" AND transaction_type = '{transaction_type}'"
    if description:
        where += f" AND description LIKE '%{description}%'"
    if bank_name:
        where += f" AND bank_name = '{bank_name}'"

    limit = min(limit, 100)
    sql = f"""
        SELECT * FROM {table}
        {where}
        ORDER BY date DESC
        LIMIT {limit}
    """
    return _run(db.conn, sql)


# ---------------------------------------------------------------------------
# get_bank_balance
# ---------------------------------------------------------------------------

def get_bank_balance(db, account_type: str = None) -> list[dict]:
    """
    Get the most recent balance for each account type.
    Returns the latest transaction's running balance per account.
    """
    table = _txn_table(db.conn)
    where = "WHERE 1=1"
    if account_type:
        where += f" AND account_type = '{account_type}'"

    sql = f"""
        SELECT account_type, date, balance, description
        FROM {table}
        {where}
        AND balance IS NOT NULL
        AND date = (
            SELECT MAX(t2.date) FROM {table} t2
            WHERE t2.account_type = {table}.account_type
            AND t2.balance IS NOT NULL
        )
        ORDER BY account_type
    """
    return _run(db.conn, sql)


# ---------------------------------------------------------------------------
# get_uncategorized_bank
# ---------------------------------------------------------------------------

def get_uncategorized_bank(db, account_type: str = None, limit: int = 50) -> list[dict]:
    """
    Find bank transactions that couldn't be categorized.
    """
    table = _txn_table(db.conn)
    where = "WHERE category = 'Uncategorized' AND transaction_type = 'DEBIT'"
    if account_type:
        where += f" AND account_type = '{account_type}'"

    limit = min(limit, 100)
    sql = f"""
        SELECT * FROM {table}
        {where}
        ORDER BY amount DESC
        LIMIT {limit}
    """
    return _run(db.conn, sql)


# ---------------------------------------------------------------------------
# get_bank_statements
# ---------------------------------------------------------------------------

def get_bank_statements(db) -> list[dict]:
    """Get all imported bank statements with metadata."""
    sql = """
        SELECT id, filename, bank_name,
               statement_period_start, statement_period_end,
               import_date, total_transactions, total_debits, total_credits
        FROM bank_statements
        ORDER BY statement_period_end DESC
    """
    return _run(db.conn, sql)


# ---------------------------------------------------------------------------
# get_loan_summary
# ---------------------------------------------------------------------------

def get_loan_summary(db, months: int = 12) -> list[dict]:
    """
    Summarize loan payments by month: total payment, principal, interest.
    """
    months = min(months, 60)
    table = "loan_transactions"
    sql = f"""
        SELECT strftime('%Y-%m', date) AS month,
               COUNT(*) AS payment_count,
               ROUND(SUM(payment_amount), 2) AS total_payment,
               ROUND(SUM(principal_amount), 2) AS total_principal,
               ROUND(SUM(interest_amount), 2) AS total_interest,
               ROUND(MIN(balance), 2) AS ending_balance
        FROM {table}
        GROUP BY month
        ORDER BY month DESC
        LIMIT {months}
    """
    return _run(db.conn, sql)


# ---------------------------------------------------------------------------
# get_loan_transactions
# ---------------------------------------------------------------------------

def get_loan_transactions(
    db,
    start_date: str = None,
    end_date: str = None,
    limit: int = 50,
) -> list[dict]:
    """
    Get individual loan payment records.
    """
    table = "loan_transactions"
    where = "WHERE 1=1"
    if start_date:
        where += f" AND date >= '{start_date}'"
    if end_date:
        where += f" AND date <= '{end_date}'"

    limit = min(limit, 100)
    sql = f"""
        SELECT * FROM {table}
        {where}
        ORDER BY date DESC
        LIMIT {limit}
    """
    return _run(db.conn, sql)


# ---------------------------------------------------------------------------
# execute_tool dispatcher
# ---------------------------------------------------------------------------

_TOOL_DISPATCH = {
    "query_bank_db": lambda db, **kw: query_bank_db(db, **kw),
    "get_bank_categories": lambda db, **kw: get_bank_categories(db, **kw),
    "summarize_bank_spending": lambda db, **kw: summarize_bank_spending(db, **kw),
    "get_bank_spending_by_month": lambda db, **kw: get_bank_spending_by_month(db, **kw),
    "find_bank_transactions": lambda db, **kw: find_bank_transactions(db, **kw),
    "get_bank_balance": lambda db, **kw: get_bank_balance(db, **kw),
    "get_uncategorized_bank": lambda db, **kw: get_uncategorized_bank(db, **kw),
    "get_bank_statements": lambda db, **kw: get_bank_statements(db, **kw),
    "get_loan_summary": lambda db, **kw: get_loan_summary(db, **kw),
    "get_loan_transactions": lambda db, **kw: get_loan_transactions(db, **kw),
}


def execute_bank_tool(name: str, db, **kwargs) -> Any:
    """Dispatch a bank tool call by name."""
    if name not in _TOOL_DISPATCH:
        raise KeyError(f"Unknown bank tool: {name}")
    return _TOOL_DISPATCH[name](db, **kwargs)


# ---------------------------------------------------------------------------
# TOOL_DEFINITIONS — OpenAI-compatible function calling schema
# ---------------------------------------------------------------------------

BANK_TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "query_bank_db",
            "description": "Run a read-only SQL query against bank statement tables (bank_transactions, bank_transactions_raw, loan_transactions, bank_statements). Use this for custom queries not covered by other bank tools.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "A SELECT SQL query against bank tables."
                    }
                },
                "required": ["sql"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_bank_categories",
            "description": "Get spending breakdown by category for bank transactions (checking and/or savings accounts). Returns transaction count, total spent, and total received per category.",
            "parameters": {
                "type": "object",
                "properties": {
                    "account_type": {
                        "type": "string",
                        "enum": ["checking", "savings"],
                        "description": "Filter by account type. Omit for both.",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_bank_spending",
            "description": "Summarize total bank spending and income with optional filters by month, category, and account type (checking/savings).",
            "parameters": {
                "type": "object",
                "properties": {
                    "month": {
                        "type": "string",
                        "description": "Filter by month in YYYY-MM format, e.g. '2025-01'.",
                    },
                    "category": {
                        "type": "string",
                        "description": "Filter by category, e.g. 'Utilities', 'Income'.",
                    },
                    "account_type": {
                        "type": "string",
                        "enum": ["checking", "savings"],
                        "description": "Filter by account type. Omit for both.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_bank_spending_by_month",
            "description": "Get monthly spending and income trends for bank accounts. Shows total spent and received per month.",
            "parameters": {
                "type": "object",
                "properties": {
                    "account_type": {
                        "type": "string",
                        "enum": ["checking", "savings"],
                        "description": "Filter by account type. Omit for both.",
                    },
                    "months": {
                        "type": "integer",
                        "description": "Number of months to return (default 12, max 60).",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_bank_transactions",
            "description": "Search bank transactions with flexible filters: account type, category, date range, amount range, transaction type (DEBIT/CREDIT), description keyword, bank name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "account_type": {
                        "type": "string",
                        "enum": ["checking", "savings"],
                        "description": "Filter by account type.",
                    },
                    "category": {"type": "string", "description": "Filter by category."},
                    "start_date": {"type": "string", "description": "Start date (YYYY-MM-DD)."},
                    "end_date": {"type": "string", "description": "End date (YYYY-MM-DD)."},
                    "min_amount": {"type": "number", "description": "Minimum amount."},
                    "max_amount": {"type": "number", "description": "Maximum amount."},
                    "transaction_type": {
                        "type": "string",
                        "enum": ["DEBIT", "CREDIT"],
                        "description": "Filter by DEBIT or CREDIT.",
                    },
                    "description": {"type": "string", "description": "Keyword search in description."},
                    "bank_name": {"type": "string", "description": "Filter by bank name, e.g. 'dcu'."},
                    "limit": {"type": "integer", "description": "Max results (default 50, max 100)."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_bank_balance",
            "description": "Get the most recent running balance for checking and/or savings accounts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "account_type": {
                        "type": "string",
                        "enum": ["checking", "savings"],
                        "description": "Filter by account type. Omit for all accounts.",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_uncategorized_bank",
            "description": "Find bank debit transactions that could not be categorized. Useful for identifying transactions that need manual review.",
            "parameters": {
                "type": "object",
                "properties": {
                    "account_type": {
                        "type": "string",
                        "enum": ["checking", "savings"],
                        "description": "Filter by account type.",
                    },
                    "limit": {"type": "integer", "description": "Max results (default 50, max 100)."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_bank_statements",
            "description": "List all imported bank statements with metadata: filename, bank name, period, transaction counts, and debit/credit totals.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_loan_summary",
            "description": "Summarize vehicle loan payments by month: total payment amount, principal paid, interest paid, and ending balance. Useful for tracking loan payoff progress.",
            "parameters": {
                "type": "object",
                "properties": {
                    "months": {
                        "type": "integer",
                        "description": "Number of months to return (default 12, max 60).",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_loan_transactions",
            "description": "Get individual loan payment records with payment amount, principal, interest, and remaining balance. Supports date range filtering.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date": {"type": "string", "description": "Start date (YYYY-MM-DD)."},
                    "end_date": {"type": "string", "description": "End date (YYYY-MM-DD)."},
                    "limit": {"type": "integer", "description": "Max results (default 50, max 100)."},
                },
            },
        },
    },
]