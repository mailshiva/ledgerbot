"""
src/agent/live_card_tools.py

Live Robinhood credit card tools via Plaid API.

Fetches current-month (or any date range) transactions directly from Plaid —
no database write, pure in-memory aggregation.

Tools exposed to the agent:
  - get_live_card_spending_by_category
  - get_live_card_spending_by_merchant
  - get_live_card_transactions_by_date
  - get_live_card_summary

Secrets pulled from macOS Keychain (service: portfolio_advisor):
  - plaid_client_id
  - plaid_secret_production
  - plaid_access_token_robinhood_cc
  - plaid_account_id_robinhood_cc

Usage (standalone test):
  python -m src.agent.live_card_tools
"""

from __future__ import annotations

import subprocess
from collections import defaultdict
from datetime import date

# ---------------------------------------------------------------------------
# Keychain helper (same pattern as config.py)
# ---------------------------------------------------------------------------

def _get_secret(account: str) -> str:
    """Fetch a secret from macOS Keychain under service 'portfolio_advisor'."""
    import os
    # Allow env var override (for Render / CI)
    env_key = account.upper()
    if env_key in os.environ:
        return os.environ[env_key]

    result = subprocess.run(
        ["security", "find-generic-password",
         "-s", "portfolio_advisor",
         "-a", account,
         "-w"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Keychain secret '{account}' not found. "
            f"Run: security add-generic-password -s portfolio_advisor -a {account} -w YOUR_VALUE"
        )
    return result.stdout.strip()


# ---------------------------------------------------------------------------
# Plaid client (lazy init)
# ---------------------------------------------------------------------------

def _get_plaid_client():
    """Return an authenticated Plaid ApiClient (production)."""
    try:
        import plaid
        from plaid.api import plaid_api
        from plaid.configuration import Configuration
        from plaid.api_client import ApiClient
    except ImportError:
        raise ImportError("plaid-python not installed. Run: pip install plaid-python")

    configuration = Configuration(
        host=plaid.Environment.Production,
        api_key={
            "clientId": _get_secret("plaid_client_id"),
            "secret":   _get_secret("plaid_secret_production"),
        }
    )
    return ApiClient(configuration)


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

def _current_month_range() -> tuple[date, date]:
    today = date.today()
    start = today.replace(day=1)
    return start, today


def _parse_date(d: str | date | None, fallback: date) -> date:
    if d is None:
        return fallback
    if isinstance(d, date):
        return d
    return date.fromisoformat(d)


# ---------------------------------------------------------------------------
# Core fetch — all transactions for a date range
# ---------------------------------------------------------------------------

def _fetch_transactions(
    start_date: str | date | None = None,
    end_date:   str | date | None = None,
) -> list[dict]:
    """
    Fetch credit card transactions from Plaid for the given date range.
    Returns a list of dicts with keys:
      date, name, merchant_name, amount, category, transaction_id, pending
    """
    from plaid.api import plaid_api
    from plaid.model.transactions_get_request import TransactionsGetRequest
    from plaid.model.transactions_get_request_options import TransactionsGetRequestOptions

    month_start, today = _current_month_range()
    start = _parse_date(start_date, month_start)
    end   = _parse_date(end_date,   today)

    access_token = _get_secret("plaid_access_token_robinhood_cc")
    account_id   = _get_secret("plaid_account_id_robinhood_cc")

    api_client = _get_plaid_client()
    client = plaid_api.PlaidApi(api_client)

    options = TransactionsGetRequestOptions(account_ids=[account_id])
    request = TransactionsGetRequest(
        access_token=access_token,
        start_date=start,
        end_date=end,
        options=options,
    )

    response = client.transactions_get(request)
    txns = response["transactions"]

    # Handle pagination
    all_txns = list(txns)
    while len(all_txns) < response["total_transactions"]:
        request = TransactionsGetRequest(
            access_token=access_token,
            start_date=start,
            end_date=end,
            options=TransactionsGetRequestOptions(
                account_ids=[account_id],
                offset=len(all_txns),
            ),
        )
        response = client.transactions_get(request)
        all_txns.extend(response["transactions"])

    # Normalise to plain dicts
    result = []
    for t in all_txns:
        # Plaid amounts: positive = money leaving account (debit), negative = credit/refund
        amount = float(t["amount"])
        result.append({
            "transaction_id": t["transaction_id"],
            "date":           str(t["date"]),
            "name":           t["name"],
            "merchant_name":  t.get("merchant_name") or t["name"],
            "amount":         amount,
            "transaction_type": "CREDIT" if amount < 0 else "DEBIT",
            "category":       (t.get("personal_finance_category") or {}).get("primary")
                              or (t.get("category") or ["Uncategorized"])[0],
            "pending":        t.get("pending", False),
        })

    return result


# ---------------------------------------------------------------------------
# Tool 1 — spending by category
# ---------------------------------------------------------------------------

def get_live_card_spending_by_category(
    start_date: str | None = None,
    end_date:   str | None = None,
    include_pending: bool = True,
) -> list[dict]:
    """
    Return spending grouped by category for the Robinhood credit card.

    Args:
        start_date: ISO date string (default: first day of current month)
        end_date:   ISO date string (default: today)
        include_pending: include pending transactions (default True)

    Returns:
        List of {category, total_spent, transaction_count} sorted by total_spent desc
    """
    txns = _fetch_transactions(start_date, end_date)

    if not include_pending:
        txns = [t for t in txns if not t["pending"]]

    # Only debits (money spent)
    debits = [t for t in txns if t["transaction_type"] == "DEBIT"]

    grouped: dict[str, dict] = defaultdict(lambda: {"total_spent": 0.0, "transaction_count": 0})
    for t in debits:
        cat = t["category"]
        grouped[cat]["total_spent"]        += t["amount"]
        grouped[cat]["transaction_count"]  += 1

    result = [
        {
            "category":          cat,
            "total_spent":       round(data["total_spent"], 2),
            "transaction_count": data["transaction_count"],
        }
        for cat, data in grouped.items()
    ]
    return sorted(result, key=lambda x: x["total_spent"], reverse=True)


# ---------------------------------------------------------------------------
# Tool 2 — spending by merchant
# ---------------------------------------------------------------------------

def get_live_card_spending_by_merchant(
    start_date: str | None = None,
    end_date:   str | None = None,
    limit: int = 20,
    include_pending: bool = True,
) -> list[dict]:
    """
    Return spending grouped by merchant for the Robinhood credit card.

    Args:
        start_date: ISO date string (default: first day of current month)
        end_date:   ISO date string (default: today)
        limit:      max merchants to return (default 20, capped at 100)
        include_pending: include pending transactions (default True)

    Returns:
        List of {merchant_name, total_spent, transaction_count} sorted by total_spent desc
    """
    limit = min(limit, 100)
    txns  = _fetch_transactions(start_date, end_date)

    if not include_pending:
        txns = [t for t in txns if not t["pending"]]

    debits = [t for t in txns if t["transaction_type"] == "DEBIT"]

    grouped: dict[str, dict] = defaultdict(lambda: {"total_spent": 0.0, "transaction_count": 0})
    for t in debits:
        merchant = t["merchant_name"]
        grouped[merchant]["total_spent"]       += t["amount"]
        grouped[merchant]["transaction_count"] += 1

    result = [
        {
            "merchant_name":     merchant,
            "total_spent":       round(data["total_spent"], 2),
            "transaction_count": data["transaction_count"],
        }
        for merchant, data in grouped.items()
    ]
    result = sorted(result, key=lambda x: x["total_spent"], reverse=True)
    return result[:limit]


# ---------------------------------------------------------------------------
# Tool 3 — transactions by date range
# ---------------------------------------------------------------------------

def get_live_card_transactions_by_date(
    start_date: str | None = None,
    end_date:   str | None = None,
    limit: int = 50,
    include_pending: bool = True,
) -> list[dict]:
    """
    Return raw transactions for the Robinhood credit card sorted by date desc.

    Args:
        start_date: ISO date string (default: first day of current month)
        end_date:   ISO date string (default: today)
        limit:      max transactions to return (default 50, capped at 500)
        include_pending: include pending transactions (default True)

    Returns:
        List of {date, merchant_name, amount, category, transaction_type, pending}
    """
    limit = min(limit, 500)
    txns  = _fetch_transactions(start_date, end_date)

    if not include_pending:
        txns = [t for t in txns if not t["pending"]]

    txns = sorted(txns, key=lambda x: x["date"], reverse=True)

    return [
        {
            "date":             t["date"],
            "merchant_name":    t["merchant_name"],
            "amount":           round(t["amount"], 2),
            "category":         t["category"],
            "transaction_type": t["transaction_type"],
            "pending":          t["pending"],
        }
        for t in txns[:limit]
    ]


# ---------------------------------------------------------------------------
# Tool 4 — summary
# ---------------------------------------------------------------------------

def get_live_card_summary(
    start_date: str | None = None,
    end_date:   str | None = None,
) -> dict:
    """
    Return a spending summary for the Robinhood credit card.

    Args:
        start_date: ISO date string (default: first day of current month)
        end_date:   ISO date string (default: today)

    Returns:
        {period_start, period_end, total_spent, total_credits,
         transaction_count, pending_count, top_category, top_merchant}
    """
    month_start, today = _current_month_range()
    start = _parse_date(start_date, month_start)
    end   = _parse_date(end_date,   today)

    txns = _fetch_transactions(start_date, end_date)

    debits  = [t for t in txns if t["transaction_type"] == "DEBIT"]
    credits = [t for t in txns if t["transaction_type"] == "CREDIT"]
    pending = [t for t in txns if t["pending"]]

    total_spent   = round(sum(t["amount"] for t in debits), 2)
    total_credits = round(abs(sum(t["amount"] for t in credits)), 2)

    # Top category
    cat_totals: dict[str, float] = defaultdict(float)
    for t in debits:
        cat_totals[t["category"]] += t["amount"]
    top_category = max(cat_totals, key=cat_totals.get) if cat_totals else None

    # Top merchant
    merchant_totals: dict[str, float] = defaultdict(float)
    for t in debits:
        merchant_totals[t["merchant_name"]] += t["amount"]
    top_merchant = max(merchant_totals, key=merchant_totals.get) if merchant_totals else None

    return {
        "period_start":      str(start),
        "period_end":        str(end),
        "total_spent":       total_spent,
        "total_credits":     total_credits,
        "net_spend":         round(total_spent - total_credits, 2),
        "transaction_count": len(txns),
        "pending_count":     len(pending),
        "top_category":      top_category,
        "top_merchant":      top_merchant,
    }


# ---------------------------------------------------------------------------
# Tool definitions (OpenAI-compatible schema for unified_tools.py)
# ---------------------------------------------------------------------------

LIVE_CARD_TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_live_card_spending_by_category",
            "description": (
                "Fetch REAL-TIME Robinhood credit card spending grouped by category directly "
                "from Plaid — includes transactions NOT yet in the database. "
                "Use for current month / recent spending questions: "
                "'How much did I spend on dining this month?', "
                "'What categories did I spend on recently?', "
                "'spending since my last statement', 'current spending by category'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date": {
                        "type": "string",
                        "description": "Start date ISO format YYYY-MM-DD (default: first of current month)"
                    },
                    "end_date": {
                        "type": "string",
                        "description": "End date ISO format YYYY-MM-DD (default: today)"
                    },
                    "include_pending": {
                        "type": "boolean",
                        "description": "Include pending transactions (default: true)"
                    },
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_live_card_spending_by_merchant",
            "description": (
                "Fetch REAL-TIME Robinhood credit card spending grouped by merchant directly "
                "from Plaid — includes transactions NOT yet in the database. "
                "Use for current / recent merchant questions: "
                "'Where did I spend the most this month?', "
                "'show me my top merchants recently', "
                "'what merchants have I used since my last statement'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date": {
                        "type": "string",
                        "description": "Start date ISO format YYYY-MM-DD (default: first of current month)"
                    },
                    "end_date": {
                        "type": "string",
                        "description": "End date ISO format YYYY-MM-DD (default: today)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max number of merchants to return (default: 20)"
                    },
                    "include_pending": {
                        "type": "boolean",
                        "description": "Include pending transactions (default: true)"
                    },
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_live_card_transactions_by_date",
            "description": (
                "Fetch REAL-TIME Robinhood credit card transactions sorted by date directly "
                "from Plaid — includes transactions NOT yet in the database. "
                "Use for recent/current transaction list questions: "
                "'show me my recent credit card transactions', "
                "'what did I buy this week?', "
                "'transactions since my last statement', 'latest charges'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date": {
                        "type": "string",
                        "description": "Start date ISO format YYYY-MM-DD (default: first of current month)"
                    },
                    "end_date": {
                        "type": "string",
                        "description": "End date ISO format YYYY-MM-DD (default: today)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max transactions to return (default: 50)"
                    },
                    "include_pending": {
                        "type": "boolean",
                        "description": "Include pending transactions (default: true)"
                    },
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_live_card_summary",
            "description": (
                "Get a REAL-TIME Robinhood credit card spending summary (total spent, credits, "
                "top category, top merchant) directly from Plaid — includes transactions NOT yet "
                "in the database. PREFER this tool when the user asks about: "
                "'current spending', 'how much have I spent this month', "
                "'spending since my last statement', 'recent credit card spending', "
                "'what is my current credit card balance/activity'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date": {
                        "type": "string",
                        "description": "Start date ISO format YYYY-MM-DD (default: first of current month)"
                    },
                    "end_date": {
                        "type": "string",
                        "description": "End date ISO format YYYY-MM-DD (default: today)"
                    },
                },
                "required": []
            }
        }
    },
]


# ---------------------------------------------------------------------------
# Dispatcher (called by unified_tools.py)
# ---------------------------------------------------------------------------

LIVE_CARD_TOOL_HANDLERS = {
    "get_live_card_spending_by_category":  get_live_card_spending_by_category,
    "get_live_card_spending_by_merchant":  get_live_card_spending_by_merchant,
    "get_live_card_transactions_by_date":  get_live_card_transactions_by_date,
    "get_live_card_summary":               get_live_card_summary,
}


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    print("=" * 60)
    print("Live Card Tools — Standalone Test")
    print("=" * 60)

    print("\n[1] Summary (current month)")
    summary = get_live_card_summary()
    print(json.dumps(summary, indent=2))

    print("\n[2] Spending by category (current month)")
    by_cat = get_live_card_spending_by_category()
    print(json.dumps(by_cat[:5], indent=2))

    print("\n[3] Top merchants (current month)")
    by_merchant = get_live_card_spending_by_merchant(limit=5)
    print(json.dumps(by_merchant, indent=2))

    print("\n[4] Recent transactions (last 10)")
    txns = get_live_card_transactions_by_date(limit=10)
    print(json.dumps(txns, indent=2))
