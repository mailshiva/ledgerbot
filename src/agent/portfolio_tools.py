"""
src/agent/portfolio_tools.py
============================
Agent tools for querying live Robinhood portfolio holdings and recommendations.

These tools fetch real-time data from SnapTrade — they do NOT query the
SQLite database (the `db` arg is accepted but unused, to keep the dispatcher
signature uniform with bank_tools and tools).

Tools:
  get_portfolio_holdings    — all current positions, sorted by equity
  get_holding_by_ticker     — detail for a single scrip / ticker
  get_portfolio_summary     — aggregate: total equity, P&L, top holdings
  get_stock_recommendation  — holding position + macro context + company news
                              + industry news assembled for LLM recommendation

The holdings source is controlled by the env var PORTFOLIO_HOLDINGS_SOURCE
(default: "snaptrade"). See src/portfolio/holdings.py for details.
"""

from __future__ import annotations

from typing import Any

from src.portfolio.holdings import get_holdings
from src.portfolio.macro import get_macro_context
from src.portfolio.news import get_company_news, get_industry_news, get_ticker_info


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def get_portfolio_holdings(
    db,                          # accepted for dispatcher uniformity, not used
    source: str | None = None,
    sort_by: str = "equity",
) -> list[dict] | dict:
    """
    Fetch all current Robinhood positions.

    Returns holdings sorted by the given field (equity, ticker, pct_change).
    """
    result = get_holdings(source=source)

    # get_holdings() returns a dict only on error
    if isinstance(result, dict):
        return result

    valid_sorts = {"equity", "ticker", "pct_change", "quantity"}
    key = sort_by if sort_by in valid_sorts else "equity"
    reverse = key != "ticker"  # alphabetical asc for ticker, desc for numbers

    return sorted(result, key=lambda h: h.get(key, 0), reverse=reverse)


def get_holding_by_ticker(
    db,
    ticker: str,
    source: str | None = None,
) -> dict:
    """
    Return position detail for a single ticker / scrip.

    Returns an error dict if the ticker is not found in the portfolio.
    """
    result = get_holdings(source=source)

    if isinstance(result, dict):
        return result

    ticker_upper = ticker.strip().upper()
    match = next((h for h in result if h["ticker"].upper() == ticker_upper), None)
    if match is None:
        held = sorted(h["ticker"] for h in result)
        return {
            "error": f"Ticker '{ticker_upper}' not found in portfolio.",
            "held_tickers": held,
        }
    return match


def get_portfolio_summary(
    db,
    source: str | None = None,
    top_n: int = 5,
) -> dict:
    """
    Aggregate portfolio summary: total equity, cost basis, unrealized P&L,
    and top N holdings by equity.
    """
    result = get_holdings(source=source)

    if isinstance(result, dict):
        return result

    if not result:
        return {"error": "No holdings found in portfolio."}

    total_equity     = sum(h["equity"] for h in result)
    total_cost_basis = sum(h["quantity"] * h["avg_cost"] for h in result)
    unrealized_pl    = total_equity - total_cost_basis
    unrealized_pct   = (unrealized_pl / total_cost_basis * 100) if total_cost_basis else 0.0

    top_holdings = sorted(result, key=lambda h: h["equity"], reverse=True)[:top_n]

    gainers = sorted(
        [h for h in result if h["pct_change"] > 0],
        key=lambda h: h["pct_change"], reverse=True,
    )[:3]
    losers = sorted(
        [h for h in result if h["pct_change"] < 0],
        key=lambda h: h["pct_change"],
    )[:3]

    return {
        "total_holdings":   len(result),
        "total_equity":     round(total_equity, 2),
        "total_cost_basis": round(total_cost_basis, 2),
        "unrealized_pl":    round(unrealized_pl, 2),
        "unrealized_pct":   round(unrealized_pct, 2),
        "top_holdings":     top_holdings,
        "top_gainers":      gainers,
        "top_losers":       losers,
    }


def get_stock_recommendation(
    db,
    ticker: str,
    source: str | None = None,
    news_count: int = 3,
) -> dict:
    """
    Assemble all context needed to recommend a BUY / HOLD / SELL decision
    for any ticker — whether or not it is currently held in the portfolio.

    Gathers:
      - Current position if held (from SnapTrade); None if not held
      - Company metadata: name, industry, sector (from yfinance)
      - Macro-economic indicators (VIX, 10Y yield, S&P 5d change, Fed rate, CPI)
      - Top {news_count} news headlines about the company
      - Top {news_count} news headlines about the company's industry

    The LLM uses this context to reason and produce a recommendation:
      - If held:     BUY MORE / HOLD / REDUCE / SELL + rationale
      - If not held: BUY / WATCH / AVOID + rationale

    Args:
        ticker:     Stock ticker, e.g. "AAPL", "NVDA"
        source:     Holdings source override (default: snaptrade)
        news_count: Number of headlines to fetch per topic (default 3)

    Returns:
        dict with keys: ticker, held, position (None if not held),
        ticker_info, macro, company_news, industry_news
    """
    ticker_upper = ticker.strip().upper()

    # 1. Check portfolio — position is None if ticker not held
    holdings_result = get_holdings(source=source)
    if isinstance(holdings_result, dict) and "error" in holdings_result:
        # Holdings fetch failed entirely (e.g. SnapTrade auth error)
        position = None
        held = False
    else:
        holdings_list = holdings_result if isinstance(holdings_result, list) else []
        position = next(
            (h for h in holdings_list if h["ticker"].upper() == ticker_upper), None
        )
        held = position is not None

    # 2. Company metadata (name, industry, sector) from yfinance
    ticker_info = get_ticker_info(ticker_upper)

    # 3. Macro context
    macro = get_macro_context()

    # 4. Company news
    company_news = get_company_news(
        ticker=ticker_upper,
        company_name=ticker_info["company_name"],
        max_headlines=news_count,
    )

    # 5. Industry news
    industry_news = get_industry_news(
        industry=ticker_info["industry"],
        sector=ticker_info["sector"],
        max_headlines=news_count,
    )

    return {
        "ticker":        ticker_upper,
        "held":          held,
        "position":      position,       # None when not held
        "ticker_info":   ticker_info,
        "macro":         macro,
        "company_news":  company_news,
        "industry_news": industry_news,
    }


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_TOOL_DISPATCH = {
    "get_portfolio_holdings":   lambda db, **kw: get_portfolio_holdings(db, **kw),
    "get_holding_by_ticker":    lambda db, **kw: get_holding_by_ticker(db, **kw),
    "get_portfolio_summary":    lambda db, **kw: get_portfolio_summary(db, **kw),
    "get_stock_recommendation": lambda db, **kw: get_stock_recommendation(db, **kw),
}


def execute_portfolio_tool(name: str, db, **kwargs) -> Any:
    """Dispatch a portfolio tool call by name."""
    if name not in _TOOL_DISPATCH:
        raise KeyError(f"Unknown portfolio tool: {name}")
    return _TOOL_DISPATCH[name](db, **kwargs)


# ---------------------------------------------------------------------------
# Tool definitions — OpenAI function-calling schema
# ---------------------------------------------------------------------------

PORTFOLIO_TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_portfolio_holdings",
            "description": (
                "Fetch all current Robinhood portfolio positions (live data) — "
                "includes both stocks AND cryptocurrency holdings (Bitcoin/BTC, "
                "Ethereum/ETH, SHIB, XRP, DOGE, etc.). "
                "Returns each holding's ticker, name, shares/units held, "
                "average cost, current price, total equity value, and percent "
                "gain/loss. Use this when the user asks what stocks or crypto they "
                "hold, their full portfolio, total investments in Robinhood, or "
                "wants to see all positions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sort_by": {
                        "type": "string",
                        "enum": ["equity", "ticker", "pct_change", "quantity"],
                        "description": "Sort holdings by this field. Default: equity (largest position first).",
                    },
                    "source": {
                        "type": "string",
                        "enum": ["snaptrade", "csv", "robin_stocks", "custom"],
                        "description": "Holdings data source. Omit to use the configured default (snaptrade).",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_holding_by_ticker",
            "description": (
                "Get position detail for a single stock or cryptocurrency by ticker symbol. "
                "Returns quantity/units held, average cost, current price, total equity "
                "value, and percent gain/loss. Use this when the user asks about "
                "a specific holding — stocks like 'my AAPL position', 'how much TSLA do I hold', "
                "or cryptocurrencies like 'my Bitcoin', 'how much ETH do I have', "
                "'what is my SHIB worth', 'show me my XRP'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol, e.g. 'AAPL', 'TSLA', 'NVDA'.",
                    },
                    "source": {
                        "type": "string",
                        "enum": ["snaptrade", "csv", "robin_stocks", "custom"],
                        "description": "Holdings data source. Omit to use the configured default.",
                    },
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_portfolio_summary",
            "description": (
                "Get an aggregate summary of the Robinhood portfolio — covers both "
                "stocks AND cryptocurrency holdings. Returns total equity value, total "
                "cost basis, unrealized profit/loss (P&L) in dollars and percent, plus "
                "top holdings by value, top gainers, and top losers. "
                "Use this when the user asks about total portfolio value, overall P&L, "
                "total amount invested in Robinhood, total crypto investment, "
                "portfolio performance, or which stocks/crypto are up/down the most."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "top_n": {
                        "type": "integer",
                        "description": "Number of top holdings to include in the summary (default 5).",
                    },
                    "source": {
                        "type": "string",
                        "enum": ["snaptrade", "csv", "robin_stocks", "custom"],
                        "description": "Holdings data source. Omit to use the configured default.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_recommendation",
            "description": (
                "Get a comprehensive recommendation for any stock or cryptocurrency ticker — "
                "whether or not it is currently held in the portfolio. Fetches: current "
                "position if held (shares/units, avg cost, current price, P&L) or marks as "
                "not held; live macro-economic indicators (VIX, 10-Year Treasury yield, "
                "S&P 500 5-day change, Fed Funds Rate, CPI); top 3 latest news headlines "
                "about the asset; top 3 latest news headlines about its industry or sector. "
                "When held, produces a BUY MORE / HOLD / REDUCE / SELL recommendation. "
                "When not held, produces a BUY / WATCH / AVOID recommendation. "
                "Use this for any question like 'should I buy AAPL', 'what do you think "
                "about NVDA', 'give me a recommendation on TSLA', 'is META worth buying', "
                "'what should I do about my Bitcoin', 'should I buy more ETH', "
                "'give me advice on my crypto'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol, e.g. 'AAPL', 'TSLA', 'NVDA'.",
                    },
                    "news_count": {
                        "type": "integer",
                        "description": "Number of news headlines to fetch per topic (default 3).",
                    },
                    "source": {
                        "type": "string",
                        "enum": ["snaptrade", "csv", "robin_stocks", "custom"],
                        "description": "Holdings data source. Omit to use the configured default (snaptrade).",
                    },
                },
                "required": ["ticker"],
            },
        },
    },
]
