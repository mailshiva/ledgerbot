"""
src/portfolio/holdings.py
=========================
Fetches live Robinhood portfolio holdings.

Adapted from portfolio-advisor/advisor/holdings.py — no cross-project imports.
Credentials are read from the *same* macOS Keychain service ("portfolio_advisor")
that the portfolio-advisor project already populated, so no new secrets are needed.

Supported sources (set env var PORTFOLIO_HOLDINGS_SOURCE):
  "snaptrade"    — Official OAuth via SnapTrade (recommended, default)
  "csv"          — Local CSV exported from Robinhood app
  "robin_stocks" — robin_stocks library
  "custom"       — Custom robinhood_client.py in portfolio-advisor project

Holdings dict shape (same as portfolio_advisor):
  {
    "ticker":        str,   e.g. "AAPL"
    "name":          str,   e.g. "Apple Inc."
    "quantity":      float, shares held
    "avg_cost":      float, average purchase price
    "current_price": float, latest market price
    "equity":        float, quantity * current_price
    "pct_change":    float, % gain/loss vs avg_cost
  }

Usage:
    from src.portfolio.holdings import get_holdings
    holdings = get_holdings()                     # uses PORTFOLIO_HOLDINGS_SOURCE env var
    holdings = get_holdings(source="csv")         # override source
    holdings = get_holdings(source="snaptrade")
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
from typing import Any

# ---------------------------------------------------------------------------
# Configuration (env-var overridable)
# ---------------------------------------------------------------------------

HOLDINGS_SOURCE   = os.getenv("PORTFOLIO_HOLDINGS_SOURCE",   "snaptrade")
HOLDINGS_CSV_PATH = os.getenv("PORTFOLIO_HOLDINGS_CSV_PATH", "holdings.csv")

# Keychain service where SnapTrade + Robinhood credentials are stored.
# This is the same service used by the portfolio-advisor project.
_KEYCHAIN_SERVICE = "portfolio_advisor"


# ---------------------------------------------------------------------------
# Secret resolution (Keychain → env var fallback)
# ---------------------------------------------------------------------------

def _get_secret(account: str, env_var: str) -> str:
    """Fetch a secret from macOS Keychain, falling back to environment variable."""
    if sys.platform == "darwin":
        try:
            result = subprocess.run(
                ["security", "find-generic-password",
                 "-a", account, "-s", _KEYCHAIN_SERVICE, "-w"],
                capture_output=True, text=True, check=True,
            )
            value = result.stdout.strip()
            if value:
                return value
        except subprocess.CalledProcessError:
            pass

    value = os.getenv(env_var, "").strip()
    if value:
        return value

    raise RuntimeError(
        f"\n  Missing secret: '{account}'\n"
        f"  Add to Keychain:  security add-generic-password "
        f"-a \"{account}\" -s \"{_KEYCHAIN_SERVICE}\" -w \"VALUE\"\n"
        f"  Or set env var:   export {env_var}=\"VALUE\"\n"
    )


def _robinhood_user() -> str:
    return _get_secret("robinhood_user", "ROBINHOOD_USER")


def _robinhood_pass() -> str:
    return _get_secret("robinhood_pass", "ROBINHOOD_PASS")


# ---------------------------------------------------------------------------
# Shared utility — live price via yfinance
# ---------------------------------------------------------------------------

def get_current_price(ticker: str) -> float:
    """Fetch live price from yfinance. Returns 0.0 if yfinance is not installed."""
    try:
        import yfinance as yf  # type: ignore
        info = yf.Ticker(ticker).fast_info
        price = info.get("lastPrice") or info.get("last_price") or 0.0
        return float(price)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Source: SnapTrade
# ---------------------------------------------------------------------------

def _from_snaptrade() -> list[dict]:
    """Fetch holdings via SnapTrade's official OAuth API."""
    from snaptrade_client import SnapTrade  # type: ignore

    client_id    = _get_secret("snaptrade_client_id",    "SNAPTRADE_CLIENT_ID")
    consumer_key = _get_secret("snaptrade_consumer_key", "SNAPTRADE_CONSUMER_KEY")
    user_id      = _get_secret("snaptrade_user_id",      "SNAPTRADE_USER_ID")
    user_secret  = _get_secret("snaptrade_user_secret",  "SNAPTRADE_USER_SECRET")
    account_id   = _get_secret("snaptrade_account_id",   "SNAPTRADE_ACCOUNT_ID")

    snaptrade     = SnapTrade(consumer_key=consumer_key, client_id=client_id)
    holdings_resp = snaptrade.account_information.get_user_account_positions(
        user_id=user_id,
        user_secret=user_secret,
        account_id=account_id,
    )
    positions = holdings_resp.body
    if not positions:
        return []

    holdings = []
    for pos in positions:
        try:
            symbol_obj = pos.get("symbol", {})

            if isinstance(symbol_obj, dict) and isinstance(symbol_obj.get("symbol"), dict):
                symbol_obj = symbol_obj["symbol"]

            if isinstance(symbol_obj, dict):
                ticker = (
                    symbol_obj.get("symbol") or
                    symbol_obj.get("raw_symbol") or
                    symbol_obj.get("ticker") or
                    "???"
                )
                if isinstance(ticker, dict):
                    ticker = ticker.get("symbol") or ticker.get("raw_symbol") or "???"
                ticker = str(ticker).strip()
                name   = symbol_obj.get("description") or symbol_obj.get("name") or ticker
            else:
                ticker = str(symbol_obj).strip()
                name   = ticker

            units         = float(pos.get("units", 0) or 0)
            avg_cost      = float(pos.get("average_purchase_price", 0) or 0)
            current_price = float(pos.get("price", 0) or 0)

            if units == 0:
                continue

            if current_price == 0:
                current_price = get_current_price(ticker)

            equity     = units * current_price
            pct_change = ((current_price - avg_cost) / avg_cost * 100) if avg_cost else 0.0

            holdings.append({
                "ticker":        ticker,
                "name":          name,
                "quantity":      units,
                "avg_cost":      round(avg_cost, 2),
                "current_price": round(current_price, 2),
                "equity":        round(equity, 2),
                "pct_change":    round(pct_change, 2),
            })
        except Exception:
            continue

    return holdings


# ---------------------------------------------------------------------------
# Source: CSV
# ---------------------------------------------------------------------------

def _from_csv(path: str) -> list[dict]:
    """Read holdings from a Robinhood CSV export. Fetches live prices via yfinance."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"CSV file not found: '{path}'. "
            f"Export from Robinhood: App → Account → Statements & History → Export, "
            f"then save as: {path}"
        )

    holdings = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ticker   = (row.get("Symbol") or row.get("symbol") or "").strip()
            name     = (row.get("Name")   or row.get("name")   or ticker).strip()
            qty      = float(row.get("Quantity")     or row.get("quantity")          or 0)
            avg_cost = float(row.get("Average Cost") or row.get("average_buy_price") or 0)

            if not ticker or qty == 0:
                continue

            current_price = get_current_price(ticker) or avg_cost
            equity        = qty * current_price
            pct_change    = ((current_price - avg_cost) / avg_cost * 100) if avg_cost else 0.0

            holdings.append({
                "ticker":        ticker,
                "name":          name,
                "quantity":      qty,
                "avg_cost":      avg_cost,
                "current_price": round(current_price, 2),
                "equity":        round(equity, 2),
                "pct_change":    round(pct_change, 2),
            })

    return holdings


# ---------------------------------------------------------------------------
# Source: robin_stocks
# ---------------------------------------------------------------------------

def _from_robin_stocks() -> list[dict]:
    """Fetch holdings via the robin_stocks library."""
    import robin_stocks.robinhood as r  # type: ignore
    r.login(_robinhood_user(), _robinhood_pass())
    raw = r.build_holdings()
    holdings = [
        {
            "ticker":        ticker,
            "name":          data.get("name", ticker),
            "quantity":      float(data.get("quantity", 0)),
            "avg_cost":      float(data.get("average_buy_price", 0)),
            "current_price": float(data.get("price", 0)),
            "equity":        float(data.get("equity", 0)),
            "pct_change":    float(data.get("percent_change", 0)),
        }
        for ticker, data in raw.items()
    ]
    r.logout()
    return holdings


# ---------------------------------------------------------------------------
# Source: Custom client (robinhood_client.py in portfolio-advisor project)
# ---------------------------------------------------------------------------

def _from_custom_client() -> list[dict]:
    """Fetch holdings via robinhood_client.py from the portfolio-advisor project."""
    import importlib.util, pathlib

    client_path = pathlib.Path(__file__).parent.parent.parent.parent / "portfolio-advisor" / "robinhood_client.py"
    if not client_path.exists():
        raise FileNotFoundError(f"robinhood_client.py not found at {client_path}")

    spec   = importlib.util.spec_from_file_location("robinhood_client", client_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    client = module.RobinhoodClient(_robinhood_user(), _robinhood_pass())
    try:
        client.login()
        return client.get_holdings()
    finally:
        client.logout()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

DEMO_HOLDINGS = [
    {"ticker": "AAPL", "name": "Apple Inc.",   "quantity": 10, "avg_cost": 170.0, "current_price": 178.5, "equity": 1785.0, "pct_change":  5.0},
    {"ticker": "NVDA", "name": "NVIDIA Corp.", "quantity": 5,  "avg_cost": 450.0, "current_price": 620.0, "equity": 3100.0, "pct_change": 37.8},
    {"ticker": "TSLA", "name": "Tesla Inc.",   "quantity": 8,  "avg_cost": 260.0, "current_price": 195.0, "equity": 1560.0, "pct_change": -25.0},
]

_SOURCES = {
    "snaptrade":    _from_snaptrade,
    "csv":          lambda: _from_csv(HOLDINGS_CSV_PATH),
    "robin_stocks": _from_robin_stocks,
    "custom":       _from_custom_client,
}


def get_holdings(source: str | None = None) -> list[dict]:
    """
    Fetch live Robinhood holdings from the configured source.

    Args:
        source: Override PORTFOLIO_HOLDINGS_SOURCE env var.
                Options: "snaptrade", "csv", "robin_stocks", "custom".

    Returns:
        List of holding dicts with ticker, name, quantity, avg_cost,
        current_price, equity, pct_change. Falls back to demo data on failure.
    """
    src = source or HOLDINGS_SOURCE
    if src not in _SOURCES:
        raise ValueError(
            f"Unknown holdings source '{src}'. "
            f"Choose from: {list(_SOURCES.keys())}"
        )
    try:
        return _SOURCES[src]()
    except Exception as exc:
        # Return demo data so the agent can still respond gracefully
        return {"error": str(exc), "demo_fallback": DEMO_HOLDINGS}
