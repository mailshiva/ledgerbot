"""
src/portfolio/macro.py
======================
Fetches macro-economic indicators used for portfolio recommendations.

Adapted from portfolio-advisor/advisor/macro.py — no cross-project imports.
Credentials (FRED API key) are read from the same macOS Keychain service
"portfolio_advisor" that the portfolio-advisor project already populated.

Indicators returned:
  vix             — CBOE Volatility Index (market fear gauge)
  treasury_10y    — 10-Year US Treasury yield (%)
  sp500_5d_change — S&P 500 5-day price change (%)
  fed_funds_rate  — Fed Funds Rate, latest (via FRED, optional)
  cpi_yoy         — CPI year-over-year change (%), latest (via FRED, optional)

yfinance is required: pip install yfinance
fredapi is optional (FRED indicators skipped if not installed):
  pip install fredapi
"""

from __future__ import annotations

import subprocess
import sys
import os


# ---------------------------------------------------------------------------
# Secret resolution (same helper pattern as holdings.py)
# ---------------------------------------------------------------------------

_KEYCHAIN_SERVICE = "portfolio_advisor"


def _get_secret(account: str, env_var: str) -> str:
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
        f"Missing secret '{account}'. "
        f"Add to Keychain: security add-generic-password "
        f"-a \"{account}\" -s \"{_KEYCHAIN_SERVICE}\" -w \"VALUE\"\n"
        f"Or set env var: export {env_var}=\"VALUE\""
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def get_macro_context() -> dict:
    """
    Fetch current macro-economic indicators.

    All fields fall back to "N/A" on error so callers are never blocked.

    Returns:
        dict with keys: vix, treasury_10y, sp500_5d_change,
                        fed_funds_rate, cpi_yoy
    """
    try:
        import yfinance as yf  # type: ignore
    except ImportError:
        return {
            "vix": "N/A",
            "treasury_10y": "N/A",
            "sp500_5d_change": "N/A",
            "fed_funds_rate": "N/A",
            "cpi_yoy": "N/A",
            "error": "yfinance not installed (pip install yfinance)",
        }

    context: dict = {}

    try:
        vix = yf.Ticker("^VIX").fast_info
        context["vix"] = round(
            float(vix.get("lastPrice", 0) or vix.get("last_price", 0)), 2
        )
    except Exception:
        context["vix"] = "N/A"

    try:
        tnx = yf.Ticker("^TNX").fast_info
        context["treasury_10y"] = round(
            float(tnx.get("lastPrice", 0) or tnx.get("last_price", 0)), 2
        )
    except Exception:
        context["treasury_10y"] = "N/A"

    try:
        spy = yf.Ticker("SPY").history(period="5d")
        week_change = (
            (spy["Close"].iloc[-1] - spy["Close"].iloc[0]) / spy["Close"].iloc[0]
        ) * 100
        context["sp500_5d_change"] = round(week_change, 2)
    except Exception:
        context["sp500_5d_change"] = "N/A"

    try:
        from fredapi import Fred  # type: ignore
        fred_key = _get_secret("fred_api_key", "FRED_API_KEY")
        fred = Fred(api_key=fred_key)
        context["fed_funds_rate"] = round(
            float(fred.get_series("FEDFUNDS").iloc[-1]), 2
        )
        context["cpi_yoy"] = round(
            float(fred.get_series("CPIAUCSL").pct_change(12).iloc[-1] * 100), 2
        )
    except Exception:
        context["fed_funds_rate"] = "N/A"
        context["cpi_yoy"] = "N/A"

    return context
