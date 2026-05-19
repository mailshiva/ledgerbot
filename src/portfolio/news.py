"""
src/portfolio/news.py
=====================
Fetches news headlines for a stock and its industry via NewsAPI.

Adapted from portfolio-advisor/advisor/news.py — no cross-project imports.
Credentials (NEWS_API_KEY) are read from the macOS Keychain service
"portfolio_advisor" or the env var NEWS_API_KEY.

Requires: pip install newsapi-python

Functions:
  get_company_news(ticker, company_name, max_headlines=3)
      → list of recent headline strings for the company

  get_industry_news(industry, sector, max_headlines=3)
      → list of recent headline strings for the industry/sector

  get_ticker_info(ticker)
      → dict with company_name, industry, sector (from yfinance)
"""

from __future__ import annotations

import os
import subprocess
import sys


# ---------------------------------------------------------------------------
# Secret resolution
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
# Ticker metadata from yfinance
# ---------------------------------------------------------------------------

def get_ticker_info(ticker: str) -> dict:
    """
    Fetch company name, industry, and sector from yfinance.

    Returns:
        dict with keys: company_name, industry, sector
        Falls back to ticker string for missing fields.
    """
    try:
        import yfinance as yf  # type: ignore
        info = yf.Ticker(ticker).info
        return {
            "company_name": info.get("longName") or info.get("shortName") or ticker,
            "industry":     info.get("industry") or "Unknown Industry",
            "sector":       info.get("sector")   or "Unknown Sector",
        }
    except Exception:
        return {
            "company_name": ticker,
            "industry":     "Unknown Industry",
            "sector":       "Unknown Sector",
        }


# ---------------------------------------------------------------------------
# News fetchers
# ---------------------------------------------------------------------------

def get_company_news(
    ticker: str,
    company_name: str,
    max_headlines: int = 3,
) -> list[str]:
    """
    Fetch the latest news headlines for a specific company.

    Queries NewsAPI using the company name (or ticker if name is long).
    Falls back to a placeholder string on error.

    Args:
        ticker:        Stock symbol, e.g. "AAPL"
        company_name:  Full company name, e.g. "Apple Inc."
        max_headlines: Maximum headlines to return (default 3)

    Returns:
        List of headline strings.
    """
    try:
        from newsapi import NewsApiClient  # type: ignore
        api_key = _get_secret("news_api_key", "NEWS_API_KEY")
        api     = NewsApiClient(api_key=api_key)

        # Use ticker when name is long to avoid noisy results
        query = f"{company_name} stock" if len(company_name) < 25 else ticker
        result = api.get_everything(
            q=query,
            language="en",
            sort_by="publishedAt",
            page_size=max_headlines,
        )
        headlines = [
            a["title"] for a in result.get("articles", []) if a.get("title")
        ]
        return headlines[:max_headlines] or [f"No recent news found for {ticker}"]
    except Exception as exc:
        return [f"News unavailable for {ticker}: {exc}"]


def get_industry_news(
    industry: str,
    sector: str,
    max_headlines: int = 3,
) -> list[str]:
    """
    Fetch the latest news headlines for an industry / sector.

    Uses the more specific 'industry' string first; if it's generic,
    falls back to 'sector'.

    Args:
        industry:      e.g. "Semiconductors", "Consumer Electronics"
        sector:        e.g. "Technology", "Healthcare"
        max_headlines: Maximum headlines to return (default 3)

    Returns:
        List of headline strings.
    """
    try:
        from newsapi import NewsApiClient  # type: ignore
        api_key = _get_secret("news_api_key", "NEWS_API_KEY")
        api     = NewsApiClient(api_key=api_key)

        # Prefer the specific industry label; fall back to sector
        query_term = industry if industry not in ("Unknown Industry", "") else sector
        query = f"{query_term} industry market"

        result = api.get_everything(
            q=query,
            language="en",
            sort_by="publishedAt",
            page_size=max_headlines,
        )
        headlines = [
            a["title"] for a in result.get("articles", []) if a.get("title")
        ]
        return headlines[:max_headlines] or [f"No recent industry news found for {query_term}"]
    except Exception as exc:
        return [f"Industry news unavailable: {exc}"]
