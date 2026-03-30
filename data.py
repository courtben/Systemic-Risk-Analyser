"""
Data fetching and caching for the Systemic Risk Dashboard.

Covers Swiss (CH), US, and UK banking institutions.
Prices sourced from Yahoo Finance via yfinance.
Market benchmark: S&P 500 (^GSPC) — used as global proxy for cross-country comparison.
"""
from __future__ import annotations

import os
import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────────

# Use an absolute path so the cache works regardless of the working directory
# (cloud runners often set cwd to something other than the project root).
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

BANKS_BY_COUNTRY: dict[str, dict[str, str]] = {
    "CH": {
        "UBSG.SW": "UBS Group",
        "CSGN.SW": "Credit Suisse",   # Historical until June 2023
        "BAER.SW": "Julius Bär",
        "VONN.SW": "Vontobel",
        "EFGN.SW": "EFG International",
        "BCVN.SW": "BCV",
    },
    "US": {
        "JPM": "JPMorgan Chase",
        "BAC": "Bank of America",
        "GS":  "Goldman Sachs",
        "WFC": "Wells Fargo",
        "C":   "Citigroup",
        "MS":  "Morgan Stanley",
    },
    "UK": {
        "HSBA.L": "HSBC",
        "BARC.L": "Barclays",
        "LLOY.L": "Lloyds Banking Group",
        "NWG.L":  "NatWest Group",
        "STAN.L": "Standard Chartered",
    },
}

COUNTRY_LABELS: dict[str, str] = {
    "CH": "Switzerland",
    "US": "United States",
    "UK": "United Kingdom",
}

# Flat lookup: ticker → display name
ALL_BANKS: dict[str, str] = {
    ticker: name
    for banks in BANKS_BY_COUNTRY.values()
    for ticker, name in banks.items()
}

# Reverse lookup: ticker → country code
BANK_COUNTRY: dict[str, str] = {
    ticker: country
    for country, banks in BANKS_BY_COUNTRY.items()
    for ticker in banks
}

BANK_COLORS: dict[str, str] = {
    # Switzerland — blues / greens / earth tones
    "UBSG.SW": "#1565c0",
    "CSGN.SW": "#b71c1c",
    "BAER.SW": "#2e7d32",
    "VONN.SW": "#6a1b9a",
    "EFGN.SW": "#4e342e",
    "BCVN.SW": "#ad1457",
    # United States — bold primary tones
    "JPM":     "#0d47a1",
    "BAC":     "#c62828",
    "GS":      "#37474f",
    "WFC":     "#f57f17",
    "C":       "#00838f",
    "MS":      "#33691e",
    # United Kingdom — purples / distinct blues
    "HSBA.L":  "#d32f2f",
    "BARC.L":  "#0277bd",
    "LLOY.L":  "#558b2f",
    "NWG.L":   "#7b1fa2",
    "STAN.L":  "#0097a7",
}

# Primary market benchmark (S&P 500 used as global proxy)
MARKET_TICKER = "^GSPC"
MARKET_NAME   = "S&P 500"

# Additional indices downloaded for the Market Data tab
EXTRA_INDICES: dict[str, str] = {
    "^SSMI": "SMI (CH)",
    "^FTSE": "FTSE 100 (UK)",
}

DEFAULT_START = "2010-01-01"
PRICE_CACHE_HOURS = 12
BS_CACHE_DAYS     = 7


# ── Prices ─────────────────────────────────────────────────────────────────────

def _download_prices(start: str, end: str) -> pd.DataFrame:
    all_tickers = list(ALL_BANKS.keys()) + [MARKET_TICKER] + list(EXTRA_INDICES.keys())
    print(f"  Downloading prices {start} → {end} for {len(ALL_BANKS)} banks ...")
    raw = yf.download(
        all_tickers, start=start, end=end,
        auto_adjust=True, progress=False,
    )
    if raw.empty:
        raise RuntimeError("yfinance returned no data")

    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw

    # Rename index tickers to friendly names
    rename = {MARKET_TICKER: MARKET_NAME}
    rename.update(EXTRA_INDICES)
    prices = prices.rename(columns=rename)
    prices = prices.dropna(axis=1, how="all")
    prices = prices.ffill(limit=5)

    bank_cols = [c for c in prices.columns if c in ALL_BANKS]
    print(f"  Got {len(prices)} rows, {len(bank_cols)}/{len(ALL_BANKS)} banks")
    return prices


def get_prices(
    start: str = DEFAULT_START,
    end: str | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")

    path = os.path.join(CACHE_DIR, "prices.parquet")
    if not force_refresh and os.path.exists(path):
        age_h = (datetime.now().timestamp() - os.path.getmtime(path)) / 3600
        if age_h < PRICE_CACHE_HOURS:
            df = pd.read_parquet(path)
            print(f"  Loaded prices from cache ({age_h:.1f}h old)")
            return df

    df = _download_prices(start, end)
    df.to_parquet(path)
    return df


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily percentage returns."""
    return prices.pct_change().iloc[1:]


# ── Balance Sheet (for SRISK) ──────────────────────────────────────────────────

def _fetch_balance_sheet_one(ticker: str, name: str) -> dict:
    try:
        t    = yf.Ticker(ticker)
        info = t.info or {}
        bs   = t.quarterly_balance_sheet

        total_liabilities = None
        if bs is not None and not bs.empty:
            for key in ["Total Liabilities Net Minority Interest",
                        "Total Liab", "Total Liabilities"]:
                if key in bs.index:
                    val = bs.loc[key].dropna()
                    if not val.empty:
                        total_liabilities = float(val.iloc[0])
                        break

            if total_liabilities is None:
                assets = equity = None
                if "Total Assets" in bs.index:
                    v = bs.loc["Total Assets"].dropna()
                    if not v.empty:
                        assets = float(v.iloc[0])
                for ek in ["Stockholders Equity", "Total Stockholder Equity",
                           "Common Stock Equity"]:
                    if ek in bs.index:
                        v = bs.loc[ek].dropna()
                        if not v.empty:
                            equity = float(v.iloc[0])
                            break
                if assets and equity:
                    total_liabilities = assets - equity

        return {
            "name":               name,
            "country":            BANK_COUNTRY.get(ticker, ""),
            "total_liabilities":  total_liabilities,
            "shares_outstanding": info.get("sharesOutstanding"),
            "market_cap":         info.get("marketCap"),
            "currency":           info.get("currency", ""),
        }
    except Exception as exc:
        print(f"    Warning – {name}: {exc}")
        return {
            "name": name, "country": BANK_COUNTRY.get(ticker, ""),
            "total_liabilities": None, "shares_outstanding": None,
            "market_cap": None, "currency": "",
        }


def get_balance_sheet(force_refresh: bool = False) -> dict:
    path = os.path.join(CACHE_DIR, "balance_sheet.json")
    if not force_refresh and os.path.exists(path):
        age_d = (datetime.now().timestamp() - os.path.getmtime(path)) / 86400
        if age_d < BS_CACHE_DAYS:
            with open(path) as f:
                data = json.load(f)
            print(f"  Loaded balance sheet from cache ({age_d:.1f}d old)")
            return data

    print("  Fetching balance sheet data ...")
    data = {}
    for ticker, name in ALL_BANKS.items():
        print(f"    {name} ({ticker})")
        data[ticker] = _fetch_balance_sheet_one(ticker, name)

    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return data


# ── Market-cap time series (for rolling SRISK) ────────────────────────────────

def build_market_cap_series(
    prices: pd.DataFrame,
    balance_sheet: dict,
) -> pd.DataFrame:
    """
    Approximate daily market cap = price × shares_outstanding.
    Constant shares outstanding is a standard simplification.
    """
    caps = {}
    for ticker in ALL_BANKS:
        if ticker not in prices.columns:
            continue
        shares = balance_sheet.get(ticker, {}).get("shares_outstanding")
        if shares:
            caps[ticker] = prices[ticker] * shares
    return pd.DataFrame(caps)
