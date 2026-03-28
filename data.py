"""
Data fetching and caching for the Swiss Banking Systemic Risk Dashboard.

Sources price data from Yahoo Finance via yfinance for Swiss banks and
the Swiss Market Index (SMI). Balance sheet data is used for SRISK.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────────

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

SWISS_BANKS: dict[str, str] = {
    "UBSG.SW": "UBS Group",
    "CSGN.SW": "Credit Suisse",   # Historical until June 2023
    "BAER.SW": "Julius Bär",
    "SLHN.SW": "Swiss Life",
    "VONN.SW": "Vontobel",
    "EFGN.SW": "EFG International",
    "BCVN.SW": "BCV",
    "ZURN.SW": "Zurich Insurance",
}

BANK_COLORS: dict[str, str] = {
    "UBSG.SW": "#1f77b4",
    "CSGN.SW": "#d62728",
    "BAER.SW": "#2ca02c",
    "SLHN.SW": "#ff7f0e",
    "VONN.SW": "#9467bd",
    "EFGN.SW": "#8c564b",
    "BCVN.SW": "#e377c2",
    "ZURN.SW": "#17becf",
}

MARKET_TICKER = "^SSMI"
MARKET_NAME = "SMI"
DEFAULT_START = "2010-01-01"

# Cache expiry (hours for prices, days for balance sheet)
PRICE_CACHE_HOURS = 12
BS_CACHE_DAYS = 7


# ── Prices ─────────────────────────────────────────────────────────────────────

def _download_prices(start: str, end: str) -> pd.DataFrame:
    tickers = list(SWISS_BANKS.keys()) + [MARKET_TICKER]
    print(f"  Downloading prices {start} → {end} ...")
    raw = yf.download(
        tickers, start=start, end=end,
        auto_adjust=True, progress=False, threads=True,
    )
    if raw.empty:
        raise RuntimeError("yfinance returned no data")

    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    prices = prices.rename(columns={MARKET_TICKER: MARKET_NAME})
    prices = prices.dropna(axis=1, how="all")

    # Forward-fill up to 5 days (handles weekends / public holidays)
    prices = prices.ffill(limit=5)

    actual = [c for c in prices.columns if c != MARKET_NAME]
    print(f"  Got {len(prices)} rows, {len(actual)} banks + market")
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
    """Simple daily percentage returns, dropping the first NaN row."""
    return prices.pct_change().iloc[1:]


# ── Balance Sheet (for SRISK) ──────────────────────────────────────────────────

def _fetch_balance_sheet_one(ticker: str, name: str) -> dict:
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        bs = t.quarterly_balance_sheet

        total_liabilities = None
        if bs is not None and not bs.empty:
            # Try direct liability key first
            for key in [
                "Total Liabilities Net Minority Interest",
                "Total Liab",
                "Total Liabilities",
            ]:
                if key in bs.index:
                    val = bs.loc[key].dropna()
                    if not val.empty:
                        total_liabilities = float(val.iloc[0])
                        break

            # Fallback: assets − equity
            if total_liabilities is None:
                assets, equity = None, None
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
            "name": name,
            "total_liabilities": total_liabilities,
            "shares_outstanding": info.get("sharesOutstanding"),
            "market_cap": info.get("marketCap"),
            "currency": info.get("currency", "CHF"),
        }
    except Exception as exc:
        print(f"    Warning – {name}: {exc}")
        return {
            "name": name,
            "total_liabilities": None,
            "shares_outstanding": None,
            "market_cap": None,
            "currency": "CHF",
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
    for ticker, name in SWISS_BANKS.items():
        print(f"    {name} ({ticker})")
        data[ticker] = _fetch_balance_sheet_one(ticker, name)

    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return data


# ── Market-cap time series (used for rolling SRISK) ───────────────────────────

def build_market_cap_series(
    prices: pd.DataFrame,
    balance_sheet: dict,
) -> pd.DataFrame:
    """
    Approximate daily market cap = price × shares_outstanding.
    Uses constant shares outstanding from the most recent yfinance info,
    which is a standard simplification for rolling SRISK.
    """
    caps = {}
    for ticker in SWISS_BANKS:
        if ticker not in prices.columns:
            continue
        shares = balance_sheet.get(ticker, {}).get("shares_outstanding")
        if shares:
            caps[ticker] = prices[ticker] * shares
    return pd.DataFrame(caps)
