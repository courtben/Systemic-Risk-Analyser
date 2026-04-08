"""
Data fetching and caching for the Systemic Risk Dashboard.

Covers major US banking institutions.
Prices sourced from Yahoo Finance via yfinance.
Market benchmark: S&P 500 (^GSPC).
Rolling window: 5 years of daily data.
"""
from __future__ import annotations

import os
import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────────

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

BANKS_BY_COUNTRY: dict[str, dict[str, str]] = {
    "US": {
        "JPM": "JPMorgan Chase",
        "BAC": "Bank of America",
        "GS":  "Goldman Sachs",
        "WFC": "Wells Fargo",
        "C":   "Citigroup",
        "MS":  "Morgan Stanley",
    },
}

# Flat lookup: ticker → display name
ALL_BANKS: dict[str, str] = {
    ticker: name
    for banks in BANKS_BY_COUNTRY.values()
    for ticker, name in banks.items()
}

BANK_COLORS: dict[str, str] = {
    "JPM": "#0d47a1",
    "BAC": "#c62828",
    "GS":  "#37474f",
    "WFC": "#f57f17",
    "C":   "#00838f",
    "MS":  "#33691e",
}

BANK_COUNTRY: dict[str, str] = {
    ticker: country
    for country, banks in BANKS_BY_COUNTRY.items()
    for ticker in banks
}

# Primary market benchmark
MARKET_TICKER = "^GSPC"
MARKET_NAME   = "S&P 500"

# Rolling 5-year window — recomputed fresh each call so the window moves
LOOKBACK_YEARS    = 5
PRICE_CACHE_HOURS = 12
BS_CACHE_DAYS     = 7


def _default_start() -> str:
    """Start date = Jan 1 of (current_year - LOOKBACK_YEARS), giving 5 full years + YTD."""
    return f"{datetime.today().year - LOOKBACK_YEARS}-01-01"


# ── Prices ─────────────────────────────────────────────────────────────────────

def _download_prices(start: str, end: str) -> pd.DataFrame:
    all_tickers = list(ALL_BANKS.keys()) + [MARKET_TICKER]
    print(f"  Downloading prices {start} to {end} for {len(ALL_BANKS)} US banks ...")
    raw = yf.download(
        all_tickers, start=start, end=end,
        auto_adjust=True, progress=False,
    )
    if raw.empty:
        raise RuntimeError("yfinance returned no data")

    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    prices = prices.rename(columns={MARKET_TICKER: MARKET_NAME})
    prices = prices.dropna(axis=1, how="all")
    prices = prices.ffill(limit=5)

    bank_cols = [c for c in prices.columns if c in ALL_BANKS]
    print(f"  Got {len(prices)} rows, {len(bank_cols)}/{len(ALL_BANKS)} banks")
    return prices


def get_prices(force_refresh: bool = False) -> pd.DataFrame:
    """
    Return 5-year daily price history for all US banks + S&P 500.

    Cache is invalidated after PRICE_CACHE_HOURS hours or when the
    cached start date is more than one month older than the rolling window.
    """
    start = _default_start()
    end   = datetime.today().strftime("%Y-%m-%d")
    path  = os.path.join(CACHE_DIR, "prices.parquet")

    if not force_refresh and os.path.exists(path):
        age_h = (datetime.now().timestamp() - os.path.getmtime(path)) / 3600
        if age_h < PRICE_CACHE_HOURS:
            df = pd.read_parquet(path)
            # Trim to the rolling 5-year window in case cache is older
            df = df.loc[df.index >= start]
            print(f"  Loaded prices from cache ({age_h:.1f}h old, {len(df)} rows)")
            return df

    df = _download_prices(start, end)
    df.to_parquet(path)
    return df


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Log daily returns: ln(P_t / P_{t-1}).  Consistent with Belluzzo (2020)."""
    return np.log(prices).diff().iloc[1:]


# ── Balance Sheet (for market cap and SRISK) ───────────────────────────────────

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
            "total_liabilities":  total_liabilities,
            "shares_outstanding": info.get("sharesOutstanding"),
            "market_cap":         info.get("marketCap"),
            "currency":           info.get("currency", "USD"),
        }
    except Exception as exc:
        print(f"    Warning – {name}: {exc}")
        return {
            "name": name,
            "total_liabilities": None, "shares_outstanding": None,
            "market_cap": None, "currency": "USD",
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


# ── Liabilities time series (quarterly, for SES / SRISK) ─────────────────────

def _fetch_liab_series(t: yf.Ticker) -> pd.Series | None:
    """Extract total-liabilities quarterly series from a yfinance Ticker."""
    LIAB_KEYS = [
        "Total Liabilities Net Minority Interest",
        "Total Liab",
        "Total Liabilities",
    ]
    parts: list[pd.Series] = []
    for bs in (t.quarterly_balance_sheet, t.balance_sheet):
        if bs is None or bs.empty:
            continue
        found = False
        for key in LIAB_KEYS:
            if key in bs.index:
                parts.append(bs.loc[key].dropna())
                found = True
                break
        if not found:
            assets_s = equity_s = None
            if "Total Assets" in bs.index:
                assets_s = bs.loc["Total Assets"].dropna()
            for ek in ("Stockholders Equity",
                       "Total Stockholder Equity",
                       "Common Stock Equity"):
                if ek in bs.index:
                    equity_s = bs.loc[ek].dropna()
                    break
            if assets_s is not None and equity_s is not None:
                idx_ = assets_s.index.union(equity_s.index)
                diff = (assets_s.reindex(idx_) - equity_s.reindex(idx_)).dropna()
                if not diff.empty:
                    parts.append(diff)

    if not parts:
        return None
    merged = pd.concat(parts)
    merged.index = pd.to_datetime(merged.index)
    merged = merged.sort_index()
    merged = merged[~merged.index.duplicated(keep="first")]
    # Keep only the 5-year window
    cutoff = pd.Timestamp(_default_start())
    merged = merged[merged.index >= cutoff - pd.DateOffset(years=1)]  # 1 extra year for bfill
    return merged if not merged.empty else None


def get_liabilities_ts(force_refresh: bool = False) -> pd.DataFrame:
    """
    Quarterly total-liabilities time series for all US banks (5-year window).

    Fetches quarterly + annual yfinance balance sheet history (~4–5 years).
    Cached as cache/liabilities_ts.parquet; refreshed every BS_CACHE_DAYS days.
    """
    path = os.path.join(CACHE_DIR, "liabilities_ts.parquet")
    if not force_refresh and os.path.exists(path):
        age_d = (datetime.now().timestamp() - os.path.getmtime(path)) / 86400
        if age_d < BS_CACHE_DAYS:
            df = pd.read_parquet(path)
            print(f"  Loaded liabilities time series from cache ({age_d:.1f}d old)")
            return df

    print("  Fetching quarterly liabilities history ...")
    records: dict[str, pd.Series] = {}
    for ticker, name in ALL_BANKS.items():
        try:
            s = _fetch_liab_series(yf.Ticker(ticker))
            if s is not None:
                records[ticker] = s
        except Exception as exc:
            print(f"    Warning – {name}: {exc}")

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records).sort_index()
    df.to_parquet(path)
    return df


def build_lb_daily(
    liab_ts: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """
    Align quarterly liabilities to the daily price index.

    Forward-fill between reporting dates; back-fill before first filing.
    """
    daily_idx = prices.index
    result: dict[str, pd.Series] = {}
    for ticker in liab_ts.columns:
        if ticker not in prices.columns:
            continue
        s = liab_ts[ticker].dropna()
        if s.empty:
            continue
        combined = s.reindex(s.index.union(daily_idx)).sort_index()
        combined = combined.ffill().bfill()
        result[ticker] = combined.reindex(daily_idx)
    return pd.DataFrame(result)


def build_lbr_daily(
    liab_ts: pd.DataFrame,
    prices: pd.DataFrame,
    fr: int = 3,
) -> pd.DataFrame:
    """
    Forward-rolled liabilities: sample every fr months, hold constant between.

    Follows forward_roll_data.m from TommasoBelluzzo/SystemicRisk.
    """
    daily_idx = prices.index
    result: dict[str, pd.Series] = {}
    for ticker in liab_ts.columns:
        if ticker not in prices.columns:
            continue
        s = liab_ts[ticker].dropna()
        if s.empty:
            continue
        monthly  = s.resample("MS").first()
        rolled   = monthly.iloc[::fr]
        combined = rolled.reindex(rolled.index.union(daily_idx)).sort_index()
        combined = combined.ffill().bfill()
        result[ticker] = combined.reindex(daily_idx)
    return pd.DataFrame(result)


def fetch_single_bank(ticker: str) -> dict | None:
    """
    Fetch all data needed to add one custom bank to the model.

    Downloads the same 5-year price window as the base banks.
    Returns a dict or None if the ticker is invalid / data unavailable.
    """
    start = _default_start()
    end   = datetime.today().strftime("%Y-%m-%d")
    try:
        t    = yf.Ticker(ticker)
        info = t.info or {}
        if not info.get("regularMarketPrice") and not info.get("currentPrice"):
            return None
        name = info.get("shortName") or info.get("longName") or ticker

        raw = yf.download(ticker, start=start, end=end,
                          auto_adjust=True, progress=False)
        if raw.empty:
            return None
        prices = raw["Close"]
        if isinstance(prices, pd.DataFrame):
            prices = prices.iloc[:, 0]
        prices.name = ticker

        bs_data = _fetch_balance_sheet_one(ticker, name)
        liab_ts = _fetch_liab_series(t)

        return {
            "ticker":        ticker,
            "name":          name,
            "prices":        prices,
            "balance_sheet": bs_data,
            "liab_ts":       liab_ts,
        }

    except Exception as exc:
        print(f"  fetch_single_bank({ticker}) failed: {exc}")
        return None


# ── Market-cap time series (price × shares outstanding) ───────────────────────

def build_market_cap_series(
    prices: pd.DataFrame,
    balance_sheet: dict,
) -> pd.DataFrame:
    """
    Approximate daily market cap = price × shares_outstanding.
    Constant shares outstanding is a standard simplification.
    """
    caps = {}
    for ticker in list(ALL_BANKS.keys()) + [
        t for t in balance_sheet if t not in ALL_BANKS
    ]:
        if ticker not in prices.columns:
            continue
        shares = balance_sheet.get(ticker, {}).get("shares_outstanding")
        if shares:
            caps[ticker] = prices[ticker] * shares
    return pd.DataFrame(caps)
