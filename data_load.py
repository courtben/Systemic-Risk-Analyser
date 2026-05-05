"""
Data fetching and caching for the Systemic Risk Dashboard.

Covers major US banking institutions — banks only
(no insurers, asset managers, or non-BHC broker-dealers).
Prices sourced from Yahoo Finance via yfinance.
Market benchmark: S&P 500 (^GSPC).
Rolling window: 5 years of daily data.
"""
from __future__ import annotations

import os
import json
import warnings
from io import StringIO
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────────

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Universe: top-~50 US bank / savings & loan holding companies by total
# assets (S&P Global Market Intelligence Q4 2025 ranking).
#
# Includes the publicly listed BHCs/SLHCs whose stock reflects US bank
# operations.  Explicitly excluded:
#   • Foreign-parent ADRs (TD, HSBC, BMO, SAN, UBS, BCS, RY, DB, MFG, CM,
#     BNP) whose listed stock reflects global group operations rather than
#     US bank operations — including them would distort the US systemic-
#     risk signal.
#   • USAA — privately held, no public ticker.
BANKS_BY_COUNTRY: dict[str, dict[str, str]] = {
    "US": {
        # Money-center / Big 4
        "JPM":   "JPMorgan Chase",
        "BAC":   "Bank of America",
        "C":     "Citigroup",
        "WFC":   "Wells Fargo",
        # Investment banks (BHCs since 2008)
        "GS":    "Goldman Sachs",
        "MS":    "Morgan Stanley",
        # Super-regional / national
        "USB":   "U.S. Bancorp",
        "PNC":   "PNC Financial",
        "TFC":   "Truist Financial",
        "COF":   "Capital One Financial",
        # Trust & custody banks
        "BK":    "BNY Mellon",
        "STT":   "State Street",
        "NTRS": "Northern Trust",
        # Brokerage / wealth-management BHCs (SLHC charter)
        "SCHW":  "Charles Schwab",
        "AMP":   "Ameriprise Financial",
        "RJF":   "Raymond James Financial",
        # Consumer-finance / payments (BHC charter)
        "AXP":   "American Express",
        "ALLY":  "Ally Financial",
        "SYF":   "Synchrony Financial",
        # Large regionals
        "FCNCA": "First Citizens BancShares",
        "FITB":  "Fifth Third Bancorp",
        "MTB":   "M&T Bank",
        "KEY":   "KeyCorp",
        "CFG":   "Citizens Financial",
        "RF":    "Regions Financial",
        "HBAN":  "Huntington Bancshares",
        "WAL":   "Western Alliance",
        # Mid-cap regionals
        "WBS":   "Webster Financial",
        "FHN":   "First Horizon",
        "EWBC":  "East West Bancorp",
        # Comerica (CMA) was acquired by Fifth Third and delisted from NYSE
        # on 2026-02-02 — its assets now show up under FITB.
        "BPOP":  "Popular Inc.",
        "UMBF":  "UMB Financial",
        "ONB":   "Old National Bancorp",
        "WTFC":  "Wintrust Financial",
        "SSB":   "SouthState",
        "COLB":  "Columbia Banking System",
        "VLY":   "Valley National Bancorp",
        "ZION":  "Zions Bancorporation",
    },
}

# Flat lookup: ticker → display name
ALL_BANKS: dict[str, str] = {
    ticker: name
    for banks in BANKS_BY_COUNTRY.values()
    for ticker, name in banks.items()
}

# Distinct colour per ticker, drawn from a modern Tailwind-inspired palette
# (500-tone primary cycle, then 700-tone darker cycle, then 400-tone lighter
# cycle). 42 entries comfortably cover the ~39-bank universe with no
# adjacent repeats and stay readable against a white canvas.
_BANK_COLOR_PALETTE: list[str] = [
    # Cycle 1 — vivid mid-tones (Tailwind 500)
    "#EF4444", "#22C55E", "#F59E0B", "#8B5CF6",
    "#EC4899", "#14B8A6", "#F97316", "#06B6D4",
    "#A855F7", "#84CC16", "#D946EF", "#6366F1",
    "#0EA5E9", "#EAB308", "#10B981", "#3B82F6",
    # Cycle 2 — deeper tones (Tailwind 700)
    "#B91C1C", "#15803D", "#B45309", "#6D28D9",
    "#BE185D", "#0F766E", "#C2410C", "#0E7490",
    "#7E22CE", "#4D7C0F", "#A21CAF", "#4338CA",
    "#0369A1", "#A16207", "#047857", "#1D4ED8",
    # Cycle 3 — softer tones (Tailwind 400)
    "#F87171", "#34D399", "#FBBF24", "#A78BFA",
    "#F472B6", "#2DD4BF", "#FB923C", "#22D3EE",
    "#C084FC", "#60A5FA",
]

BANK_COLORS: dict[str, str] = {
    ticker: _BANK_COLOR_PALETTE[i % len(_BANK_COLOR_PALETTE)]
    for i, ticker in enumerate(ALL_BANKS)
}

BANK_COUNTRY: dict[str, str] = {
    ticker: country
    for country, banks in BANKS_BY_COUNTRY.items()
    for ticker in banks
}

# Human-readable country / region labels (shown in the UI)
COUNTRY_LABELS: dict[str, str] = {
    "US": "United States",
}

# Primary market benchmark
MARKET_TICKER = "^GSPC"
MARKET_NAME   = "S&P 500"

# Rolling 5-year window — recomputed fresh each call so the window moves
LOOKBACK_YEARS    = 5
PRICE_CACHE_HOURS = 12
BS_CACHE_DAYS     = 7

SEPARATE_ACCOUNT_FACTOR = 0.40
STATE_CACHE_DAYS = 7

# BAA10YM (Moody's BAA minus 10-Year Treasury) has no yfinance equivalent — FRED only
FRED_SERIES: dict[str, str] = {
    "credit_spread": "BAA10YM",
}

# Rate / yield tickers available directly from Yahoo Finance (daily, no API key)
# ZQ=F: 30-Day Fed Funds futures — implied rate = 100 − price
YF_RATE_TICKERS: dict[str, str] = {
    "ffr":       "ZQ=F",   # 30-Day Fed Funds Futures → 100 − price ≈ effective FFR
    "yield_10y": "^TNX",   # 10-Year Treasury Note Yield (%)
    "yield_3m":  "^IRX",   # 13-Week T-Bill Rate (%) — proxy for 3-month yield
    "vix":       "^VIX",   # CBOE Volatility Index
}


def _default_start() -> str:
    """Start date = Jan 1 of (current_year - LOOKBACK_YEARS), giving 5 full years + YTD."""
    return f"{datetime.today().year - LOOKBACK_YEARS}-01-01"


# ── Prices ─────────────────────────────────────────────────────────────────────

def _download_prices(start: str, end: str) -> pd.DataFrame:
    all_tickers = list(ALL_BANKS.keys()) + [MARKET_TICKER]
    print(f"  Downloading prices {start} to {end} for {len(ALL_BANKS)} banks ...")
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

        separate_accounts = None
        if bs is not None and not bs.empty:
            for key in [
                "Separate Account Assets",
                "Separate Account Business",
                "Separate Accounts",
            ]:
                if key in bs.index:
                    val = bs.loc[key].dropna()
                    if not val.empty:
                        separate_accounts = float(val.iloc[0])
                        break

        shares_ts = None
        try:
            shares_hist = t.get_shares_full(start=_default_start())
            if shares_hist is not None and len(shares_hist) > 0:
                shares_hist = shares_hist.dropna()
                if len(shares_hist) > 0:
                    shares_hist.index = pd.to_datetime(shares_hist.index)
                    shares_hist = shares_hist.sort_index()
                    shares_hist = shares_hist[~shares_hist.index.duplicated(keep="last")]
                    shares_ts = {
                        str(k): float(v)
                        for k, v in shares_hist.items()
                        if pd.notna(v)
                    }
        except Exception:
            shares_ts = None

        return {
            "name":               name,
            "total_liabilities":  total_liabilities,
            "separate_accounts":  separate_accounts,
            "shares_outstanding": info.get("sharesOutstanding"),
            "shares_ts":          shares_ts,
            "market_cap":         info.get("marketCap"),
            "currency":           info.get("currency", "USD"),
        }
    except Exception as exc:
        print(f"    Warning – {name}: {exc}")
        return {
            "name": name,
            "total_liabilities": None, "separate_accounts": None,
            "shares_outstanding": None,
            "shares_ts": None,
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


def _fetch_separate_account_series(t: yf.Ticker) -> pd.Series | None:
    """Extract separate-account quarterly series when available."""
    keys = [
        "Separate Account Assets",
        "Separate Account Business",
        "Separate Accounts",
    ]
    parts: list[pd.Series] = []
    for bs in (t.quarterly_balance_sheet, t.balance_sheet):
        if bs is None or bs.empty:
            continue
        for key in keys:
            if key in bs.index:
                parts.append(bs.loc[key].dropna())
                break

    if not parts:
        return None

    merged = pd.concat(parts)
    merged.index = pd.to_datetime(merged.index)
    merged = merged.sort_index()
    merged = merged[~merged.index.duplicated(keep="first")]
    cutoff = pd.Timestamp(_default_start())
    merged = merged[merged.index >= cutoff - pd.DateOffset(years=1)]
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


def get_separate_accounts_ts(force_refresh: bool = False) -> pd.DataFrame:
    """
    Quarterly separate-account history where reported.

    Mostly relevant for insurers; many banks will have no data and are omitted.
    """
    path = os.path.join(CACHE_DIR, "separate_accounts_ts.parquet")
    if not force_refresh and os.path.exists(path):
        age_d = (datetime.now().timestamp() - os.path.getmtime(path)) / 86400
        if age_d < BS_CACHE_DAYS:
            df = pd.read_parquet(path)
            print(f"  Loaded separate-accounts time series from cache ({age_d:.1f}d old)")
            return df

    print("  Fetching separate-account history ...")
    records: dict[str, pd.Series] = {}
    for ticker, name in ALL_BANKS.items():
        try:
            s = _fetch_separate_account_series(yf.Ticker(ticker))
            if s is not None:
                records[ticker] = s
        except Exception as exc:
            print(f"    Warning – {name}: {exc}")

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records).sort_index()
    df.to_parquet(path)
    return df


def _fred_series(code: str, start: str) -> pd.Series:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={code}&cosd={start}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.text))
    # FRED changed column name from "DATE" to "observation_date" — detect dynamically
    cols = df.columns.tolist()
    date_col = next((c for c in cols if "date" in c.lower()), None)
    val_col  = next((c for c in cols if c != date_col), None)
    if date_col is None or val_col is None:
        raise ValueError(f"Unexpected FRED CSV columns: {cols}")
    s = pd.to_numeric(df[val_col], errors="coerce")
    s.index = pd.to_datetime(df[date_col])
    return s.dropna().rename(code)


def get_state_variables(prices: pd.DataFrame, force_refresh: bool = False) -> pd.DataFrame:
    """
    Daily systemic state variables aligned to the price index.

    Rate / yield variables (VIX, 10Y yield, 3M T-bill, FFR) are sourced from
    Yahoo Finance via a single batch download.  Only the BAA credit spread,
    which has no yfinance equivalent, is still fetched from FRED.
    """
    path = os.path.join(CACHE_DIR, "state_variables.parquet")
    if not force_refresh and os.path.exists(path):
        age_d = (datetime.now().timestamp() - os.path.getmtime(path)) / 86400
        if age_d < STATE_CACHE_DAYS:
            df = pd.read_parquet(path)
            df.index = pd.to_datetime(df.index)
            return df.reindex(prices.index).ffill().bfill()

    start = _default_start()
    end   = datetime.today().strftime("%Y-%m-%d")
    state: dict[str, pd.Series] = {}

    # ── yfinance: rates / yields (single batch download) ──────────────────────
    print("  Fetching rate/yield state variables from Yahoo Finance ...")
    try:
        raw = yf.download(
            list(YF_RATE_TICKERS.values()),
            start=start, end=end,
            auto_adjust=True, progress=False,
        )
        close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
        for name, ticker in YF_RATE_TICKERS.items():
            if ticker not in close.columns:
                print(f"  Warning – {ticker} not in download result")
                continue
            s = close[ticker].dropna().astype(float)
            if name == "ffr":
                s = 100.0 - s   # futures price → implied rate (%)
            state[name] = s.rename(name)
    except Exception as exc:
        print(f"  Warning – yfinance rate download failed: {exc}")

    # ── FRED: credit spread only (BAA10YM — no yfinance equivalent) ───────────
    for name, code in FRED_SERIES.items():
        try:
            state[name] = _fred_series(code, start)
        except Exception as exc:
            print(f"  Warning – FRED {code}: {exc}")

    if not state:
        return pd.DataFrame(index=prices.index)

    state_df = pd.DataFrame(state).sort_index()
    if state_df.empty:
        return pd.DataFrame(index=prices.index)

    if {"yield_10y", "yield_3m"} <= set(state_df.columns):
        state_df["yield_spread"] = state_df["yield_10y"] - state_df["yield_3m"]

    if MARKET_NAME in prices.columns:
        market_rets = compute_returns(prices[[MARKET_NAME]])[MARKET_NAME]
        state_df["market_excess_proxy"] = market_rets.reindex(state_df.index).ffill()

    state_df = state_df.reindex(prices.index).ffill().bfill()
    state_df.to_parquet(path)
    return state_df


def build_lb_daily(
    liab_ts: pd.DataFrame,
    prices: pd.DataFrame,
    separate_accounts_ts: pd.DataFrame | None = None,
    sf: float = SEPARATE_ACCOUNT_FACTOR,
) -> pd.DataFrame:
    """
    Align quarterly liabilities to the daily price index.

    Forward-fill between reporting dates; back-fill before first filing.
    """
    daily_idx = prices.index
    sa_daily: dict[str, pd.Series] = {}
    if separate_accounts_ts is not None and not separate_accounts_ts.empty:
        for ticker in separate_accounts_ts.columns:
            if ticker not in prices.columns:
                continue
            s = separate_accounts_ts[ticker].dropna()
            if s.empty:
                continue
            combined = s.reindex(s.index.union(daily_idx)).sort_index()
            combined = combined.ffill().bfill()
            sa_daily[ticker] = combined.reindex(daily_idx)

    result: dict[str, pd.Series] = {}
    for ticker in liab_ts.columns:
        if ticker not in prices.columns:
            continue
        s = liab_ts[ticker].dropna()
        if s.empty:
            continue
        combined = s.reindex(s.index.union(daily_idx)).sort_index()
        combined = combined.ffill().bfill()
        series = combined.reindex(daily_idx)
        if ticker in sa_daily:
            series = series - ((1.0 - sf) * sa_daily[ticker])
        result[ticker] = series
    return pd.DataFrame(result)


def build_lbr_daily(
    liab_ts: pd.DataFrame,
    prices: pd.DataFrame,
    separate_accounts_ts: pd.DataFrame | None = None,
    sf: float = SEPARATE_ACCOUNT_FACTOR,
    fr: int = 3,
) -> pd.DataFrame:
    """
    Forward-rolled liabilities using the MATLAB forward_roll_data.m logic.

    The original workflow rolls an already-daily liability series forward across
    observed month buckets, rather than resampling filings to synthetic monthly
    points first.
    """
    if fr == 0:
        return build_lb_daily(liab_ts, prices, separate_accounts_ts=separate_accounts_ts, sf=sf)

    daily_idx = prices.index
    lb_daily = build_lb_daily(liab_ts, prices, separate_accounts_ts=separate_accounts_ts, sf=sf)
    result: dict[str, pd.Series] = {}
    month_keys = daily_idx.to_period("M")
    first_mask = np.r_[True, month_keys[1:] != month_keys[:-1]]
    first_idx = np.flatnonzero(first_mask)

    if len(first_idx) == 0:
        return pd.DataFrame(index=daily_idx)

    seq_starts = first_idx[::fr]
    segment_ends = np.concatenate([seq_starts[1:] - 1, [len(daily_idx) - 1]])

    for ticker in lb_daily.columns:
        if ticker not in prices.columns:
            continue
        values = lb_daily[ticker].to_numpy(dtype=float)
        if len(values) != len(daily_idx):
            continue
        rolled = np.full(len(daily_idx), np.nan, dtype=float)
        for start, end in zip(seq_starts, segment_ends):
            rolled[start : end + 1] = values[start]
        result[ticker] = pd.Series(rolled, index=daily_idx)
    return pd.DataFrame(result, index=daily_idx)


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
        sa_ts = _fetch_separate_account_series(t)

        return {
            "ticker":        ticker,
            "name":          name,
            "prices":        prices,
            "balance_sheet": bs_data,
            "liab_ts":       liab_ts,
            "separate_accounts_ts": sa_ts,
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
    Daily market cap = price × shares outstanding.

    Prefers time-varying share-count history when available and falls back to a
    constant shares-outstanding snapshot otherwise.
    """
    caps = {}
    for ticker in list(ALL_BANKS.keys()) + [
        t for t in balance_sheet if t not in ALL_BANKS
    ]:
        if ticker not in prices.columns:
            continue
        meta = balance_sheet.get(ticker, {})
        shares_series = None

        raw_shares_ts = meta.get("shares_ts")
        if isinstance(raw_shares_ts, dict) and raw_shares_ts:
            try:
                shares_series = pd.Series(raw_shares_ts, dtype=float)
                shares_series.index = pd.to_datetime(shares_series.index)
                shares_series = shares_series.sort_index()
                shares_series = shares_series[~shares_series.index.duplicated(keep="last")]
                shares_series = shares_series.reindex(
                    shares_series.index.union(prices.index)
                ).sort_index().ffill().bfill()
                shares_series = shares_series.reindex(prices.index)
            except Exception:
                shares_series = None

        if shares_series is None:
            shares = meta.get("shares_outstanding")
            if shares:
                shares_series = pd.Series(float(shares), index=prices.index)

        if shares_series is not None:
            caps[ticker] = prices[ticker] * shares_series
    return pd.DataFrame(caps)
