"""
Systemic risk measures for Swiss banking institutions.

Implements the three canonical market-based measures:
  - MES   : Marginal Expected Shortfall (Acharya et al. 2017)
  - ΔCoVaR: Delta Conditional Value at Risk (Adrian & Brunnermeier 2016)
  - SRISK : Systemic Risk / Capital Shortfall (Brownlees & Engle 2017)

Methodology follows TimoDimi/SystemicRisk (CoVaR/MES joint model)
adapted to a rolling-window OLS/quantile-regression framework.
"""

import os
import warnings
import numpy as np
import pandas as pd
from statsmodels.regression.quantile_regression import QuantReg

warnings.filterwarnings("ignore")

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)


# ── MES ────────────────────────────────────────────────────────────────────────

def compute_mes(
    bank_returns: pd.Series,
    market_returns: pd.Series,
    window: int = 252,
    alpha: float = 0.05,
) -> pd.Series:
    """
    Rolling Marginal Expected Shortfall.

        MES_i(t) = −E[ R_i,t | R_m,t < VaR_m^α ]

    Measures the expected fractional loss of bank i when the market
    falls below its α-th percentile.  Returned as a positive number
    (loss convention).
    """
    idx = bank_returns.index.intersection(market_returns.index)
    bank = bank_returns.reindex(idx).values
    mkt  = market_returns.reindex(idx).values
    n    = len(idx)

    mes = np.full(n, np.nan)
    for t in range(window, n):
        wb, wm  = bank[t - window:t], mkt[t - window:t]
        thresh  = np.quantile(wm, alpha)
        bad     = wm < thresh
        if bad.sum() >= max(5, int(window * alpha * 0.5)):
            mes[t] = -wb[bad].mean()   # flip sign: loss is positive

    return pd.Series(mes, index=idx, name=bank_returns.name)


# ── CoVaR / ΔCoVaR ────────────────────────────────────────────────────────────

def _quantile_reg(y: np.ndarray, x: np.ndarray, q: float) -> np.ndarray:
    """
    Linear quantile regression  y = a + b*x  at quantile q.
    Returns [intercept, slope].
    """
    X = np.column_stack([np.ones(len(y)), x])
    res = QuantReg(y, X).fit(q=q, max_iter=200, p_tol=1e-4, verbose=False)
    return res.params          # [a, b]


def compute_delta_covar(
    bank_returns: pd.Series,
    market_returns: pd.Series,
    window: int = 252,
    alpha: float = 0.05,
    step: int   = 5,
) -> tuple[pd.Series, pd.Series]:
    """
    Rolling CoVaR and ΔCoVaR via quantile regression.

    Model (Adrian & Brunnermeier 2016):
        R_m = a^q + b^q · R_i + ε^q     fitted at quantile q = alpha

    CoVaR_i^α(t) = â^α + b̂^α · VaR_i^α(t)
    ΔCoVaR_i(t)  = CoVaR_i^α(t) − CoVaR_i^{0.5}(t)

    ΔCoVaR captures the *marginal* contribution of bank i to system
    tail risk (difference between stressed and median state).

    Parameters
    ----------
    step : int
        Stride between estimation windows (weekly = 5 is standard).
        Values are linearly interpolated between computed points.

    Returns
    -------
    covar      : pd.Series (negative — VaR convention)
    delta_covar: pd.Series (negative — more negative = more systemic)
    """
    idx  = bank_returns.index.intersection(market_returns.index)
    bank = bank_returns.reindex(idx).values
    mkt  = market_returns.reindex(idx).values
    n    = len(idx)

    covar_arr  = np.full(n, np.nan)
    dcovar_arr = np.full(n, np.nan)

    for t in range(window, n, step):
        wb, wm = bank[t - window:t], mkt[t - window:t]
        try:
            params_q   = _quantile_reg(wm, wb, alpha)
            params_med = _quantile_reg(wm, wb, 0.5)
        except Exception:
            continue

        var_q   = np.quantile(wb, alpha)
        var_med = np.median(wb)

        covar_q   = params_q[0]   + params_q[1]   * var_q
        covar_med = params_med[0] + params_med[1] * var_med

        covar_arr[t]  = covar_q
        dcovar_arr[t] = covar_q - covar_med

    covar  = pd.Series(covar_arr,  index=idx).interpolate("linear")
    dcovar = pd.Series(dcovar_arr, index=idx).interpolate("linear")
    return covar, dcovar


# ── SRISK ──────────────────────────────────────────────────────────────────────

def compute_srisk(
    mes_series: pd.Series,
    market_cap_series: pd.Series,
    total_liabilities: float | None,
    k: float = 0.08,
    h: int   = 22,
) -> pd.Series:
    """
    Rolling SRISK (Brownlees & Engle 2017).

        LRMES_i(t) = 1 − exp(−h · MES_i(t))
        SRISK_i(t) = max(0,  k · D_i  −  (1−k) · (1 − LRMES_i(t)) · W_i(t))

    where
        D_i  = book value of total liabilities (assumed constant)
        W_i  = market capitalization (time-varying: price × shares)
        k    = prudential capital ratio (8 %)
        h    = LRMES stress horizon in trading days (22 ≈ 1 month)

    Returns NaN for banks without balance sheet data.
    """
    if total_liabilities is None:
        return pd.Series(np.nan, index=mes_series.index)

    idx = mes_series.index.intersection(market_cap_series.index)
    mes = mes_series.reindex(idx).clip(0, 0.99)
    mkt = market_cap_series.reindex(idx)

    lrmes = 1.0 - np.exp(-h * mes)
    srisk = k * total_liabilities - (1.0 - k) * (1.0 - lrmes) * mkt
    srisk = srisk.clip(lower=0)
    return srisk


# ── Orchestration ──────────────────────────────────────────────────────────────

def compute_all(
    returns: pd.DataFrame,
    market_cap_ts: pd.DataFrame,
    balance_sheet: dict,
    window: int = 252,
    alpha: float = 0.05,
    step: int   = 5,
    force_refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Compute MES, ΔCoVaR, and SRISK for all available banks.

    Results are cached to disk as parquet files under ./cache/.
    Pass force_refresh=True to ignore the cache.

    Returns
    -------
    dict with keys 'mes', 'covar', 'delta_covar', 'srisk'
    Each value is a pd.DataFrame with bank tickers as columns.
    """
    cache_paths = {
        k: os.path.join(CACHE_DIR, f"measures_{k}.parquet")
        for k in ("mes", "covar", "delta_covar", "srisk")
    }

    # Return cached results if all files exist and are fresh (<12 h)
    if not force_refresh and all(os.path.exists(p) for p in cache_paths.values()):
        oldest = min(os.path.getmtime(p) for p in cache_paths.values())
        age_h  = (pd.Timestamp.now().timestamp() - oldest) / 3600
        if age_h < 12:
            print(f"  Loaded measures from cache ({age_h:.1f}h old)")
            return {k: pd.read_parquet(p) for k, p in cache_paths.items()}

    from data import ALL_BANKS, MARKET_NAME

    mkt_ret = returns[MARKET_NAME] if MARKET_NAME in returns.columns else \
              returns[returns.columns.intersection(["SMI", "S&P 500", "Market"])].iloc[:, 0]
    bank_cols = [c for c in returns.columns if c in ALL_BANKS]

    mes_d, covar_d, dcovar_d, srisk_d = {}, {}, {}, {}

    for ticker in bank_cols:
        name = ALL_BANKS.get(ticker, ticker)
        print(f"  {name} ({ticker}) ...")

        bank_ret = returns[ticker].dropna()

        mes_d[ticker] = compute_mes(bank_ret, mkt_ret, window, alpha)

        covar_s, dcovar_s = compute_delta_covar(
            bank_ret, mkt_ret, window, alpha, step
        )
        covar_d[ticker]  = covar_s
        dcovar_d[ticker] = dcovar_s

        mc_ts = market_cap_ts.get(ticker)  # may be None for some banks
        total_liab = balance_sheet.get(ticker, {}).get("total_liabilities")
        if mc_ts is not None:
            srisk_d[ticker] = compute_srisk(mes_d[ticker], mc_ts, total_liab)
        else:
            srisk_d[ticker] = pd.Series(np.nan, index=mes_d[ticker].index)

    result = {
        "mes":         pd.DataFrame(mes_d),
        "covar":       pd.DataFrame(covar_d),
        "delta_covar": pd.DataFrame(dcovar_d),
        "srisk":       pd.DataFrame(srisk_d),
    }
    for k, df in result.items():
        df.to_parquet(cache_paths[k])

    return result
