"""
Systemic risk measures — translated from TommasoBelluzzo/SystemicRisk (MATLAB).

Implements DCC-GJR-GARCH methodology for five canonical market-based measures:
  - MES      : Marginal Expected Shortfall
  - SES      : Systemic Expected Shortfall
  - CoVaR    : Conditional Value at Risk
  - ΔCoVaR   : Delta Conditional Value at Risk
  - SRISK    : Systemic Risk / Capital Shortfall

References:
  Acharya et al. (2010, 2017), Adrian & Brunnermeier (2016),
  Brownlees & Engle (2017), Belluzzo (2020) github.com/TommasoBelluzzo/SystemicRisk
"""
from __future__ import annotations

import os
import json
import hashlib
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# ── Global parameters ─────────────────────────────────────────────────────────

CAR = 0.08   # prudential capital adequacy ratio (Brownlees & Engle 2017)
D   = 0.40   # LRMES stress horizon: 40 % market decline
FR  = 3      # forward-roll frequency in months


# ── GJR-GARCH(1,1,1) ─────────────────────────────────────────────────────────

def _fit_gjrgarch(r: np.ndarray, label: str = "series") -> np.ndarray:
    """
    Fit GJR-GARCH(1,1,1) with zero mean.  Returns conditional variance h[t].

    Uses the `arch` package when available; falls back to EWMA(λ=0.94).
    The fallback is explicitly reported because it is not model-equivalent.
    Returns are passed in decimal units; internally scaled ×100 for numerics.
    """
    r = np.asarray(r, dtype=float)
    try:
        from arch import arch_model  # type: ignore
        am  = arch_model(r * 100.0, mean="Zero", vol="GARCH", p=1, o=1, q=1)
        res = am.fit(disp="off", show_warning=False)
        h   = (res.conditional_volatility / 100.0) ** 2
        return h
    except Exception as exc:
        msg = (
            f"GJR-GARCH fit failed for {label}; falling back to EWMA(0.94). "
            f"Reason: {type(exc).__name__}: {exc}"
        )
        warnings.warn(msg, RuntimeWarning)
        print(f"  Warning: {msg}")
        lam  = 0.94
        h    = np.empty(len(r))
        h[0] = np.var(r) if len(r) > 1 else 1e-6
        for t in range(1, len(r)):
            h[t] = lam * h[t - 1] + (1.0 - lam) * r[t - 1] ** 2
        return h


# ── DCC(1,1) ──────────────────────────────────────────────────────────────────

def _fit_dcc(eps: np.ndarray) -> tuple[float, float]:
    """
    Fit DCC(1,1) parameters (a, b) via conditional log-likelihood.

    eps : (T, 2) array of standardised GARCH residuals.

    The DCC contribution to the log-likelihood (correlation part only):
        ℓ_t = −½ [log(1−ρ²) + (e₁²+e₂²−2ρe₁e₂)/(1−ρ²) − e₁² − e₂²]
    """
    Q_bar = np.cov(eps.T)

    def neg_ll(params: np.ndarray) -> float:
        a, b = float(params[0]), float(params[1])
        if a <= 0.0 or b <= 0.0 or a + b >= 1.0:
            return 1e10
        Q     = Q_bar.copy()
        Q_tgt = (1.0 - a - b) * Q_bar
        ll    = 0.0
        for t in range(1, len(eps)):
            Q = Q_tgt + a * np.outer(eps[t - 1], eps[t - 1]) + b * Q
            rho = np.clip(Q[0, 1] / np.sqrt(Q[0, 0] * Q[1, 1]), -0.9999, 0.9999)
            e1, e2 = eps[t, 0], eps[t, 1]
            ll += (
                np.log(1.0 - rho ** 2)
                + (e1 ** 2 + e2 ** 2 - 2.0 * rho * e1 * e2) / (1.0 - rho ** 2)
                - e1 ** 2 - e2 ** 2
            )
        return ll

    res = minimize(
        neg_ll,
        x0     = [0.01, 0.95],
        method = "L-BFGS-B",
        bounds = [(1e-6, 0.5), (1e-6, 0.9999)],
        options = {"ftol": 1e-8, "maxiter": 200},
    )
    return float(res.x[0]), float(res.x[1])


def _dcc_path(eps: np.ndarray, a: float, b: float) -> np.ndarray:
    """Compute DCC correlation path ρ[t] from fitted parameters (a, b)."""
    Q_bar = np.cov(eps.T)
    Q_tgt = (1.0 - a - b) * Q_bar
    Q     = Q_bar.copy()
    T     = len(eps)
    rho   = np.empty(T)
    for t in range(T):
        if t > 0:
            Q = Q_tgt + a * np.outer(eps[t - 1], eps[t - 1]) + b * Q
        rho[t] = np.clip(Q[0, 1] / np.sqrt(Q[0, 0] * Q[1, 1]), -0.9999, 0.9999)
    return rho


def dcc_gjrgarch(
    rm: np.ndarray,
    rf: np.ndarray,
    market_label: str = "market",
    firm_label: str = "firm",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bivariate DCC-GJR-GARCH(1,1,1) with zero mean.

    Returns
    -------
    sm  : conditional volatility of market (std dev, decimal daily)
    sf  : conditional volatility of firm   (std dev, decimal daily)
    rho : DCC conditional correlation ρ[t]
    """
    hm  = _fit_gjrgarch(rm, market_label)
    hf  = _fit_gjrgarch(rf, firm_label)
    sm  = np.sqrt(np.maximum(hm, 1e-12))
    sf  = np.sqrt(np.maximum(hf, 1e-12))
    eps = np.column_stack([rm / sm, rf / sf])
    a, b = _fit_dcc(eps)
    rho  = _dcc_path(eps, a, b)
    return sm, sf, rho


def _quantile_regression(y: np.ndarray, x: np.ndarray, alpha: float) -> np.ndarray:
    """
    Iteratively reweighted least squares quantile regression.

    Mirrors the solver used in the upstream MATLAB implementation.
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    x = np.asarray(x, dtype=float)
    n = len(y)
    if x.ndim == 1:
        x = x.reshape(n, 1)

    X = np.column_stack([np.ones(n), x])
    X_star = X.copy()
    beta = np.ones(X.shape[1], dtype=float)

    diff = np.inf
    i = 0
    while diff > 1e-6 and i < 1000:
        beta_prev = beta.copy()
        Xt_star = X_star.T
        lhs = Xt_star @ X
        rhs = Xt_star @ y
        try:
            beta = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(lhs, rhs, rcond=None)[0]

        rsd = y - X @ beta
        rsd[np.abs(rsd) < 1e-6] = 1e-6
        neg = rsd < 0
        pos = rsd > 0
        rsd[neg] = alpha * rsd[neg]
        rsd[pos] = (1.0 - alpha) * rsd[pos]
        rsd = np.abs(rsd)

        X_star = X / rsd[:, None]
        diff = float(np.max(np.abs(beta - beta_prev)))
        i += 1

    return beta


# ── CoVaR / ΔCoVaR ────────────────────────────────────────────────────────────

def compute_covar_dcovar(
    bank_returns: pd.Series,
    market_returns: pd.Series,
    sm: np.ndarray,
    sf: np.ndarray,
    rho: np.ndarray,
    state_vars: pd.DataFrame | None = None,
    alpha: float = 0.05,
) -> tuple[pd.Series, pd.Series]:
    """
    CoVaR and ΔCoVaR via quantile regression on DCC-standardised residuals.

    Model (Adrian & Brunnermeier 2016 / Belluzzo 2020):
        rm_0 = b₀ + b₁ · rf_0   at quantile α

    CoVaR_i(t)  = b₀ + b₁ · VaR_i(t)        VaR_i(t) = sf(t) · c_i
    ΔCoVaR_i(t) = b₁ · (VaR_i(t) − Median_i)

    Returns positive loss convention (larger = more systemic).
    """
    idx  = bank_returns.index.intersection(market_returns.index)
    rm   = market_returns.reindex(idx).values
    rf   = bank_returns.reindex(idx).values
    nmin = min(len(idx), len(sm), len(sf), len(rho))

    rm, rf       = rm[-nmin:], rf[-nmin:]
    sm_v, sf_v   = sm[-nmin:], sf[-nmin:]
    rho_v        = rho[-nmin:]
    idx_out      = idx[-nmin:]

    rm_0 = rm - rm.mean()
    rf_0 = rf - rf.mean()

    sv_v = None
    if state_vars is not None and not state_vars.empty:
        sv_df = state_vars.reindex(idx_out)
        sv_df = sv_df.dropna(axis=1, how="all")
        if not sv_df.empty:
            sv_v = sv_df.to_numpy(dtype=float)

    c_firm = np.quantile(rf_0 / sf_v, alpha)                 # scalar, < 0
    var_i  = sf_v * c_firm                                   # time-varying, < 0

    if sv_v is None:
        b = _quantile_regression(rm_0, rf_0, alpha)
        covar = b[0] + b[1] * var_i
    else:
        b = _quantile_regression(rm_0[1:], np.column_stack([rf_0[1:], sv_v[:-1]]), alpha)
        covar = b[0] + b[1] * var_i[1:]
        for i in range(sv_v.shape[1]):
            covar = covar + b[i + 2] * sv_v[:-1, i]
        covar = np.concatenate([[covar[0]], covar])

    dcovar = b[1] * (var_i - np.median(rf_0))

    covar_s  = pd.Series(-np.minimum(covar,  0.0), index=idx_out,
                         name=bank_returns.name)
    dcovar_s = pd.Series(-np.minimum(dcovar, 0.0), index=idx_out,
                         name=bank_returns.name)
    return covar_s, dcovar_s


# ── MES / LRMES ───────────────────────────────────────────────────────────────

def compute_mes_lrmes(
    bank_returns: pd.Series,
    market_returns: pd.Series,
    sm: np.ndarray,
    sf: np.ndarray,
    rho: np.ndarray,
    alpha: float = 0.05,
    d: float = D,
) -> tuple[pd.Series, pd.Series]:
    """
    MES via Silverman-kernel conditional expectation (Belluzzo 2020).

    k₁, k₂ are global scalars estimated from the full sample.
    Time variation enters entirely through sm(t), sf(t), ρ(t).

        MES(t)   = −min(sf(t)·ρ(t)·k₁  +  sf(t)·z(t)·k₂,  0)
        β(t)     = ρ(t) · sf(t) / sm(t)
        LRMES(t) = 1 − exp(log(1−d) · β(t))

    Returns
    -------
    mes   : pd.Series  (positive, daily)
    lrmes : pd.Series  (fraction, 0–1)
    """
    idx  = bank_returns.index.intersection(market_returns.index)
    rm   = market_returns.reindex(idx).values
    rf   = bank_returns.reindex(idx).values
    nmin = min(len(idx), len(sm), len(sf), len(rho))

    rm, rf       = rm[-nmin:], rf[-nmin:]
    sm_v, sf_v   = sm[-nmin:], sf[-nmin:]
    rho_v        = rho[-nmin:]
    idx_out      = idx[-nmin:]

    rm_0   = rm - rm.mean()
    rf_0   = rf - rf.mean()

    c_mkt  = np.quantile(rm_0, alpha)                         # scalar, < 0
    u      = rm_0 / sm_v                                      # standardised market
    z_v    = np.sqrt(np.maximum(1.0 - rho_v ** 2, 1e-8))
    x      = (rf_0 / sf_v - rho_v * u) / z_v

    # Silverman bandwidth on standardised market residuals
    u_std  = np.std(u, ddof=1) if nmin > 1 else 0.0
    u_iqr  = stats.iqr(u, rng=(25, 75), scale=1.0)
    r0_s   = min(u_std, u_iqr / 1.349) if u_iqr > 0 else u_std
    h_bw   = r0_s * (4.0 / (3.0 * nmin)) ** 0.2
    if not np.isfinite(h_bw) or h_bw <= 0:
        h_bw = 1e-6
    f      = stats.norm.cdf((c_mkt / sm_v - u) / h_bw)

    sum_f  = f.sum()
    k1     = float((u * f).sum() / sum_f) if sum_f > 0 else 0.0
    k2     = float((x * f).sum() / sum_f) if sum_f > 0 else 0.0

    mes_raw  = sf_v * rho_v * k1 + sf_v * z_v * k2
    mes_vals = -np.minimum(mes_raw, 0.0)

    beta_v   = rho_v * (sf_v / sm_v)
    lrmes_v  = 1.0 - np.exp(np.log(1.0 - d) * beta_v)

    mes_s   = pd.Series(mes_vals,                    index=idx_out, name=bank_returns.name)
    lrmes_s = pd.Series(np.clip(lrmes_v, 0.0, 1.0), index=idx_out, name=bank_returns.name)
    return mes_s, lrmes_s


# ── SES ───────────────────────────────────────────────────────────────────────

def compute_ses(
    lb: pd.Series,
    cp: pd.Series,
    car: float = CAR,
) -> pd.Series:
    """
    Systemic Expected Shortfall (Belluzzo / Acharya et al. 2010).

        SES(t) = max(0,  car·lb(t)·(1+Δlb/lb)  −  (1−car)·cp(t)·(1+Δcp/cp))

    lb : total liabilities  (daily, forward-filled from quarterly filings)
    cp : market capitalisation (daily, price × shares)
    """
    idx  = lb.index.intersection(cp.index)
    lb_  = lb.reindex(idx).replace(0.0, np.nan)
    cp_  = cp.reindex(idx).replace(0.0, np.nan)

    lb_pc = lb_.pct_change().fillna(0.0)
    eq_pc = cp_.pct_change().fillna(0.0)

    ses = car * lb_ * (1.0 + lb_pc) - (1.0 - car) * cp_ * (1.0 + eq_pc)
    return ses.clip(lower=0.0)


# ── SRISK ─────────────────────────────────────────────────────────────────────

def compute_srisk(
    lrmes: pd.Series,
    lbr: pd.Series,
    cp: pd.Series,
    car: float = CAR,
) -> pd.Series:
    """
    SRISK (Brownlees & Engle 2017 / Belluzzo 2020).

        SRISK(t) = max(0,  car·lbr(t)  −  (1−car)·(1−LRMES(t))·cp(t))

    lrmes : long-run MES fraction (0–1)
    lbr   : forward-rolled liabilities (quarterly step function)
    cp    : market capitalisation (daily)
    """
    idx   = lrmes.index.intersection(lbr.index).intersection(cp.index)
    lr    = lrmes.reindex(idx).clip(0.0, 1.0)
    lb    = lbr.reindex(idx)
    cap   = cp.reindex(idx)

    srisk = car * lb - (1.0 - car) * (1.0 - lr) * cap
    return srisk.clip(lower=0.0)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_dcc_cache() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    """Load cached DCC outputs (sm, sf, rho) if available."""
    paths = {k: os.path.join(CACHE_DIR, f"{k}.parquet")
             for k in ("dcc_sm", "dcc_sf", "dcc_rho")}
    if not all(os.path.exists(p) for p in paths.values()):
        return None
    return (
        pd.read_parquet(paths["dcc_sm"]),
        pd.read_parquet(paths["dcc_sf"]),
        pd.read_parquet(paths["dcc_rho"]),
    )


def _save_dcc_cache(
    sm_d: dict, sf_d: dict, rho_d: dict,
    index_map: dict[str, pd.Index],
) -> None:
    """Persist DCC outputs to parquet."""
    def _to_df(d: dict) -> pd.DataFrame:
        series = {t: pd.Series(v, index=index_map[t]) for t, v in d.items()}
        return pd.DataFrame(series)

    _to_df(sm_d).to_parquet(os.path.join(CACHE_DIR, "dcc_sm.parquet"))
    _to_df(sf_d).to_parquet(os.path.join(CACHE_DIR, "dcc_sf.parquet"))
    _to_df(rho_d).to_parquet(os.path.join(CACHE_DIR, "dcc_rho.parquet"))


def update_dcc_cache_column(key: str, ticker: str, values: np.ndarray, index: pd.Index) -> None:
    """Merge or create one DCC cache column without dropping non-overlapping dates."""
    path = os.path.join(CACHE_DIR, f"{key}.parquet")
    new_s = pd.Series(values, index=index, name=ticker)
    if os.path.exists(path):
        existing = pd.read_parquet(path)
        combined_index = existing.index.union(index)
        existing = existing.reindex(combined_index)
        existing[ticker] = new_s.reindex(combined_index)
        existing.to_parquet(path)
    else:
        pd.DataFrame({ticker: new_s}).to_parquet(path)


def _cache_signature(
    returns: pd.DataFrame,
    market_cap_ts: pd.DataFrame,
    lb_daily: pd.DataFrame | None,
    lbr_daily: pd.DataFrame | None,
    state_vars: pd.DataFrame | None,
    alpha: float,
    d: float,
    car: float,
) -> str:
    payload = {
        "returns_index_start": str(returns.index.min()) if not returns.empty else None,
        "returns_index_end": str(returns.index.max()) if not returns.empty else None,
        "returns_cols": sorted(map(str, returns.columns.tolist())),
        "market_cap_cols": sorted(map(str, market_cap_ts.columns.tolist())) if market_cap_ts is not None else [],
        "lb_cols": sorted(map(str, lb_daily.columns.tolist())) if lb_daily is not None else [],
        "lbr_cols": sorted(map(str, lbr_daily.columns.tolist())) if lbr_daily is not None else [],
        "state_cols": sorted(map(str, state_vars.columns.tolist())) if state_vars is not None else [],
        "alpha": round(float(alpha), 8),
        "d": round(float(d), 8),
        "car": round(float(car), 8),
        "version": 2,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _cache_meta_path() -> str:
    return os.path.join(CACHE_DIR, "measures_meta.json")


# ── Orchestration ──────────────────────────────────────────────────────────────

def compute_all(
    returns: pd.DataFrame,
    market_cap_ts: pd.DataFrame,
    lb_daily: pd.DataFrame | None,
    lbr_daily: pd.DataFrame | None,
    balance_sheet: dict,
    state_vars: pd.DataFrame | None = None,
    alpha: float = 0.05,
    d: float = D,
    car: float = CAR,
    force_refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Compute MES, SES, CoVaR, ΔCoVaR, and SRISK for all available banks.

    DCC-GJR-GARCH is fitted per bank (market + firm bivariate).
    Results and DCC outputs are cached as parquet files under ./cache/.
    Pass force_refresh=True to bypass the cache.

    Returns
    -------
    dict with keys 'mes', 'ses', 'covar', 'delta_covar', 'srisk'.
    Each value is a pd.DataFrame with bank tickers as columns.
    """
    measure_keys = ("mes", "ses", "covar", "delta_covar", "srisk")
    cache = {k: os.path.join(CACHE_DIR, f"measures_{k}.parquet")
             for k in measure_keys}
    dcc_paths = [os.path.join(CACHE_DIR, f"{k}.parquet")
                 for k in ("dcc_sm", "dcc_sf", "dcc_rho")]
    meta_path = _cache_meta_path()
    signature = _cache_signature(
        returns, market_cap_ts, lb_daily, lbr_daily, state_vars, alpha, d, car
    )

    all_cache = list(cache.values()) + dcc_paths
    if not force_refresh and all(os.path.exists(p) for p in all_cache) and os.path.exists(meta_path):
        oldest = min(os.path.getmtime(p) for p in all_cache)
        age_h  = (pd.Timestamp.now().timestamp() - oldest) / 3600
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except Exception:
            meta = {}
        if age_h < 12 and meta.get("signature") == signature:
            print(f"  Loaded measures from cache ({age_h:.1f}h old)")
            return {k: pd.read_parquet(p) for k, p in cache.items()}

    from data_load import ALL_BANKS, MARKET_NAME

    mkt_ret   = (returns[MARKET_NAME] if MARKET_NAME in returns.columns
                 else returns[returns.columns.intersection(
                     ["SMI", "S&P 500", "Market"])].iloc[:, 0])
    bank_cols = [c for c in returns.columns if c in ALL_BANKS]

    mes_d, ses_d, covar_d, dcovar_d, srisk_d = {}, {}, {}, {}, {}
    sm_d, sf_d, rho_d, idx_map = {}, {}, {}, {}

    for ticker in bank_cols:
        name = ALL_BANKS.get(ticker, ticker)
        print(f"  {name} ({ticker}) ...")

        bank_ret = returns[ticker].dropna()
        idx      = bank_ret.index.intersection(mkt_ret.index)
        rm       = mkt_ret.reindex(idx).values
        rf       = bank_ret.reindex(idx).values

        # DCC-GJR-GARCH
        sm, sf, rho = dcc_gjrgarch(rm, rf, market_label=MARKET_NAME, firm_label=ticker)

        nmin    = min(len(idx), len(sm))
        idx_out = idx[-nmin:]

        sm_d[ticker]  = sm[-nmin:]
        sf_d[ticker]  = sf[-nmin:]
        rho_d[ticker] = rho[-nmin:]
        idx_map[ticker] = idx_out

        covar_s, dcovar_s = compute_covar_dcovar(
            bank_ret, mkt_ret, sm, sf, rho, state_vars=state_vars, alpha=alpha)
        covar_d[ticker]  = covar_s
        dcovar_d[ticker] = dcovar_s

        mes_s, lrmes_s = compute_mes_lrmes(
            bank_ret, mkt_ret, sm, sf, rho, alpha, d)
        mes_d[ticker] = mes_s

        cp_ts  = market_cap_ts.get(ticker)
        lb_ts  = lb_daily.get(ticker)  if lb_daily  is not None else None
        lbr_ts = lbr_daily.get(ticker) if lbr_daily is not None else None

        if cp_ts is not None and lb_ts is not None:
            ses_d[ticker]   = compute_ses(lb_ts, cp_ts, car)
        else:
            ses_d[ticker]   = pd.Series(np.nan, index=mes_s.index)

        if cp_ts is not None and lbr_ts is not None:
            srisk_d[ticker] = compute_srisk(lrmes_s, lbr_ts, cp_ts, car)
        else:
            srisk_d[ticker] = pd.Series(np.nan, index=mes_s.index)

    result = {
        "mes":         pd.DataFrame(mes_d),
        "ses":         pd.DataFrame(ses_d),
        "covar":       pd.DataFrame(covar_d),
        "delta_covar": pd.DataFrame(dcovar_d),
        "srisk":       pd.DataFrame(srisk_d),
    }
    for k, df in result.items():
        df.to_parquet(cache[k])

    _save_dcc_cache(sm_d, sf_d, rho_d, idx_map)
    with open(meta_path, "w") as f:
        json.dump({"signature": signature}, f)
    return result


# ── Fast recompute for interactive alpha changes ────────────────────────────────

def recompute_for_alpha(
    returns: pd.DataFrame,
    market_cap_ts: pd.DataFrame,
    lb_daily: pd.DataFrame | None,
    lbr_daily: pd.DataFrame | None,
    balance_sheet: dict,
    state_vars: pd.DataFrame | None = None,
    alpha: float = 0.05,
    d: float = D,
    car: float = CAR,
) -> dict[str, pd.DataFrame]:
    """
    Recompute all measures for a new alpha using cached DCC outputs.

    Avoids refitting DCC-GJR-GARCH (the computationally expensive step).
    Falls back to a full compute_all() if the DCC cache is absent.
    """
    cached = _load_dcc_cache()
    if cached is None:
        return compute_all(
            returns, market_cap_ts, lb_daily, lbr_daily, balance_sheet,
            state_vars=state_vars, alpha=alpha, d=d, car=car,
        )

    dcc_sm, dcc_sf, dcc_rho = cached

    from data_load import ALL_BANKS, MARKET_NAME

    mkt_ret   = (returns[MARKET_NAME] if MARKET_NAME in returns.columns
                 else returns[returns.columns.intersection(
                     ["SMI", "S&P 500", "Market"])].iloc[:, 0])
    bank_cols = [c for c in returns.columns if c in ALL_BANKS]

    mes_d, ses_d, covar_d, dcovar_d, srisk_d = {}, {}, {}, {}, {}

    for ticker in bank_cols:
        if ticker not in dcc_sm.columns:
            continue

        bank_ret = returns[ticker].dropna()
        sm  = dcc_sm[ticker].dropna().values
        sf  = dcc_sf[ticker].dropna().values
        rho = dcc_rho[ticker].dropna().values
        if len(sm) == 0 or len(sf) == 0 or len(rho) == 0:
            continue

        covar_s, dcovar_s = compute_covar_dcovar(
            bank_ret, mkt_ret, sm, sf, rho, state_vars=state_vars, alpha=alpha)
        covar_d[ticker]  = covar_s
        dcovar_d[ticker] = dcovar_s

        mes_s, lrmes_s = compute_mes_lrmes(
            bank_ret, mkt_ret, sm, sf, rho, alpha, d)
        mes_d[ticker] = mes_s

        cp_ts  = market_cap_ts.get(ticker)
        lb_ts  = lb_daily.get(ticker)  if lb_daily  is not None else None
        lbr_ts = lbr_daily.get(ticker) if lbr_daily is not None else None

        if cp_ts is not None and lb_ts is not None:
            ses_d[ticker] = compute_ses(lb_ts, cp_ts, car)
        else:
            ses_d[ticker] = pd.Series(np.nan, index=mes_s.index)

        if cp_ts is not None and lbr_ts is not None:
            srisk_d[ticker] = compute_srisk(lrmes_s, lbr_ts, cp_ts, car)
        else:
            srisk_d[ticker] = pd.Series(np.nan, index=mes_s.index)

    return {
        "mes":         pd.DataFrame(mes_d),
        "ses":         pd.DataFrame(ses_d),
        "covar":       pd.DataFrame(covar_d),
        "delta_covar": pd.DataFrame(dcovar_d),
        "srisk":       pd.DataFrame(srisk_d),
    }
