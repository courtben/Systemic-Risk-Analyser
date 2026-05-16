"""
Systemic Risk Dashboard — US Banks
====================================
Interactive Dash application visualising MES, DeltaCoVaR, and SRISK
for major US banking institutions. Custom banks can be added by ticker.

Run:
    pip install -r requirements.txt
    python app.py
Then open http://127.0.0.1:8050
"""

import sys
import warnings
import threading
import numpy as np

# Windows consoles default to cp1252, which can't render Δ / α / ρ used in our
# startup logs. Reconfigure to UTF-8 where supported so `python app.py` doesn't
# crash on the first print.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import pandas as pd
import plotly.graph_objects as go
import os
from datetime import datetime, timezone

import io
import json
import dash
from dash import dcc, html, Input, Output, State, no_update, dash_table, ALL
import dash_bootstrap_components as dbc

warnings.filterwarnings("ignore")

# ── Default Plotly graph config ──────────────────────────────────────────────
# Surfaces a visible toolbar on every chart with a high-quality PNG export
# button ("Save as image"). Lasso/box-select are removed since they're not
# meaningful for any of our charts, and Plotly's brand logo is hidden.
_DEFAULT_GRAPH_CONFIG = {
    "displayModeBar": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    "toImageButtonOptions": {
        "format": "png",
        "filename": "systemic-risk-chart",
        "height": 700,
        "width": 1200,
        "scale": 2,
    },
}

_original_dcc_graph = dcc.Graph

def _Graph(*args, **kwargs):
    """Drop-in replacement for dcc.Graph that injects our default config."""
    if "config" not in kwargs:
        kwargs["config"] = _DEFAULT_GRAPH_CONFIG
    return _original_dcc_graph(*args, **kwargs)

dcc.Graph = _Graph

# ── Load / compute data at startup ────────────────────────────────────────────

print("=" * 60)
print("Systemic Risk Dashboard  —  US Banks")
print("=" * 60)

import data_load as D
import systemic_measures as M

print("\n[1/7] Fetching price data ...")
PRICES = D.get_prices()

print("\n[2/7] Computing returns ...")
RETURNS = D.compute_returns(PRICES)

print("\n[3/7] Fetching balance sheet data ...")
BS = D.get_balance_sheet()

print("\n[4/7] Fetching liabilities and separate-account time series ...")
LIAB_TS      = D.get_liabilities_ts()
SEP_ACCT_TS  = D.get_separate_accounts_ts()
LB_DAILY     = D.build_lb_daily(LIAB_TS, PRICES, SEP_ACCT_TS)
LBR_DAILY    = D.build_lbr_daily(LIAB_TS, PRICES, SEP_ACCT_TS)

print("\n[5/7] Fetching systemic state variables ...")
STATE_VARS = D.get_state_variables(PRICES)

print("\n[6/7] Computing systemic risk measures (DCC-GJR-GARCH) ...")
MC_TS    = D.build_market_cap_series(PRICES, BS)
# Returns dict with keys: 'mes', 'ses', 'covar', 'delta_covar', 'srisk'
MEASURES = M.compute_all(RETURNS, MC_TS, LB_DAILY, LBR_DAILY, BS, state_vars=STATE_VARS)

# ── Quasi-leverage = (Book Liabilities + Market Cap) / Market Cap ────────────
def _build_leverage(lb_daily: pd.DataFrame, mc_ts: pd.DataFrame) -> pd.DataFrame:
    common = [c for c in lb_daily.columns if c in mc_ts.columns]
    if not common:
        return pd.DataFrame()
    idx = lb_daily.index.union(mc_ts.index)
    lb  = lb_daily[common].reindex(idx).ffill()
    mc  = mc_ts[common].reindex(idx).ffill()
    mc_safe = mc.where(mc > 0, np.nan)
    return (lb + mc_safe) / mc_safe

LEVERAGE = _build_leverage(LB_DAILY, MC_TS)

# ── DCC time-varying correlation (ρ_t) ───────────────────────────────────────
def _load_dcc_rho() -> pd.DataFrame:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "dcc_rho.parquet")
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception as e:
        print(f"  Warning: could not load dcc_rho.parquet: {e}")
        return pd.DataFrame()

DCC_RHO = _load_dcc_rho()

# ── α-grid for ΔCoVaR ────────────────────────────────────────────────────────
# ΔCoVaR uses quantile regression (IRLS) which is too slow to recompute
# interactively.
ALPHA_GRID = [0.01, 0.025, 0.05, 0.075, 0.10]

def _nearest_alpha(a: float) -> float:
    return min(ALPHA_GRID, key=lambda g: abs(g - a))

def _build_dcovar_grid() -> dict[float, pd.DataFrame]:
    out = {0.05: MEASURES["delta_covar"]}
    for a in ALPHA_GRID:
        if abs(a - 0.05) < 1e-9:
            continue
        print(f"  ΔCoVaR α = {a:.3f} ...")
        try:
            res = M.recompute_for_alpha(
                RETURNS, MC_TS, LB_DAILY, LBR_DAILY, BS,
                state_vars=STATE_VARS, alpha=a,
            )
            out[round(a, 4)] = res["delta_covar"]
        except Exception as e:
            print(f"    skipped: {e}")
    return out

print("\n[7/7] Precomputing ΔCoVaR α-grid (1%, 2.5%, 7.5%, 10%) ...")
DCOVAR_BY_ALPHA = _build_dcovar_grid()

# ── Last-refresh persistence ─────────────────────────────────────────────────
# Stored in cache/ so the timestamp survives container restarts and Render
# deploys.  When the file is missing (very first boot, or after a fresh
# clone without committed cache) we fall back to the current UTC time.

_LAST_REFRESH_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "cache", "last_refresh.json"
)


def _now_utc_str() -> str:
    """Format the current UTC moment for display in the navbar."""
    return pd.Timestamp.utcnow().tz_localize(None).strftime("%Y-%m-%d %H:%M UTC")


def _read_last_refresh() -> str:
    """Return the persisted last-refresh timestamp, or the current UTC time."""
    try:
        with open(_LAST_REFRESH_PATH, "r", encoding="utf-8") as f:
            return json.load(f).get("timestamp_utc", _now_utc_str())
    except Exception:
        return _now_utc_str()


def _write_last_refresh(ts: str) -> None:
    """Persist the last-refresh timestamp so it survives restarts."""
    try:
        os.makedirs(os.path.dirname(_LAST_REFRESH_PATH), exist_ok=True)
        with open(_LAST_REFRESH_PATH, "w", encoding="utf-8") as f:
            json.dump({"timestamp_utc": ts}, f)
    except Exception as exc:
        print(f"[Refresh] could not persist last-refresh: {exc}")


LAST_UPDATED = _read_last_refresh()
print(f"\nReady  ({LAST_UPDATED})")
print("=" * 60)

# ── Constants ─────────────────────────────────────────────────────────────────

ALL_BANKS      = D.ALL_BANKS
BANK_COLORS    = D.BANK_COLORS
BANK_COUNTRY   = D.BANK_COUNTRY
COUNTRY_LABELS = D.COUNTRY_LABELS
MARKET_NAME    = D.MARKET_NAME

# ── Presentation layer (style tokens, chart builders) ─────────────────────────
# Extracted into charts.py; re-exported here under the original private names
# to keep existing call sites in callbacks unchanged.
import charts as C

PLOTLY_TEMPLATE = C.PLOTLY_TEMPLATE
BG_PAGE         = C.BG_PAGE
BG_CARD         = C.BG_CARD
BG_HEADER       = C.BG_HEADER
BORDER          = C.BORDER
TEXT_MUTED      = C.TEXT_MUTED
TEXT_MAIN       = C.TEXT_MAIN
ACCENT_BLUE      = C.ACCENT_BLUE
ACCENT_BLUE_DARK = C.ACCENT_BLUE_DARK
ACCENT_RED       = C.ACCENT_RED
ACCENT_GREEN     = C.ACCENT_GREEN
ACCENT_AMBER     = C.ACCENT_AMBER
NEUTRAL_GREY     = C.NEUTRAL_GREY
CRISIS_PERIODS  = C.CRISIS_PERIODS

# Format / layout helpers — keep underscore-prefixed aliases for callbacks
_add_crisis_overlays = C.add_crisis_overlays
_base_layout         = C.base_layout
_name                = C.name_for
_color               = C.color_for
_fmt_bn              = C.fmt_bn
_fmt_pct             = C.fmt_pct
_fmt_ratio           = C.fmt_ratio
_fmt_pct_raw         = C.fmt_pct_raw
_fmt_bn_x1           = C.fmt_bn_x1

# Chart builders
delta_ranking_bar    = C.delta_ranking_bar
ranking_bar          = C.ranking_bar
timeseries_chart     = C.timeseries_chart
_srisk_bar_generic   = C.srisk_bar_generic
srisk_pie            = C.srisk_pie
srisk_stacked_area   = C.srisk_stacked_area
price_chart          = C.price_chart
corr_heatmap         = C.corr_heatmap
return_hist          = C.return_hist
market_dcc_chart     = C.market_dcc_chart
kpi_card             = C.kpi_card


# ── Data-selection helpers (stay in app.py) ──────────────────────────────────

def _slice(df: pd.DataFrame, start, end, tickers=None) -> pd.DataFrame:
    out = df.loc[str(start):str(end)]
    if tickers:
        out = out[[c for c in tickers if c in out.columns]]
    return out


def _latest_row(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    valid = df.dropna(how="all")
    if valid.empty:
        return pd.Series(dtype=float)
    return valid.iloc[-1]


# ── App layout ────────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    suppress_callback_exceptions=True,
    title="Systemic Risk Dashboard",
)
# Expose the Flask WSGI server — required by gunicorn on Plotly Cloud:
#   gunicorn app:server
server = app.server

DATE_MIN       = RETURNS.index.min().date()
DATE_MAX       = RETURNS.index.max().date()
DATE_DEF_START = DATE_MIN  # 5-year window, start from beginning

# ── Header ────────────────────────────────────────────────────────────────────

header = dbc.Navbar(
    dbc.Container([
        html.Div([
            html.H5("Systemic Risk Dashboard — US Banks",
                    className="mb-0",
                    style={"color": TEXT_MAIN, "fontWeight": "700", "fontSize": "1.05rem"}),
            html.Small(
                "MES · ΔCoVaR · SRISK  |  "
                "Acharya et al. (2017) · Adrian & Brunnermeier (2016) · Brownlees & Engle (2017)",
                className="text-muted",
            ),
        ]),
        html.Div([
            html.Div(id="refresh-progress", className="me-3"),
            html.Div([
                html.Span("Last refreshed",
                          className="d-block text-muted",
                          style={"fontSize": "0.66rem",
                                 "letterSpacing": "0.08em",
                                 "textTransform": "uppercase",
                                 "lineHeight": "1"}),
                html.Span(LAST_UPDATED, id="updated-ts",
                          style={"fontSize": "0.85rem",
                                 "fontWeight": "600",
                                 "color": TEXT_MAIN}),
            ], className="text-end"),
        ], className="d-flex align-items-center"),
    ], fluid=True, className="d-flex justify-content-between align-items-center"),
    color="white",
    style={"borderBottom": f"1px solid {BORDER}",
           "boxShadow": "0 1px 4px rgba(0,0,0,0.08)"},
    sticky="top",
)

# ── Controls ──────────────────────────────────────────────────────────────────

_PRESET_BUTTONS = [
    ("preset-6m",  "6M",  0.5),
    ("preset-1y",  "1Y",  1),
    ("preset-2y",  "2Y",  2),
    ("preset-5y",  "5Y",  5),
    ("preset-all", "All", None),
]

_LBL_STYLE = {"fontSize": "0.72rem", "fontWeight": "600"}

controls = dbc.Container([
    # Row 1 — Date range + Quick range
    dbc.Row([
        dbc.Col([
            html.Label("Date range", className="text-muted mb-0", style=_LBL_STYLE),
            dcc.DatePickerRange(
                id="date-range",
                min_date_allowed=DATE_MIN, max_date_allowed=DATE_MAX,
                start_date=DATE_DEF_START, end_date=DATE_MAX,
                display_format="YYYY-MM-DD",
                style={"fontSize": "0.8rem"},
            ),
        ], xs=12, md=4),
        dbc.Col([
            html.Label("Quick range", className="text-muted mb-0", style=_LBL_STYLE),
            html.Div([
                dbc.ButtonGroup([
                    dbc.Button(label, id=pid, size="sm",
                               color="primary", outline=True, n_clicks=0)
                    for pid, label, _ in _PRESET_BUTTONS
                ], size="sm", className="me-2"),
                dbc.Button("⟲ Reset all",
                           id="btn-reset-defaults",
                           size="sm",
                           color="secondary",
                           outline=True,
                           n_clicks=0,
                           title="Reset date range, banks, α, k and d to defaults"),
            ], className="d-flex flex-wrap align-items-center"),
        ], xs=12, md=8),
    ], className="gy-1 align-items-end"),

    # Row 2 — Banks + Add bank + Selection (All/None)
    dbc.Row([
        dbc.Col([
            html.Label("Banks", className="text-muted mb-0", style=_LBL_STYLE),
            dcc.Dropdown(
                id="bank-select",
                multi=True,
                placeholder="Select banks ...",
                style={"fontSize": "0.8rem"},
            ),
        ], xs=12, md=6),
        dbc.Col([
            html.Label("Add bank by ticker", className="text-muted mb-0", style=_LBL_STYLE),
            dbc.InputGroup([
                dbc.Input(
                    id="custom-ticker-input",
                    placeholder="e.g. BRK-B, USB, TFC ...",
                    type="text",
                    size="sm",
                    style={"fontSize": "0.8rem"},
                    debounce=False,
                ),
                dbc.Button("Add", id="btn-add-bank", color="primary",
                           size="sm", n_clicks=0),
            ], size="sm"),
            dcc.Loading(
                type="dot",
                color=C.ACCENT_BLUE,
                children=html.Div(id="add-bank-status",
                                  style={"fontSize": "0.7rem", "minHeight": "0"}),
            ),
        ], xs=12, md=4),
        dbc.Col([
            html.Label("Selection", className="text-muted mb-0 d-block", style=_LBL_STYLE),
            dbc.ButtonGroup([
                dbc.Button("All", id="btn-all", size="sm",
                           color="secondary", outline=True),
                dbc.Button("None", id="btn-none", size="sm",
                           color="secondary", outline=True),
            ], size="sm"),
        ], xs=12, md=2),
    ], className="gy-1 mt-1 align-items-end"),

    # Row 3 — Critical value α (inline, compact)
    dbc.Row([
        dbc.Col(
            html.Div([
                html.Span(
                    id="alpha-label",
                    children="Critical Value α = 5.0%  (worst 5% of market days)",
                    className="text-muted me-3",
                    style={"fontSize": "0.72rem", "fontWeight": "600"},
                ),
                dbc.RadioItems(
                    id="alpha-select",
                    options=[
                        {"label": f"{a*100:g}%", "value": a} for a in ALPHA_GRID
                    ],
                    value=0.05,
                    inline=True,
                    className="d-inline-flex mb-0",
                    inputClassName="me-1",
                    labelClassName="me-3 mb-0",
                    labelStyle={"fontSize": "0.8rem", "fontWeight": "500"},
                ),
            ], className="d-flex align-items-center flex-wrap"),
            xs=12,
        ),
    ], className="gy-0 mt-1"),
], fluid=True, className="py-1 px-3",
   style={"backgroundColor": "#f8f9fa"})


# ── Collapsible wrapper around the filter bar ────────────────────────────────
# The whole "Date range / Banks / α" block can be hidden so plots get more
# vertical space. A thin toggle bar at the top flips a dbc.Collapse and
# swaps the caret glyph (▼ ↔ ▶).

controls_section = html.Div([
    html.Div([
        dbc.Button(
            [
                html.Span("▼", id="filters-caret",
                          style={"display": "inline-block",
                                 "transition": "transform 0.15s",
                                 "marginRight": "6px"}),
                "Filters",
            ],
            id="btn-toggle-filters",
            color="link",
            size="sm",
            n_clicks=0,
            className="p-0 text-decoration-none",
            style={"fontSize": "0.78rem", "fontWeight": "600",
                   "color": TEXT_MAIN},
        ),
    ], className="px-3 py-1",
       style={"backgroundColor": "#f8f9fa"}),
    dbc.Collapse(
        controls,
        id="filters-collapse",
        is_open=True,
    ),
], style={"borderBottom": f"1px solid {BORDER}"})


# ── Tab content ───────────────────────────────────────────────────────────────

_card = {"backgroundColor": BG_CARD,
         "border": f"1px solid {BORDER}",
         "borderRadius": "6px",
         "padding": "12px 16px",
         "marginBottom": "16px"}

# ── Start tab ────────────────────────────────────────────────────────────────
# Landing page: brief orientation, the three core measures with live KPIs,
# tab navigation cards, and quick reference material (risk pills, crisis
# bands, data scope). Built once at import time; the three KPI cards are
# populated by the existing Overview callback so values match the rest of
# the dashboard exactly.

_TABS_META: list[tuple[str, str, str]] = [
    # (tab_id, label, 1–2 sentence description)
    ("tab-overview",
     "Overview",
     "Overview of current systemic risk levels, "
     "highlighting the banks with the highest contributions and the largest one-week shifts."),
    ("tab-ts",
     "Time Series",
     "Historical developement of systemic risk measures over time."),
    ("tab-srisk",
     "SRISK",
     "Stress-test SRISK live by tuning the capital ratio and "
     "market-decline parameters."),
    ("tab-market",
     "Market & Correlation",
     "Performance and correlations of the individual banks compared to the market (S&P 500)."),
    ("tab-methodology",
     "Methodology",
     "Data sources, references and formulas used to calculate risk measures."),
]


def _measure_explainer(title: str, body: str) -> html.Div:
    return html.Div([
        html.H6(title, className="mb-1 mt-0",
                style={"fontWeight": "700", "color": TEXT_MAIN,
                       "fontSize": "0.92rem"}),
        html.P(body, className="text-muted mb-2",
               style={"fontSize": "0.78rem", "lineHeight": "1.35"}),
    ])


def _tab_link_row(tab_id: str, label: str, desc: str) -> html.Div:
    """Single full-width clickable row for the vertical tab-bookmark list."""
    inner = dbc.Card(
        dbc.CardBody([
            html.Div([
                html.Div([
                    html.Div(label, style={"fontWeight": "700",
                                           "fontSize": "0.88rem",
                                           "color": TEXT_MAIN,
                                           "lineHeight": "1.2"}),
                    html.Div(desc, className="text-muted",
                             style={"fontSize": "0.72rem",
                                    "lineHeight": "1.3"}),
                ], style={"flex": "1", "minWidth": 0}),
                html.Span("→",
                          style={"color": ACCENT_BLUE,
                                 "fontWeight": "700",
                                 "fontSize": "1.05rem",
                                 "marginLeft": "8px"}),
            ], className="d-flex align-items-center"),
        ], style={"padding": "8px 12px"}),
        style={"border": f"1px solid {BORDER}",
               "borderLeft": f"4px solid {ACCENT_BLUE}",
               "boxShadow": "0 1px 3px rgba(0,0,0,0.05)"},
    )
    return html.Div(
        inner,
        id={"type": "start-tab-link", "tab": tab_id},
        n_clicks=0,
        className="start-tab-link mb-2",
        style={"cursor": "pointer",
               "transition": "transform 0.12s, box-shadow 0.12s"},
    )


def _risk_pill(label: str, color: str) -> html.Span:
    return html.Span(
        label,
        style={
            "backgroundColor": color + "1a",
            "color": color,
            "border": f"1px solid {color}33",
            "borderRadius": "999px",
            "padding": "1px 10px",
            "fontSize": "0.72rem",
            "fontWeight": "700",
            "letterSpacing": "0.04em",
            "textTransform": "uppercase",
            "whiteSpace": "nowrap",
            "marginRight": "10px",
        },
    )


_compact_card = {**_card, "padding": "10px 14px", "marginBottom": "10px"}


start_layout = dbc.Container([
    # ── Top row: summary (left) + tab bookmarks stacked (right) ──────────
    dbc.Row([
        # Left — dashboard summary (centered content)
        dbc.Col(
            html.Div([
                html.H3("Systemic Risk Analyser",
                        className="text-center",
                        style={"fontWeight": "800",
                               "color": TEXT_MAIN,
                               "marginBottom": "6px"}),
                html.P(
                    "A data-driven dashboard for exploring systemic risk dynamics in the U.S. banking sector "
                    "through interactive visualisations, econometric risk models, and established systemic risk indicators (MES, ΔCoVaR, SRISK).",
                    className="mb-0 text-center mx-auto",
                    style={"fontSize": "0.88rem", "lineHeight": "1.45",
                           "color": TEXT_MAIN, "maxWidth": "640px"},
                ),
            ], style={**_compact_card,
                      "borderLeft": f"4px solid {ACCENT_BLUE}",
                      "height": "100%",
                      "display": "flex",
                      "flexDirection": "column",
                      "justifyContent": "center"}),
            xs=12, lg=5, className="mb-2",
        ),
        # Right — vertical tab bookmark list
        dbc.Col([
            html.H6("Explore the dashboard",
                    className="mb-2",
                    style={"fontWeight": "700", "color": TEXT_MAIN}),
            html.Div([
                _tab_link_row(tab_id, label, desc)
                for tab_id, label, desc in _TABS_META
            ]),
        ], xs=12, lg=7, className="mb-2"),
    ], className="mt-2"),

    html.Hr(className="my-2"),

    # ── Three core measures (explainers above live KPIs) ─────────────────
    html.H6("The three core measures",
            className="mt-1 mb-2",
            style={"fontWeight": "700", "color": TEXT_MAIN}),
    dbc.Row([
        dbc.Col([
            _measure_explainer(
                "MES — Marginal Expected Shortfall",
                "How much a bank's stock typically loses on a bad day for "
                "the market. Higher = hit harder by market downturns."),
            html.Div(id="kpi-start-mes", className="mt-auto"),
        ], xs=12, md=4, className="mb-2 d-flex flex-column"),
        dbc.Col([
            _measure_explainer(
                "ΔCoVaR — Conditional VaR contribution",
                "How much the market's downside risk grows when this bank "
                "is in trouble. More negative = larger systemic footprint."),
            html.Div(id="kpi-start-covar", className="mt-auto"),
        ], xs=12, md=4, className="mb-2 d-flex flex-column"),
        dbc.Col([
            _measure_explainer(
                "SRISK — Capital shortfall under stress",
                "Extra capital needed if the market crashed. "
                "Positive values = likely needs a rescue."),
            html.Div(id="kpi-start-srisk", className="mt-auto"),
        ], xs=12, md=4, className="mb-2 d-flex flex-column"),
    ]),

    html.Hr(className="my-2"),

    # ── Risk indicator scale — single tight strip ───────────────────────
    html.Div([
        html.Span("Risk indicator scale",
                  style={"fontWeight": "700",
                         "fontSize": "0.78rem",
                         "letterSpacing": "0.05em",
                         "textTransform": "uppercase",
                         "color": TEXT_MAIN,
                         "marginRight": "16px"}),
        html.Span(
            "rolling 500-obs percentile of the latest aggregate:",
            className="text-muted",
            style={"fontSize": "0.78rem", "marginRight": "16px"},
        ),
        *[
            html.Span([
                _risk_pill(label, color),
                html.Span(
                    band,
                    className="text-muted",
                    style={"fontSize": "0.78rem", "marginRight": "18px"},
                ),
            ])
            for label, color, band in [
                ("Low",    ACCENT_GREEN, "below the 70th percentile"),
                ("Medium", ACCENT_AMBER, "70th – 90th percentile"),
                ("High",   ACCENT_RED,   "at or above the 90th percentile"),
            ]
        ],
    ], style={**_compact_card,
              "display": "flex",
              "alignItems": "center",
              "flexWrap": "wrap",
              "rowGap": "6px",
              "marginTop": "4px"}),

    # ── Data scope footer ──────────────────────────────────────────────────
    html.Div(
        html.Small([
            html.B("Scope: "),
            f"{len(D.ALL_BANKS)} U.S. banks · daily data "
            f"{DATE_MIN.isoformat()} → {DATE_MAX.isoformat()} · benchmark "
            f"{D.MARKET_NAME}. ",
            html.Span(
                "Educational / research tool — not investment advice.",
                className="text-muted",
            ),
        ], className="text-muted",
           style={"fontSize": "0.74rem"}),
        className="text-center mt-1 mb-2",
    ),

], fluid=True, className="px-3")


overview_layout = dbc.Container([
    dbc.Row([
        dbc.Col(id="kpi-mes",      xs=6, md=4, lg=True, className="mb-3"),
        dbc.Col(id="kpi-lrmes",    xs=6, md=4, lg=True, className="mb-3"),
        dbc.Col(id="kpi-covar",    xs=6, md=4, lg=True, className="mb-3"),
        dbc.Col(id="kpi-srisk",    xs=6, md=4, lg=True, className="mb-3"),
        dbc.Col(id="kpi-leverage", xs=6, md=4, lg=True, className="mb-3"),
    ], className="mt-3"),

    html.Hr(className="my-2"),

    # ── Top 10 ranking row ────────────────────────────────────────────────
    html.H6("Top 10",
            className="mt-2 mb-2",
            style={"fontWeight": "700", "color": TEXT_MAIN}),
    dbc.Row([
        dbc.Col(dcc.Graph(id="chart-mes-rank"),   xs=12, md=4, className="mb-3"),
        dbc.Col(dcc.Graph(id="chart-srisk-rank"), xs=12, md=4, className="mb-3"),
        dbc.Col(dcc.Graph(id="chart-covar-rank"), xs=12, md=4, className="mb-3"),
    ]),

    html.Hr(className="my-2"),

    # ── 7-day change row ──────────────────────────────────────────────────
    html.H6("7-day change",
            className="mt-2 mb-2",
            style={"fontWeight": "700", "color": TEXT_MAIN}),
    dbc.Row([
        dbc.Col(dcc.Graph(id="chart-mes-dw"),   xs=12, md=4, className="mb-3"),
        dbc.Col(dcc.Graph(id="chart-srisk-dw"), xs=12, md=4, className="mb-3"),
        dbc.Col(dcc.Graph(id="chart-covar-dw"), xs=12, md=4, className="mb-3"),
    ]),

    html.Hr(className="my-2"),

    dbc.Row([
        dbc.Col([
            html.Div([
                dbc.Button(
                    [html.Span("▸", id="risk-summary-caret",
                               style={"display": "inline-block",
                                      "transition": "transform 0.15s",
                                      "marginRight": "6px"}),
                     "Risk Summary"],
                    id="btn-risk-summary",
                    color="link",
                    size="sm",
                    className="p-0 text-decoration-none me-3",
                    style={"fontSize": "0.82rem", "fontWeight": "600",
                           "color": TEXT_MAIN},
                    n_clicks=0,
                ),
                html.Span("Snapshot date:", className="text-muted me-2",
                          style={"fontSize": "0.78rem"}),
                dcc.DatePickerSingle(
                    id="risk-summary-date",
                    min_date_allowed=DATE_MIN,
                    max_date_allowed=DATE_MAX,
                    display_format="YYYY-MM-DD",
                    placeholder="Latest in range",
                    clearable=True,
                    style={"fontSize": "0.78rem"},
                ),
                dbc.Button("Download CSV", id="btn-download-overview",
                           size="sm", color="primary", outline=True,
                           className="float-end",
                           style={"fontSize": "0.75rem"}),
                dcc.Download(id="download-overview"),
            ], className="mb-2 d-flex align-items-center flex-wrap"),
            dbc.Collapse(
                html.Div(id="risk-table"),
                id="risk-summary-collapse",
                is_open=False,
            ),
        ], xs=12, className="mb-3"),
    ]),
], fluid=True, style={"backgroundColor": BG_PAGE})

timeseries_layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.Label("Measure", className="text-muted mb-1",
                       style={"fontSize": "0.78rem", "fontWeight": "600"}),
            dcc.Dropdown(
                id="ts-measure",
                options=[
                    {"label": "MES — Marginal Expected Shortfall",       "value": "mes"},
                    {"label": "LRMES — Long-Run MES (40% decline)",       "value": "lrmes"},
                    {"label": "ΔCoVaR — Conditional VaR contribution",   "value": "delta_covar"},
                    {"label": "CoVaR (level)",                            "value": "covar"},
                ],
                value="mes", clearable=False,
                style={"fontSize": "0.85rem"},
            ),
        ], xs=12, md=3, className="mt-3 mb-2"),
        dbc.Col([
            html.Label("Overlay market returns", className="text-muted mb-1",
                       style={"fontSize": "0.78rem", "fontWeight": "600"}),
            dbc.Checklist(
                id="ts-overlay",
                options=[{"label": f" Show {MARKET_NAME} daily returns", "value": "show"}],
                value=["show"], switch=True, className="mt-1",
            ),
        ], xs=12, md=2, className="mt-3 mb-2"),
        dbc.Col([
            html.Label("Crisis overlays", className="text-muted mb-1",
                       style={"fontSize": "0.78rem", "fontWeight": "600"}),
            dbc.Checklist(
                id="ts-crises",
                options=[{"label": " Shade crisis periods", "value": "show"}],
                value=["show"], switch=True, className="mt-1",
            ),
        ], xs=12, md=2, className="mt-3 mb-2"),
        dbc.Col([
            html.Label("Traces", className="text-muted mb-1",
                       style={"fontSize": "0.78rem", "fontWeight": "600"}),
            dbc.Checklist(
                id="ts-traces",
                options=[
                    {"label": " Individual",  "value": "individual"},
                    {"label": " Aggregate",   "value": "aggregate"},
                ],
                value=["individual", "aggregate"],
                switch=True, inline=True, className="mt-1",
            ),
        ], xs=12, md=3, className="mt-3 mb-2"),
        dbc.Col([
            html.Label("\u00a0", className="text-muted mb-1 d-block",
                       style={"fontSize": "0.78rem"}),
            dbc.Button("Download CSV", id="btn-download-ts",
                       size="sm", color="primary", outline=True),
            dcc.Download(id="download-ts"),
        ], xs=12, md=2, className="mt-3 mb-2"),
    ]),

    html.Hr(className="my-2"),

    dbc.Row([
        dbc.Col([
            html.P(
                "Daily values of the chosen risk measure for each selected "
                "bank. Use it to spot when risk has spiked historically "
                "and which banks moved the most.",
                className="text-muted mb-1 mt-1",
                style={"fontSize": "0.78rem"},
            ),
            dcc.Graph(id="chart-timeseries"),
        ], xs=12),
    ]),

    html.Hr(className="my-2"),

    dbc.Row([
        dbc.Col([
            html.Div([
                html.Ul([
                    html.Li([
                        html.B("MES"),
                        " — expected daily loss in a bank's stock when the "
                        "market is having one of its worst α% of days. "
                        "Higher MES means the bank gets hit harder when "
                        "the market falls."]),
                    html.Li([
                        html.B("LRMES"),
                        " — expected drop in a bank's stock if the market "
                        "falls by d over the medium term — the long-horizon "
                        "counterpart to MES. Higher LRMES means a larger "
                        "expected loss under sustained market stress."]),
                    html.Li([
                        html.B("ΔCoVaR"),
                        " — how much the market's downside risk grows when "
                        "a specific bank moves from normal to distressed. "
                        "A more negative value means the bank contributes "
                        "more to system-wide tail risk."]),
                    html.Li([
                        html.B("CoVaR"),
                        " — the market's Value-at-Risk conditional on a "
                        "specific bank being in a particular state (median "
                        "or distressed); the building block for ΔCoVaR. "
                        "More negative means deeper potential market "
                        "losses conditional on bank stress."]),
                ], className="mb-0 ps-3 text-muted",
                   style={"fontSize": "0.82rem", "lineHeight": "1.45"}),
            ], style=_card),
        ], xs=12, className="mb-3"),
    ]),
], fluid=True, style={"backgroundColor": BG_PAGE})

srisk_layout = dbc.Container([
    # ── Tab heading with hover-popover info icon (same style as the
    # KPI ⓘ on the Start / Overview tabs). Popover body is populated by
    # populate_srisk_info_popover() at page load.
    html.Div([
        html.H5("SRISK",
                style={"fontWeight": "700", "color": TEXT_MAIN,
                       "fontSize": "1.05rem", "marginBottom": 0,
                       "display": "inline-block"}),
        html.Span(
            "i",
            id="srisk-info-icon",
            className="kpi-info-icon",
            style={
                "cursor": "help",
                "color": ACCENT_BLUE,
                "border": f"1.5px solid {ACCENT_BLUE}",
                "borderRadius": "50%",
                "width": "18px",
                "height": "18px",
                "display": "inline-flex",
                "alignItems": "center",
                "justifyContent": "center",
                "fontSize": "0.74rem",
                "fontStyle": "italic",
                "fontWeight": "700",
                "fontFamily": "Georgia, 'Times New Roman', serif",
                "marginLeft": "8px",
                "verticalAlign": "middle",
                "lineHeight": "1",
                "userSelect": "none",
                "transition": "background-color 0.15s, color 0.15s",
            },
        ),
        dbc.Popover(
            id="srisk-info-popover",
            target="srisk-info-icon",
            trigger="hover focus",
            placement="auto",
            className="kpi-info-popover",
        ),
    ], className="mt-3 mb-2"),

    # ── Stress parameters (global): k, d sliders + Download ──────────────────
    # k and d affect both the cross-sectional and time-series views below,
    # so they sit at the top of the tab as global what-if controls.
    dbc.Row([
        dbc.Col([
            html.Label(
                id="srisk-k-label",
                children="Prudential capital ratio  k = 8.0%",
                className="text-muted mb-1",
                style={"fontSize": "0.78rem", "fontWeight": "600"},
            ),
            dcc.Slider(
                id="srisk-k-slider",
                min=3, max=15, step=0.5,
                value=8,
                marks={i: f"{i}%" for i in range(3, 16, 2)},
                tooltip={"placement": "bottom", "always_visible": False},
                # Only fire callback on mouse-release — without this, every
                # intermediate drag position triggers a SRISK recompute
                # (10+ callbacks per drag), making the UI feel frozen.
                updatemode="mouseup",
                className="mb-0",
            ),
            html.Small(
                "k is the minimum equity to capital ratio a bank is "
                "assumed to need in stress. Higher k "
                "raises the implied shortfall, lower k shrinks it. "
                "Brownlees & Engle (2017) use 8%.",
                className="text-muted",
                style={"fontSize": "0.74rem"},
            ),
        ], xs=12, md=6, className="mt-2 mb-3"),
        dbc.Col([
            html.Label(
                id="srisk-d-label",
                children="LRMES decline threshold  d = 40.0%",
                className="text-muted mb-1",
                style={"fontSize": "0.78rem", "fontWeight": "600"},
            ),
            dcc.Slider(
                id="srisk-d-slider",
                min=10, max=60, step=2.5,
                value=40,
                marks={i: f"{i}%" for i in range(10, 61, 10)},
                tooltip={"placement": "bottom", "always_visible": False},
                # Same rationale as srisk-k-slider: defer the callback
                # until the user releases the mouse.
                updatemode="mouseup",
                className="mb-0",
            ),
            html.Small(
                "d is the market decline threshold used to define LRMES. Higher d"
                "increases SRISK, lower d lowers SRISK."
                "Brownlees & Engle (2017) use 40%.",
                className="text-muted",
                style={"fontSize": "0.74rem"},
            ),
        ], xs=12, md=6, className="mt-2 mb-3"),
    ]),

    html.Hr(className="my-2"),

    # ── Section 1: Cross-sectional view (bar + pie) ─────────────────────────
    # Normalisation toggle and download sit inline with the section header.
    dbc.Row([
        dbc.Col([
            html.H6("Cross-sectional SRISK",
                    className="mb-0 mt-2",
                    style={"fontWeight": "700", "fontSize": "0.92rem"}),
        ], xs=12, md=4),
        dbc.Col([
            dbc.RadioItems(
                id="srisk-norm",
                options=[
                    {"label": " USD bn",         "value": "abs"},
                    {"label": " % of aggregate", "value": "pct_agg"},
                    {"label": " % of mkt cap",   "value": "pct_mc"},
                ],
                value="abs",
                inline=True,
                inputStyle={"marginRight": "4px"},
                labelStyle={"marginRight": "12px", "fontSize": "0.82rem"},
                className="mt-2",
            ),
        ], xs=12, md=6),
        dbc.Col([
            dbc.Button("Download CSV", id="btn-download-srisk",
                       size="sm", color="primary", outline=True,
                       className="float-end mt-2"),
            dcc.Download(id="download-srisk"),
        ], xs=12, md=2),
    ], className="align-items-center"),
    dbc.Row([
        dbc.Col([
            html.P(
                "Cross-sectional SRISK for the 10 largest contibuting banks.",
                className="text-muted mb-1 mt-1",
                style={"fontSize": "0.78rem"},
            ),
            dcc.Graph(id="chart-srisk-bar"),
        ], xs=12, md=7, className="mb-3"),
        dbc.Col([
            html.P(
                "Share of aggregate SRISK across institutions. "
                "Highlights systemic risk concentration.",
                className="text-muted mb-1 mt-1",
                style={"fontSize": "0.78rem"},
            ),
            dcc.Graph(id="chart-srisk-pie"),
        ], xs=12, md=5, className="mb-3"),
    ]),
    html.Hr(className="my-2"),

    # ── Section 2: Time view (aggregate or stacked) ────────────────────────
    dbc.Row([
        dbc.Col([
            html.H6("SRISK over time",
                    className="mb-0 mt-2",
                    style={"fontWeight": "700", "fontSize": "0.92rem"}),
        ], xs=12, md=4),
        dbc.Col([
            dbc.RadioItems(
                id="srisk-ts-mode",
                options=[
                    {"label": " Aggregate",     "value": "aggregate"},
                    {"label": " Stacked by bank", "value": "stacked"},
                ],
                value="aggregate",
                inline=True,
                inputStyle={"marginRight": "4px"},
                labelStyle={"marginRight": "12px", "fontSize": "0.82rem"},
                className="mt-2",
            ),
        ], xs=12, md=8),
    ], className="align-items-center mb-1"),
    dbc.Row([
        dbc.Col([
            html.P(
                "SRISK through time for the selected banks. "
                "Aggregate view shows total system-level stress; stacked view breaks it down by institution to reveal shifting contributions.",
                className="text-muted mb-1",
                style={"fontSize": "0.78rem"},
            ),
            dcc.Graph(id="chart-srisk-ts"),
        ], xs=12, className="mb-3"),
    ]),
], fluid=True, style={"backgroundColor": BG_PAGE})

# ── Market Data + DCC Correlation (combined) tab ─────────────────────────────

market_dcc_layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            dbc.Checklist(
                id="market-dcc-crises",
                options=[{"label": " Shade crisis periods", "value": "show"}],
                value=["show"], switch=True,
                className="mb-2 mt-3",
            ),
        ], xs=12, md=4),
        dbc.Col([
            dbc.Button("Download DCC CSV", id="btn-download-dcc",
                       size="sm", color="primary", outline=True,
                       className="float-end mt-3"),
            dcc.Download(id="download-dcc"),
        ], xs=12, md=8),
    ]),

    html.Hr(className="my-2"),

    dbc.Row([
        dbc.Col([
            html.P(
                "Rebased prices, each bank's correlation with the market, and the "
                "cross-bank average correlation. Monitor correlations over time.",
                className="text-muted mb-1 mt-1",
                style={"fontSize": "0.78rem"},
            ),
            dcc.Graph(id="chart-market-dcc"),
        ], xs=12, md=7, className="mb-3"),
        dbc.Col([
            html.P(
                "How strongly each pair of banks' daily returns moves "
                "together. Warmer cells = higher co-movement = more "
                "contagion risk in a downturn.",
                className="text-muted mb-1 mt-1",
                style={"fontSize": "0.78rem"},
            ),
            dcc.Graph(id="chart-corr"),
        ], xs=12, md=5, className="mb-3"),
    ]),
], fluid=True, style={"backgroundColor": BG_PAGE})


# ── Methodology tab ──────────────────────────────────────────────────────────
# Each model / measure is rendered as a uniform card:
#   Title  →  LaTeX formula (MathJax)  →  1–2 sentence description
#         →  parameter bullets (1 bullet per symbol).
# Cards sit in a 2-column responsive grid so the tab scans like a reference
# sheet rather than a wall of prose.

def _methodology_card(title: str, formula_latex: str, description: str,
                      params: list[tuple[str, str]]) -> dbc.Card:
    """Render one methodology card (without dbc.Col wrapper).

    Used inside the Methodology tab's grid (wrapped in dbc.Col), as the
    body of every KPI hover-popover, and inline at the top of the SRISK tab.
    """
    params_md = "\n".join(f"- **{sym}** &mdash; {meaning}"
                          for sym, meaning in params)
    return dbc.Card(
        dbc.CardBody([
            html.H6(title, className="mb-2",
                    style={"fontWeight": "700",
                           "color": TEXT_MAIN,
                           "fontSize": "0.95rem"}),
            html.Div(
                dcc.Markdown(formula_latex, mathjax=True,
                             className="mb-0 methodology-formula",
                             style={"fontSize": "0.95rem"}),
                style={"backgroundColor": "#f8f9fa",
                       "border": f"1px solid {BORDER}",
                       "borderRadius": "6px",
                       "padding": "10px 14px",
                       "textAlign": "center",
                       "marginBottom": "10px",
                       "overflowX": "auto",
                       "maxWidth": "100%"},
            ),
            html.P(description, className="mb-2",
                   style={"fontSize": "0.83rem",
                          "lineHeight": "1.45",
                          "color": TEXT_MAIN}),
            html.Div("Parameters",
                     style={"fontSize": "0.7rem",
                            "letterSpacing": "0.06em",
                            "textTransform": "uppercase",
                            "color": ACCENT_BLUE,
                            "fontWeight": "700",
                            "marginBottom": "2px"}),
            dcc.Markdown(params_md, mathjax=True,
                         className="mb-0 methodology-params",
                         style={"fontSize": "0.8rem",
                                "lineHeight": "1.4"}),
        ], style={"padding": "14px 16px"}),
        style={"borderLeft": f"4px solid {ACCENT_BLUE}",
               "boxShadow": "0 1px 3px rgba(0,0,0,0.06)",
               "height": "100%"},
        className="mb-3",
    )


def _measure_card(title: str, formula_latex: str, description: str,
                  params: list[tuple[str, str]]) -> dbc.Col:
    """dbc.Col-wrapped methodology card, used in the Methodology grid."""
    return dbc.Col(
        _methodology_card(title, formula_latex, description, params),
        xs=12, lg=6,
    )


# Single source of truth for every methodology card.
_METHODOLOGY_DATA: dict[str, dict] = {
    "gjr_garch": dict(
        title="GJR-GARCH(1,1,1) — conditional volatility",
        formula_latex=(
            r"$$h_i(t) = \omega + \alpha_g\,\varepsilon_i(t-1)^2 "
            r"+ \gamma\,\varepsilon_i(t-1)^2\,\mathbb{1}_{\{\varepsilon_i(t-1)<0\}} "
            r"+ \beta_g\,h_i(t-1), \quad \sigma_i(t) = \sqrt{h_i(t)}$$"
        ),
        description=(
            "Estimates a bank's day-to-day stock volatility, giving extra "
            "weight to negative shocks since losses enlarge future swings "
            "more than equal-sized gains. Higher conditional volatility "
            "means the stock is expected to swing more sharply that day. "
            "An analogous independent fit produces $\\sigma_m(t)$ for the "
            "market."
        ),
        params=[
            (r"$\omega$",                       "long-run variance floor"),
            (r"$\alpha_g$",                     "ARCH effect — sensitivity to past shock size"),
            (r"$\gamma$",                       "leverage effect — extra amplification for negative shocks"),
            (r"$\beta_g$",                      "GARCH persistence of conditional variance"),
            (r"$\varepsilon_i(t)$",             "bank $i$'s return innovation (equals $r_i(t)$ under the zero-mean assumption)"),
            (r"$h_i(t),\ \sigma_i(t)$",         "bank $i$'s conditional variance and volatility"),
        ],
    ),
    "dcc": dict(
        title="DCC(1,1) — dynamic conditional correlation",
        formula_latex=(
            r"$$Q_i(t) = (1-a-b)\bar Q_i + a\,\varepsilon_i(t-1)\varepsilon_i(t-1)^{\top} "
            r"+ b\,Q_i(t-1), \quad "
            r"\rho_i(t) = \frac{Q_i(t)^{(1,2)}}{\sqrt{Q_i(t)^{(1,1)}\,Q_i(t)^{(2,2)}}}$$"
        ),
        description=(
            "Tracks how closely each bank's daily returns move with the "
            "market over time, letting the correlation drift and slowly "
            "revert to its long-run average. Higher correlation means the "
            "bank is moving more in lockstep with the market."
        ),
        params=[
            (r"$\bar Q_i$",         "unconditional covariance of standardised residuals for bank $i$"),
            (r"$\varepsilon_i(t)$", "standardised residual vector $[r_m(t)/\\sigma_m(t),\\ r_i(t)/\\sigma_i(t)]^{\\top}$"),
            (r"$a$",                "DCC shock sensitivity"),
            (r"$b$",                "DCC correlation persistence"),
            (r"$\rho_i(t)$",        "dynamic conditional correlation between bank $i$ and the market"),
        ],
    ),
    "mes": dict(
        title="MES — Marginal Expected Shortfall",
        formula_latex=(
            r"$$\mathrm{MES}_i(t) = -\min\!\left(\sigma_i(t)\,\rho_i(t)\,k_{1,i} "
            r"+ \sigma_i(t)\sqrt{1-\rho_i(t)^2}\,k_{2,i},\; 0\right)$$"
        ),
        description=(
            "Expected daily loss in a bank's stock when the market is "
            "having one of its worst α% of days. Higher MES means the "
            "bank gets hit harder when the market falls."
        ),
        params=[
            (r"$\sigma_i(t)$",      "bank $i$'s conditional volatility (GJR-GARCH)"),
            (r"$\rho_i(t)$",        "DCC conditional correlation between bank $i$ and the market"),
            (r"$k_{1,i},\ k_{2,i}$","kernel-weighted means of standardised market and idiosyncratic residuals for bank $i$"),
            (r"$\alpha$",           "market tail probability"),
        ],
    ),
    "lrmes": dict(
        title="LRMES — Long-Run Marginal Expected Shortfall",
        formula_latex=(
            r"$$\beta_i(t) = \rho_i(t)\,\frac{\sigma_i(t)}{\sigma_m(t)}, "
            r"\quad \mathrm{LRMES}_i(t) = 1 - (1-d)^{\beta_i(t)}$$"
        ),
        description=(
            "Expected drop in a bank's stock if the market falls by d "
            "over the medium term — the long-horizon counterpart to MES. "
            "Higher LRMES means a larger expected loss under sustained "
            "market stress."
        ),
        params=[
            (r"$\beta_i(t)$",       "DCC-implied conditional market beta for bank $i$"),
            (r"$\rho_i(t)$",        "DCC conditional correlation between bank $i$ and the market"),
            (r"$\sigma_i(t)$",      "bank $i$'s conditional volatility"),
            (r"$\sigma_m(t)$",      "market conditional volatility"),
            (r"$d$",                "market-decline threshold"),
        ],
    ),
    "covar_level": dict(
        title="CoVaR — Conditional Value-at-Risk (level)",
        formula_latex=(
            r"$$\mathrm{CoVaR}_i(t) = b_{0,i} + b_{1,i}\cdot\mathrm{VaR}_i(t) "
            r"+ \gamma_i'\,\mathbf{M}(t-1), "
            r"\quad \mathrm{VaR}_i(t) = \sigma_i(t)\cdot c_i$$"
        ),
        description=(
            "The market's Value-at-Risk conditional on a specific bank "
            "being in a particular state (median or distressed), with "
            "lagged macro state controls. More negative means deeper "
            "potential market losses conditional on bank stress."
        ),
        params=[
            (r"$b_{0,i},\ b_{1,i}$",        "quantile-regression intercept and slope for bank $i$ at level α"),
            (r"$\gamma_i$",                 "coefficient vector on the lagged macro state variables for bank $i$"),
            (r"$\mathbf{M}(t-1)$",          "state vector: VIX, 10Y yield, 3M T-bill, Fed Funds, BAA−10Y credit spread (lagged 1 day)"),
            (r"$\sigma_i(t)$",              "bank $i$'s conditional volatility (GJR-GARCH)"),
            (r"$c_i$",                      "$\\alpha$-quantile of $\\sigma_i(t)$-standardised returns of bank $i$"),
            (r"$\mathrm{VaR}_i(t)$",        "time-varying Value-at-Risk of bank $i$"),
            (r"$\alpha$",                   "tail probability — adjustable in the topbar"),
        ],
    ),
    "covar": dict(
        title="ΔCoVaR — Delta Conditional Value-at-Risk",
        formula_latex=(
            r"$$\Delta\mathrm{CoVaR}_i(t) = b_{1,i}\bigl(\mathrm{VaR}_i(t) - "
            r"\mathrm{Median}_i\bigr), \quad \mathrm{VaR}_i(t) = "
            r"\sigma_i(t)\cdot c_i$$"
        ),
        description=(
            "Measures how much the market's downside risk grows when a "
            "specific bank moves from normal to distressed. A more "
            "negative value means the bank contributes more to system-wide "
            "tail risk."
        ),
        params=[
            (r"$b_{1,i}$",            "quantile-regression slope for bank $i$ at level α"),
            (r"$\sigma_i(t)$",        "bank $i$'s conditional volatility"),
            (r"$c_i$",                "$\\alpha$-quantile of $\\sigma_i(t)$-standardised returns of bank $i$"),
            (r"$\mathrm{VaR}_i(t)$",  "time-varying Value-at-Risk of bank $i$"),
            (r"$\mathrm{Median}_i$",  "median of demeaned returns of bank $i$ (normal-state benchmark)"),
            (r"$\alpha$",             "tail probability — adjustable in the topbar"),
        ],
    ),
    "srisk": dict(
        title="SRISK — Capital Shortfall under Stress",
        formula_latex=(
            r"$$\mathrm{SRISK}_i(t) = \max\!\left(0,\; k\,\tilde D_i(t) - "
            r"(1-k)(1-\mathrm{LRMES}_i(t))\,W_i(t)\right)$$"
        ),
        description=(
            "Extra capital a bank would need to stay above its required "
            "cushion if the market crashed by d. Positive SRISK means the "
            "bank would likely need a bailout in a severe downturn."
        ),
        params=[
            (r"$k$",                       "prudential capital ratio"),
            (r"$\tilde D_i(t)$",           "bank $i$'s forward-rolled book liabilities (quarterly step function)"),
            (r"$W_i(t)$",                  "bank $i$'s market capitalisation"),
            (r"$\mathrm{LRMES}_i(t)$",     "expected equity loss fraction for bank $i$ under stress"),
            (r"$d$",                       "market-decline threshold"),
        ],
    ),
    "lvg": dict(
        title="LVG — Quasi-Leverage",
        formula_latex=(
            r"$$\mathrm{LVG}_i(t) = \frac{D_i(t) + W_i(t)}{W_i(t)}$$"
        ),
        description=(
            "A simple leverage ratio — total assets (liabilities + equity) "
            "divided by equity. Higher LVG means a thinner equity cushion "
            "against losses."
        ),
        params=[
            (r"$D_i(t)$",            "bank $i$'s book liabilities (quarterly filings, forward-filled daily)"),
            (r"$W_i(t)$",            "bank $i$'s market capitalisation"),
            (r"$D_i(t)+W_i(t)$",     "bank $i$'s quasi-total assets"),
        ],
    ),
}


def _methodology_card_for(key: str) -> dbc.Card:
    """Fresh methodology card (no dbc.Col wrapper) for the given measure
    key — used by the KPI hover popovers."""
    return _methodology_card(**_METHODOLOGY_DATA[key])


_VOL_CORR_CARDS = [_measure_card(**_METHODOLOGY_DATA[k])
                   for k in ("gjr_garch", "dcc")]

_MEASURE_CARDS = [_measure_card(**_METHODOLOGY_DATA[k])
                  for k in ("mes", "lrmes",
                            "covar_level", "covar",
                            "srisk", "lvg")]


# Per-source cache age — read once at module load from each cache file's
# mtime. The APScheduler job rewrites these files daily at 06:00 UTC.
def _cache_mtime(path: str) -> str:
    try:
        ts = os.path.getmtime(path)
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M UTC")
    except Exception:
        return "unknown"


_CACHE_AGE = {
    "prices":          _cache_mtime("cache/prices.parquet"),
    "balance_sheet":   _cache_mtime("cache/balance_sheet.json"),
    "state_variables": _cache_mtime("cache/state_variables.parquet"),
}


def _section_header(title: str) -> html.Div:
    """Section title + horizontal rule — used by every Methodology block."""
    return html.Div([
        html.H5(title, className="mb-1",
                style={"fontWeight": "700", "color": TEXT_MAIN,
                       "fontSize": "1.05rem"}),
        html.Hr(style={"marginTop": "4px", "marginBottom": "12px",
                       "borderTopWidth": "2px",
                       "borderTopColor": ACCENT_BLUE,
                       "opacity": 0.6}),
    ], className="mt-3")


methodology_layout = dbc.Container([
    # Intro
    html.Div([
        html.H4("Methodology",
                className="mb-2",
                style={"fontWeight": "800", "color": TEXT_MAIN}),
        html.P(
            "All measures are estimated from a bivariate "
            "DCC-GJR-GARCH(1,1,1) model with zero mean, where the market "
            f"index ({MARKET_NAME}) and each firm's daily log return form "
            "the two-variable system.",
            className="text-muted mb-0",
            style={"fontSize": "0.88rem", "lineHeight": "1.5",
                   "maxWidth": "900px"},
        ),
    ], className="mt-3 mb-2"),

    # ── 1. Data sources ──────────────────────────────────────────────────
    _section_header("Data sources"),
    html.Div([
        html.Ul([
            html.Li([html.B("Prices & balance sheets: "),
                     "Yahoo Finance (yfinance)."]),
            html.Li([html.B("Rates, yields, VIX: "),
                     "Yahoo Finance (ZQ=F, ^TNX, ^IRX, ^VIX)."]),
            html.Li([html.B("Credit spread (BAA10YM): "),
                     "FRED."]),
        ], className="mb-2", style={"fontSize": "0.85rem",
                                    "lineHeight": "1.55"}),
        html.P([html.B("Refresh frequency: "),
                "Daily at 06:00 UTC."],
               className="text-muted mb-1",
               style={"fontSize": "0.82rem", "lineHeight": "1.5"}),
        html.Div([
            html.Div(html.B("Last cache update"),
                     className="text-muted",
                     style={"fontSize": "0.78rem",
                            "letterSpacing": "0.05em",
                            "textTransform": "uppercase",
                            "marginBottom": "2px"}),
            html.Ul([
                html.Li([html.B("Prices: "),
                         html.Span(_CACHE_AGE["prices"],
                                   style={"fontFamily": "ui-monospace, "
                                                        "SFMono-Regular, "
                                                        "Menlo, monospace"})]),
                html.Li([html.B("Balance sheets: "),
                         html.Span(_CACHE_AGE["balance_sheet"],
                                   style={"fontFamily": "ui-monospace, "
                                                        "SFMono-Regular, "
                                                        "Menlo, monospace"})]),
                html.Li([html.B("State variables: "),
                         html.Span(_CACHE_AGE["state_variables"],
                                   style={"fontFamily": "ui-monospace, "
                                                        "SFMono-Regular, "
                                                        "Menlo, monospace"})]),
            ], className="mb-0 text-muted",
               style={"fontSize": "0.82rem", "lineHeight": "1.5"}),
        ]),
    ], style=_card),

    # ── 2. References ────────────────────────────────────────────────────
    _section_header("References"),
    html.Div([
        html.Ol([
            html.Li([
                "Acharya, V. V., Pedersen, L. H., Philippon, T., & Richardson, M. (2017). ",
                html.I("Measuring Systemic Risk. "),
                "Review of Financial Studies 30(1): 2–47."]),
            html.Li([
                "Adrian, T., & Brunnermeier, M. K. (2016). ",
                html.I("CoVaR. "),
                "American Economic Review 106(7): 1705–1741."]),
            html.Li([
                "Brownlees, C., & Engle, R. F. (2017). ",
                html.I("SRISK: A Conditional Capital Shortfall Measure "
                       "of Systemic Risk. "),
                "Review of Financial Studies 30(1): 48–79."]),
            html.Li([
                "Belluzzo, T. (2020). ",
                html.A("github.com/TommasoBelluzzo/SystemicRisk",
                       href="https://github.com/TommasoBelluzzo/SystemicRisk",
                       target="_blank"),
                " — MATLAB reference implementation."]),
            html.Li([
                "Court, B., Gisiger, M., & Mezabrovschi, S. (2025). ",
                html.I("Systemic Risk Analyzer (focus banks)."),
                " ZHAW Master's thesis."]),
        ], className="mb-0", style={"fontSize": "0.85rem",
                                    "lineHeight": "1.55"}),
    ], style=_card),

    # ── 3. Risk measures ─────────────────────────────────────────────────
    _section_header("Risk measures"),
    # Sub-section: Volatility & correlation models
    html.H6("Volatility & correlation models",
            className="mb-2",
            style={"fontWeight": "700", "color": TEXT_MAIN,
                   "fontSize": "0.95rem"}),
    dbc.Row(_VOL_CORR_CARDS, className="g-3"),

    # Sub-section: Systemic risk measures
    html.H6("Systemic risk measures",
            className="mb-2 mt-3",
            style={"fontWeight": "700", "color": TEXT_MAIN,
                   "fontSize": "0.95rem"}),
    dbc.Row(_MEASURE_CARDS, className="g-3 mb-3"),

], fluid=True, style={"backgroundColor": BG_PAGE})

# ── Main layout ───────────────────────────────────────────────────────────────

_loading_bar = html.Div(
    dbc.Progress(
        value=100, animated=True, striped=False, color="primary",
        style={"height": "3px", "borderRadius": 0},
        className="p-0",
    ),
    style={
        "position": "fixed", "top": 0, "left": 0, "right": 0,
        "zIndex": 9999, "margin": 0, "padding": 0,
    },
)

app.layout = html.Div([
    header,
    controls_section,
    dcc.Store(id="refresh-store",      data=0),
    dcc.Store(id="alpha-store",        data=0.05),
    dcc.Store(id="custom-banks-store", data={}),
    # Polls the background refresh worker continuously.  Always enabled so
    # the daily APScheduler-triggered refresh is picked up without any user
    # interaction.  2 s feels live during an in-flight refresh while keeping
    # idle traffic low.
    dcc.Interval(id="refresh-interval", interval=2000,
                 disabled=False, n_intervals=0),
    dcc.Loading(
        id="loading-main",
        custom_spinner=_loading_bar,
        overlay_style={"opacity": 0, "backgroundColor": "transparent"},
        children=dbc.Tabs([
            dbc.Tab(start_layout,         label="Start",                tab_id="tab-start"),
            dbc.Tab(overview_layout,    label="Overview",        tab_id="tab-overview"),
            dbc.Tab(timeseries_layout,    label="Time Series",          tab_id="tab-ts"),
            dbc.Tab(srisk_layout,         label="SRISK",               tab_id="tab-srisk"),
            dbc.Tab(market_dcc_layout,    label="Market & Correlation", tab_id="tab-market"),
            dbc.Tab(methodology_layout,   label="Methodology",          tab_id="tab-methodology"),
        ], id="main-tabs", active_tab="tab-start",
           style={"paddingLeft": "1rem", "backgroundColor": "#f8f9fa",
                  "borderBottom": f"1px solid {BORDER}"}),
    ),
], style={"backgroundColor": BG_PAGE, "minHeight": "100vh"})


# ── Callbacks ─────────────────────────────────────────────────────────────────

# ── Filter bar collapse toggle ───────────────────────────────────────────────

@app.callback(
    Output("filters-collapse", "is_open"),
    Output("filters-caret",    "style"),
    Input("btn-toggle-filters", "n_clicks"),
    State("filters-collapse",   "is_open"),
    prevent_initial_call=True,
)
def toggle_filters(_n_clicks, is_open):
    new_open = not is_open
    caret_style = {
        "display": "inline-block",
        "transition": "transform 0.15s",
        "marginRight": "6px",
        # Rotate ▼ → ▶ when collapsed (90° counter-clockwise so the arrow
        # points right, signalling the panel can be expanded again).
        "transform": "rotate(0deg)" if new_open else "rotate(-90deg)",
    }
    return new_open, caret_style


# ── SRISK methodology popover (populated once on page load) ────────────────
# The popover body is the same methodology card shown on hover from every
# KPI's ⓘ icon. It's built lazily here because _methodology_card_for /
# _METHODOLOGY_DATA are defined further down the file, after srisk_layout.

@app.callback(
    Output("srisk-info-popover", "children"),
    Input("main-tabs",           "active_tab"),
)
def populate_srisk_info_popover(_active_tab):
    return dbc.PopoverBody(
        _methodology_card_for("srisk"),
        style={"padding": 0, "backgroundColor": "transparent"},
    )


# ── Time-range presets ─────────────────────────────────────────────────────────

@app.callback(
    Output("date-range", "start_date", allow_duplicate=True),
    Output("date-range", "end_date",   allow_duplicate=True),
    [Input(pid, "n_clicks") for pid, _, _ in _PRESET_BUTTONS],
    prevent_initial_call=True,
)
def apply_preset_range(*_clicks):
    from dash import callback_context
    if not callback_context.triggered:
        return no_update, no_update
    triggered = callback_context.triggered[0]["prop_id"].split(".")[0]
    years_map = {pid: yrs for pid, _, yrs in _PRESET_BUTTONS}
    if triggered not in years_map:
        return no_update, no_update
    yrs = years_map[triggered]
    end = DATE_MAX
    if yrs is None:
        start = DATE_MIN
    else:
        target = pd.Timestamp(end) - pd.DateOffset(years=int(yrs)) if yrs >= 1 \
                 else pd.Timestamp(end) - pd.DateOffset(months=6)
        start = max(target.date(), DATE_MIN)
    return start, end


# ── Reset to defaults ─────────────────────────────────────────────────────────

@app.callback(
    Output("date-range",      "start_date", allow_duplicate=True),
    Output("date-range",      "end_date",   allow_duplicate=True),
    Output("alpha-select",    "value",      allow_duplicate=True),
    Output("srisk-k-slider",  "value",      allow_duplicate=True),
    Output("srisk-d-slider",  "value",      allow_duplicate=True),
    Output("bank-select",     "value",      allow_duplicate=True),
    Input("btn-reset-defaults", "n_clicks"),
    prevent_initial_call=True,
)
def reset_to_defaults(n_clicks):
    """Restore controls to their default state.

    Defaults: full date range, α = 5%, k = 8%, d = 40%, all visible banks selected.
    """
    if not n_clicks:
        return no_update, no_update, no_update, no_update, no_update, no_update
    default_banks = [t for t in ALL_BANKS if t in RETURNS.columns]
    return DATE_DEF_START, DATE_MAX, 0.05, 8, 40, default_banks


# ── Bank dropdown ──────────────────────────────────────────────────────────────

@app.callback(
    Output("bank-select", "options"),
    Output("bank-select", "value", allow_duplicate=True),
    Input("btn-all",             "n_clicks"),
    Input("btn-none",            "n_clicks"),
    Input("custom-banks-store",  "data"),
    State("bank-select",         "value"),
    prevent_initial_call="initial_duplicate",
)
def update_bank_options(_all, _none, custom_banks, current_values):
    from dash import callback_context
    triggered = (callback_context.triggered[0]["prop_id"]
                 if callback_context.triggered else "")

    all_banks = {**ALL_BANKS, **(custom_banks or {})}

    def _origin_label(ticker: str) -> str:
        # Base banks are all US now; custom-added tickers are tagged "Custom"
        return "Custom" if ticker not in BANK_COUNTRY else COUNTRY_LABELS.get(
            BANK_COUNTRY[ticker], BANK_COUNTRY[ticker]
        )

    options = [
        {
            "label": f"{name} ({ticker}) \u00b7 {_origin_label(ticker)}",
            "value": ticker,
        }
        for ticker, name in all_banks.items()
        if ticker in RETURNS.columns
    ]
    valid = [o["value"] for o in options]

    if "btn-none" in triggered:
        return options, []

    if "custom-banks-store" in triggered:
        # Keep current selection (intersected with the new visible set) and
        # auto-select any newly added tickers.
        current = [t for t in (current_values or []) if t in valid]
        new_tickers = [t for t in (custom_banks or {}) if t not in current]
        return options, current + new_tickers

    # Initial load or All: select everything visible
    return options, valid


# ── Add custom bank ────────────────────────────────────────────────────────────

@app.callback(
    Output("custom-banks-store", "data"),
    Output("add-bank-status",    "children"),
    Input("btn-add-bank",        "n_clicks"),
    State("custom-ticker-input", "value"),
    State("custom-banks-store",  "data"),
    prevent_initial_call=True,
)
def add_custom_bank(n_clicks, ticker_input, custom_banks):
    global PRICES, RETURNS, MC_TS, BS, LIAB_TS, SEP_ACCT_TS, LB_DAILY, LBR_DAILY, STATE_VARS, MEASURES, ALL_BANKS, LEVERAGE

    ticker = (ticker_input or "").strip().upper()
    if not ticker:
        return no_update, dbc.Alert(
            "Please enter a ticker symbol.", color="warning",
            className="py-1 mb-0", style={"fontSize": "0.8rem"})

    if ticker in RETURNS.columns:
        return no_update, dbc.Alert(
            f"{ticker} is already in the model.", color="info",
            className="py-1 mb-0", style={"fontSize": "0.8rem"})

    print(f"\n[Add Bank] Fetching data for {ticker} ...")
    bank_data = D.fetch_single_bank(ticker)

    if bank_data is None:
        return no_update, dbc.Alert(
            f"Could not fetch data for '{ticker}'. Check the ticker symbol.",
            color="danger", className="py-1 mb-0", style={"fontSize": "0.8rem"})

    name = bank_data["name"]

    # ── Prices & returns ──────────────────────────────────────────────────────
    new_prices  = bank_data["prices"].reindex(PRICES.index)
    PRICES[ticker] = new_prices
    new_returns = D.compute_returns(new_prices.to_frame()).iloc[:, 0]
    RETURNS[ticker] = new_returns

    # ── Balance sheet ─────────────────────────────────────────────────────────
    BS[ticker] = bank_data["balance_sheet"]

    # ── Market cap time series ────────────────────────────────────────────────
    new_mc = D.build_market_cap_series(PRICES[[ticker]], {ticker: bank_data["balance_sheet"]})
    if ticker in new_mc.columns:
        MC_TS[ticker] = new_mc[ticker]

    # ── Liabilities ───────────────────────────────────────────────────────────
    if bank_data["liab_ts"] is not None:
        liab_s = bank_data["liab_ts"]
        liab_s.name = ticker
        new_liab_df = pd.DataFrame({ticker: liab_s})
        new_sa_df = pd.DataFrame()
        if bank_data.get("separate_accounts_ts") is not None:
            sa_s = bank_data["separate_accounts_ts"]
            sa_s.name = ticker
            new_sa_df = pd.DataFrame({ticker: sa_s})
            SEP_ACCT_TS = (SEP_ACCT_TS.reindex(SEP_ACCT_TS.index.union(sa_s.index))
                                      .sort_index())
            SEP_ACCT_TS[ticker] = sa_s.reindex(SEP_ACCT_TS.index)
        # Merge into LIAB_TS
        LIAB_TS = (LIAB_TS.reindex(LIAB_TS.index.union(liab_s.index))
                           .sort_index())
        LIAB_TS[ticker] = liab_s.reindex(LIAB_TS.index)
        # Build daily series for this bank
        lb_one  = D.build_lb_daily(new_liab_df,  PRICES[[ticker]], new_sa_df)
        lbr_one = D.build_lbr_daily(new_liab_df, PRICES[[ticker]], new_sa_df)
        if ticker in lb_one.columns:
            LB_DAILY[ticker]  = lb_one[ticker]
        if ticker in lbr_one.columns:
            LBR_DAILY[ticker] = lbr_one[ticker]

    # ── DCC-GJR-GARCH + measures ──────────────────────────────────────────────
    print(f"  Computing DCC-GJR-GARCH for {name} ...")
    mkt_ret  = RETURNS[MARKET_NAME]
    bank_ret = RETURNS[ticker].dropna()
    idx      = bank_ret.index.intersection(mkt_ret.index)
    rm, rf   = mkt_ret.reindex(idx).values, bank_ret.reindex(idx).values

    sm, sf, rho = M.dcc_gjrgarch(rm, rf, market_label=MARKET_NAME, firm_label=ticker)

    # Persist DCC outputs to cache so recompute_for_alpha can find this bank
    nmin    = min(len(idx), len(sm))
    idx_out = idx[-nmin:]
    for dcc_key, arr in [("dcc_sm", sm[-nmin:]),
                          ("dcc_sf", sf[-nmin:]),
                          ("dcc_rho", rho[-nmin:])]:
        try:
            M.update_dcc_cache_column(dcc_key, ticker, arr, idx_out)
        except Exception as _e:
            print(f"  Warning: DCC cache update failed for {dcc_key}: {_e}")

    alpha = 0.05
    state_vars = STATE_VARS.reindex(idx_out) if not STATE_VARS.empty else None
    covar_s, dcovar_s = M.compute_covar_dcovar(bank_ret, mkt_ret, sm, sf, rho, state_vars=state_vars, alpha=alpha)
    mes_s,   lrmes_s  = M.compute_mes_lrmes(   bank_ret, mkt_ret, sm, sf, rho, alpha)

    cp_ts  = MC_TS.get(ticker)
    lb_ts  = LB_DAILY.get(ticker)  if LB_DAILY  is not None else None
    lbr_ts = LBR_DAILY.get(ticker) if LBR_DAILY is not None else None

    ses_s   = (M.compute_ses(lb_ts, cp_ts)
               if cp_ts is not None and lb_ts is not None
               else pd.Series(np.nan, index=mes_s.index))
    srisk_s = (M.compute_srisk(lrmes_s, lbr_ts, cp_ts)
               if cp_ts is not None and lbr_ts is not None
               else pd.Series(np.nan, index=mes_s.index))

    MEASURES["mes"][ticker]         = mes_s
    if "lrmes" not in MEASURES:
        MEASURES["lrmes"] = pd.DataFrame(index=mes_s.index)
    MEASURES["lrmes"][ticker]       = lrmes_s
    MEASURES["ses"][ticker]         = ses_s
    MEASURES["covar"][ticker]       = covar_s
    MEASURES["delta_covar"][ticker] = dcovar_s
    MEASURES["srisk"][ticker]       = srisk_s

    # ── Rebuild leverage for all banks including the new one ──────────────────
    LEVERAGE = _build_leverage(LB_DAILY, MC_TS)

    # ── Update display name lookup ────────────────────────────────────────────
    ALL_BANKS[ticker] = name

    print(f"[Add Bank] Done: {name} ({ticker})")
    new_custom = {**(custom_banks or {}), ticker: name}
    msg = dbc.Alert(
        f"Added {name} ({ticker})", color="success", dismissable=True,
        className="py-1 mb-0", style={"fontSize": "0.8rem"})
    return new_custom, msg


# ── Refresh (background worker + progress polling) ────────────────────────────
# Heavy refresh work runs in a daemon thread so the Dash event loop stays
# responsive. A module-level state dict (protected by a lock) tracks progress;
# a dcc.Interval polls it and, when the worker finishes, bumps refresh-store
# to force every tab callback to re-render with the fresh globals.

_REFRESH_STATE: dict = {
    "running":      False,
    "step":         "Idle",
    "progress":     0.0,     # 0.0 – 1.0
    "error":        None,
    "completed_at": None,    # timestamp of last successful refresh
    "seq":          0,       # incremented on every completion (drives UI redraw)
}
_REFRESH_LOCK = threading.Lock()


def _set_refresh_state(**kwargs) -> None:
    with _REFRESH_LOCK:
        _REFRESH_STATE.update(kwargs)


def _get_refresh_state() -> dict:
    with _REFRESH_LOCK:
        return dict(_REFRESH_STATE)


def _add_custom_banks_inline(custom_banks: dict | None) -> None:
    """Re-add any user-added custom banks after a full refresh wiped globals."""
    global PRICES, RETURNS, BS, MC_TS, LIAB_TS, SEP_ACCT_TS, LB_DAILY, LBR_DAILY, MEASURES
    for ticker, name in (custom_banks or {}).items():
        if ticker in RETURNS.columns:
            continue
        bank_data = D.fetch_single_bank(ticker)
        if bank_data is None:
            continue
        PRICES[ticker]   = bank_data["prices"].reindex(PRICES.index)
        RETURNS[ticker]  = D.compute_returns(PRICES[[ticker]]).iloc[:, 0]
        BS[ticker]       = bank_data["balance_sheet"]
        ALL_BANKS[ticker] = name
        new_mc = D.build_market_cap_series(PRICES[[ticker]], {ticker: BS[ticker]})
        if ticker in new_mc.columns:
            MC_TS[ticker] = new_mc[ticker]
        if bank_data["liab_ts"] is not None:
            liab_s = bank_data["liab_ts"]
            LIAB_TS[ticker] = liab_s.reindex(LIAB_TS.index)
            sa_df = pd.DataFrame()
            if bank_data.get("separate_accounts_ts") is not None:
                sa_s = bank_data["separate_accounts_ts"]
                SEP_ACCT_TS[ticker] = sa_s.reindex(SEP_ACCT_TS.index)
                sa_df = pd.DataFrame({ticker: sa_s})
            lb_one  = D.build_lb_daily( pd.DataFrame({ticker: liab_s}), PRICES[[ticker]], sa_df)
            lbr_one = D.build_lbr_daily(pd.DataFrame({ticker: liab_s}), PRICES[[ticker]], sa_df)
            if ticker in lb_one.columns:
                LB_DAILY[ticker]  = lb_one[ticker]
            if ticker in lbr_one.columns:
                LBR_DAILY[ticker] = lbr_one[ticker]
        # Compute systemic measures for the custom bank
        mkt_ret  = RETURNS[MARKET_NAME]
        bank_ret = RETURNS[ticker].dropna()
        idx      = bank_ret.index.intersection(mkt_ret.index)
        sm, sf, rho = M.dcc_gjrgarch(
            mkt_ret.reindex(idx).values,
            bank_ret.reindex(idx).values,
            market_label=MARKET_NAME,
            firm_label=ticker,
        )
        covar_s, dcovar_s = M.compute_covar_dcovar(bank_ret, mkt_ret, sm, sf, rho, state_vars=STATE_VARS)
        mes_s,   lrmes_s  = M.compute_mes_lrmes(   bank_ret, mkt_ret, sm, sf, rho)
        cp_ts  = MC_TS.get(ticker)
        lb_ts  = LB_DAILY.get(ticker)  if LB_DAILY  is not None else None
        lbr_ts = LBR_DAILY.get(ticker) if LBR_DAILY is not None else None
        MEASURES["mes"][ticker]         = mes_s
        if "lrmes" not in MEASURES:
            MEASURES["lrmes"] = pd.DataFrame(index=mes_s.index)
        MEASURES["lrmes"][ticker]       = lrmes_s
        MEASURES["ses"][ticker]         = (M.compute_ses(lb_ts, cp_ts)
                                           if cp_ts is not None and lb_ts is not None
                                           else pd.Series(np.nan, index=mes_s.index))
        MEASURES["covar"][ticker]       = covar_s
        MEASURES["delta_covar"][ticker] = dcovar_s
        MEASURES["srisk"][ticker]       = (M.compute_srisk(lrmes_s, lbr_ts, cp_ts)
                                           if cp_ts is not None and lbr_ts is not None
                                           else pd.Series(np.nan, index=mes_s.index))


def _run_refresh_work(custom_banks: dict | None) -> None:
    """Perform heavy refresh off the main thread; publish progress via _REFRESH_STATE."""
    global PRICES, RETURNS, BS, MC_TS, LIAB_TS, SEP_ACCT_TS, LB_DAILY, LBR_DAILY
    global STATE_VARS, MEASURES, LAST_UPDATED, LEVERAGE, DCOVAR_BY_ALPHA

    try:
        print("\n[Refresh] Starting background refresh ...")
        _set_refresh_state(running=True, error=None,
                           progress=0.03, step="Fetching prices")
        PRICES = D.get_prices(force_refresh=True)

        _set_refresh_state(progress=0.12, step="Computing returns")
        RETURNS = D.compute_returns(PRICES)

        _set_refresh_state(progress=0.18, step="Fetching balance sheets")
        BS = D.get_balance_sheet(force_refresh=True)

        _set_refresh_state(progress=0.28, step="Fetching liabilities")
        LIAB_TS     = D.get_liabilities_ts(force_refresh=True)
        SEP_ACCT_TS = D.get_separate_accounts_ts(force_refresh=True)
        LB_DAILY    = D.build_lb_daily(LIAB_TS, PRICES, SEP_ACCT_TS)
        LBR_DAILY   = D.build_lbr_daily(LIAB_TS, PRICES, SEP_ACCT_TS)

        _set_refresh_state(progress=0.42, step="Fetching state variables")
        STATE_VARS = D.get_state_variables(PRICES, force_refresh=True)

        _set_refresh_state(progress=0.48, step="Building market-cap series")
        MC_TS = D.build_market_cap_series(PRICES, BS)

        _set_refresh_state(progress=0.55, step="Fitting DCC-GJR-GARCH per bank")
        MEASURES = M.compute_all(
            RETURNS, MC_TS, LB_DAILY, LBR_DAILY, BS,
            state_vars=STATE_VARS, force_refresh=True,
        )

        _set_refresh_state(progress=0.88, step="Re-adding custom banks")
        _add_custom_banks_inline(custom_banks)

        _set_refresh_state(progress=0.94, step="Precomputing α-grid for ΔCoVaR")
        DCOVAR_BY_ALPHA = _build_dcovar_grid()

        LEVERAGE = _build_leverage(LB_DAILY, MC_TS)
        LAST_UPDATED = _now_utc_str()
        _write_last_refresh(LAST_UPDATED)

        # Bump seq so the poll callback can fire refresh-store exactly once.
        with _REFRESH_LOCK:
            _REFRESH_STATE.update(
                running=False, progress=1.0, step="Done",
                completed_at=LAST_UPDATED, error=None,
                seq=_REFRESH_STATE["seq"] + 1,
            )
        print(f"[Refresh] Done ({LAST_UPDATED}).")
    except Exception as exc:
        print(f"[Refresh] Failed: {type(exc).__name__}: {exc}")
        _set_refresh_state(
            running=False, step="Failed",
            error=f"{type(exc).__name__}: {exc}",
        )


# ── Daily refresh scheduler ──────────────────────────────────────────────────
# Runs once per day at 06:00 UTC (~01:00 US/Eastern).  Replaces the manual
# "Refresh" button.  APScheduler's BackgroundScheduler uses a worker thread
# so it doesn't block Dash callbacks.
#
# Multi-process safety: APScheduler must not run in more than one process,
# or daily jobs would fire concurrently.  Procfile pins `--workers=1`; the
# DISABLE_SCHEDULER env var is an additional belt-and-braces guard for
# local debugging or when running a separate worker container.

_SCHEDULER_STARTED = False


def _start_scheduler() -> None:
    global _SCHEDULER_STARTED
    if _SCHEDULER_STARTED or os.environ.get("DISABLE_SCHEDULER") == "1":
        return
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        from apscheduler.triggers.cron import CronTrigger
    except ImportError:
        print("[Scheduler] APScheduler not installed — daily refresh disabled. "
              "Add `APScheduler` to requirements.txt to enable.")
        return

    sched = BackgroundScheduler(timezone="UTC")
    sched.add_job(
        # Custom banks are session state; the daily cron refreshes the
        # default universe only.  Users with custom banks can still trigger
        # bank-specific recomputes via the Add-bank flow which calls into
        # the same per-ticker pipeline.
        func=lambda: _run_refresh_work({}),
        trigger=CronTrigger(hour=6, minute=0, timezone="UTC"),
        id="daily-refresh",
        misfire_grace_time=3600,  # tolerate up-to-1h delay (e.g., dyno wake)
        coalesce=True,            # collapse missed runs into one
        max_instances=1,          # never overlap with an in-flight refresh
    )
    sched.start()
    _SCHEDULER_STARTED = True
    print("[Scheduler] Daily refresh scheduled at 06:00 UTC.")


_start_scheduler()


def _refresh_progress_view(state: dict):
    """Render the small progress indicator shown next to the navbar timestamp.
    Visible only while the daily refresh job is in flight; idle and error
    states render nothing or a one-line error message respectively."""
    if state.get("running"):
        pct = int(round(state.get("progress", 0.0) * 100))
        return html.Div([
            dbc.Progress(
                value=pct, animated=True, striped=True, color="primary",
                style={"height": "6px", "width": "140px"},
                className="mb-1",
            ),
            html.Small(
                f"{state.get('step', '')} ({pct}%)",
                className="text-muted",
                style={"fontSize": "0.72rem"},
            ),
        ], style={"display": "inline-block", "minWidth": "140px"})
    if state.get("error"):
        return html.Small(
            f"Refresh failed: {state['error']}",
            className="text-danger",
            style={"fontSize": "0.75rem"},
        )
    return ""  # idle — nothing to show


# Track which scheduler-completed refresh we've already broadcast to clients
# so the always-on poll callback bumps refresh-store exactly once per refresh.
_LAST_BROADCAST_SEQ = 0


@app.callback(
    Output("refresh-progress", "children"),
    Output("refresh-store",    "data",     allow_duplicate=True),
    Output("updated-ts",       "children", allow_duplicate=True),
    Input("refresh-interval",  "n_intervals"),
    State("refresh-store",     "data"),
    prevent_initial_call=True,
)
def poll_refresh(_n, current):
    """Poll the refresh worker; while running, show progress.  When the
    background scheduler completes a new refresh (state['seq'] advances),
    bump refresh-store exactly once so every chart re-renders with fresh
    data, and update the navbar timestamp."""
    global _LAST_BROADCAST_SEQ
    state = _get_refresh_state()
    view  = _refresh_progress_view(state)
    seq   = int(state.get("seq", 0))

    if state["running"]:
        return view, no_update, no_update

    # Idle: only emit when a NEW completed seq is present.
    if seq > _LAST_BROADCAST_SEQ and state.get("completed_at"):
        _LAST_BROADCAST_SEQ = seq
        new_counter = (current or 0) + 1
        return view, new_counter, state["completed_at"]

    # No new refresh to broadcast — quiet.
    return view, no_update, no_update


# ── Alpha recompute ────────────────────────────────────────────────────────────

@app.callback(
    Output("alpha-store", "data"),
    Output("alpha-label", "children"),
    Input("alpha-select", "value"),
    prevent_initial_call=True,
)
def update_alpha(alpha):
    """Update MES + ΔCoVaR for a new α; everything else is α-invariant.

    Performance-critical path. The previous implementation called
    M.recompute_for_alpha which re-ran the IRLS quantile regression for
    ΔCoVaR (only to be overwritten by the precomputed grid value), plus
    the full LRMES / SES / SRISK pipeline (all α-invariant). On the free
    Render tier this took 10–30 s per α click.

    The optimised path:
      1. ΔCoVaR — swap in the frame from DCOVAR_BY_ALPHA (precomputed at
         startup for the full grid).
      2. MES   — recompute from cached DCC outputs (ρ, σ_f, σ_m). This is
         the only computation that genuinely depends on α: the kernel
         weights k₁, k₂ are reweighted to the new tail probability.
      3. LRMES, SES, SRISK — left untouched. They're α-invariant.
    """
    global MEASURES
    alpha_pct = alpha * 100.0
    print(f"\n[α] Updating MES & ΔCoVaR for α={alpha:.3f}")

    # 1) ΔCoVaR: O(1) frame swap from the precomputed grid.
    dcovar_df = DCOVAR_BY_ALPHA.get(round(alpha, 4))
    if dcovar_df is not None and not dcovar_df.empty:
        # Preserve the existing index so dependent slices align.
        base = MEASURES.get("delta_covar")
        if base is not None and not base.empty:
            updated = base.copy()
            for col in dcovar_df.columns:
                updated[col] = dcovar_df[col].reindex(updated.index)
            MEASURES["delta_covar"] = updated
        else:
            MEASURES["delta_covar"] = dcovar_df

    # 2) MES: recompute per-bank from cached DCC components.
    cached = M._load_dcc_cache()
    if cached is not None:
        dcc_sm, dcc_sf, dcc_rho = cached
        mkt_ret = (RETURNS[MARKET_NAME] if MARKET_NAME in RETURNS.columns
                   else RETURNS[RETURNS.columns.intersection(
                       ["SMI", "S&P 500", "Market"])].iloc[:, 0])
        bank_cols = [c for c in RETURNS.columns
                     if c in MEASURES.get("mes", pd.DataFrame()).columns]
        new_mes_frame = MEASURES["mes"].copy()
        for ticker in bank_cols:
            if ticker not in dcc_sm.columns:
                continue
            try:
                bank_ret = RETURNS[ticker].dropna()
                sm  = dcc_sm[ticker].dropna().values
                sf  = dcc_sf[ticker].dropna().values
                rho = dcc_rho[ticker].dropna().values
                if len(sm) == 0 or len(sf) == 0 or len(rho) == 0:
                    continue
                mes_s, _ = M.compute_mes_lrmes(
                    bank_ret, mkt_ret, sm, sf, rho, alpha=alpha
                )
                new_mes_frame[ticker] = mes_s.reindex(new_mes_frame.index)
            except Exception as exc:
                print(f"    [α] MES recompute failed for {ticker}: "
                      f"{type(exc).__name__}: {exc}")
        MEASURES["mes"] = new_mes_frame

    print("[α] Done.")
    return alpha, (
        f"Critical Value α = {alpha_pct:.1f}% (worst {alpha_pct:.1f}% of market days)"
    )


# ── Start-tab bookmark navigation ─────────────────────────────────────────────
# Clicking any of the 5 tab cards on the Start tab activates the matching
# tab in the main Tabs component. Pattern-matched ID — single callback
# handles every card.

@app.callback(
    Output("main-tabs", "active_tab", allow_duplicate=True),
    Input({"type": "start-tab-link", "tab": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def jump_to_tab_from_start(_clicks):
    from dash import callback_context
    if not callback_context.triggered:
        return no_update
    # Skip the spurious "all clicks = 0" fire that can happen on first render.
    if not any(c for c in (_clicks or []) if c):
        return no_update
    triggered = callback_context.triggered_id
    if not isinstance(triggered, dict):
        return no_update
    return triggered.get("tab", no_update)


# ── Overview tab ──────────────────────────────────────────────────────────────

@app.callback(
    Output("kpi-mes",          "children"),
    Output("kpi-lrmes",        "children"),
    Output("kpi-covar",        "children"),
    Output("kpi-srisk",        "children"),
    Output("kpi-leverage",     "children"),
    Output("kpi-start-mes",    "children"),
    Output("kpi-start-covar",  "children"),
    Output("kpi-start-srisk",  "children"),
    Output("chart-mes-rank",   "figure"),
    Output("chart-srisk-rank", "figure"),
    Output("chart-covar-rank", "figure"),
    Output("chart-mes-dw",     "figure"),
    Output("chart-srisk-dw",   "figure"),
    Output("chart-covar-dw",   "figure"),
    Output("risk-table",       "children"),
    Input("date-range",        "start_date"),
    Input("date-range",        "end_date"),
    Input("bank-select",       "value"),
    Input("refresh-store",     "data"),
    Input("alpha-store",       "data"),
    Input("risk-summary-date", "date"),
)
def update_overview(start, end, tickers, _refresh, _alpha, snap_date):
    tickers = tickers or []
    mes_df    = _slice(MEASURES["mes"],         start, end, tickers)
    lrmes_df  = _slice(MEASURES.get("lrmes", pd.DataFrame()), start, end, tickers)
    dcovar_df = _slice(MEASURES["delta_covar"], start, end, tickers)
    srisk_df  = _slice(MEASURES["srisk"],       start, end, tickers)
    lvg_df    = _slice(LEVERAGE,                start, end, tickers) if not LEVERAGE.empty else pd.DataFrame()

    latest_mes   = _latest_row(mes_df)
    latest_lrmes = _latest_row(lrmes_df)
    latest_covar = _latest_row(dcovar_df)
    latest_srisk = _latest_row(srisk_df)
    latest_lvg   = _latest_row(lvg_df)

    agg_mes   = latest_mes.mean()
    agg_lrmes = latest_lrmes.mean() if not latest_lrmes.empty else float("nan")
    agg_covar = latest_covar.mean()
    agg_srisk = latest_srisk.sum()
    agg_lvg   = latest_lvg.mean() if not latest_lvg.empty else float("nan")

    # ── 7-day delta for the aggregate KPI values ──────────────────────────────
    # Compares the latest cross-sectional aggregate (mean / sum) to the value
    # 7 calendar days earlier, so KPI cards reflect a recent trend.
    #
    # NOTE: We deliberately do NOT use DataFrame.asof here. DataFrame.asof
    # walks backward until it finds a row with ALL columns non-NaN, which
    # collapses to all-NaN whenever any single column is entirely empty
    # (e.g. a freshly added bank without balance-sheet data yet). Series.asof
    # (per-column) ignores NaN values column-wise and returns the most
    # recent non-NaN observation ≤ prev_dt for each column independently.
    def _agg_7d_delta(df: pd.DataFrame, agg: str) -> float:
        if df is None or df.empty:
            return float("nan")
        d = df.dropna(how="all")
        if d.empty:
            return float("nan")
        last_dt = d.index[-1]
        prev_dt = last_dt - pd.Timedelta(days=7)
        try:
            prev_row = d.apply(lambda s: s.asof(prev_dt))
        except Exception:
            return float("nan")
        cur = d.iloc[-1]
        if agg == "mean":
            return float(cur.mean(skipna=True)) - float(prev_row.mean(skipna=True))
        if agg == "sum":
            return float(cur.sum(skipna=True)) - float(prev_row.sum(skipna=True))
        if agg == "abs_mean":
            return float(cur.abs().mean(skipna=True)) - float(prev_row.abs().mean(skipna=True))
        return float("nan")

    d_mes_kpi   = _agg_7d_delta(mes_df,    "mean")
    d_lrmes_kpi = _agg_7d_delta(lrmes_df,  "mean")
    d_cov_kpi   = _agg_7d_delta(dcovar_df, "abs_mean")
    d_srisk_kpi = _agg_7d_delta(srisk_df,  "sum")
    d_lvg_kpi   = _agg_7d_delta(lvg_df,    "mean")

    def _badge(delta: float, fmt: str = "pp") -> tuple[str | None, str]:
        """Format a delta value as (text, direction) for kpi_card."""
        if pd.isna(delta) or delta == 0:
            return None, "neutral"
        direction = "up" if delta > 0 else "down"
        if fmt == "pp":
            text = f"{delta * 100:+.2f} pp vs 7d ago"
        elif fmt == "bn":
            text = f"{delta / 1e9:+.2f} bn vs 7d ago"
        elif fmt == "ratio":
            text = f"{delta:+.2f} vs 7d ago"
        else:
            text = f"{delta:+.4f} vs 7d ago"
        return text, direction

    mes_badge,   mes_dir   = _badge(d_mes_kpi,   "pp")
    lrmes_badge, lrmes_dir = _badge(d_lrmes_kpi, "pp")
    cov_badge,   cov_dir   = _badge(d_cov_kpi,   "pp")
    srisk_badge, srisk_dir = _badge(d_srisk_kpi, "bn")
    lvg_badge,   lvg_dir   = _badge(d_lvg_kpi,   "ratio")

    # ── Risk classification: rolling-500 percentile of the daily aggregate ───
    # Build a per-day aggregate series across the (filtered) banks using the
    # same aggregation as the KPI value, then compute the percentile rank of
    # the latest aggregate within the last 500 observations:
    #   < 0.7        → Low
    #   0.7  – 0.9   → Medium
    #   ≥ 0.9        → High
    # Higher values mean more risk for every measure shown here, so the
    # classification reads as severity directly.
    def _classify_risk(
        df: pd.DataFrame,
        agg: str = "mean",
        window: int = 500,
        min_obs: int = 30,
        fmt_fn=None,
        abs_for_value: bool = False,
    ) -> tuple[str | None, str | None]:
        if df is None or df.empty:
            return None, None
        d = df.dropna(how="all")
        if d.empty:
            return None, None
        if agg == "mean":
            s = d.mean(axis=1, skipna=True)
        elif agg == "sum":
            s = d.sum(axis=1, skipna=True, min_count=1)
        elif agg == "abs_mean":
            s = d.abs().mean(axis=1, skipna=True)
        else:
            return None, None
        s = s.dropna()
        if len(s) < min_obs:
            return None, None
        win = s.tail(window)
        latest = float(win.iloc[-1])
        pct = float((win <= latest).sum()) / float(len(win))
        if pct >= 0.9:
            level = "High"
        elif pct >= 0.7:
            level = "Medium"
        else:
            level = "Low"
        q70 = float(win.quantile(0.70))
        q90 = float(win.quantile(0.90))
        # |ΔCoVaR| is displayed as the absolute value, so format the tooltip
        # threshold values with the same sign convention.
        if abs_for_value:
            latest = abs(latest)
            q70    = abs(q70)
            q90    = abs(q90)
        f = fmt_fn or (lambda x: f"{x:,.4f}")
        tip = (
            f"Rolling {len(win)}-obs percentile: {pct * 100:.0f}%\n"
            f"Current: {f(latest)}\n"
            f"P70 (Medium ≥): {f(q70)}\n"
            f"P90 (High ≥): {f(q90)}\n"
            f"Bands: Low <70%  ·  Medium 70–90%  ·  High ≥90%"
        )
        return level, tip

    mes_risk,   mes_tip   = _classify_risk(mes_df,    "mean",     fmt_fn=_fmt_pct)
    lrmes_risk, lrmes_tip = _classify_risk(lrmes_df,  "mean",     fmt_fn=_fmt_pct)
    cov_risk,   cov_tip   = _classify_risk(dcovar_df, "abs_mean", fmt_fn=_fmt_pct, abs_for_value=True)
    srisk_risk, srisk_tip = _classify_risk(srisk_df,  "sum",      fmt_fn=_fmt_bn)
    lvg_risk,   lvg_tip   = _classify_risk(lvg_df,    "mean",     fmt_fn=_fmt_ratio)

    _ACCENT = ACCENT_BLUE
    # Hover popover bodies — the full methodology card for each measure.
    # Build a fresh component per call so Dash doesn't try to deduplicate
    # the same instance across Overview + Start tab outputs.
    _info_mes   = _methodology_card_for("mes")
    _info_lrmes = _methodology_card_for("lrmes")
    _info_cov   = _methodology_card_for("covar")
    _info_srisk = _methodology_card_for("srisk")
    _info_lvg   = _methodology_card_for("lvg")

    kpi_mes  = kpi_card("Avg. MES",
                        _fmt_pct(agg_mes),
                        "Mean 1-day tail loss",
                        _ACCENT,
                        delta_text=mes_badge, delta_direction=mes_dir,
                        risk_level=mes_risk, risk_tooltip=mes_tip,
                        info_content=_info_mes,
                        info_id="info-icon-overview-mes")
    kpi_lrmes = kpi_card("Avg. LRMES",
                        _fmt_pct(agg_lrmes),
                        "Long-run MES at d-decline scenario",
                        _ACCENT,
                        delta_text=lrmes_badge, delta_direction=lrmes_dir,
                        risk_level=lrmes_risk, risk_tooltip=lrmes_tip,
                        info_content=_info_lrmes,
                        info_id="info-icon-overview-lrmes")
    kpi_cov  = kpi_card("Avg. |ΔCoVaR|",
                        _fmt_pct(abs(agg_covar) if not pd.isna(agg_covar) else float("nan")),
                        "Mean marginal systemic contribution",
                        _ACCENT,
                        delta_text=cov_badge, delta_direction=cov_dir,
                        risk_level=cov_risk, risk_tooltip=cov_tip,
                        info_content=_info_cov,
                        info_id="info-icon-overview-covar")
    kpi_srisk = kpi_card("Total SRISK",
                         _fmt_bn(agg_srisk if agg_srisk > 0 else float("nan")),
                         "Aggregate capital shortfall estimate",
                         _ACCENT,
                         delta_text=srisk_badge, delta_direction=srisk_dir,
                         risk_level=srisk_risk, risk_tooltip=srisk_tip,
                         info_content=_info_srisk,
                         info_id="info-icon-overview-srisk")
    kpi_lvg  = kpi_card("Avg. Leverage",
                        _fmt_ratio(agg_lvg),
                        "(Liabilities + Market Cap) / Market Cap",
                        _ACCENT,
                        delta_text=lvg_badge, delta_direction=lvg_dir,
                        risk_level=lvg_risk, risk_tooltip=lvg_tip,
                        info_content=_info_lvg,
                        info_id="info-icon-overview-lvg")

    # Overview ranking bars — show only the top 10 banks per measure
    _TOP_N = 10
    _mes_top   = latest_mes.dropna().nlargest(_TOP_N)
    _covar_abs = latest_covar.abs().dropna()
    _covar_top = _covar_abs.nlargest(_TOP_N)
    _srisk_pos = latest_srisk[latest_srisk > 0].dropna()
    _srisk_top = _srisk_pos.nlargest(_TOP_N)

    fig_mes   = ranking_bar(_mes_top,   "MES",       "", )
    fig_srisk = ranking_bar(_srisk_top, "SRISK",     "", fmt_fn=_fmt_bn)
    fig_covar = ranking_bar(_covar_top, "|ΔCoVaR|",  "", )

    # ── Δ-last-week / Δ-last-month deltas ─────────────────────────────────────
    # Per-column Series.asof is used instead of DataFrame.asof: the latter
    # walks backward until it finds a row with all non-NaN columns, which
    # collapses to NaN whenever any single bank's column is entirely empty
    # (e.g. a freshly added bank without balance-sheet data yet). Per-column
    # asof ignores NaN values column-wise and returns the most recent
    # non-NaN observation ≤ prev_dt for each bank independently.
    def _delta(df: pd.DataFrame, days: int) -> pd.Series:
        if df.empty:
            return pd.Series(dtype=float)
        last = df.dropna(how="all")
        if last.empty:
            return pd.Series(dtype=float)
        last_dt = last.index[-1]
        prev_dt = last_dt - pd.Timedelta(days=days)
        try:
            prev = last.apply(lambda s: s.asof(prev_dt))
        except Exception:
            prev = pd.Series(np.nan, index=last.columns)
        return last.iloc[-1] - prev

    d_mes_w  = _delta(mes_df,    7)
    d_mes_m  = _delta(mes_df,    30)
    d_cov_w  = _delta(dcovar_df, 7)
    d_sri_w  = _delta(srisk_df,  7)
    d_sri_m  = _delta(srisk_df,  30)

    # ── Snapshot row for the Risk Summary table ──────────────────────────────
    # If the user picks a snapshot date, use that row; otherwise fall back to
    # the latest available row in the selected range. KPIs and ranking charts
    # always use the latest row.
    def _row_at(df: pd.DataFrame, date_str: str | None) -> pd.Series:
        if df is None or df.empty:
            return pd.Series(dtype=float)
        if not date_str:
            return _latest_row(df)
        try:
            target = pd.Timestamp(date_str)
            return df.asof(target)
        except Exception:
            return _latest_row(df)

    snap_mes   = _row_at(mes_df,    snap_date)
    snap_lrmes = _row_at(lrmes_df,  snap_date)
    snap_covar = _row_at(dcovar_df, snap_date)
    snap_srisk = _row_at(srisk_df,  snap_date)
    snap_lvg   = _row_at(lvg_df,    snap_date)

    # Summary table as sortable DataTable with Δ-week / Δ-month columns
    all_tickers = sorted(
        set(latest_mes.index) | set(latest_covar.index)
    )
    table_rows = []
    for t in all_tickers:
        table_rows.append({
            "bank":      _name(t),
            "ticker":    t,
            "mes":       float(snap_mes.get(t,   np.nan)) if pd.notna(snap_mes.get(t,   np.nan)) else None,
            "mes_dw":    float(d_mes_w.get(t,    np.nan)) if pd.notna(d_mes_w.get(t,    np.nan)) else None,
            "mes_dm":    float(d_mes_m.get(t,    np.nan)) if pd.notna(d_mes_m.get(t,    np.nan)) else None,
            "lrmes":     float(snap_lrmes.get(t, np.nan)) if pd.notna(snap_lrmes.get(t, np.nan)) else None,
            "covar":     float(snap_covar.get(t, np.nan)) if pd.notna(snap_covar.get(t, np.nan)) else None,
            "covar_dw":  float(d_cov_w.get(t,    np.nan)) if pd.notna(d_cov_w.get(t,    np.nan)) else None,
            "srisk_bn":  (float(snap_srisk.get(t, np.nan)) / 1e9) if pd.notna(snap_srisk.get(t, np.nan)) else None,
            "srisk_dw":  (float(d_sri_w.get(t,   np.nan)) / 1e9) if pd.notna(d_sri_w.get(t,   np.nan)) else None,
            "srisk_dm":  (float(d_sri_m.get(t,   np.nan)) / 1e9) if pd.notna(d_sri_m.get(t,   np.nan)) else None,
            "lvg":       float(snap_lvg.get(t,   np.nan)) if pd.notna(snap_lvg.get(t,   np.nan)) else None,
        })

    # Note: dash_table.FormatTemplate.percentage(2) returns a Format object,
    # NOT a plain dict — can't be spread with {**...}. Use plain dict specifiers
    # throughout so column formats are composable.
    pct_fmt     = dash_table.FormatTemplate.percentage(2)
    pct_signed  = {"specifier": "+.2%"}
    num_fmt2    = {"specifier": ",.2f"}
    num_fmtp    = {"specifier": "+,.2f"}  # signed Δ cells

    columns = [
        {"name": "Bank",         "id": "bank"},
        {"name": "Ticker",       "id": "ticker"},
        {"name": "MES",          "id": "mes",       "type": "numeric", "format": pct_fmt},
        {"name": "Δ MES (1w)",   "id": "mes_dw",    "type": "numeric", "format": pct_signed},
        {"name": "Δ MES (1m)",   "id": "mes_dm",    "type": "numeric", "format": pct_signed},
        {"name": "LRMES",        "id": "lrmes",     "type": "numeric", "format": pct_fmt},
        {"name": "ΔCoVaR",       "id": "covar",     "type": "numeric", "format": pct_fmt},
        {"name": "Δ ΔCoVaR (1w)","id": "covar_dw",  "type": "numeric", "format": pct_signed},
        {"name": "SRISK (bn)",   "id": "srisk_bn",  "type": "numeric", "format": num_fmt2},
        {"name": "Δ SRISK (1w)", "id": "srisk_dw",  "type": "numeric", "format": num_fmtp},
        {"name": "Δ SRISK (1m)", "id": "srisk_dm",  "type": "numeric", "format": num_fmtp},
        {"name": "LVG",          "id": "lvg",       "type": "numeric", "format": num_fmt2},
    ]

    table = dash_table.DataTable(
        id="risk-datatable",
        columns=columns,
        data=table_rows,
        sort_action="native",
        filter_action="native",
        page_size=25,
        style_cell={
            "fontSize": "0.82rem",
            "padding": "6px 10px",
            "fontFamily": "system-ui, -apple-system, Segoe UI, sans-serif",
            "textAlign": "right",
        },
        style_cell_conditional=[
            {"if": {"column_id": "bank"},   "textAlign": "left",  "fontWeight": "500"},
            {"if": {"column_id": "ticker"}, "textAlign": "left",  "color": TEXT_MUTED},
        ],
        style_header={
            "backgroundColor": "#f8f9fa",
            "fontWeight": "700",
            "borderBottom": f"2px solid {BORDER}",
            "fontSize": "0.78rem",
        },
        style_data_conditional=[
            # Positive deltas (worsening for MES/ΔCoVaR/SRISK) in red
            {"if": {"filter_query": "{mes_dw} > 0",   "column_id": "mes_dw"},   "color": C.ACCENT_RED},
            {"if": {"filter_query": "{mes_dw} < 0",   "column_id": "mes_dw"},   "color": C.ACCENT_GREEN},
            {"if": {"filter_query": "{mes_dm} > 0",   "column_id": "mes_dm"},   "color": C.ACCENT_RED},
            {"if": {"filter_query": "{mes_dm} < 0",   "column_id": "mes_dm"},   "color": C.ACCENT_GREEN},
            {"if": {"filter_query": "{covar_dw} > 0", "column_id": "covar_dw"}, "color": C.ACCENT_RED},
            {"if": {"filter_query": "{covar_dw} < 0", "column_id": "covar_dw"}, "color": C.ACCENT_GREEN},
            {"if": {"filter_query": "{srisk_dw} > 0", "column_id": "srisk_dw"}, "color": C.ACCENT_RED},
            {"if": {"filter_query": "{srisk_dw} < 0", "column_id": "srisk_dw"}, "color": C.ACCENT_GREEN},
            {"if": {"filter_query": "{srisk_dm} > 0", "column_id": "srisk_dm"}, "color": C.ACCENT_RED},
            {"if": {"filter_query": "{srisk_dm} < 0", "column_id": "srisk_dm"}, "color": C.ACCENT_GREEN},
        ],
        style_table={"overflowX": "auto", "backgroundColor": BG_CARD},
    )

    # Δ-ranking bars — top 10 movers by absolute change
    def _top_by_abs(s: pd.Series, n: int = _TOP_N) -> pd.Series:
        s = s.dropna()
        if s.empty:
            return s
        top_idx = s.abs().nlargest(n).index
        return s.reindex(top_idx)

    fig_mes_dw   = delta_ranking_bar(_top_by_abs(d_mes_w), "Δ MES",     "")
    fig_srisk_dw = delta_ranking_bar(_top_by_abs(d_sri_w), "Δ SRISK",   "", divide_bn=True)
    fig_covar_dw = delta_ranking_bar(_top_by_abs(d_cov_w), "Δ ΔCoVaR",  "")

    # Start tab mirrors the Overview KPI cards for the three core measures.
    # Build separate component instances (including fresh methodology
    # cards for the popovers) so Dash isn't asked to render the same
    # subtree twice in the layout.
    kpi_start_mes = kpi_card("Avg. MES",
                             _fmt_pct(agg_mes),
                             "Mean 1-day tail loss",
                             _ACCENT,
                             delta_text=mes_badge, delta_direction=mes_dir,
                             risk_level=mes_risk, risk_tooltip=mes_tip,
                             info_content=_methodology_card_for("mes"),
                             info_id="info-icon-start-mes")
    kpi_start_cov = kpi_card("Avg. |ΔCoVaR|",
                             _fmt_pct(abs(agg_covar) if not pd.isna(agg_covar) else float("nan")),
                             "Mean marginal systemic contribution",
                             _ACCENT,
                             delta_text=cov_badge, delta_direction=cov_dir,
                             risk_level=cov_risk, risk_tooltip=cov_tip,
                             info_content=_methodology_card_for("covar"),
                             info_id="info-icon-start-covar")
    kpi_start_srisk = kpi_card("Total SRISK",
                               _fmt_bn(agg_srisk if agg_srisk > 0 else float("nan")),
                               "Aggregate capital shortfall estimate",
                               _ACCENT,
                               delta_text=srisk_badge, delta_direction=srisk_dir,
                               risk_level=srisk_risk, risk_tooltip=srisk_tip,
                               info_content=_methodology_card_for("srisk"),
                               info_id="info-icon-start-srisk")

    return (kpi_mes, kpi_lrmes, kpi_cov, kpi_srisk, kpi_lvg,
            kpi_start_mes, kpi_start_cov, kpi_start_srisk,
            fig_mes, fig_srisk, fig_covar,
            fig_mes_dw, fig_srisk_dw, fig_covar_dw,
            table)


# ── Risk Summary collapse toggle ──────────────────────────────────────────────

@app.callback(
    Output("risk-summary-collapse", "is_open"),
    Output("risk-summary-caret",    "children"),
    Input("btn-risk-summary",       "n_clicks"),
    State("risk-summary-collapse",  "is_open"),
    prevent_initial_call=True,
)
def toggle_risk_summary(_n_clicks, is_open):
    new_open = not is_open
    caret = "▾" if new_open else "▸"
    return new_open, caret


# ── Time series tab ───────────────────────────────────────────────────────────

@app.callback(
    Output("chart-timeseries", "figure"),
    Input("date-range",    "start_date"),
    Input("date-range",    "end_date"),
    Input("bank-select",   "value"),
    Input("ts-measure",    "value"),
    Input("ts-overlay",    "value"),
    Input("ts-crises",     "value"),
    Input("ts-traces",     "value"),
    Input("refresh-store", "data"),
    Input("alpha-store",   "data"),
)
def update_timeseries(start, end, tickers, measure, overlay, crises,
                      traces, _refresh, _alpha):
    tickers = tickers or []
    traces = traces or []
    show_individual = "individual" in traces
    show_aggregate  = "aggregate"  in traces
    df = _slice(MEASURES[measure], start, end, tickers)

    labels = {
        "mes":         ("MES",       "MES (loss fraction)"),
        "lrmes":       ("LRMES",     "LRMES (fraction, 1-month stress)"),
        "delta_covar": ("ΔCoVaR",    "ΔCoVaR"),
        "covar":       ("CoVaR",     "CoVaR"),
    }
    label, ylabel = labels.get(measure, (measure, measure))

    _AGG_MODE = {
        "mes":         ("mean", "Mean"),
        "lrmes":       ("mean", "Mean"),
        "delta_covar": ("mean", "Mean"),
        "covar":       ("mean", "Mean"),
    }
    agg_series = None
    agg_label  = ""
    agg_hover  = "%{y:.4f}"
    if not df.empty and show_aggregate:
        mode, prefix = _AGG_MODE.get(measure, ("mean", "Mean"))
        if mode == "sum":
            agg_series = df.sum(axis=1, min_count=1)
            agg_hover  = "%{y:,.2s}"  # SI suffix (k / M / G) for USD totals
        else:
            agg_series = df.mean(axis=1)
        n_sel = sum(1 for _ in df.columns)
        agg_label = f"{prefix} {label} — {n_sel} selected"

    mkt = None
    if overlay and "show" in overlay:
        r = _slice(RETURNS[[MARKET_NAME]], start, end)
        if MARKET_NAME in r.columns:
            mkt = r[MARKET_NAME]

    # Individual traces are drawn from df.columns — pass an empty frame when
    # the user has hidden them (but keep df's index so the aggregate still
    # plots on the same x-axis).
    plot_df = df if show_individual else df.iloc[:, 0:0]

    show_crises = bool(crises and "show" in crises)
    return timeseries_chart(
        plot_df, f"Rolling {label}", ylabel,
        market_ret=mkt,
        show_crises=show_crises,
        data_start=start, data_end=end,
        aggregate=agg_series,
        aggregate_label=agg_label,
        aggregate_hover_fmt=agg_hover,
    )


# ── SRISK tab ─────────────────────────────────────────────────────────────────


_D_BASE = 0.40  # LRMES decline threshold used to build MEASURES["lrmes"].


def _compute_srisk_df(k: float, d: float = _D_BASE) -> pd.DataFrame:
    """
    Recompute SRISK across all banks for user-selected prudential ratio k and
    LRMES decline threshold d.

    Vectorised equivalent of :func:`systemic_measures.compute_srisk` applied
    column-wise:

        SRISK(t) = max(0, k·LBR(t) − (1−k)·(1−LRMES(t; d))·MC(t))

    LRMES is rescaled analytically from the cached d = 40 % series via
        LRMES(t; d) = 1 − (1 − LRMES₀(t))^(log(1−d) / log(1−0.40))
    which is exact under the Brownlees–Engle / Belluzzo closed form
    (β(t) is independent of d).

    Returns an empty DataFrame if the upstream global series aren't populated
    yet (pre-refresh state).
    """
    lrmes_df = MEASURES.get("lrmes")
    if (lrmes_df is None or lrmes_df.empty
            or LBR_DAILY is None or LBR_DAILY.empty
            or MC_TS is None or MC_TS.empty):
        return pd.DataFrame()
    cols = [c for c in lrmes_df.columns
            if c in LBR_DAILY.columns and c in MC_TS.columns]
    if not cols:
        return pd.DataFrame()
    idx = (lrmes_df.index
           .intersection(LBR_DAILY.index)
           .intersection(MC_TS.index))
    lr  = lrmes_df.reindex(index=idx, columns=cols).clip(0.0, 1.0)
    # Rescale cached LRMES (built with d = _D_BASE) to the requested d.
    if not np.isclose(d, _D_BASE):
        d = float(np.clip(d, 1e-4, 0.999))
        exponent = np.log(1.0 - d) / np.log(1.0 - _D_BASE)
        lr = 1.0 - np.power(np.clip(1.0 - lr, 1e-12, 1.0), exponent)
        lr = lr.clip(0.0, 1.0)
    lb  = LBR_DAILY.reindex(index=idx, columns=cols)
    cap = MC_TS.reindex(index=idx, columns=cols)
    srisk = k * lb - (1.0 - k) * (1.0 - lr) * cap
    return srisk.clip(lower=0.0)


@app.callback(
    Output("chart-srisk-bar", "figure"),
    Output("chart-srisk-pie", "figure"),
    Output("chart-srisk-ts",  "figure"),
    Output("srisk-k-label",   "children"),
    Output("srisk-d-label",   "children"),
    Input("date-range",    "start_date"),
    Input("date-range",    "end_date"),
    Input("bank-select",   "value"),
    Input("srisk-norm",    "value"),
    Input("srisk-k-slider","value"),
    Input("srisk-d-slider","value"),
    Input("srisk-ts-mode", "value"),
    Input("refresh-store", "data"),
    # Note: alpha-store is intentionally NOT an input here. SRISK and LRMES
    # are α-invariant under the closed-form approximation (LRMES depends on
    # ρ/σ/d only, not α). Wiring α to this callback would force an
    # expensive 3-figure rebuild on every α click for no observable change.
)
def update_srisk(start, end, tickers, norm, k_pct, d_pct, ts_mode, _refresh):
    tickers = tickers or []
    # Convert slider values (percent) to ratios; recompute SRISK for this (k, d).
    k = float(k_pct) / 100.0 if k_pct is not None else 0.08
    d = float(d_pct) / 100.0 if d_pct is not None else _D_BASE
    srisk_all = _compute_srisk_df(k, d)
    if srisk_all.empty:
        srisk_all = MEASURES.get("srisk", pd.DataFrame())
    df      = _slice(srisk_all, start, end, tickers)
    latest  = _latest_row(df)
    mc_latest = _latest_row(_slice(MC_TS, start, end, tickers)) if not MC_TS.empty else pd.Series(dtype=float)

    if norm == "pct_agg":
        total = latest.sum()
        latest_disp = (latest / total * 100.0) if total > 0 else latest * np.nan
        title_bar = "SRISK by Bank — Top 10 (% of aggregate SRISK)"
        xlabel_bar = "% of aggregate SRISK"
        fmt_bar = _fmt_pct_raw
    elif norm == "pct_mc":
        mc_aligned = mc_latest.reindex(latest.index)
        latest_disp = (latest / mc_aligned.where(mc_aligned > 0, np.nan)) * 100.0
        title_bar = "SRISK by Bank — Top 10 (% of own market cap)"
        xlabel_bar = "% of market cap"
        fmt_bar = _fmt_pct_raw
    else:  # 'abs'
        latest_disp = latest / 1e9
        title_bar = "SRISK by Bank — Top 10 (USD bn)"
        xlabel_bar = "SRISK (bn USD)"
        fmt_bar = _fmt_bn_x1  # already in bn

    # Restrict the bar chart to the top 10 banks by displayed value.
    bar_series = latest_disp.dropna().sort_values(ascending=False).head(10)
    fig_bar = _srisk_bar_generic(bar_series, title_bar, xlabel_bar, fmt_bar, norm)
    # Pie: top 5 individual + 'Other' bucket for the rest.
    fig_pie = srisk_pie(latest, top_n=5)

    # ── SRISK over time: aggregate or stacked ────────────────────────────────
    if ts_mode == "stacked":
        mc_cols  = [c for c in df.columns if c in MC_TS.columns]
        mc_total = (MC_TS[mc_cols].reindex(df.index).ffill().sum(axis=1)
                    if mc_cols else None)
        y_unit = {"abs": "bn", "pct_agg": "pct_agg", "pct_mc": "pct_mc"}.get(norm, "bn")
        fig_ts = srisk_stacked_area(
            df,
            y_unit=y_unit,
            total_mc_ts=mc_total,
            top_n=10,
            show_crises=True,
            data_start=start, data_end=end,
        )
    else:  # aggregate
        agg = df.sum(axis=1)
        if norm == "pct_mc":
            mc_total = MC_TS[[c for c in df.columns if c in MC_TS.columns]].reindex(df.index).ffill().sum(axis=1)
            y_ts = (agg / mc_total.where(mc_total > 0, np.nan)) * 100.0
            y_label = "Aggregate SRISK (% of total market cap)"
            y_hover = "%{y:.2f}%"
        else:
            y_ts = agg.values / 1e9
            y_label = "Aggregate SRISK (bn USD)"
            y_hover = "%{y:.1f} bn"

        fig_ts = go.Figure(go.Scatter(
            x=agg.index, y=y_ts,
            fill="tozeroy",
            fillcolor="rgba(74, 141, 220, 0.12)",
            line=dict(color=C.ACCENT_BLUE, width=2),
            name="Aggregate",
            hovertemplate=(
                f"Date: %{{x|%Y-%m-%d}}<br>"
                f"Bank: Aggregate<br>"
                f"Value: {y_hover}<extra></extra>"
            ),
        ))
        _add_crisis_overlays(fig_ts, start, end)
        fig_ts.update_layout(
            title=dict(text="Aggregate SRISK over Time",
                       font=dict(size=C.CHART_TITLE_SIZE)),
            yaxis_title=y_label,
            height=360,
            **_base_layout(),
        )

    k_label = f"Prudential capital ratio  k = {k * 100:.1f}%"
    d_label = f"LRMES decline threshold  d = {d * 100:.1f}%"
    return fig_bar, fig_pie, fig_ts, k_label, d_label


# ── Market data tab ───────────────────────────────────────────────────────────

# ── Market Data + DCC Correlation (combined) tab ─────────────────────────────

@app.callback(
    Output("chart-market-dcc", "figure"),
    Output("chart-corr",       "figure"),
    Input("date-range",        "start_date"),
    Input("date-range",        "end_date"),
    Input("bank-select",       "value"),
    Input("market-dcc-crises", "value"),
    Input("refresh-store",     "data"),
    # Note: alpha-store is intentionally NOT an input here. The market
    # price index, DCC correlation matrix, and pairwise return correlation
    # are all derived from raw returns / DCC outputs and do not depend on
    # the tail probability α.
)
def update_market_dcc_tab(start, end, tickers, crises, _refresh):
    tickers = tickers or []
    keep    = tickers + [MARKET_NAME]
    prices  = _slice(PRICES,  start, end, keep)
    rets    = _slice(RETURNS, start, end, tickers)
    dcc_df  = _slice(DCC_RHO, start, end, tickers) if not DCC_RHO.empty else pd.DataFrame()

    show_crises = bool(crises and "show" in crises)

    return (
        market_dcc_chart(prices, dcc_df, show_crises, start, end),
        corr_heatmap(rets),
    )


@app.callback(
    Output("download-dcc", "data"),
    Input("btn-download-dcc", "n_clicks"),
    State("date-range",  "start_date"),
    State("date-range",  "end_date"),
    State("bank-select", "value"),
    prevent_initial_call=True,
)
def download_dcc_csv(_n, start, end, tickers):
    tickers = tickers or []
    if DCC_RHO.empty:
        return no_update
    df = _slice(DCC_RHO, start, end, tickers)
    if df.empty:
        return no_update
    df.index.name = "Date"
    stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
    return _csv_download(df, f"dcc_rho_{stamp}.csv")


# ── CSV downloads ─────────────────────────────────────────────────────────────

def _csv_download(df: pd.DataFrame, filename: str) -> dict:
    buf = io.StringIO()
    df.to_csv(buf, index=True)
    return dict(content=buf.getvalue(), filename=filename)


@app.callback(
    Output("download-overview", "data"),
    Input("btn-download-overview", "n_clicks"),
    State("date-range",  "start_date"),
    State("date-range",  "end_date"),
    State("bank-select", "value"),
    prevent_initial_call=True,
)
def download_overview_csv(_n, start, end, tickers):
    tickers = tickers or []
    frames = []
    for key in ("mes", "lrmes", "delta_covar", "srisk"):
        m_df = MEASURES.get(key)
        if m_df is None or m_df.empty:
            continue
        sub = _slice(m_df, start, end, tickers)
        latest = _latest_row(sub).rename(key.upper().replace("_", ""))
        frames.append(latest)
    if not LEVERAGE.empty:
        lvg_sub = _slice(LEVERAGE, start, end, tickers)
        frames.append(_latest_row(lvg_sub).rename("LVG"))
    if not frames:
        return no_update
    summary = pd.concat(frames, axis=1)
    summary.insert(0, "Bank", [_name(t) for t in summary.index])
    summary.index.name = "Ticker"
    stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
    return _csv_download(summary, f"systemic_risk_overview_{stamp}.csv")


@app.callback(
    Output("download-ts", "data"),
    Input("btn-download-ts", "n_clicks"),
    State("date-range",  "start_date"),
    State("date-range",  "end_date"),
    State("bank-select", "value"),
    State("ts-measure",  "value"),
    prevent_initial_call=True,
)
def download_ts_csv(_n, start, end, tickers, measure):
    tickers = tickers or []
    df = _slice(MEASURES.get(measure, pd.DataFrame()), start, end, tickers)
    if df.empty:
        return no_update
    df.index.name = "Date"
    stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
    return _csv_download(df, f"systemic_risk_{measure}_{stamp}.csv")


@app.callback(
    Output("download-srisk", "data"),
    Input("btn-download-srisk", "n_clicks"),
    State("date-range",  "start_date"),
    State("date-range",  "end_date"),
    State("bank-select", "value"),
    State("srisk-k-slider", "value"),
    State("srisk-d-slider", "value"),
    prevent_initial_call=True,
)
def download_srisk_csv(_n, start, end, tickers, k_pct, d_pct):
    tickers = tickers or []
    k = float(k_pct) / 100.0 if k_pct is not None else 0.08
    d = float(d_pct) / 100.0 if d_pct is not None else _D_BASE
    srisk_all = _compute_srisk_df(k, d)
    if srisk_all.empty:
        srisk_all = MEASURES.get("srisk", pd.DataFrame())
    df = _slice(srisk_all, start, end, tickers)
    if df.empty:
        return no_update
    df = df.copy()
    df["Aggregate"] = df.sum(axis=1)
    df.index.name = "Date"
    stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
    return _csv_download(
        df,
        f"srisk_timeseries_k{int(round(k*1000))}bp_d{int(round(d*1000))}bp_{stamp}.csv",
    )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug = False)
