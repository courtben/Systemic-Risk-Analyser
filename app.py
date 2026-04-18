"""
Systemic Risk Dashboard — US Banks
====================================
Interactive Dash application visualising MES, SES, DeltaCoVaR, and SRISK
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
from plotly.subplots import make_subplots

import io
import dash
from dash import dcc, html, Input, Output, State, no_update, dash_table
import dash_bootstrap_components as dbc

warnings.filterwarnings("ignore")

# ── Load / compute data at startup ────────────────────────────────────────────

print("=" * 60)
print("Systemic Risk Dashboard  —  US Banks")
print("=" * 60)

import data_load as D
import systemic_measures as M

print("\n[1/4] Fetching price data ...")
PRICES = D.get_prices()

print("\n[2/4] Computing returns ...")
RETURNS = D.compute_returns(PRICES)

print("\n[3/5] Fetching balance sheet data ...")
BS = D.get_balance_sheet()

print("\n[4/6] Fetching liabilities and separate-account time series ...")
LIAB_TS      = D.get_liabilities_ts()
SEP_ACCT_TS  = D.get_separate_accounts_ts()
LB_DAILY     = D.build_lb_daily(LIAB_TS, PRICES, SEP_ACCT_TS)
LBR_DAILY    = D.build_lbr_daily(LIAB_TS, PRICES, SEP_ACCT_TS)

print("\n[5/6] Fetching systemic state variables ...")
STATE_VARS = D.get_state_variables(PRICES)

print("\n[6/6] Computing systemic risk measures (DCC-GJR-GARCH) ...")
MC_TS    = D.build_market_cap_series(PRICES, BS)
# Returns dict with keys: 'mes', 'ses', 'covar', 'delta_covar', 'srisk'
MEASURES = M.compute_all(RETURNS, MC_TS, LB_DAILY, LBR_DAILY, BS, state_vars=STATE_VARS)

# ── Quasi-leverage = (Book Liabilities + Market Cap) / Market Cap ────────────
# Matches the "LVG" column on NYU Stern V-Lab.
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
# interactively. Precompute at a small grid and snap the slider to nearest.
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

LAST_UPDATED = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
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
CRISIS_PERIODS  = C.CRISIS_PERIODS

# Format / layout helpers — keep underscore-prefixed aliases for callbacks
_add_crisis_overlays = C.add_crisis_overlays
_base_layout         = C.base_layout
_name                = C.name_for
_color               = C.color_for
_fmt_bn              = C.fmt_bn
_fmt_pct             = C.fmt_pct
_fmt_ratio           = C.fmt_ratio
_fmt_ses             = C.fmt_ses
_fmt_pct_raw         = C.fmt_pct_raw
_fmt_bn_x1           = C.fmt_bn_x1

# Chart builders
delta_ranking_bar    = C.delta_ranking_bar
ranking_bar          = C.ranking_bar
timeseries_chart     = C.timeseries_chart
_srisk_bar_generic   = C.srisk_bar_generic
srisk_bar            = C.srisk_bar
srisk_pie            = C.srisk_pie
price_chart          = C.price_chart
corr_heatmap         = C.corr_heatmap
return_hist          = C.return_hist
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
                "MES · SES · ΔCoVaR · SRISK  |  "
                "Acharya et al. (2010/2017) · Adrian & Brunnermeier (2016) · Brownlees & Engle (2017)",
                className="text-muted",
            ),
        ]),
        html.Div([
            html.Small(f"Updated: {LAST_UPDATED}", id="updated-ts",
                       className="text-muted me-3"),
            html.Div(id="refresh-progress", className="me-2"),
            dbc.Button("Refresh", id="btn-refresh", size="sm",
                       color="primary", outline=True),
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

controls = dbc.Container([
    dbc.Row([
        # Date range
        dbc.Col([
            html.Label("Date range", className="text-muted mb-1",
                       style={"fontSize": "0.78rem", "fontWeight": "600"}),
            dcc.DatePickerRange(
                id="date-range",
                min_date_allowed=DATE_MIN, max_date_allowed=DATE_MAX,
                start_date=DATE_DEF_START, end_date=DATE_MAX,
                display_format="YYYY-MM-DD",
                style={"fontSize": "0.85rem"},
            ),
        ], xs=12, md=4, className="mb-2"),

        # Time-range presets
        dbc.Col([
            html.Label("Quick range", className="text-muted mb-1",
                       style={"fontSize": "0.78rem", "fontWeight": "600"}),
            dbc.ButtonGroup([
                dbc.Button(label, id=pid, size="sm",
                           color="primary", outline=True, n_clicks=0)
                for pid, label, _ in _PRESET_BUTTONS
            ], size="sm"),
        ], xs=12, md=6, className="mb-2"),

        # Select / deselect all
        dbc.Col([
            html.Label("Selection", className="text-muted mb-1 d-block",
                       style={"fontSize": "0.78rem", "fontWeight": "600"}),
            dbc.ButtonGroup([
                dbc.Button("All", id="btn-all", size="sm",
                           color="secondary", outline=True),
                dbc.Button("None", id="btn-none", size="sm",
                           color="secondary", outline=True),
            ]),
        ], xs=12, md=2, className="mb-2"),
    ], className="gy-0 align-items-end"),

    # Bank selector (second row, full width)
    dbc.Row([
        dbc.Col([
            html.Label("Banks", className="text-muted mb-1",
                       style={"fontSize": "0.78rem", "fontWeight": "600"}),
            dcc.Dropdown(
                id="bank-select",
                multi=True,
                placeholder="Select banks ...",
                style={"fontSize": "0.85rem"},
            ),
        ], xs=12),
    ], className="mt-1"),

    # Critical value slider (third row)
    dbc.Row([
        dbc.Col([
            html.Label(
                id="alpha-label",
                children="Critical Value α = 5.0%  (worst 5% of market days)",
                className="text-muted mb-1",
                style={"fontSize": "0.78rem", "fontWeight": "600"},
            ),
            dcc.Slider(
                id="alpha-slider",
                min=1, max=10, step=0.5,
                value=5,
                marks={i: f"{i}%" for i in range(1, 11)},
                tooltip={"placement": "bottom", "always_visible": False},
                className="mb-0",
            ),
            html.Small(
                "Changing α recomputes MES, LRMES, SES & SRISK interactively. "
                "ΔCoVaR is precomputed at α ∈ {1, 2.5, 5, 7.5, 10}% and snaps "
                "to the nearest grid point.",
                className="text-muted",
                style={"fontSize": "0.74rem"},
            ),
        ], xs=12, md=8, className="mb-2"),
    ], className="mt-2"),

    # Add custom bank (fourth row)
    dbc.Row([
        dbc.Col([
            html.Label("Add bank by ticker", className="text-muted mb-1",
                       style={"fontSize": "0.78rem", "fontWeight": "600"}),
            dbc.InputGroup([
                dbc.Input(
                    id="custom-ticker-input",
                    placeholder="e.g. BRK-B, USB, TFC ...",
                    type="text",
                    size="sm",
                    style={"fontSize": "0.85rem"},
                    debounce=False,
                ),
                dbc.Button("Add Bank", id="btn-add-bank", color="primary",
                           size="sm", n_clicks=0),
            ], size="sm"),
            dcc.Loading(
                type="dot",
                color="#0d47a1",
                children=html.Div(id="add-bank-status", className="mt-1"),
            ),
        ], xs=12, md=7),
    ], className="mt-2 mb-1"),
], fluid=True, className="py-2 px-3",
   style={"backgroundColor": "#f8f9fa",
          "borderBottom": f"1px solid {BORDER}"})

# ── Tab content ───────────────────────────────────────────────────────────────

_card = {"backgroundColor": BG_CARD,
         "border": f"1px solid {BORDER}",
         "borderRadius": "6px",
         "padding": "12px 16px",
         "marginBottom": "16px"}

overview_layout = dbc.Container([
    dbc.Row([
        dbc.Col(id="kpi-mes",      xs=6, md=4, lg=2, className="mb-3"),
        dbc.Col(id="kpi-lrmes",    xs=6, md=4, lg=2, className="mb-3"),
        dbc.Col(id="kpi-ses",      xs=6, md=4, lg=2, className="mb-3"),
        dbc.Col(id="kpi-covar",    xs=6, md=4, lg=2, className="mb-3"),
        dbc.Col(id="kpi-srisk",    xs=6, md=4, lg=2, className="mb-3"),
        dbc.Col(id="kpi-leverage", xs=6, md=4, lg=2, className="mb-3"),
    ], className="mt-3"),
    dbc.Row([
        dbc.Col(dcc.Graph(id="chart-mes-rank"),   xs=12, md=4, className="mb-3"),
        dbc.Col(dcc.Graph(id="chart-ses-rank"),   xs=12, md=4, className="mb-3"),
        dbc.Col(dcc.Graph(id="chart-covar-rank"), xs=12, md=4, className="mb-3"),
    ]),
    dbc.Row([
        dbc.Col([
            html.P("Change over last 7 trading days", className="text-muted mb-1 mt-2",
                   style={"fontSize": "0.82rem", "fontWeight": "600"}),
        ], xs=12),
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id="chart-mes-dw"),   xs=12, md=4, className="mb-3"),
        dbc.Col(dcc.Graph(id="chart-srisk-dw"), xs=12, md=4, className="mb-3"),
        dbc.Col(dcc.Graph(id="chart-covar-dw"), xs=12, md=4, className="mb-3"),
    ]),
    dbc.Row([
        dbc.Col([
            html.Div([
                html.P("Risk Summary (latest date in selected range)",
                       className="text-muted mb-0 d-inline-block",
                       style={"fontSize": "0.82rem", "fontWeight": "600"}),
                dbc.Button("Download CSV", id="btn-download-overview",
                           size="sm", color="primary", outline=True,
                           className="float-end",
                           style={"fontSize": "0.75rem"}),
                dcc.Download(id="download-overview"),
            ], className="mb-2"),
            html.Div(id="risk-table"),
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
                    {"label": "SES — Systemic Expected Shortfall",        "value": "ses"},
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
    dbc.Row([dbc.Col(dcc.Graph(id="chart-timeseries"), xs=12)]),
    dbc.Row([
        dbc.Col([
            html.Div([
                html.P([
                    html.B("MES"), " — expected fractional loss of the bank when the "
                    f"market ({MARKET_NAME}) falls below its α-th percentile (set above). "
                    "Higher MES = greater tail sensitivity.",
                    html.Br(),
                    html.B("SES"), " — capital shortfall estimate (Acharya et al. 2010). "
                    "SES = max(0, k·D·(1+ΔD/D) − (1−k)·W·(1+ΔW/W)), k = 8% capital ratio. "
                    "D = liabilities, W = market cap.  Reported in USD billions.",
                    html.Br(),
                    html.B("ΔCoVaR"), " — market's VaR when the bank is in stress minus "
                    "its VaR at its median state (Adrian & Brunnermeier 2016). "
                    "More negative ΔCoVaR = larger systemic footprint. ",
                    html.Span("Snapped to nearest α ∈ {1, 2.5, 5, 7.5, 10}%.",
                              style={"fontStyle": "italic"}),
                ], className="mb-0 text-muted", style={"fontSize": "0.82rem"}),
            ], style=_card),
        ], xs=12, className="mb-3"),
    ]),
], fluid=True, style={"backgroundColor": BG_PAGE})

srisk_layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.Label("Normalisation", className="text-muted mb-1",
                       style={"fontSize": "0.78rem", "fontWeight": "600"}),
            dbc.RadioItems(
                id="srisk-norm",
                options=[
                    {"label": " Absolute (USD bn)",      "value": "abs"},
                    {"label": " % of aggregate SRISK",    "value": "pct_agg"},
                    {"label": " % of own market cap",     "value": "pct_mc"},
                ],
                value="abs",
                inline=True,
                inputStyle={"marginRight": "4px"},
                labelStyle={"marginRight": "16px", "fontSize": "0.85rem"},
            ),
        ], xs=12, md=9, className="mt-3"),
        dbc.Col([
            dbc.Button("Download CSV", id="btn-download-srisk",
                       size="sm", color="primary", outline=True,
                       className="float-end mt-4"),
            dcc.Download(id="download-srisk"),
        ], xs=12, md=3),
    ]),
    # Prudential capital ratio k — user-selectable for what-if analysis
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
                className="mb-0",
            ),
            html.Small(
                "k is the minimum equity / (equity + debt) ratio a bank is "
                "assumed to need in stress. SRISK rescales linearly: higher k "
                "raises the implied shortfall; lower k shrinks it. "
                "Brownlees & Engle (2017) use 8%.",
                className="text-muted",
                style={"fontSize": "0.74rem"},
            ),
        ], xs=12, md=9, className="mt-2 mb-3"),
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id="chart-srisk-bar"), xs=12, md=7, className="mb-3"),
        dbc.Col(dcc.Graph(id="chart-srisk-pie"), xs=12, md=5, className="mb-3"),
    ]),
    dbc.Row([dbc.Col(dcc.Graph(id="chart-srisk-ts"), xs=12, className="mb-3")]),
    dbc.Row([
        dbc.Col([
            html.Div([
                html.P([
                    html.B("Formula: "), "SRISK = max(0, k·D − (1−k)·(1−LRMES)·W)",
                    html.Br(),
                    "k = prudential capital ratio (slider above; default 8%) · "
                    "D = book liabilities · W = market cap · "
                    "LRMES = 1 − exp(−22·MES)  [1-month horizon approximation]",
                    html.Br(),
                    html.Span("Note: SRISK values are in USD.",
                              className="text-warning",
                              style={"fontWeight": "600"}),
                ], className="mb-0 text-muted", style={"fontSize": "0.82rem"}),
            ], style=_card),
        ], xs=12, className="mb-3"),
    ]),
], fluid=True, style={"backgroundColor": BG_PAGE})

market_layout = dbc.Container([
    dbc.Row([dbc.Col(dcc.Graph(id="chart-prices"), xs=12, className="mt-3 mb-3")]),
    dbc.Row([
        dbc.Col(dcc.Graph(id="chart-returns-hist"), xs=12, md=6, className="mb-3"),
        dbc.Col(dcc.Graph(id="chart-corr"),         xs=12, md=6, className="mb-3"),
    ]),
], fluid=True, style={"backgroundColor": BG_PAGE})


# ── DCC Correlation tab ──────────────────────────────────────────────────────

dcc_layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.Div([
                html.P([
                    html.B("DCC(1,1) time-varying correlation ρ(t) "),
                    "between each bank and the market index. Rising ρ during "
                    "stress episodes indicates comovement — the 'one-factor' "
                    "regime in which diversification benefits collapse.",
                ], className="text-muted mb-2",
                   style={"fontSize": "0.85rem"}),
            ], style=_card, className="mt-3"),
        ], xs=12),
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Checklist(
                id="dcc-crises",
                options=[{"label": " Shade crisis periods", "value": "show"}],
                value=["show"], switch=True,
                className="mb-2",
            ),
        ], xs=12, md=4),
        dbc.Col([
            dbc.Button("Download CSV", id="btn-download-dcc",
                       size="sm", color="primary", outline=True,
                       className="float-end"),
            dcc.Download(id="download-dcc"),
        ], xs=12, md=8),
    ]),
    dbc.Row([dbc.Col(dcc.Graph(id="chart-dcc-rho"), xs=12, className="mb-3")]),
    dbc.Row([
        dbc.Col([
            html.Div([
                html.P([
                    html.B("Average ρ across selected banks"),
                    " — the mean correlation line is a simple systemic-risk "
                    "barometer: sustained elevation above the long-run mean "
                    "often precedes or accompanies crisis periods.",
                ], className="text-muted mb-0",
                   style={"fontSize": "0.82rem"}),
            ], style=_card),
        ], xs=12, className="mb-3"),
    ]),
    dbc.Row([dbc.Col(dcc.Graph(id="chart-dcc-avg"), xs=12, className="mb-3")]),
], fluid=True, style={"backgroundColor": BG_PAGE})


# ── Methodology tab ──────────────────────────────────────────────────────────

def _formula_block(name: str, formula: str, note: str = "") -> html.Div:
    return html.Div([
        html.P(name, className="mb-1",
               style={"fontWeight": "700", "fontSize": "0.95rem",
                      "color": TEXT_MAIN}),
        html.Pre(formula,
                 style={"backgroundColor": "#f8f9fa",
                        "padding": "10px 14px", "borderRadius": "4px",
                        "border": f"1px solid {BORDER}",
                        "fontSize": "0.85rem", "marginBottom": "6px",
                        "whiteSpace": "pre-wrap"}),
        html.P(note, className="text-muted mb-3",
               style={"fontSize": "0.82rem"}) if note else None,
    ])


methodology_layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H5("Methodology & References", className="mb-3",
                        style={"fontWeight": "700"}),
                html.P(
                    "All five measures are estimated from the bivariate "
                    "DCC-GJR-GARCH(1,1,1) model with zero mean. The market index "
                    f"({MARKET_NAME}) and each firm's daily log return form the "
                    "two-variable system.",
                    className="text-muted mb-3",
                    style={"fontSize": "0.88rem"}),

                html.Hr(),
                html.H6("Volatility & correlation", className="mb-2 mt-3",
                        style={"fontWeight": "700"}),
                _formula_block(
                    "GJR-GARCH(1,1,1)",
                    "h_t = ω + α·ε²_{t-1} + γ·ε²_{t-1}·𝟙{ε_{t-1}<0} + β·h_{t-1}",
                    "Fitted independently to market and firm log returns."),
                _formula_block(
                    "DCC(1,1) correlation",
                    "Q_t = (1−a−b)·Q̄ + a·ε_{t-1}ε'_{t-1} + b·Q_{t-1}\n"
                    "ρ_t = Q_t[0,1] / √(Q_t[0,0]·Q_t[1,1])",
                    "(a, b) maximise the Gaussian conditional log-likelihood."),

                html.Hr(),
                html.H6("Systemic risk measures", className="mb-2 mt-3",
                        style={"fontWeight": "700"}),
                _formula_block(
                    "MES — Marginal Expected Shortfall",
                    "MES_i(t) = −min( σ_f(t)·ρ(t)·k₁ + σ_f(t)·√(1−ρ²)·k₂ , 0 )",
                    "k₁, k₂ are Silverman-kernel weighted expectations of the "
                    "standardised residuals conditional on the market return "
                    "falling below its α-th percentile."),
                _formula_block(
                    "LRMES — Long-Run MES",
                    "β(t)    = ρ(t) · σ_f(t) / σ_m(t)\n"
                    "LRMES(t) = 1 − exp(log(1−d) · β(t))",
                    "Horizon stress d = 40% market decline; no simulation (closed-form approximation)."),
                _formula_block(
                    "SES — Systemic Expected Shortfall",
                    "SES(t) = max(0, k·D(t)·(1+ΔD/D) − (1−k)·W(t)·(1+ΔW/W))",
                    "k = 8% prudential ratio; D = book liabilities; W = market cap. "
                    "SES = 0 is binding only when market equity exceeds the "
                    "capital requirement, i.e. 'Not binding'."),
                _formula_block(
                    "CoVaR / ΔCoVaR",
                    "r_m = b₀ + b₁·r_f  (quantile regression at α)\n"
                    "VaR_i(t)    = σ_f(t)·c_i     with c_i = α-quantile of r_f/σ_f\n"
                    "CoVaR_i(t)  = b₀ + b₁·VaR_i(t)\n"
                    "ΔCoVaR_i(t) = b₁ · (VaR_i(t) − median_i)",
                    "IRLS quantile regression (mirrors Belluzzo 2020 MATLAB). "
                    "Currently cached at α = 5%; the α-slider does not recompute ΔCoVaR."),
                _formula_block(
                    "SRISK — Capital shortfall under stress",
                    "SRISK(t) = max(0, k·D̃(t) − (1−k)·(1−LRMES(t))·W(t))",
                    "D̃ = forward-rolled liabilities (quarterly step function, "
                    "matches Belluzzo's forward_roll_data.m). "
                    "k = 8%, LRMES as above."),
                _formula_block(
                    "LVG — Quasi-leverage",
                    "LVG(t) = (D(t) + W(t)) / W(t)",
                    "Matches NYU Stern V-Lab's LVG column."),

                html.Hr(),
                html.H6("Data sources", className="mb-2 mt-3",
                        style={"fontWeight": "700"}),
                html.Ul([
                    html.Li([html.B("Prices, balance sheets, separate accounts:"),
                             " Yahoo Finance via yfinance."]),
                    html.Li([html.B("Rates & yields (Fed Funds, 10Y, 3M, VIX):"),
                             " Yahoo Finance (ZQ=F, ^TNX, ^IRX, ^VIX)."]),
                    html.Li([html.B("Credit spread (BAA10YM):"),
                             " FRED — no yfinance equivalent."]),
                ], style={"fontSize": "0.85rem", "color": TEXT_MUTED}),

                html.Hr(),
                html.H6("References", className="mb-2 mt-3",
                        style={"fontWeight": "700"}),
                html.Ol([
                    html.Li([
                        "Acharya, V. V., Pedersen, L. H., Philippon, T., & "
                        "Richardson, M. (2010, 2017). ",
                        html.I("Measuring Systemic Risk. "),
                        "Review of Financial Studies 30(1): 2–47."]),
                    html.Li([
                        "Adrian, T., & Brunnermeier, M. K. (2016). ",
                        html.I("CoVaR. "),
                        "American Economic Review 106(7): 1705–1741."]),
                    html.Li([
                        "Brownlees, C., & Engle, R. F. (2017). ",
                        html.I("SRISK: A Conditional Capital Shortfall Measure of Systemic Risk. "),
                        "Review of Financial Studies 30(1): 48–79."]),
                    html.Li([
                        "Engle, R. F. (2002). ",
                        html.I("Dynamic Conditional Correlation. "),
                        "Journal of Business & Economic Statistics 20(3): 339–350."]),
                    html.Li([
                        "Glosten, L. R., Jagannathan, R., & Runkle, D. E. (1993). ",
                        html.I("On the Relation between the Expected Value and the "
                               "Volatility of the Nominal Excess Return on Stocks. "),
                        "Journal of Finance 48(5): 1779–1801."]),
                    html.Li([
                        "Belluzzo, T. (2020). ",
                        html.A("github.com/TommasoBelluzzo/SystemicRisk",
                               href="https://github.com/TommasoBelluzzo/SystemicRisk",
                               target="_blank"),
                        " — MATLAB reference implementation."]),
                    html.Li([
                        "NYU Stern V-Lab — ",
                        html.A("vlab.stern.nyu.edu/srisk",
                               href="https://vlab.stern.nyu.edu/srisk",
                               target="_blank"),
                        " — published SRISK estimates for ~1 200 firms."]),
                ], style={"fontSize": "0.85rem"}),

                html.Hr(),
                html.H6("Parameters", className="mb-2 mt-3",
                        style={"fontWeight": "700"}),
                html.Ul([
                    html.Li(["Prudential capital ratio ", html.B("k = 8%")]),
                    html.Li(["LRMES stress threshold ", html.B("d = 40% (1-month)")]),
                    html.Li(["Forward-roll frequency ", html.B("3 months (quarterly filings)")]),
                    html.Li(["Default critical value ", html.B("α = 5%"),
                             " (adjustable via slider for MES, LRMES, SES, SRISK)"]),
                    html.Li(["Estimation window: rolling 5 years of daily log returns"]),
                ], style={"fontSize": "0.85rem", "color": TEXT_MUTED}),
            ], style=_card),
        ], xs=12),
    ]),
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
    controls,
    dcc.Store(id="refresh-store",      data=0),
    dcc.Store(id="alpha-store",        data=0.05),
    dcc.Store(id="custom-banks-store", data={}),
    # Polls the background refresh worker; starts disabled, enabled by btn-refresh.
    dcc.Interval(id="refresh-interval", interval=500,
                 disabled=True, n_intervals=0),
    dcc.Loading(
        id="loading-main",
        custom_spinner=_loading_bar,
        overlay_style={"opacity": 0, "backgroundColor": "transparent"},
        children=dbc.Tabs([
            dbc.Tab(overview_layout,    label="Overview",        tab_id="tab-overview"),
            dbc.Tab(timeseries_layout,  label="Time Series",     tab_id="tab-ts"),
            dbc.Tab(srisk_layout,       label="SRISK",           tab_id="tab-srisk"),
            dbc.Tab(dcc_layout,         label="DCC Correlation", tab_id="tab-dcc"),
            dbc.Tab(market_layout,      label="Market Data",     tab_id="tab-market"),
            dbc.Tab(methodology_layout, label="Methodology",     tab_id="tab-methodology"),
        ], id="main-tabs", active_tab="tab-overview",
           style={"paddingLeft": "1rem", "backgroundColor": "#f8f9fa",
                  "borderBottom": f"1px solid {BORDER}"}),
    ),
], style={"backgroundColor": BG_PAGE, "minHeight": "100vh"})


# ── Callbacks ─────────────────────────────────────────────────────────────────

# ── Time-range presets ─────────────────────────────────────────────────────────

@app.callback(
    Output("date-range", "start_date"),
    Output("date-range", "end_date"),
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


# ── Bank dropdown ──────────────────────────────────────────────────────────────

@app.callback(
    Output("bank-select", "options"),
    Output("bank-select", "value"),
    Input("btn-all",             "n_clicks"),
    Input("btn-none",            "n_clicks"),
    Input("custom-banks-store",  "data"),
    State("bank-select",         "value"),
    prevent_initial_call=False,
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
        LAST_UPDATED = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")

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


def _refresh_progress_view(state: dict):
    """Render the small progress indicator shown next to the Refresh button."""
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


@app.callback(
    Output("refresh-interval", "disabled"),
    Output("refresh-progress", "children", allow_duplicate=True),
    Input("btn-refresh",        "n_clicks"),
    State("custom-banks-store", "data"),
    prevent_initial_call=True,
)
def start_refresh(n_clicks, custom_banks):
    """Start the background worker (if not already running) and enable polling."""
    if not n_clicks:
        return no_update, no_update
    state = _get_refresh_state()
    if state["running"]:
        # Already refreshing — keep polling, but don't spawn a second thread.
        return False, _refresh_progress_view(state)
    threading.Thread(
        target=_run_refresh_work,
        args=(custom_banks,),
        daemon=True,
    ).start()
    return False, _refresh_progress_view(_get_refresh_state())


@app.callback(
    Output("refresh-progress", "children"),
    Output("refresh-interval", "disabled", allow_duplicate=True),
    Output("refresh-store",    "data",     allow_duplicate=True),
    Output("updated-ts",       "children", allow_duplicate=True),
    Input("refresh-interval",  "n_intervals"),
    State("refresh-store",     "data"),
    prevent_initial_call=True,
)
def poll_refresh(_n, current):
    """Poll the refresh worker; when it finishes, bump refresh-store once."""
    state = _get_refresh_state()
    view  = _refresh_progress_view(state)
    if state["running"]:
        return view, False, no_update, no_update
    # Worker is not running: either finished successfully, errored, or never
    # started.  Disable the interval and, on success, bump refresh-store.
    if state.get("error"):
        return view, True, no_update, no_update
    if state.get("completed_at"):
        new_counter = (current or 0) + 1
        ts_text = f"Updated: {state['completed_at']}"
        return view, True, new_counter, ts_text
    # No refresh has ever run — stop polling quietly.
    return view, True, no_update, no_update


# ── Alpha recompute ────────────────────────────────────────────────────────────

@app.callback(
    Output("alpha-store", "data"),
    Output("alpha-label", "children"),
    Input("alpha-slider", "value"),
    prevent_initial_call=True,
)
def update_alpha(alpha_pct):
    """Recompute MES/LRMES/SES/SRISK for new α; snap ΔCoVaR to nearest grid point."""
    global MEASURES
    alpha = alpha_pct / 100.0
    snap  = _nearest_alpha(alpha)
    print(f"\n[α] Recomputing MES/LRMES/SES/SRISK for α={alpha:.3f}; ΔCoVaR snapped to {snap:.3f}")
    new = M.recompute_for_alpha(RETURNS, MC_TS, LB_DAILY, LBR_DAILY, BS, state_vars=STATE_VARS, alpha=alpha)
    # Swap in the precomputed ΔCoVaR for the nearest grid alpha
    snap_df = DCOVAR_BY_ALPHA.get(round(snap, 4))
    if snap_df is not None:
        new["delta_covar"] = snap_df
    for key in new:
        base = MEASURES.get(key)
        if base is None or base.empty:
            MEASURES[key] = new[key]
            continue
        updated = base.copy()
        for col in new[key].columns:
            updated[col] = new[key][col].reindex(updated.index)
        MEASURES[key] = updated
    print("[α] Done.")
    return alpha, (
        f"Critical Value α = {alpha_pct:.1f}% (worst {alpha_pct:.1f}% of market days). "
        f"ΔCoVaR uses nearest grid α = {snap*100:.1f}%."
    )


# ── Overview tab ──────────────────────────────────────────────────────────────

@app.callback(
    Output("kpi-mes",          "children"),
    Output("kpi-lrmes",        "children"),
    Output("kpi-ses",          "children"),
    Output("kpi-covar",        "children"),
    Output("kpi-srisk",        "children"),
    Output("kpi-leverage",     "children"),
    Output("chart-mes-rank",   "figure"),
    Output("chart-ses-rank",   "figure"),
    Output("chart-covar-rank", "figure"),
    Output("chart-mes-dw",     "figure"),
    Output("chart-srisk-dw",   "figure"),
    Output("chart-covar-dw",   "figure"),
    Output("risk-table",       "children"),
    Input("date-range",    "start_date"),
    Input("date-range",    "end_date"),
    Input("bank-select",   "value"),
    Input("refresh-store", "data"),
    Input("alpha-store",   "data"),
)
def update_overview(start, end, tickers, _refresh, _alpha):
    tickers = tickers or []
    mes_df    = _slice(MEASURES["mes"],         start, end, tickers)
    lrmes_df  = _slice(MEASURES.get("lrmes", pd.DataFrame()), start, end, tickers)
    ses_df    = _slice(MEASURES["ses"],         start, end, tickers)
    dcovar_df = _slice(MEASURES["delta_covar"], start, end, tickers)
    srisk_df  = _slice(MEASURES["srisk"],       start, end, tickers)
    lvg_df    = _slice(LEVERAGE,                start, end, tickers) if not LEVERAGE.empty else pd.DataFrame()

    latest_mes   = _latest_row(mes_df)
    latest_lrmes = _latest_row(lrmes_df)
    latest_ses   = _latest_row(ses_df)
    latest_covar = _latest_row(dcovar_df)
    latest_srisk = _latest_row(srisk_df)
    latest_lvg   = _latest_row(lvg_df)

    agg_mes   = latest_mes.mean()
    agg_lrmes = latest_lrmes.mean() if not latest_lrmes.empty else float("nan")
    agg_ses   = latest_ses[latest_ses > 0].mean() if (latest_ses > 0).any() else float("nan")
    agg_covar = latest_covar.mean()
    agg_srisk = latest_srisk.sum()
    agg_lvg   = latest_lvg.mean() if not latest_lvg.empty else float("nan")

    kpi_mes  = kpi_card("Avg. MES (latest)",
                        _fmt_pct(agg_mes),
                        "Mean 1-day tail loss",
                        "#c62828")
    kpi_lrmes = kpi_card("Avg. LRMES (latest)",
                        _fmt_pct(agg_lrmes),
                        "Long-run MES at 40% market decline",
                        "#ad1457")
    kpi_ses  = kpi_card("Avg. SES (binding)",
                        _fmt_ses(agg_ses),
                        "Capital shortfall (mean of banks with SES > 0)",
                        "#6a1b9a")
    kpi_cov  = kpi_card("Avg. |ΔCoVaR| (latest)",
                        _fmt_pct(abs(agg_covar) if not pd.isna(agg_covar) else float("nan")),
                        "Mean marginal systemic contribution",
                        "#e65100")
    kpi_srisk = kpi_card("Total SRISK (latest)",
                         _fmt_bn(agg_srisk if agg_srisk > 0 else float("nan")),
                         "Aggregate capital shortfall estimate",
                         "#2e7d32")
    kpi_lvg  = kpi_card("Avg. Leverage (LVG)",
                        _fmt_ratio(agg_lvg),
                        "(Liabilities + Market Cap) / Market Cap",
                        "#0277bd")

    # Overview ranking bars — show only the top 10 banks per measure
    _TOP_N = 10
    _mes_top   = latest_mes.dropna().nlargest(_TOP_N)
    _ses_pos   = latest_ses[latest_ses > 0].dropna()
    _ses_top   = _ses_pos.nlargest(_TOP_N)
    _covar_abs = latest_covar.abs().dropna()
    _covar_top = _covar_abs.nlargest(_TOP_N)

    fig_mes   = ranking_bar(_mes_top,   "MES Ranking — Top 10 (latest)",        "MES")
    fig_ses   = ranking_bar(_ses_top,   "SES Ranking — Top 10 (latest)",        "SES (capital shortfall)", fmt_fn=_fmt_bn)
    fig_covar = ranking_bar(_covar_top, "|ΔCoVaR| Ranking — Top 10 (latest)",   "|ΔCoVaR|")

    # ── Δ-last-week / Δ-last-month deltas ─────────────────────────────────────
    # asof returns the last value on or before the given date.
    def _delta(df: pd.DataFrame, days: int) -> pd.Series:
        if df.empty:
            return pd.Series(dtype=float)
        last = df.dropna(how="all")
        if last.empty:
            return pd.Series(dtype=float)
        last_dt  = last.index[-1]
        prev_dt  = last_dt - pd.Timedelta(days=days)
        try:
            prev = last.asof(prev_dt)
        except Exception:
            prev = pd.Series(np.nan, index=last.columns)
        return last.iloc[-1] - prev

    d_mes_w  = _delta(mes_df,    7)
    d_mes_m  = _delta(mes_df,    30)
    d_cov_w  = _delta(dcovar_df, 7)
    d_sri_w  = _delta(srisk_df,  7)
    d_sri_m  = _delta(srisk_df,  30)

    # Summary table as sortable DataTable with Δ-week / Δ-month columns
    all_tickers = sorted(
        set(latest_mes.index) | set(latest_ses.index) | set(latest_covar.index)
    )
    table_rows = []
    for t in all_tickers:
        table_rows.append({
            "bank":      _name(t),
            "ticker":    t,
            "mes":       float(latest_mes.get(t,   np.nan)) if pd.notna(latest_mes.get(t,   np.nan)) else None,
            "mes_dw":    float(d_mes_w.get(t,      np.nan)) if pd.notna(d_mes_w.get(t,      np.nan)) else None,
            "mes_dm":    float(d_mes_m.get(t,      np.nan)) if pd.notna(d_mes_m.get(t,      np.nan)) else None,
            "lrmes":     float(latest_lrmes.get(t, np.nan)) if pd.notna(latest_lrmes.get(t, np.nan)) else None,
            "ses_bn":    (float(latest_ses.get(t,  np.nan)) / 1e9) if pd.notna(latest_ses.get(t,  np.nan)) else None,
            "covar":     float(latest_covar.get(t, np.nan)) if pd.notna(latest_covar.get(t, np.nan)) else None,
            "covar_dw":  float(d_cov_w.get(t,      np.nan)) if pd.notna(d_cov_w.get(t,      np.nan)) else None,
            "srisk_bn":  (float(latest_srisk.get(t, np.nan)) / 1e9) if pd.notna(latest_srisk.get(t, np.nan)) else None,
            "srisk_dw":  (float(d_sri_w.get(t,    np.nan)) / 1e9) if pd.notna(d_sri_w.get(t,    np.nan)) else None,
            "srisk_dm":  (float(d_sri_m.get(t,    np.nan)) / 1e9) if pd.notna(d_sri_m.get(t,    np.nan)) else None,
            "lvg":       float(latest_lvg.get(t,   np.nan)) if pd.notna(latest_lvg.get(t,   np.nan)) else None,
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
        {"name": "SES (bn)",     "id": "ses_bn",    "type": "numeric", "format": num_fmt2},
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
            {"if": {"filter_query": "{mes_dw} > 0",   "column_id": "mes_dw"},   "color": "#c62828"},
            {"if": {"filter_query": "{mes_dw} < 0",   "column_id": "mes_dw"},   "color": "#2e7d32"},
            {"if": {"filter_query": "{mes_dm} > 0",   "column_id": "mes_dm"},   "color": "#c62828"},
            {"if": {"filter_query": "{mes_dm} < 0",   "column_id": "mes_dm"},   "color": "#2e7d32"},
            {"if": {"filter_query": "{covar_dw} > 0", "column_id": "covar_dw"}, "color": "#c62828"},
            {"if": {"filter_query": "{covar_dw} < 0", "column_id": "covar_dw"}, "color": "#2e7d32"},
            {"if": {"filter_query": "{srisk_dw} > 0", "column_id": "srisk_dw"}, "color": "#c62828"},
            {"if": {"filter_query": "{srisk_dw} < 0", "column_id": "srisk_dw"}, "color": "#2e7d32"},
            {"if": {"filter_query": "{srisk_dm} > 0", "column_id": "srisk_dm"}, "color": "#c62828"},
            {"if": {"filter_query": "{srisk_dm} < 0", "column_id": "srisk_dm"}, "color": "#2e7d32"},
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

    fig_mes_dw   = delta_ranking_bar(_top_by_abs(d_mes_w), "Δ MES (1w) — Top 10",     "Δ MES (pp)")
    fig_srisk_dw = delta_ranking_bar(_top_by_abs(d_sri_w), "Δ SRISK (1w) — Top 10",   "Δ SRISK (bn)", divide_bn=True)
    fig_covar_dw = delta_ranking_bar(_top_by_abs(d_cov_w), "Δ ΔCoVaR (1w) — Top 10", "Δ ΔCoVaR (pp)")

    return (kpi_mes, kpi_lrmes, kpi_ses, kpi_cov, kpi_srisk, kpi_lvg,
            fig_mes, fig_ses, fig_covar,
            fig_mes_dw, fig_srisk_dw, fig_covar_dw,
            table)


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
        "ses":         ("SES",       "SES (capital shortfall)"),
        "delta_covar": ("ΔCoVaR",    "ΔCoVaR"),
        "covar":       ("CoVaR",     "CoVaR"),
    }
    label, ylabel = labels.get(measure, (measure, measure))

    # Portfolio aggregate:
    #   SES is a USD capital-shortfall dollar amount → cumulative across banks.
    #   Everything else (MES, LRMES, ΔCoVaR, CoVaR) is a fraction/level → mean.
    _AGG_MODE = {
        "mes":         ("mean", "Mean"),
        "lrmes":       ("mean", "Mean"),
        "ses":         ("sum",  "Total"),
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


def _compute_srisk_df(k: float) -> pd.DataFrame:
    """
    Recompute SRISK across all banks for a user-selected prudential ratio k.

    Vectorised equivalent of :func:`systemic_measures.compute_srisk` applied
    column-wise:

        SRISK(t) = max(0, k·LBR(t) − (1−k)·(1−LRMES(t))·MC(t))

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
    lb  = LBR_DAILY.reindex(index=idx, columns=cols)
    cap = MC_TS.reindex(index=idx, columns=cols)
    srisk = k * lb - (1.0 - k) * (1.0 - lr) * cap
    return srisk.clip(lower=0.0)


@app.callback(
    Output("chart-srisk-bar", "figure"),
    Output("chart-srisk-pie", "figure"),
    Output("chart-srisk-ts",  "figure"),
    Output("srisk-k-label",   "children"),
    Input("date-range",    "start_date"),
    Input("date-range",    "end_date"),
    Input("bank-select",   "value"),
    Input("srisk-norm",    "value"),
    Input("srisk-k-slider","value"),
    Input("refresh-store", "data"),
    Input("alpha-store",   "data"),
)
def update_srisk(start, end, tickers, norm, k_pct, _refresh, _alpha):
    tickers = tickers or []
    # Convert slider (percent) to ratio; recompute SRISK for this k.
    k = float(k_pct) / 100.0 if k_pct is not None else 0.08
    srisk_all = _compute_srisk_df(k)
    if srisk_all.empty:
        srisk_all = MEASURES.get("srisk", pd.DataFrame())
    df      = _slice(srisk_all, start, end, tickers)
    latest  = _latest_row(df)
    mc_latest = _latest_row(_slice(MC_TS, start, end, tickers)) if not MC_TS.empty else pd.Series(dtype=float)

    if norm == "pct_agg":
        total = latest.sum()
        latest_disp = (latest / total * 100.0) if total > 0 else latest * np.nan
        title_bar = "SRISK by Bank (% of aggregate SRISK)"
        xlabel_bar = "% of aggregate SRISK"
        fmt_bar = _fmt_pct_raw
    elif norm == "pct_mc":
        mc_aligned = mc_latest.reindex(latest.index)
        latest_disp = (latest / mc_aligned.where(mc_aligned > 0, np.nan)) * 100.0
        title_bar = "SRISK by Bank (% of own market cap)"
        xlabel_bar = "% of market cap"
        fmt_bar = _fmt_pct_raw
    else:  # 'abs'
        latest_disp = latest / 1e9
        title_bar = "SRISK by Bank (USD bn)"
        xlabel_bar = "SRISK (bn USD)"
        fmt_bar = _fmt_bn_x1  # already in bn

    fig_bar = _srisk_bar_generic(latest_disp, title_bar, xlabel_bar, fmt_bar, norm)
    fig_pie = srisk_pie(latest)  # pie always uses absolute values for share

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
        fillcolor="rgba(198,40,40,0.12)",
        line=dict(color="#c62828", width=2),
        hovertemplate=f"Date: %{{x}}<br>Total: {y_hover}<extra></extra>",
    ))
    _add_crisis_overlays(fig_ts, start, end)
    fig_ts.update_layout(
        title=dict(text="Aggregate SRISK over Time",
                   font=dict(size=14)),
        yaxis_title=y_label,
        height=300,
        **_base_layout(),
    )

    k_label = f"Prudential capital ratio  k = {k * 100:.1f}%"
    return fig_bar, fig_pie, fig_ts, k_label


# ── Market data tab ───────────────────────────────────────────────────────────

@app.callback(
    Output("chart-prices",       "figure"),
    Output("chart-returns-hist", "figure"),
    Output("chart-corr",         "figure"),
    Input("date-range",    "start_date"),
    Input("date-range",    "end_date"),
    Input("bank-select",   "value"),
    Input("refresh-store", "data"),
    Input("alpha-store",   "data"),
)
def update_market(start, end, tickers, _refresh, _alpha):
    tickers = tickers or []
    keep    = tickers + [MARKET_NAME]
    prices  = _slice(PRICES,   start, end, keep)
    rets    = _slice(RETURNS,  start, end, tickers)

    return price_chart(prices), return_hist(rets, tickers), corr_heatmap(rets)


# ── DCC Correlation tab ───────────────────────────────────────────────────────

@app.callback(
    Output("chart-dcc-rho", "figure"),
    Output("chart-dcc-avg", "figure"),
    Input("date-range",    "start_date"),
    Input("date-range",    "end_date"),
    Input("bank-select",   "value"),
    Input("dcc-crises",    "value"),
    Input("refresh-store", "data"),
)
def update_dcc_tab(start, end, tickers, crises, _refresh):
    tickers = tickers or []
    empty = go.Figure().update_layout(title="No DCC ρ data", **_base_layout())
    if DCC_RHO.empty:
        return empty, empty

    df = _slice(DCC_RHO, start, end, tickers)
    if df.empty:
        return empty, empty

    show_crises = bool(crises and "show" in crises)

    # ρ per bank
    fig_rho = go.Figure()
    for ticker in df.columns:
        s = df[ticker].dropna()
        if s.empty:
            continue
        fig_rho.add_trace(go.Scatter(
            x=s.index, y=s.values,
            name=_name(ticker),
            line=dict(color=_color(ticker), width=1.6),
            hovertemplate=f"{_name(ticker)}: %{{y:.3f}}<extra></extra>",
        ))
    if show_crises:
        _add_crisis_overlays(fig_rho, start, end)
    fig_rho.update_layout(
        title=dict(text=f"DCC ρ(t) — Correlation with {MARKET_NAME}",
                   font=dict(size=14)),
        yaxis_title="ρ(t)",
        yaxis=dict(range=[-0.2, 1.0]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, font_size=11),
        height=420,
        **_base_layout(),
    )

    # Mean ρ across selected banks
    avg = df.mean(axis=1)
    fig_avg = go.Figure(go.Scatter(
        x=avg.index, y=avg.values,
        line=dict(color="#0d47a1", width=2),
        fill="tozeroy",
        fillcolor="rgba(13, 71, 161, 0.08)",
        hovertemplate="Date: %{x}<br>Mean ρ: %{y:.3f}<extra></extra>",
    ))
    if not avg.empty:
        fig_avg.add_hline(y=float(avg.mean()), line_dash="dot",
                          line_color="#555",
                          annotation_text="Sample mean",
                          annotation_position="top left",
                          annotation_font_size=10)
    if show_crises:
        _add_crisis_overlays(fig_avg, start, end)
    fig_avg.update_layout(
        title=dict(text="Average DCC ρ across selected banks",
                   font=dict(size=14)),
        yaxis_title="Mean ρ(t)",
        yaxis=dict(range=[-0.2, 1.0]),
        height=300,
        **_base_layout(),
    )
    return fig_rho, fig_avg


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
    for key in ("mes", "lrmes", "ses", "delta_covar", "srisk"):
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
    prevent_initial_call=True,
)
def download_srisk_csv(_n, start, end, tickers, k_pct):
    tickers = tickers or []
    k = float(k_pct) / 100.0 if k_pct is not None else 0.08
    srisk_all = _compute_srisk_df(k)
    if srisk_all.empty:
        srisk_all = MEASURES.get("srisk", pd.DataFrame())
    df = _slice(srisk_all, start, end, tickers)
    if df.empty:
        return no_update
    df = df.copy()
    df["Aggregate"] = df.sum(axis=1)
    df.index.name = "Date"
    stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
    return _csv_download(df, f"srisk_timeseries_k{int(round(k*1000))}bp_{stamp}.csv")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug = False)
