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

import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import os
from plotly.subplots import make_subplots

import dash
from dash import dcc, html, Input, Output, State
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

LAST_UPDATED = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
print(f"\nReady  ({LAST_UPDATED})")
print("=" * 60)

# ── Constants ─────────────────────────────────────────────────────────────────

ALL_BANKS   = D.ALL_BANKS
BANK_COLORS = D.BANK_COLORS
MARKET_NAME = D.MARKET_NAME

PLOTLY_TEMPLATE = "plotly_white"

# Style tokens (light theme)
BG_PAGE    = "#f4f6f9"
BG_CARD    = "#ffffff"
BG_HEADER  = "#ffffff"
BORDER     = "#dee2e6"
TEXT_MUTED = "#6c757d"
TEXT_MAIN  = "#212529"

# ── Helper functions ──────────────────────────────────────────────────────────

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


def _name(ticker: str) -> str:
    return ALL_BANKS.get(ticker, ticker)


def _color(ticker: str) -> str:
    return BANK_COLORS.get(ticker, "#aaaaaa")


def _fmt_bn(x) -> str:
    if pd.isna(x):
        return "N/A"
    if x == 0:
        return "0.00 bn"
    return f"{x / 1e9:.2f} bn"


def _fmt_pct(x) -> str:
    return "N/A" if pd.isna(x) else f"{x * 100:.2f}%"


# ── Chart builders ────────────────────────────────────────────────────────────

def _base_layout(**kwargs) -> dict:
    return dict(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor=BG_CARD,
        plot_bgcolor=BG_CARD,
        font=dict(color=TEXT_MAIN, size=12),
        margin=dict(l=10, r=10, t=45, b=30),
        **kwargs,
    )


def ranking_bar(series: pd.Series, title: str, xlabel: str,
                fmt_fn=_fmt_pct) -> go.Figure:
    s      = series.dropna().sort_values(ascending=False)
    colors = [_color(t) for t in s.index]
    labels = [_name(t) for t in s.index]
    text   = [fmt_fn(v) for v in s.values]
    # Scale to billions for bn-formatted charts so x-axis is readable
    xvals  = s.values / 1e9 if fmt_fn is _fmt_bn else s.values

    fig = go.Figure(go.Bar(
        x=xvals, y=labels,
        orientation="h",
        marker_color=colors,
        marker_line_width=0,
        text=text, textposition="auto",
        insidetextanchor="middle",
        hovertemplate="%{y}: %{text}<extra></extra>",
    ))
    base = _base_layout()
    base["margin"] = dict(l=10, r=90, t=45, b=30)
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis_title=xlabel,
        yaxis=dict(autorange="reversed"),
        height=300,
        **base,
    )
    return fig


def timeseries_chart(
    df: pd.DataFrame, title: str, ylabel: str,
    market_ret: pd.Series | None = None,
) -> go.Figure:
    rows = 2 if market_ret is not None else 1
    fig  = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        row_heights=[0.75, 0.25] if rows == 2 else [1.0],
        vertical_spacing=0.06,
    )

    for ticker in df.columns:
        s = df[ticker].dropna()
        lbl = _name(ticker)
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values,
            name=lbl,
            line=dict(color=_color(ticker), width=1.8),
            hovertemplate=f"{_name(ticker)}: %{{y:.4f}}<extra></extra>",
        ), row=1, col=1)

    if market_ret is not None:
        clrs = np.where(market_ret.values >= 0, "#2e7d32", "#c62828")
        fig.add_trace(go.Bar(
            x=market_ret.index, y=market_ret.values,
            name=MARKET_NAME,
            marker_color=clrs,
            opacity=0.55,
            hovertemplate=f"{MARKET_NAME}: %{{y:.4f}}<extra></extra>",
        ), row=2, col=1)
        fig.update_yaxes(title_text="Mkt Return", row=2, col=1, title_font_size=11)

    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        yaxis_title=ylabel,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, font_size=11),
        height=480,
        **_base_layout(),
    )
    return fig


def srisk_bar(series: pd.Series, title: str) -> go.Figure:
    s = series.dropna().sort_values(ascending=False)
    if s.empty:
        return go.Figure().update_layout(title="No SRISK data",
                                          **_base_layout())
    labels = [_name(t) for t in s.index]

    fig = go.Figure(go.Bar(
        x=s.values / 1e9, y=labels,
        orientation="h",
        marker_color=[_color(t) for t in s.index],
        marker_line_width=0,
        text=[_fmt_bn(v) for v in s.values],
        textposition="auto",
        insidetextanchor="middle",
        hovertemplate="%{y}: %{text}<extra></extra>",
    ))
    base = _base_layout()
    base["margin"] = dict(l=10, r=80, t=45, b=30)
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis_title="SRISK (bn, native currency)",
        yaxis=dict(autorange="reversed"),
        height=320,
        **base,
    )
    return fig


def srisk_pie(series: pd.Series) -> go.Figure:
    s = series.dropna()
    s = s[s > 0]
    if s.empty:
        return go.Figure().update_layout(title="No positive SRISK", **_base_layout())
    labels = [_name(t) for t in s.index]
    fig = go.Figure(go.Pie(
        labels=labels, values=s.values,
        marker_colors=[_color(t) for t in s.index],
        textinfo="label+percent",
        hovertemplate="%{label}: %{value:.2e} (%{percent})<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="SRISK Share (%)", font=dict(size=14)),
        height=320,
        **_base_layout(),
    )
    return fig


def price_chart(prices: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for ticker in prices.columns:
        if ticker not in ALL_BANKS:
            continue
        s = prices[ticker].dropna()
        if len(s) < 2:
            continue
        rebased = s / s.iloc[0] * 100
        lbl = _name(ticker)
        fig.add_trace(go.Scatter(
            x=rebased.index, y=rebased.values,
            name=lbl,
            line=dict(color=_color(ticker), width=1.6),
        ))

    # Market index
    if MARKET_NAME in prices.columns:
        s = prices[MARKET_NAME].dropna()
        if len(s) >= 2:
            rebased = s / s.iloc[0] * 100
            fig.add_trace(go.Scatter(
                x=rebased.index, y=rebased.values,
                name=MARKET_NAME,
                line=dict(color="#333333", width=2, dash="dot"),
            ))

    fig.update_layout(
        title=dict(text="Rebased Price Performance (100 = start of period)",
                   font=dict(size=14)),
        yaxis_title="Index (start = 100)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, font_size=11),
        height=380,
        **_base_layout(),
    )
    return fig


def corr_heatmap(returns: pd.DataFrame) -> go.Figure:
    cols  = [c for c in returns.columns if c in ALL_BANKS]
    if not cols:
        return go.Figure().update_layout(title="No data", **_base_layout())
    corr  = returns[cols].corr()
    names = [_name(t) for t in corr.columns]

    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=names, y=names,
        colorscale="RdBu_r",
        zmin=-1, zmax=1,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        hovertemplate="%{x} / %{y}: %{z:.2f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Return Correlation Matrix", font=dict(size=14)),
        height=420,
        **_base_layout(),
    )
    return fig


def return_hist(returns: pd.DataFrame, tickers: list) -> go.Figure:
    fig = go.Figure()
    for ticker in tickers:
        if ticker not in returns.columns:
            continue
        s = returns[ticker].dropna() * 100
        lbl = _name(ticker)
        fig.add_trace(go.Histogram(
            x=s.values, name=lbl, opacity=0.6, nbinsx=80,
            marker_color=_color(ticker),
        ))
    fig.update_layout(
        title=dict(text="Daily Return Distribution (%)", font=dict(size=14)),
        xaxis_title="Daily Return (%)",
        barmode="overlay",
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, font_size=11),
        height=300,
        **_base_layout(),
    )
    return fig


# ── KPI card ──────────────────────────────────────────────────────────────────

def kpi_card(title: str, value: str, subtitle: str, accent: str) -> dbc.Card:
    return dbc.Card([
        dbc.CardBody([
            html.P(title, className="mb-1 text-muted",
                   style={"fontSize": "0.78rem", "fontWeight": "600",
                          "letterSpacing": "0.05em", "textTransform": "uppercase"}),
            html.H4(value, style={"color": accent, "fontWeight": "700",
                                   "marginBottom": "2px"}),
            html.P(subtitle, className="mb-0 text-muted",
                   style={"fontSize": "0.78rem"}),
        ])
    ], style={"backgroundColor": BG_CARD, "border": f"1px solid {BORDER}",
              "borderLeft": f"4px solid {accent}",
              "boxShadow": "0 1px 3px rgba(0,0,0,0.06)"})


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
            html.Small(f"Updated: {LAST_UPDATED}", className="text-muted me-3"),
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
        ], xs=12, md=5, className="mb-2"),

        # Select / deselect all
        dbc.Col([
            html.Label("\u00a0", className="mb-1 d-block",
                       style={"fontSize": "0.78rem"}),
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
                "Changing α recomputes MES, SES & SRISK interactively. "
                "ΔCoVaR uses the cached α=5% value (quantile regression is too slow to rerun on-the-fly).",
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
        dbc.Col(id="kpi-mes",   xs=12, md=3, className="mb-3"),
        dbc.Col(id="kpi-ses",   xs=12, md=3, className="mb-3"),
        dbc.Col(id="kpi-covar", xs=12, md=3, className="mb-3"),
        dbc.Col(id="kpi-srisk", xs=12, md=3, className="mb-3"),
    ], className="mt-3"),
    dbc.Row([
        dbc.Col(dcc.Graph(id="chart-mes-rank"),   xs=12, md=4, className="mb-3"),
        dbc.Col(dcc.Graph(id="chart-ses-rank"),   xs=12, md=4, className="mb-3"),
        dbc.Col(dcc.Graph(id="chart-covar-rank"), xs=12, md=4, className="mb-3"),
    ]),
    dbc.Row([
        dbc.Col([
            html.P("Risk Summary (latest date in selected range)",
                   className="text-muted mb-2",
                   style={"fontSize": "0.82rem", "fontWeight": "600"}),
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
                    {"label": "SES — Systemic Expected Shortfall",        "value": "ses"},
                    {"label": "ΔCoVaR — Conditional VaR contribution",   "value": "delta_covar"},
                    {"label": "CoVaR (level)",                            "value": "covar"},
                ],
                value="mes", clearable=False,
                style={"fontSize": "0.85rem"},
            ),
        ], xs=12, md=5, className="mt-3 mb-2"),
        dbc.Col([
            html.Label("Overlay market returns", className="text-muted mb-1",
                       style={"fontSize": "0.78rem", "fontWeight": "600"}),
            dbc.Checklist(
                id="ts-overlay",
                options=[{"label": f" Show {MARKET_NAME} daily returns", "value": "show"}],
                value=["show"], switch=True, className="mt-1",
            ),
        ], xs=12, md=4, className="mt-3 mb-2"),
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
                    html.Span("Uses cached α = 5%; not recomputed when slider changes.",
                              style={"fontStyle": "italic"}),
                ], className="mb-0 text-muted", style={"fontSize": "0.82rem"}),
            ], style=_card),
        ], xs=12, className="mb-3"),
    ]),
], fluid=True, style={"backgroundColor": BG_PAGE})

srisk_layout = dbc.Container([
    dbc.Row([
        dbc.Col(dcc.Graph(id="chart-srisk-bar"), xs=12, md=7, className="mt-3 mb-3"),
        dbc.Col(dcc.Graph(id="chart-srisk-pie"), xs=12, md=5, className="mt-3 mb-3"),
    ]),
    dbc.Row([dbc.Col(dcc.Graph(id="chart-srisk-ts"), xs=12, className="mb-3")]),
    dbc.Row([
        dbc.Col([
            html.Div([
                html.P([
                    html.B("Formula: "), "SRISK = max(0, k·D − (1−k)·(1−LRMES)·W)",
                    html.Br(),
                    "k = 8% prudential capital ratio · "
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
    dcc.Loading(
        id="loading-main",
        custom_spinner=_loading_bar,
        overlay_style={"opacity": 0, "backgroundColor": "transparent"},
        children=dbc.Tabs([
            dbc.Tab(overview_layout,   label="Overview",    tab_id="tab-overview"),
            dbc.Tab(timeseries_layout, label="Time Series", tab_id="tab-ts"),
            dbc.Tab(srisk_layout,      label="SRISK",       tab_id="tab-srisk"),
            dbc.Tab(market_layout,     label="Market Data", tab_id="tab-market"),
        ], id="main-tabs", active_tab="tab-overview",
           style={"paddingLeft": "1rem", "backgroundColor": "#f8f9fa",
                  "borderBottom": f"1px solid {BORDER}"}),
    ),
], style={"backgroundColor": BG_PAGE, "minHeight": "100vh"})


# ── Callbacks ─────────────────────────────────────────────────────────────────

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
    options = [
        {"label": f"{name}  ({ticker})", "value": ticker}
        for ticker, name in all_banks.items()
        if ticker in RETURNS.columns
    ]
    valid = [o["value"] for o in options]

    if "btn-none" in triggered:
        return options, []

    if "custom-banks-store" in triggered:
        # Keep current selection and auto-select any newly added tickers
        current = set(current_values or [])
        new_tickers = [t for t in (custom_banks or {}) if t not in current]
        return options, list(current) + new_tickers

    # Initial load or All: select everything
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
    global PRICES, RETURNS, MC_TS, BS, LIAB_TS, SEP_ACCT_TS, LB_DAILY, LBR_DAILY, STATE_VARS, MEASURES, ALL_BANKS
    from dash import no_update

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
    MEASURES["ses"][ticker]         = ses_s
    MEASURES["covar"][ticker]       = covar_s
    MEASURES["delta_covar"][ticker] = dcovar_s
    MEASURES["srisk"][ticker]       = srisk_s

    # ── Update display name lookup ────────────────────────────────────────────
    ALL_BANKS[ticker] = name

    print(f"[Add Bank] Done: {name} ({ticker})")
    new_custom = {**(custom_banks or {}), ticker: name}
    msg = dbc.Alert(
        f"Added {name} ({ticker})", color="success", dismissable=True,
        className="py-1 mb-0", style={"fontSize": "0.8rem"})
    return new_custom, msg


# ── Refresh ────────────────────────────────────────────────────────────────────

@app.callback(
    Output("refresh-store", "data"),
    Input("btn-refresh", "n_clicks"),
    State("refresh-store",      "data"),
    State("custom-banks-store", "data"),
    prevent_initial_call=True,
)
def refresh_data(n_clicks, current, custom_banks):
    global PRICES, RETURNS, BS, MC_TS, LIAB_TS, SEP_ACCT_TS, LB_DAILY, LBR_DAILY, STATE_VARS, MEASURES, LAST_UPDATED, ALL_BANKS
    print("\n[Refresh] Re-fetching data ...")
    PRICES    = D.get_prices(force_refresh=True)
    RETURNS   = D.compute_returns(PRICES)
    BS        = D.get_balance_sheet(force_refresh=True)
    LIAB_TS   = D.get_liabilities_ts(force_refresh=True)
    SEP_ACCT_TS = D.get_separate_accounts_ts(force_refresh=True)
    LB_DAILY  = D.build_lb_daily(LIAB_TS, PRICES, SEP_ACCT_TS)
    LBR_DAILY = D.build_lbr_daily(LIAB_TS, PRICES, SEP_ACCT_TS)
    STATE_VARS = D.get_state_variables(PRICES, force_refresh=True)
    MC_TS     = D.build_market_cap_series(PRICES, BS)
    MEASURES  = M.compute_all(RETURNS, MC_TS, LB_DAILY, LBR_DAILY, BS, state_vars=STATE_VARS, force_refresh=True)

    # Re-add any custom banks the user had added this session
    for ticker, name in (custom_banks or {}).items():
        if ticker not in RETURNS.columns:
            bank_data = D.fetch_single_bank(ticker)
            if bank_data is None:
                continue
            PRICES[ticker]  = bank_data["prices"].reindex(PRICES.index)
            RETURNS[ticker] = D.compute_returns(PRICES[[ticker]]).iloc[:, 0]
            BS[ticker]      = bank_data["balance_sheet"]
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
            # Compute measures for the custom bank
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
            MEASURES["ses"][ticker]         = (M.compute_ses(lb_ts, cp_ts)
                                               if cp_ts is not None and lb_ts is not None
                                               else pd.Series(np.nan, index=mes_s.index))
            MEASURES["covar"][ticker]       = covar_s
            MEASURES["delta_covar"][ticker] = dcovar_s
            MEASURES["srisk"][ticker]       = (M.compute_srisk(lrmes_s, lbr_ts, cp_ts)
                                               if cp_ts is not None and lbr_ts is not None
                                               else pd.Series(np.nan, index=mes_s.index))

    LAST_UPDATED = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    print("[Refresh] Done.")
    return (current or 0) + 1


# ── Alpha recompute ────────────────────────────────────────────────────────────

@app.callback(
    Output("alpha-store", "data"),
    Output("alpha-label", "children"),
    Input("alpha-slider", "value"),
    prevent_initial_call=True,
)
def update_alpha(alpha_pct):
    """Recompute MES, SES, SRISK for new alpha. ΔCoVaR keeps cached value."""
    global MEASURES
    alpha = alpha_pct / 100.0
    print(f"\n[α] Recomputing MES/SES/SRISK for α={alpha:.3f} ...")
    new = M.recompute_for_alpha(RETURNS, MC_TS, LB_DAILY, LBR_DAILY, BS, state_vars=STATE_VARS, alpha=alpha)
    # Merge column-by-column so custom banks not in `new` are preserved
    for key in new:
        updated = MEASURES[key].copy()
        for col in new[key].columns:
            updated[col] = new[key][col].reindex(updated.index)
        MEASURES[key] = updated
    print("[α] Done.")
    return alpha, f"Critical Value α = {alpha_pct:.1f}%  (worst {alpha_pct:.1f}% of market days)"


# ── Overview tab ──────────────────────────────────────────────────────────────

@app.callback(
    Output("kpi-mes",          "children"),
    Output("kpi-ses",          "children"),
    Output("kpi-covar",        "children"),
    Output("kpi-srisk",        "children"),
    Output("chart-mes-rank",   "figure"),
    Output("chart-ses-rank",   "figure"),
    Output("chart-covar-rank", "figure"),
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
    ses_df    = _slice(MEASURES["ses"],         start, end, tickers)
    dcovar_df = _slice(MEASURES["delta_covar"], start, end, tickers)
    srisk_df  = _slice(MEASURES["srisk"],       start, end, tickers)

    latest_mes   = _latest_row(mes_df)
    latest_ses   = _latest_row(ses_df)
    latest_covar = _latest_row(dcovar_df)
    latest_srisk = _latest_row(srisk_df)

    agg_mes   = latest_mes.mean()
    agg_ses   = latest_ses.mean()
    agg_covar = latest_covar.mean()
    agg_srisk = latest_srisk.sum()

    kpi_mes  = kpi_card("Avg. MES (latest)",
                        _fmt_pct(agg_mes),
                        "Mean expected loss on market crash days",
                        "#c62828")
    kpi_ses  = kpi_card("Avg. SES (latest)",
                        _fmt_bn(agg_ses if not pd.isna(agg_ses) and agg_ses > 0 else float("nan")),
                        "Capital shortfall estimate (Acharya et al. 2010)",
                        "#6a1b9a")
    kpi_cov  = kpi_card("Avg. |ΔCoVaR| (latest)",
                        _fmt_pct(abs(agg_covar) if not pd.isna(agg_covar) else float("nan")),
                        "Mean marginal systemic contribution",
                        "#e65100")
    kpi_srisk = kpi_card("Total SRISK (latest, native ccy)",
                         _fmt_bn(agg_srisk if agg_srisk > 0 else float("nan")),
                         "Aggregate capital shortfall estimate",
                         "#2e7d32")

    fig_mes   = ranking_bar(latest_mes,         "MES Ranking (latest)",         "MES")
    fig_ses   = ranking_bar(latest_ses[latest_ses > 0], "SES Ranking (latest)",  "SES (capital shortfall)", fmt_fn=_fmt_bn)
    fig_covar = ranking_bar(latest_covar.abs(), "|ΔCoVaR| Ranking (latest)",   "|ΔCoVaR|")

    # Summary table
    all_tickers = sorted(
        set(latest_mes.index) | set(latest_ses.index) | set(latest_covar.index)
    )
    rows = []
    for t in all_tickers:
        rows.append(html.Tr([
            html.Td(html.Span("●", style={"color": _color(t)})),
            html.Td(_name(t), style={"fontWeight": "500"}),
            html.Td(t, className="text-muted", style={"fontSize": "0.8rem"}),
            html.Td(_fmt_pct(latest_mes.get(t,   float("nan")))),
            html.Td(_fmt_bn(latest_ses.get(t,    float("nan")))),
            html.Td(_fmt_pct(latest_covar.get(t, float("nan")))),
            html.Td(_fmt_bn(latest_srisk.get(t,  float("nan")))),
        ]))

    table = dbc.Table(
        [
            html.Thead(html.Tr([
                html.Th(""), html.Th("Bank"),
                html.Th("Ticker"), html.Th("MES"), html.Th("SES (bn)"),
                html.Th("ΔCoVaR"), html.Th("SRISK"),
            ]), style={"backgroundColor": "#f8f9fa"}),
            html.Tbody(rows),
        ],
        bordered=True, hover=True, responsive=True, size="sm",
        style={"fontSize": "0.85rem", "backgroundColor": BG_CARD},
    )

    return kpi_mes, kpi_ses, kpi_cov, kpi_srisk, fig_mes, fig_ses, fig_covar, table


# ── Time series tab ───────────────────────────────────────────────────────────

@app.callback(
    Output("chart-timeseries", "figure"),
    Input("date-range",    "start_date"),
    Input("date-range",    "end_date"),
    Input("bank-select",   "value"),
    Input("ts-measure",    "value"),
    Input("ts-overlay",    "value"),
    Input("refresh-store", "data"),
    Input("alpha-store",   "data"),
)
def update_timeseries(start, end, tickers, measure, overlay, _refresh, _alpha):
    tickers = tickers or []
    df = _slice(MEASURES[measure], start, end, tickers)

    labels = {
        "mes":         ("MES",       "MES (loss fraction)"),
        "ses":         ("SES",       "SES (leverage-scaled)"),
        "delta_covar": ("ΔCoVaR",    "ΔCoVaR"),
        "covar":       ("CoVaR",     "CoVaR"),
    }
    label, ylabel = labels.get(measure, (measure, measure))

    mkt = None
    if overlay and "show" in overlay:
        r = _slice(RETURNS[[MARKET_NAME]], start, end)
        if MARKET_NAME in r.columns:
            mkt = r[MARKET_NAME]

    return timeseries_chart(df, f"Rolling {label}", ylabel, market_ret=mkt)


# ── SRISK tab ─────────────────────────────────────────────────────────────────

@app.callback(
    Output("chart-srisk-bar", "figure"),
    Output("chart-srisk-pie", "figure"),
    Output("chart-srisk-ts",  "figure"),
    Input("date-range",    "start_date"),
    Input("date-range",    "end_date"),
    Input("bank-select",   "value"),
    Input("refresh-store", "data"),
    Input("alpha-store",   "data"),
)
def update_srisk(start, end, tickers, _refresh, _alpha):
    tickers = tickers or []
    df      = _slice(MEASURES["srisk"], start, end, tickers)
    latest  = _latest_row(df)

    fig_bar = srisk_bar(latest, "SRISK by Bank (latest, native currency)")
    fig_pie = srisk_pie(latest)

    agg = df.sum(axis=1)
    fig_ts = go.Figure(go.Scatter(
        x=agg.index, y=agg.values / 1e9,
        fill="tozeroy",
        fillcolor="rgba(198,40,40,0.12)",
        line=dict(color="#c62828", width=2),
        hovertemplate="Date: %{x}<br>Total: %{y:.1f} bn<extra></extra>",
    ))
    fig_ts.update_layout(
        title=dict(text="Aggregate SRISK over Time (sum, native currencies)",
                   font=dict(size=14)),
        yaxis_title="Aggregate SRISK (bn)",
        height=300,
        **_base_layout(),
    )

    return fig_bar, fig_pie, fig_ts


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


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug = False)
