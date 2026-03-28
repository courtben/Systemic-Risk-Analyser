"""
Swiss Banking Systemic Risk Dashboard
======================================
Interactive Dash application visualising MES, ΔCoVaR, and SRISK for
major Swiss financial institutions.

Run:
    pip install -r requirements.txt
    python app.py
Then open http://127.0.0.1:8050 in your browser.
"""

import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import date, timedelta

import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc

warnings.filterwarnings("ignore")

# ── Load / compute data at startup ────────────────────────────────────────────

print("=" * 60)
print("Swiss Banking Systemic Risk Dashboard")
print("=" * 60)

import data as D
import measures as M

print("\n[1/4] Fetching price data ...")
PRICES = D.get_prices()

print("\n[2/4] Computing returns ...")
RETURNS = D.compute_returns(PRICES)

print("\n[3/4] Fetching balance sheet data ...")
BS = D.get_balance_sheet()

print("\n[4/4] Computing systemic risk measures ...")
MC_TS = D.build_market_cap_series(PRICES, BS)
MEASURES = M.compute_all(RETURNS, MC_TS, BS)

LAST_UPDATED = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
print(f"\nDashboard ready  ({LAST_UPDATED})")
print("=" * 60)

# ── Constants ─────────────────────────────────────────────────────────────────

BANKS       = D.SWISS_BANKS
BANK_COLORS = D.BANK_COLORS
MARKET_NAME = D.MARKET_NAME

# Dropdown options (only banks that have at least some data)
available_banks = [
    {"label": name, "value": ticker}
    for ticker, name in BANKS.items()
    if ticker in RETURNS.columns
]

PLOTLY_TEMPLATE = "plotly_dark"

# ── Helper functions ──────────────────────────────────────────────────────────

def _slice(df: pd.DataFrame, start, end, tickers=None) -> pd.DataFrame:
    """Date-slice a DataFrame and optionally filter columns."""
    out = df.loc[str(start):str(end)]
    if tickers:
        out = out[[c for c in tickers if c in out.columns]]
    return out


def _latest_row(df: pd.DataFrame) -> pd.Series:
    """Return the last non-all-NaN row."""
    return df.dropna(how="all").iloc[-1]


def _bank_name(ticker: str) -> str:
    return BANKS.get(ticker, ticker)


def _color(ticker: str) -> str:
    return BANK_COLORS.get(ticker, "#aaaaaa")


def _format_bn(x) -> str:
    if pd.isna(x):
        return "N/A"
    return f"{x / 1e9:.2f} bn CHF"


def _format_pct(x) -> str:
    if pd.isna(x):
        return "N/A"
    return f"{x * 100:.2f}%"


# ── Chart builders ────────────────────────────────────────────────────────────

def ranking_bar(series: pd.Series, title: str, xlabel: str,
                color_fn=None, fmt_fn=_format_pct) -> go.Figure:
    s = series.dropna().sort_values(ascending=False)
    colors = [color_fn(t) if color_fn else "#1f77b4" for t in s.index]
    labels = [_bank_name(t) for t in s.index]
    text   = [fmt_fn(v) for v in s.values]

    fig = go.Figure(go.Bar(
        x=s.values, y=labels,
        orientation="h",
        marker_color=colors,
        text=text, textposition="outside",
        hovertemplate="%{y}: %{text}<extra></extra>",
    ))
    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis=dict(autorange="reversed"),
        template=PLOTLY_TEMPLATE,
        margin=dict(l=10, r=80, t=40, b=30),
        height=280,
    )
    return fig


def timeseries_chart(
    df: pd.DataFrame, title: str, ylabel: str,
    fmt_fn=_format_pct, market_ret: pd.Series | None = None,
) -> go.Figure:
    rows = 2 if market_ret is not None else 1
    fig  = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        row_heights=[0.75, 0.25] if rows == 2 else [1.0],
        vertical_spacing=0.05,
    )

    for ticker in df.columns:
        s = df[ticker].dropna()
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values,
            name=_bank_name(ticker),
            line=dict(color=_color(ticker), width=1.5),
            hovertemplate=f"{_bank_name(ticker)}: %{{y:.3f}}<extra></extra>",
        ), row=1, col=1)

    if market_ret is not None:
        fig.add_trace(go.Bar(
            x=market_ret.index, y=market_ret.values,
            name=MARKET_NAME,
            marker_color=np.where(market_ret.values >= 0, "#27ae60", "#e74c3c"),
            opacity=0.6,
            hovertemplate="SMI return: %{y:.3f}<extra></extra>",
            showlegend=True,
        ), row=2, col=1)
        fig.update_yaxes(title_text="SMI Return", row=2, col=1)

    fig.update_layout(
        title=title,
        yaxis_title=ylabel,
        template=PLOTLY_TEMPLATE,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=50, b=30),
        height=480,
    )
    return fig


def srisk_bar(series: pd.Series, title: str) -> go.Figure:
    s = series.dropna().sort_values(ascending=False)
    if s.empty:
        return go.Figure().update_layout(title="No SRISK data available",
                                          template=PLOTLY_TEMPLATE)
    colors = [_color(t) for t in s.index]
    labels = [_bank_name(t) for t in s.index]
    text   = [_format_bn(v) for v in s.values]

    fig = go.Figure(go.Bar(
        x=s.values / 1e9, y=labels,
        orientation="h",
        marker_color=colors,
        text=text, textposition="outside",
        hovertemplate="%{y}: %{text}<extra></extra>",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="SRISK (bn CHF)",
        yaxis=dict(autorange="reversed"),
        template=PLOTLY_TEMPLATE,
        margin=dict(l=10, r=100, t=40, b=30),
        height=280,
    )
    return fig


def srisk_pie(series: pd.Series) -> go.Figure:
    s = series.dropna()
    s = s[s > 0]
    if s.empty:
        return go.Figure().update_layout(title="No positive SRISK",
                                          template=PLOTLY_TEMPLATE)
    labels = [_bank_name(t) for t in s.index]
    colors = [_color(t) for t in s.index]

    fig = go.Figure(go.Pie(
        labels=labels, values=s.values,
        marker_colors=colors,
        textinfo="label+percent",
        hovertemplate="%{label}: %{value:.2e} CHF (%{percent})<extra></extra>",
    ))
    fig.update_layout(
        title="SRISK Share (%)",
        template=PLOTLY_TEMPLATE,
        margin=dict(l=10, r=10, t=40, b=10),
        height=280,
    )
    return fig


def price_performance_chart(prices: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for ticker in prices.columns:
        if ticker == MARKET_NAME:
            continue
        s = prices[ticker].dropna()
        if s.empty:
            continue
        rebased = s / s.iloc[0] * 100
        fig.add_trace(go.Scatter(
            x=rebased.index, y=rebased.values,
            name=_bank_name(ticker),
            line=dict(color=_color(ticker), width=1.5),
        ))

    # Add market index
    if MARKET_NAME in prices.columns:
        s = prices[MARKET_NAME].dropna()
        rebased = s / s.iloc[0] * 100
        fig.add_trace(go.Scatter(
            x=rebased.index, y=rebased.values,
            name=MARKET_NAME,
            line=dict(color="#ffffff", width=2, dash="dash"),
        ))

    fig.update_layout(
        title="Rebased Price Performance (100 = start of period)",
        yaxis_title="Index (Start = 100)",
        template=PLOTLY_TEMPLATE,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=50, b=30),
        height=380,
    )
    return fig


def correlation_heatmap(returns: pd.DataFrame) -> go.Figure:
    cols  = [c for c in returns.columns if c != MARKET_NAME]
    corr  = returns[cols].corr()
    names = [_bank_name(t) for t in corr.columns]

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
        title="Return Correlation Matrix",
        template=PLOTLY_TEMPLATE,
        margin=dict(l=10, r=10, t=40, b=10),
        height=380,
    )
    return fig


def return_hist(returns: pd.DataFrame, tickers: list) -> go.Figure:
    fig = go.Figure()
    for ticker in tickers:
        if ticker not in returns.columns:
            continue
        s = returns[ticker].dropna() * 100
        fig.add_trace(go.Histogram(
            x=s.values,
            name=_bank_name(ticker),
            opacity=0.6,
            nbinsx=80,
            marker_color=_color(ticker),
        ))
    fig.update_layout(
        title="Daily Return Distribution (%)",
        xaxis_title="Daily Return (%)",
        barmode="overlay",
        template=PLOTLY_TEMPLATE,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=50, b=30),
        height=300,
    )
    return fig


# ── KPI card helper ───────────────────────────────────────────────────────────

def kpi_card(title: str, value: str, subtitle: str = "", color: str = "#1f77b4"):
    return dbc.Card([
        dbc.CardBody([
            html.P(title, className="text-muted mb-1",
                   style={"fontSize": "0.85rem", "fontWeight": "600", "letterSpacing": "0.04em"}),
            html.H4(value, style={"color": color, "fontWeight": "700", "marginBottom": "2px"}),
            html.P(subtitle, className="text-muted mb-0",
                   style={"fontSize": "0.78rem"}),
        ])
    ], style={"backgroundColor": "#16213e", "border": "1px solid #0f3460"})


# ── App layout ────────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    suppress_callback_exceptions=True,
    title="Swiss Banking Systemic Risk",
)

# Date bounds
DATE_MIN = RETURNS.index.min().date()
DATE_MAX = RETURNS.index.max().date()
DATE_DEF_START = max(DATE_MIN, date(2015, 1, 1))

# ── Header ────────────────────────────────────────────────────────────────────
header = dbc.Navbar(
    dbc.Container([
        html.Div([
            html.H5("🏦 Swiss Banking — Systemic Risk Monitor",
                    className="mb-0", style={"color": "#e0e0e0", "fontWeight": "700"}),
            html.Small(f"Data: Yahoo Finance / yfinance  ·  "
                       f"Methodology: Acharya et al. (2017), Adrian & Brunnermeier (2016)",
                       className="text-muted"),
        ]),
        html.Div([
            html.Small(f"Last updated: {LAST_UPDATED}", className="text-muted me-3"),
            dbc.Button("⟳ Refresh", id="btn-refresh", size="sm",
                       color="secondary", outline=True),
        ], className="d-flex align-items-center"),
    ], fluid=True, className="d-flex justify-content-between align-items-center"),
    color="dark", dark=True, sticky="top",
    style={"borderBottom": "2px solid #0f3460"},
)

# ── Control bar ───────────────────────────────────────────────────────────────
controls = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.Label("Date range", className="text-muted mb-1",
                       style={"fontSize": "0.8rem"}),
            dcc.DatePickerRange(
                id="date-range",
                min_date_allowed=DATE_MIN,
                max_date_allowed=DATE_MAX,
                start_date=DATE_DEF_START,
                end_date=DATE_MAX,
                display_format="YYYY-MM-DD",
                style={"fontSize": "0.85rem"},
            ),
        ], xs=12, md=5, className="mb-2"),

        dbc.Col([
            html.Label("Banks", className="text-muted mb-1",
                       style={"fontSize": "0.8rem"}),
            dcc.Dropdown(
                id="bank-select",
                options=available_banks,
                value=[b["value"] for b in available_banks],
                multi=True,
                placeholder="Select banks ...",
                style={"fontSize": "0.85rem"},
            ),
        ], xs=12, md=5, className="mb-2"),

        dbc.Col([
            html.Label("Significance α", className="text-muted mb-1",
                       style={"fontSize": "0.8rem"}),
            dcc.Dropdown(
                id="alpha-select",
                options=[
                    {"label": "1%",  "value": 0.01},
                    {"label": "5%",  "value": 0.05},
                    {"label": "10%", "value": 0.10},
                ],
                value=0.05, clearable=False,
                style={"fontSize": "0.85rem"},
            ),
        ], xs=12, md=2, className="mb-2"),
    ], className="gy-1"),
], fluid=True, className="py-2 px-3",
   style={"backgroundColor": "#0d0d1a", "borderBottom": "1px solid #0f3460"})

# ── Hidden store for refresh trigger ─────────────────────────────────────────
stores = html.Div([
    dcc.Store(id="refresh-store", data=0),
    dcc.Interval(id="noop-interval", interval=9999999, n_intervals=0),
])

# ── Tab layouts ───────────────────────────────────────────────────────────────

overview_layout = dbc.Container([
    # KPI row
    dbc.Row([
        dbc.Col(id="kpi-mes",    xs=12, md=4, className="mb-3"),
        dbc.Col(id="kpi-covar",  xs=12, md=4, className="mb-3"),
        dbc.Col(id="kpi-srisk",  xs=12, md=4, className="mb-3"),
    ], className="mt-3"),
    # Rankings
    dbc.Row([
        dbc.Col(dcc.Graph(id="chart-mes-rank"),   xs=12, md=6, className="mb-3"),
        dbc.Col(dcc.Graph(id="chart-covar-rank"), xs=12, md=6, className="mb-3"),
    ]),
    # Risk table
    dbc.Row([
        dbc.Col([
            html.H6("Risk Summary Table",
                    className="text-muted mb-2", style={"fontSize": "0.85rem"}),
            html.Div(id="risk-table"),
        ], xs=12),
    ], className="mb-3"),
], fluid=True)

timeseries_layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.Label("Measure", className="text-muted mb-1",
                       style={"fontSize": "0.8rem"}),
            dcc.Dropdown(
                id="ts-measure",
                options=[
                    {"label": "MES — Marginal Expected Shortfall",
                     "value": "mes"},
                    {"label": "ΔCoVaR — Conditional Value at Risk",
                     "value": "delta_covar"},
                    {"label": "CoVaR (level)",
                     "value": "covar"},
                ],
                value="mes", clearable=False,
                style={"fontSize": "0.85rem"},
            ),
        ], xs=12, md=4, className="mt-3 mb-2"),
        dbc.Col([
            html.Label("Overlay SMI returns", className="text-muted mb-1",
                       style={"fontSize": "0.8rem"}),
            dbc.Checklist(
                id="ts-overlay",
                options=[{"label": " Show SMI daily returns", "value": "show"}],
                value=["show"],
                switch=True,
            ),
        ], xs=12, md=4, className="mt-3 mb-2"),
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id="chart-timeseries"), xs=12),
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Interpretation", className="card-title text-info"),
                    html.P([
                        html.B("MES"), " — expected fractional loss of the bank "
                        "when the market index falls below its 5th-percentile threshold. "
                        "Higher MES = higher contribution to tail risk.",
                        html.Br(),
                        html.B("ΔCoVaR"), " — difference between the market's VaR when "
                        "the bank is in stress versus at its median state (Adrian & Brunnermeier 2016). "
                        "More negative ΔCoVaR = larger systemic footprint.",
                    ], className="mb-0", style={"fontSize": "0.82rem"}),
                ])
            ], style={"backgroundColor": "#16213e", "border": "1px solid #0f3460"}),
        ], xs=12, className="mb-3"),
    ]),
], fluid=True)

srisk_layout = dbc.Container([
    dbc.Row([
        dbc.Col(dcc.Graph(id="chart-srisk-bar"), xs=12, md=7, className="mt-3 mb-3"),
        dbc.Col(dcc.Graph(id="chart-srisk-pie"), xs=12, md=5, className="mt-3 mb-3"),
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id="chart-srisk-ts"), xs=12, className="mb-3"),
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("SRISK Methodology", className="card-title text-info"),
                    html.P([
                        "SRISK = max(0, k·D − (1−k)·(1−LRMES)·W)",
                        html.Br(),
                        "where k = 8% (prudential capital ratio), "
                        "D = book liabilities, W = market cap, "
                        "LRMES = 1 − exp(−22·MES) (1-month horizon approximation).",
                        html.Br(),
                        "SRISK represents the expected capital shortfall conditional on a market crisis. "
                        "Balance sheet data sourced from Yahoo Finance (quarterly, latest available).",
                    ], className="mb-0", style={"fontSize": "0.82rem"}),
                ])
            ], style={"backgroundColor": "#16213e", "border": "1px solid #0f3460"}),
        ], xs=12, className="mb-3"),
    ]),
], fluid=True)

market_layout = dbc.Container([
    dbc.Row([
        dbc.Col(dcc.Graph(id="chart-prices"), xs=12, className="mt-3 mb-3"),
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id="chart-returns-hist"), xs=12, md=6, className="mb-3"),
        dbc.Col(dcc.Graph(id="chart-corr"),         xs=12, md=6, className="mb-3"),
    ]),
], fluid=True)

# ── Main layout ───────────────────────────────────────────────────────────────

app.layout = html.Div([
    header,
    controls,
    stores,
    dbc.Tabs([
        dbc.Tab(overview_layout,    label="Overview",    tab_id="tab-overview"),
        dbc.Tab(timeseries_layout,  label="Time Series", tab_id="tab-ts"),
        dbc.Tab(srisk_layout,       label="SRISK",       tab_id="tab-srisk"),
        dbc.Tab(market_layout,      label="Market Data", tab_id="tab-market"),
    ], id="main-tabs", active_tab="tab-overview",
       style={"backgroundColor": "#0d0d1a",
              "borderBottom": "1px solid #0f3460",
              "paddingLeft": "1rem"}),
], style={"backgroundColor": "#0a0a1a", "minHeight": "100vh"})


# ── Callbacks ─────────────────────────────────────────────────────────────────

@app.callback(
    Output("refresh-store", "data"),
    Input("btn-refresh", "n_clicks"),
    State("refresh-store", "data"),
    prevent_initial_call=True,
)
def refresh_data(n_clicks, current):
    """Re-download prices and recompute all measures on button click."""
    global PRICES, RETURNS, BS, MC_TS, MEASURES, LAST_UPDATED
    print("\n[Refresh] Re-fetching data ...")
    PRICES   = D.get_prices(force_refresh=True)
    RETURNS  = D.compute_returns(PRICES)
    BS       = D.get_balance_sheet(force_refresh=True)
    MC_TS    = D.build_market_cap_series(PRICES, BS)
    MEASURES = M.compute_all(RETURNS, MC_TS, BS, force_refresh=True)
    LAST_UPDATED = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    print("[Refresh] Done.")
    return (current or 0) + 1


# ── Overview tab ──────────────────────────────────────────────────────────────

@app.callback(
    Output("kpi-mes",         "children"),
    Output("kpi-covar",       "children"),
    Output("kpi-srisk",       "children"),
    Output("chart-mes-rank",  "figure"),
    Output("chart-covar-rank","figure"),
    Output("risk-table",      "children"),
    Input("date-range",   "start_date"),
    Input("date-range",   "end_date"),
    Input("bank-select",  "value"),
    Input("refresh-store","data"),
)
def update_overview(start, end, tickers, _refresh):
    mes_df    = _slice(MEASURES["mes"],         start, end, tickers)
    dcovar_df = _slice(MEASURES["delta_covar"], start, end, tickers)
    srisk_df  = _slice(MEASURES["srisk"],       start, end, tickers)

    latest_mes    = _latest_row(mes_df)    if not mes_df.empty    else pd.Series(dtype=float)
    latest_covar  = _latest_row(dcovar_df) if not dcovar_df.empty else pd.Series(dtype=float)
    latest_srisk  = _latest_row(srisk_df)  if not srisk_df.empty  else pd.Series(dtype=float)

    agg_mes   = latest_mes.mean()
    agg_covar = latest_covar.mean()
    agg_srisk = latest_srisk.sum()

    kpi1 = kpi_card(
        "Avg. MES (latest)",
        _format_pct(agg_mes),
        "Mean expected loss on market crash days",
        "#e94560",
    )
    kpi2 = kpi_card(
        "Avg. |ΔCoVaR| (latest)",
        _format_pct(abs(agg_covar) if not pd.isna(agg_covar) else np.nan),
        "Mean marginal systemic contribution",
        "#f39c12",
    )
    kpi3 = kpi_card(
        "Total SRISK (latest)",
        _format_bn(agg_srisk if agg_srisk > 0 else np.nan),
        "Aggregate capital shortfall estimate",
        "#27ae60",
    )

    fig_mes   = ranking_bar(latest_mes,          "MES Ranking (latest)",    "MES",    _color)
    fig_covar = ranking_bar(latest_covar.abs(),  "|ΔCoVaR| Ranking (latest)", "|ΔCoVaR|", _color)

    # Summary table
    rows = []
    all_tickers = sorted(
        set(latest_mes.index) | set(latest_covar.index) | set(latest_srisk.index)
    )
    for t in all_tickers:
        mes_v   = latest_mes.get(t,   np.nan)
        cov_v   = latest_covar.get(t, np.nan)
        srisk_v = latest_srisk.get(t, np.nan)
        rows.append(html.Tr([
            html.Td(html.Span(
                "●", style={"color": _color(t), "marginRight": "6px"}
            )),
            html.Td(_bank_name(t)),
            html.Td(t, style={"color": "#888"}),
            html.Td(_format_pct(mes_v)),
            html.Td(_format_pct(cov_v)),
            html.Td(_format_bn(srisk_v)),
        ]))

    table = dbc.Table(
        [
            html.Thead(html.Tr([
                html.Th(""),
                html.Th("Bank"),
                html.Th("Ticker"),
                html.Th("MES"),
                html.Th("ΔCoVaR"),
                html.Th("SRISK"),
            ])),
            html.Tbody(rows),
        ],
        bordered=False, hover=True, responsive=True,
        style={"fontSize": "0.85rem"},
    )

    return kpi1, kpi2, kpi3, fig_mes, fig_covar, table


# ── Time-series tab ───────────────────────────────────────────────────────────

@app.callback(
    Output("chart-timeseries", "figure"),
    Input("date-range",  "start_date"),
    Input("date-range",  "end_date"),
    Input("bank-select", "value"),
    Input("ts-measure",  "value"),
    Input("ts-overlay",  "value"),
    Input("refresh-store","data"),
)
def update_timeseries(start, end, tickers, measure, overlay, _refresh):
    df = _slice(MEASURES[measure], start, end, tickers)

    measure_labels = {
        "mes":         ("MES",    "MES (loss fraction)"),
        "delta_covar": ("ΔCoVaR", "ΔCoVaR"),
        "covar":       ("CoVaR",  "CoVaR"),
    }
    label, ylabel = measure_labels.get(measure, (measure, measure))

    mkt = None
    if overlay and "show" in overlay:
        ret_slice = _slice(RETURNS[[MARKET_NAME]], start, end)
        if MARKET_NAME in ret_slice.columns:
            mkt = ret_slice[MARKET_NAME]

    return timeseries_chart(df, f"Rolling {label} — Swiss Banks", ylabel,
                            market_ret=mkt)


# ── SRISK tab ─────────────────────────────────────────────────────────────────

@app.callback(
    Output("chart-srisk-bar", "figure"),
    Output("chart-srisk-pie", "figure"),
    Output("chart-srisk-ts",  "figure"),
    Input("date-range",   "start_date"),
    Input("date-range",   "end_date"),
    Input("bank-select",  "value"),
    Input("refresh-store","data"),
)
def update_srisk(start, end, tickers, _refresh):
    df = _slice(MEASURES["srisk"], start, end, tickers)

    latest = _latest_row(df) if not df.empty else pd.Series(dtype=float)

    fig_bar = srisk_bar(latest, "SRISK by Bank (latest, bn CHF)")
    fig_pie = srisk_pie(latest)

    # Aggregate SRISK over time
    agg = df.sum(axis=1)
    fig_ts = go.Figure(go.Scatter(
        x=agg.index, y=agg.values / 1e9,
        fill="tozeroy",
        line=dict(color="#e94560", width=2),
        hovertemplate="Date: %{x}<br>Total SRISK: %{y:.1f} bn CHF<extra></extra>",
    ))
    fig_ts.update_layout(
        title="Aggregate SRISK over Time",
        yaxis_title="Aggregate SRISK (bn CHF)",
        template=PLOTLY_TEMPLATE,
        margin=dict(l=10, r=10, t=50, b=30),
        height=320,
    )

    return fig_bar, fig_pie, fig_ts


# ── Market data tab ───────────────────────────────────────────────────────────

@app.callback(
    Output("chart-prices",       "figure"),
    Output("chart-returns-hist", "figure"),
    Output("chart-corr",         "figure"),
    Input("date-range",   "start_date"),
    Input("date-range",   "end_date"),
    Input("bank-select",  "value"),
    Input("refresh-store","data"),
)
def update_market(start, end, tickers, _refresh):
    price_slice = _slice(PRICES, start, end, (tickers or []) + [MARKET_NAME])
    ret_slice   = _slice(RETURNS, start, end, tickers)

    fig_prices = price_performance_chart(price_slice)
    fig_hist   = return_hist(ret_slice, tickers or [])
    fig_corr   = correlation_heatmap(ret_slice)

    return fig_prices, fig_hist, fig_corr


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8050)
