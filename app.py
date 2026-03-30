"""
Systemic Risk Dashboard — Switzerland, US & UK Banks
======================================================
Interactive Dash application visualising MES, DeltaCoVaR, and SRISK
for major banking institutions across three countries.

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
print("Systemic Risk Dashboard  —  CH / US / UK Banks")
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
MC_TS    = D.build_market_cap_series(PRICES, BS)
MEASURES = M.compute_all(RETURNS, MC_TS, BS)

LAST_UPDATED = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
print(f"\nReady  ({LAST_UPDATED})")
print("=" * 60)

# ── Constants ─────────────────────────────────────────────────────────────────

BANKS_BY_COUNTRY = D.BANKS_BY_COUNTRY
ALL_BANKS        = D.ALL_BANKS
BANK_COLORS      = D.BANK_COLORS
BANK_COUNTRY     = D.BANK_COUNTRY
COUNTRY_LABELS   = D.COUNTRY_LABELS
MARKET_NAME      = D.MARKET_NAME

PLOTLY_TEMPLATE  = "plotly_white"

# Country flag emoji map
COUNTRY_FLAGS = {"CH": "🇨🇭", "US": "🇺🇸", "UK": "🇬🇧"}

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
    return df.dropna(how="all").iloc[-1] if not df.empty else pd.Series(dtype=float)


def _name(ticker: str) -> str:
    return ALL_BANKS.get(ticker, ticker)


def _color(ticker: str) -> str:
    return BANK_COLORS.get(ticker, "#aaaaaa")


def _fmt_bn(x) -> str:
    return "N/A" if pd.isna(x) or x == 0 else f"{x / 1e9:.2f} bn"


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
    labels = [f"{COUNTRY_FLAGS.get(BANK_COUNTRY.get(t,''), '')} {_name(t)}"
              for t in s.index]
    text   = [fmt_fn(v) for v in s.values]

    fig = go.Figure(go.Bar(
        x=s.values, y=labels,
        orientation="h",
        marker_color=colors,
        marker_line_width=0,
        text=text, textposition="outside",
        hovertemplate="%{y}: %{text}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis_title=xlabel,
        yaxis=dict(autorange="reversed"),
        height=300,
        **_base_layout(),
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
        lbl = f"{COUNTRY_FLAGS.get(BANK_COUNTRY.get(ticker,''),'')} {_name(ticker)}"
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
    labels = [f"{COUNTRY_FLAGS.get(BANK_COUNTRY.get(t,''),'')} {_name(t)}"
              for t in s.index]

    fig = go.Figure(go.Bar(
        x=s.values / 1e9, y=labels,
        orientation="h",
        marker_color=[_color(t) for t in s.index],
        marker_line_width=0,
        text=[_fmt_bn(v) for v in s.values],
        textposition="outside",
        hovertemplate="%{y}: %{text}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis_title="SRISK (bn, native currency)",
        yaxis=dict(autorange="reversed"),
        height=320,
        margin=dict(l=10, r=100, t=45, b=30),
        **_base_layout(),
    )
    return fig


def srisk_pie(series: pd.Series) -> go.Figure:
    s = series.dropna()
    s = s[s > 0]
    if s.empty:
        return go.Figure().update_layout(title="No positive SRISK", **_base_layout())
    labels = [f"{COUNTRY_FLAGS.get(BANK_COUNTRY.get(t,''),'')} {_name(t)}"
              for t in s.index]
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
        lbl = f"{COUNTRY_FLAGS.get(BANK_COUNTRY.get(ticker,''),'')} {_name(ticker)}"
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
    names = [f"{COUNTRY_FLAGS.get(BANK_COUNTRY.get(t,''),'')} {_name(t)}"
             for t in corr.columns]

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
        lbl = f"{COUNTRY_FLAGS.get(BANK_COUNTRY.get(ticker,''),'')} {_name(ticker)}"
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
DATE_DEF_START = max(DATE_MIN, pd.Timestamp("2015-01-01").date())

# ── Header ────────────────────────────────────────────────────────────────────

header = dbc.Navbar(
    dbc.Container([
        html.Div([
            html.H5("Systemic Risk Dashboard — CH / US / UK Banks",
                    className="mb-0",
                    style={"color": TEXT_MAIN, "fontWeight": "700", "fontSize": "1.05rem"}),
            html.Small(
                "MES · ΔCoVaR · SRISK  |  "
                "Acharya et al. (2017) · Adrian & Brunnermeier (2016) · Brownlees & Engle (2017)",
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

        # Country filter
        dbc.Col([
            html.Label("Country", className="text-muted mb-1",
                       style={"fontSize": "0.78rem", "fontWeight": "600"}),
            dcc.Checklist(
                id="country-select",
                options=[
                    {"label": "  🇨🇭 Switzerland", "value": "CH"},
                    {"label": "  🇺🇸 United States", "value": "US"},
                    {"label": "  🇬🇧 United Kingdom", "value": "UK"},
                ],
                value=["CH"],
                inline=True,
                className="mt-1",
                inputStyle={"marginRight": "4px", "cursor": "pointer"},
                labelStyle={"marginRight": "18px", "fontSize": "0.88rem",
                            "cursor": "pointer"},
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
        dbc.Col(id="kpi-mes",   xs=12, md=4, className="mb-3"),
        dbc.Col(id="kpi-covar", xs=12, md=4, className="mb-3"),
        dbc.Col(id="kpi-srisk", xs=12, md=4, className="mb-3"),
    ], className="mt-3"),
    dbc.Row([
        dbc.Col(dcc.Graph(id="chart-mes-rank"),   xs=12, md=6, className="mb-3"),
        dbc.Col(dcc.Graph(id="chart-covar-rank"), xs=12, md=6, className="mb-3"),
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
                    {"label": "MES — Marginal Expected Shortfall",  "value": "mes"},
                    {"label": "ΔCoVaR — Conditional VaR contribution", "value": "delta_covar"},
                    {"label": "CoVaR (level)",                        "value": "covar"},
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
                    f"market ({MARKET_NAME}) falls below its α-th percentile threshold. "
                    "Higher MES = greater contribution to tail risk.",
                    html.Br(),
                    html.B("ΔCoVaR"), " — market's VaR when the bank is in stress minus "
                    "its VaR when the bank is at its median state (Adrian & Brunnermeier 2016). "
                    "More negative ΔCoVaR = larger systemic footprint.",
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
                    html.Span("Note: SRISK values are in each bank's native reporting currency "
                              "(CHF / USD / GBP) and are not directly comparable across countries.",
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

app.layout = html.Div([
    header,
    controls,
    dcc.Store(id="refresh-store", data=0),
    dbc.Tabs([
        dbc.Tab(overview_layout,   label="Overview",    tab_id="tab-overview"),
        dbc.Tab(timeseries_layout, label="Time Series", tab_id="tab-ts"),
        dbc.Tab(srisk_layout,      label="SRISK",       tab_id="tab-srisk"),
        dbc.Tab(market_layout,     label="Market Data", tab_id="tab-market"),
    ], id="main-tabs", active_tab="tab-overview",
       style={"paddingLeft": "1rem", "backgroundColor": "#f8f9fa",
              "borderBottom": f"1px solid {BORDER}"}),
], style={"backgroundColor": BG_PAGE, "minHeight": "100vh"})


# ── Callbacks ─────────────────────────────────────────────────────────────────

# ── Country → bank dropdown ────────────────────────────────────────────────────

@app.callback(
    Output("bank-select", "options"),
    Output("bank-select", "value"),
    Input("country-select", "value"),
    Input("btn-all",  "n_clicks"),
    Input("btn-none", "n_clicks"),
    State("bank-select", "value"),
    prevent_initial_call=False,
)
def update_bank_options(countries, _all, _none, current_values):
    from dash import callback_context
    triggered = (callback_context.triggered[0]["prop_id"]
                 if callback_context.triggered else "country-select.value")

    # Build option list for selected countries
    options = []
    for country in (countries or []):
        flag  = COUNTRY_FLAGS.get(country, "")
        label = COUNTRY_LABELS.get(country, country)
        # Group header (disabled separator — plain string label required)
        options.append({
            "label": f"── {flag} {label} ──",
            "value": f"__header_{country}__",
            "disabled": True,
        })
        for ticker, name in BANKS_BY_COUNTRY[country].items():
            if ticker in RETURNS.columns:
                options.append({
                    "label": f"{flag} {name}  ({ticker})",
                    "value": ticker,
                })

    valid = [o["value"] for o in options if not o.get("disabled")]

    if "btn-all" in triggered:
        return options, valid
    if "btn-none" in triggered:
        return options, []

    # On country change: select all banks in newly selected countries
    return options, valid


# ── Refresh ────────────────────────────────────────────────────────────────────

@app.callback(
    Output("refresh-store", "data"),
    Input("btn-refresh", "n_clicks"),
    State("refresh-store", "data"),
    prevent_initial_call=True,
)
def refresh_data(n_clicks, current):
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
    Output("kpi-mes",          "children"),
    Output("kpi-covar",        "children"),
    Output("kpi-srisk",        "children"),
    Output("chart-mes-rank",   "figure"),
    Output("chart-covar-rank", "figure"),
    Output("risk-table",       "children"),
    Input("date-range",    "start_date"),
    Input("date-range",    "end_date"),
    Input("bank-select",   "value"),
    Input("refresh-store", "data"),
)
def update_overview(start, end, tickers, _):
    tickers = tickers or []
    mes_df    = _slice(MEASURES["mes"],         start, end, tickers)
    dcovar_df = _slice(MEASURES["delta_covar"], start, end, tickers)
    srisk_df  = _slice(MEASURES["srisk"],       start, end, tickers)

    latest_mes   = _latest_row(mes_df)
    latest_covar = _latest_row(dcovar_df)
    latest_srisk = _latest_row(srisk_df)

    agg_mes   = latest_mes.mean()
    agg_covar = latest_covar.mean()
    agg_srisk = latest_srisk.sum()

    kpi1 = kpi_card("Avg. MES (latest)",
                    _fmt_pct(agg_mes),
                    "Mean expected loss on market crash days",
                    "#c62828")
    kpi2 = kpi_card("Avg. |DeltaCoVaR| (latest)",
                    _fmt_pct(abs(agg_covar) if not pd.isna(agg_covar) else float("nan")),
                    "Mean marginal systemic contribution",
                    "#e65100")
    kpi3 = kpi_card("Total SRISK (latest, native ccy)",
                    _fmt_bn(agg_srisk if agg_srisk > 0 else float("nan")),
                    "Aggregate capital shortfall estimate",
                    "#2e7d32")

    fig_mes   = ranking_bar(latest_mes,        "MES Ranking (latest)",     "MES")
    fig_covar = ranking_bar(latest_covar.abs(), "|DeltaCoVaR| Ranking (latest)", "|DeltaCoVaR|")

    # Summary table
    rows = []
    for t in sorted(set(latest_mes.index) | set(latest_covar.index)):
        rows.append(html.Tr([
            html.Td(html.Span("●", style={"color": _color(t)})),
            html.Td(COUNTRY_FLAGS.get(BANK_COUNTRY.get(t, ""), "")),
            html.Td(_name(t), style={"fontWeight": "500"}),
            html.Td(t, className="text-muted", style={"fontSize": "0.8rem"}),
            html.Td(_fmt_pct(latest_mes.get(t, float("nan")))),
            html.Td(_fmt_pct(latest_covar.get(t, float("nan")))),
            html.Td(_fmt_bn(latest_srisk.get(t, float("nan")))),
        ]))

    table = dbc.Table(
        [
            html.Thead(html.Tr([
                html.Th(""), html.Th(""), html.Th("Bank"),
                html.Th("Ticker"), html.Th("MES"),
                html.Th("DeltaCoVaR"), html.Th("SRISK"),
            ]), style={"backgroundColor": "#f8f9fa"}),
            html.Tbody(rows),
        ],
        bordered=True, hover=True, responsive=True, size="sm",
        style={"fontSize": "0.85rem", "backgroundColor": BG_CARD},
    )

    return kpi1, kpi2, kpi3, fig_mes, fig_covar, table


# ── Time series tab ───────────────────────────────────────────────────────────

@app.callback(
    Output("chart-timeseries", "figure"),
    Input("date-range",    "start_date"),
    Input("date-range",    "end_date"),
    Input("bank-select",   "value"),
    Input("ts-measure",    "value"),
    Input("ts-overlay",    "value"),
    Input("refresh-store", "data"),
)
def update_timeseries(start, end, tickers, measure, overlay, _):
    tickers = tickers or []
    df = _slice(MEASURES[measure], start, end, tickers)

    labels = {
        "mes":         ("MES",    "MES (loss fraction)"),
        "delta_covar": ("DeltaCoVaR", "DeltaCoVaR"),
        "covar":       ("CoVaR",  "CoVaR"),
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
)
def update_srisk(start, end, tickers, _):
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
)
def update_market(start, end, tickers, _):
    tickers = tickers or []
    keep    = tickers + [MARKET_NAME]
    prices  = _slice(PRICES,   start, end, keep)
    rets    = _slice(RETURNS,  start, end, tickers)

    return price_chart(prices), return_hist(rets, tickers), corr_heatmap(rets)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run_server(host="0.0.0.0", port=port)
