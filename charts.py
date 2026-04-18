"""
Chart builders, style tokens, and presentation helpers.

Pure presentation layer — no callbacks, no data fetching. All functions take
DataFrames / Series already filtered to the view and return Plotly figures or
Dash components.

Bank name + colour lookups live in ``data_load``; this module references them
through :data:`data_load.ALL_BANKS` / :data:`data_load.BANK_COLORS`, both of
which are mutated in place when the user adds a custom bank at runtime, so
lookups stay consistent without re-importing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dash import html
import dash_bootstrap_components as dbc

import data_load as _D

# ── Style tokens (light theme) ────────────────────────────────────────────────

PLOTLY_TEMPLATE = "plotly_white"

BG_PAGE    = "#f4f6f9"
BG_CARD    = "#ffffff"
BG_HEADER  = "#ffffff"
BORDER     = "#dee2e6"
TEXT_MUTED = "#6c757d"
TEXT_MAIN  = "#212529"

# ── Crisis period overlays ────────────────────────────────────────────────────
# Notable systemic-stress episodes; timings inspired by Acharya, Brunnermeier &
# Pierret (2025) and common literature conventions.
CRISIS_PERIODS: list[tuple[str, str, str, str]] = [
    # (start,        end,          label,         fillcolor)
    ("2007-08-01", "2009-06-30", "GFC",          "rgba(198, 40, 40, 0.09)"),
    ("2010-04-01", "2012-12-31", "Euro Crisis",  "rgba(230, 81,  0, 0.09)"),
    ("2020-02-20", "2020-04-30", "COVID-19",     "rgba(106, 27, 154, 0.10)"),
    ("2022-01-01", "2022-10-31", "Rate Shock",   "rgba(245,127, 23, 0.09)"),
    ("2023-03-01", "2023-05-31", "SVB / CS",     "rgba(198, 40, 40, 0.10)"),
    ("2025-04-02", "2025-06-30", "Liberation Day / Tariffs", "rgba(21, 101, 192, 0.10)"),
]


def add_crisis_overlays(fig: go.Figure, data_start=None, data_end=None, row=None) -> None:
    """Add shaded vrects for crisis windows that intersect the data range."""
    if data_start is None or data_end is None:
        return
    try:
        ds = pd.to_datetime(data_start)
        de = pd.to_datetime(data_end)
    except Exception:
        return
    for s, e, label, color in CRISIS_PERIODS:
        cs = pd.to_datetime(s)
        ce = pd.to_datetime(e)
        if ce < ds or cs > de:
            continue
        cs_clip = max(cs, ds)
        ce_clip = min(ce, de)
        kwargs = dict(
            x0=cs_clip, x1=ce_clip,
            fillcolor=color, line_width=0, layer="below",
            annotation_text=label,
            annotation_position="top left",
            annotation_font_size=9,
            annotation_font_color="#555",
        )
        if row is not None:
            fig.add_vrect(**kwargs, row=row, col=1)
        else:
            fig.add_vrect(**kwargs)


# ── Lookup helpers ────────────────────────────────────────────────────────────

def name_for(ticker: str) -> str:
    return _D.ALL_BANKS.get(ticker, ticker)


def color_for(ticker: str) -> str:
    return _D.BANK_COLORS.get(ticker, "#aaaaaa")


# ── Format helpers ────────────────────────────────────────────────────────────

def fmt_bn(x) -> str:
    if pd.isna(x):
        return "N/A"
    if x == 0:
        return "0.00 bn"
    return f"{x / 1e9:.2f} bn"


def fmt_pct(x) -> str:
    return "N/A" if pd.isna(x) else f"{x * 100:.2f}%"


def fmt_ratio(x) -> str:
    return "N/A" if pd.isna(x) else f"{x:.2f}x"


def fmt_ses(x) -> str:
    """SES: show 'Not binding' for zero/NaN shortfalls, billions otherwise."""
    if pd.isna(x):
        return "N/A"
    if x <= 0:
        return "Not binding"
    return f"{x / 1e9:.2f} bn"


def fmt_pct_raw(x) -> str:
    """Already in percent (e.g. 12.3 → '12.30%')."""
    return "N/A" if pd.isna(x) else f"{x:.2f}%"


def fmt_bn_x1(x) -> str:
    """Already scaled to billions (e.g. 24.5 → '24.50 bn')."""
    return "N/A" if pd.isna(x) else f"{x:.2f} bn"


# ── Layout helper ─────────────────────────────────────────────────────────────

def base_layout(**kwargs) -> dict:
    return dict(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor=BG_CARD,
        plot_bgcolor=BG_CARD,
        font=dict(color=TEXT_MAIN, size=12),
        margin=dict(l=10, r=10, t=45, b=30),
        **kwargs,
    )


# ── Chart builders ────────────────────────────────────────────────────────────

def delta_ranking_bar(series: pd.Series, title: str, xlabel: str,
                      fmt_fn=fmt_pct, divide_bn: bool = False) -> go.Figure:
    """Signed change ranking: red = worsening (positive), green = improving (negative)."""
    s = series.dropna()
    if s.empty:
        return go.Figure().update_layout(title=f"{title} — no data", **base_layout())
    s = s.reindex(s.abs().sort_values(ascending=False).index)
    labels = [name_for(t) for t in s.index]
    xvals  = s.values / 1e9 if divide_bn else s.values
    colors = ["#c62828" if v > 0 else "#2e7d32" for v in s.values]
    if divide_bn:
        text = [f"{v/1e9:+,.2f} bn" for v in s.values]
    else:
        text = [f"{v*100:+.2f} pp" for v in s.values]
    fig = go.Figure(go.Bar(
        x=xvals, y=labels,
        orientation="h",
        marker_color=colors,
        marker_line_width=0,
        text=text, textposition="auto",
        insidetextanchor="middle",
        hovertemplate="%{y}: %{text}<extra></extra>",
    ))
    base = base_layout()
    base["margin"] = dict(l=10, r=90, t=45, b=30)
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis_title=xlabel,
        yaxis=dict(autorange="reversed"),
        height=300,
        **base,
    )
    fig.add_vline(x=0, line_width=1, line_color="#aaa")
    return fig


def ranking_bar(series: pd.Series, title: str, xlabel: str,
                fmt_fn=fmt_pct) -> go.Figure:
    s      = series.dropna().sort_values(ascending=False)
    colors = [color_for(t) for t in s.index]
    labels = [name_for(t) for t in s.index]
    text   = [fmt_fn(v) for v in s.values]
    # Scale to billions for bn-formatted charts so x-axis is readable
    xvals  = s.values / 1e9 if fmt_fn is fmt_bn else s.values

    fig = go.Figure(go.Bar(
        x=xvals, y=labels,
        orientation="h",
        marker_color=colors,
        marker_line_width=0,
        text=text, textposition="auto",
        insidetextanchor="middle",
        hovertemplate="%{y}: %{text}<extra></extra>",
    ))
    base = base_layout()
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
    show_crises: bool = False,
    data_start=None, data_end=None,
    aggregate: pd.Series | None = None,
    aggregate_label: str = "Portfolio",
    aggregate_hover_fmt: str = "%{y:.4f}",
) -> go.Figure:
    rows = 2 if market_ret is not None else 1
    fig  = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        row_heights=[0.75, 0.25] if rows == 2 else [1.0],
        vertical_spacing=0.06,
    )

    # Per-bank traces — slightly thinner + more translucent so the aggregate
    # line sits visually on top.
    has_agg = aggregate is not None and not aggregate.dropna().empty
    bank_width   = 1.4 if has_agg else 1.8
    bank_opacity = 0.55 if has_agg else 1.0
    for ticker in df.columns:
        s = df[ticker].dropna()
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values,
            name=name_for(ticker),
            line=dict(color=color_for(ticker), width=bank_width),
            opacity=bank_opacity,
            hovertemplate=f"{name_for(ticker)}: %{{y:.4f}}<extra></extra>",
        ), row=1, col=1)

    # Portfolio aggregate — added last so it paints on top. Thick dark line
    # with a soft white halo (two traces: wide white underneath, narrow
    # black on top) to stay legible against coloured bank lines.
    if has_agg:
        agg_s = aggregate.dropna()
        fig.add_trace(go.Scatter(
            x=agg_s.index, y=agg_s.values,
            name=aggregate_label,
            line=dict(color="#ffffff", width=6.0),
            opacity=0.85,
            hoverinfo="skip",
            showlegend=False,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=agg_s.index, y=agg_s.values,
            name=aggregate_label,
            line=dict(color="#0d1b5e", width=3.2),
            hovertemplate=f"<b>{aggregate_label}</b>: {aggregate_hover_fmt}<extra></extra>",
        ), row=1, col=1)

    if market_ret is not None:
        clrs = np.where(market_ret.values >= 0, "#2e7d32", "#c62828")
        fig.add_trace(go.Bar(
            x=market_ret.index, y=market_ret.values,
            name=_D.MARKET_NAME,
            marker_color=clrs,
            opacity=0.55,
            hovertemplate=f"{_D.MARKET_NAME}: %{{y:.4f}}<extra></extra>",
        ), row=2, col=1)
        fig.update_yaxes(title_text="Mkt Return", row=2, col=1, title_font_size=11)

    if show_crises:
        add_crisis_overlays(fig, data_start, data_end, row=1 if rows == 2 else None)

    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        yaxis_title=ylabel,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, font_size=11),
        height=480,
        **base_layout(),
    )
    return fig


def srisk_bar_generic(series: pd.Series, title: str, xlabel: str,
                      fmt_fn, norm: str) -> go.Figure:
    s = series.dropna().sort_values(ascending=False)
    if s.empty:
        return go.Figure().update_layout(title="No SRISK data", **base_layout())
    labels = [name_for(t) for t in s.index]
    text   = [fmt_fn(v) for v in s.values]
    fig = go.Figure(go.Bar(
        x=s.values, y=labels,
        orientation="h",
        marker_color=[color_for(t) for t in s.index],
        marker_line_width=0,
        text=text, textposition="auto",
        insidetextanchor="middle",
        hovertemplate="%{y}: %{text}<extra></extra>",
    ))
    base = base_layout()
    base["margin"] = dict(l=10, r=90, t=45, b=30)
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis_title=xlabel,
        yaxis=dict(autorange="reversed"),
        height=320,
        **base,
    )
    return fig


def srisk_bar(series: pd.Series, title: str) -> go.Figure:
    s = series.dropna().sort_values(ascending=False)
    if s.empty:
        return go.Figure().update_layout(title="No SRISK data", **base_layout())
    labels = [name_for(t) for t in s.index]

    fig = go.Figure(go.Bar(
        x=s.values / 1e9, y=labels,
        orientation="h",
        marker_color=[color_for(t) for t in s.index],
        marker_line_width=0,
        text=[fmt_bn(v) for v in s.values],
        textposition="auto",
        insidetextanchor="middle",
        hovertemplate="%{y}: %{text}<extra></extra>",
    ))
    base = base_layout()
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
        return go.Figure().update_layout(title="No positive SRISK", **base_layout())
    labels = [name_for(t) for t in s.index]
    fig = go.Figure(go.Pie(
        labels=labels, values=s.values,
        marker_colors=[color_for(t) for t in s.index],
        textinfo="label+percent",
        hovertemplate="%{label}: %{value:.2e} (%{percent})<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="SRISK Share (%)", font=dict(size=14)),
        height=320,
        **base_layout(),
    )
    return fig


def price_chart(prices: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for ticker in prices.columns:
        if ticker not in _D.ALL_BANKS:
            continue
        s = prices[ticker].dropna()
        if len(s) < 2:
            continue
        rebased = s / s.iloc[0] * 100
        fig.add_trace(go.Scatter(
            x=rebased.index, y=rebased.values,
            name=name_for(ticker),
            line=dict(color=color_for(ticker), width=1.6),
        ))

    # Market index
    if _D.MARKET_NAME in prices.columns:
        s = prices[_D.MARKET_NAME].dropna()
        if len(s) >= 2:
            rebased = s / s.iloc[0] * 100
            fig.add_trace(go.Scatter(
                x=rebased.index, y=rebased.values,
                name=_D.MARKET_NAME,
                line=dict(color="#333333", width=2, dash="dot"),
            ))

    fig.update_layout(
        title=dict(text="Rebased Price Performance (100 = start of period)",
                   font=dict(size=14)),
        yaxis_title="Index (start = 100)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, font_size=11),
        height=380,
        **base_layout(),
    )
    return fig


def corr_heatmap(returns: pd.DataFrame) -> go.Figure:
    cols  = [c for c in returns.columns if c in _D.ALL_BANKS]
    if not cols:
        return go.Figure().update_layout(title="No data", **base_layout())
    corr  = returns[cols].corr()
    names = [name_for(t) for t in corr.columns]

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
        **base_layout(),
    )
    return fig


def return_hist(returns: pd.DataFrame, tickers: list) -> go.Figure:
    fig = go.Figure()
    for ticker in tickers:
        if ticker not in returns.columns:
            continue
        s = returns[ticker].dropna() * 100
        fig.add_trace(go.Histogram(
            x=s.values, name=name_for(ticker), opacity=0.6, nbinsx=80,
            marker_color=color_for(ticker),
        ))
    fig.update_layout(
        title=dict(text="Daily Return Distribution (%)", font=dict(size=14)),
        xaxis_title="Daily Return (%)",
        barmode="overlay",
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, font_size=11),
        height=300,
        **base_layout(),
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
