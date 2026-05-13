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

# Unified palette — one tone per role across every tab. Inspired by a
# modern admin-dashboard reference design.
#
#   ACCENT_BLUE       primary blue  — KPI accents, ranking bars, primary CTAs
#   ACCENT_BLUE_DARK  deeper blue   — portfolio aggregate / overlay lines
#   ACCENT_RED        single red    — worsening deltas, "High" risk, negative bars
#   ACCENT_GREEN      single green  — improving deltas, "Low" risk, positive bars
#   ACCENT_AMBER      single amber  — "Medium" risk pill
#   NEUTRAL_GREY      muted grey    — "Other" slice, axis baselines
ACCENT_BLUE      = "#4A8DDC"
ACCENT_BLUE_DARK = "#5E6EED"
ACCENT_RED       = "#FF0854"
ACCENT_GREEN     = "#00D284"
ACCENT_AMBER     = "#F59E0B"
NEUTRAL_GREY     = "#b0b7c3"

# Typography tokens — one size for every piece of in-chart text (axis titles,
# tick labels, legend, annotations, in-bar value labels, pie labels) so the
# dashboard reads as a single visual language. Chart titles stay slightly
# larger so they remain visually distinct.
#
# Text colour: black-on-accent rather than white. Every brand colour in the
# palette (#4A8DDC blue, #FF0854 red, #00D284 green, #F59E0B amber, plus the
# vivid Tailwind 400-500 bank palette) is a mid-to-light tone where dark
# text gives noticeably stronger WCAG contrast (5–10:1) than white (2–3.5:1).
CHART_FONT_SIZE  = 12
CHART_TITLE_SIZE = 14
CHART_TEXT_COLOR = "#212529"

# ── Crisis period overlays ────────────────────────────────────────────────────
# Notable systemic-stress episodes; timings inspired by Acharya, Brunnermeier &
# Pierret (2025) and common literature conventions.
CRISIS_PERIODS: list[tuple[str, str, str, str]] = [
    # (start,        end,          label,         fillcolor)
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
            annotation_font_size=CHART_FONT_SIZE,
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
        font=dict(color=TEXT_MAIN, size=CHART_FONT_SIZE),
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
    labels = [f"{name_for(t)} ({t})" for t in s.index]
    xvals  = s.values / 1e9 if divide_bn else s.values
    colors = [ACCENT_RED if v > 0 else ACCENT_GREEN for v in s.values]
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
        textfont=dict(size=CHART_FONT_SIZE, color=CHART_TEXT_COLOR),
        insidetextfont=dict(size=CHART_FONT_SIZE, color=CHART_TEXT_COLOR),
        outsidetextfont=dict(size=CHART_FONT_SIZE, color=CHART_TEXT_COLOR),
        hovertemplate="%{y}: %{text}<extra></extra>",
    ))
    base = base_layout()
    longest = max((len(lab) for lab in labels), default=10)
    left_margin = max(120, min(220, int(7.0 * longest)))
    base["margin"] = dict(l=left_margin, r=70, t=45, b=30)
    fig.update_layout(
        title=dict(text=title, font=dict(size=CHART_TITLE_SIZE),
                   x=0.5, xanchor="center"),
        xaxis_title=xlabel or None,
        yaxis=dict(autorange="reversed", tickfont=dict(size=CHART_FONT_SIZE)),
        height=320,
        **base,
    )
    fig.add_vline(x=0, line_width=1, line_color="#aaa")
    return fig


def ranking_bar(series: pd.Series, title: str, xlabel: str,
                fmt_fn=fmt_pct) -> go.Figure:
    s      = series.dropna().sort_values(ascending=False)
    # Compose "Bank Name (TICKER)" labels so users see both the readable
    # name and the symbol used for filtering / API calls.
    labels = [f"{name_for(t)} ({t})" for t in s.index]
    text   = [fmt_fn(v) for v in s.values]
    # Scale to billions for bn-formatted charts so x-axis is readable
    xvals  = s.values / 1e9 if fmt_fn is fmt_bn else s.values

    fig = go.Figure(go.Bar(
        x=xvals, y=labels,
        orientation="h",
        marker_color=ACCENT_BLUE,
        marker_line_width=0,
        text=text, textposition="auto",
        insidetextanchor="middle",
        textfont=dict(size=CHART_FONT_SIZE, color=CHART_TEXT_COLOR),
        insidetextfont=dict(size=CHART_FONT_SIZE, color=CHART_TEXT_COLOR),
        outsidetextfont=dict(size=CHART_FONT_SIZE, color=CHART_TEXT_COLOR),
        hovertemplate="%{y}: %{text}<extra></extra>",
    ))
    base = base_layout()
    # Dynamic left margin based on the longest label so names don't clip
    # at narrower column widths (overview has 3 charts side-by-side).
    longest = max((len(lab) for lab in labels), default=10)
    left_margin = max(120, min(220, int(7.0 * longest)))
    base["margin"] = dict(l=left_margin, r=70, t=45, b=30)
    fig.update_layout(
        title=dict(text=title, font=dict(size=CHART_TITLE_SIZE),
                   x=0.5, xanchor="center"),
        xaxis_title=xlabel or None,
        yaxis=dict(autorange="reversed", tickfont=dict(size=CHART_FONT_SIZE)),
        height=320,
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
    # Pull the numeric precision from the aggregate hover template (e.g.
    # "%{y:.4f}" → ".4f") so per-bank hovers share the same formatting.
    bank_hover_fmt = aggregate_hover_fmt
    for ticker in df.columns:
        s = df[ticker].dropna()
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values,
            name=name_for(ticker),
            line=dict(color=color_for(ticker), width=bank_width),
            opacity=bank_opacity,
            showlegend=False,
            hovertemplate=(
                f"Date: %{{x|%Y-%m-%d}}<br>"
                f"Bank: {name_for(ticker)}<br>"
                f"Value: {bank_hover_fmt}<extra></extra>"
            ),
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
            line=dict(color=ACCENT_BLUE_DARK, width=3.2),
            showlegend=False,
            hovertemplate=(
                f"Date: %{{x|%Y-%m-%d}}<br>"
                f"Bank: {aggregate_label}<br>"
                f"Value: {aggregate_hover_fmt}<extra></extra>"
            ),
        ), row=1, col=1)

    if market_ret is not None:
        clrs = np.where(market_ret.values >= 0, ACCENT_GREEN, ACCENT_RED)
        fig.add_trace(go.Bar(
            x=market_ret.index, y=market_ret.values,
            name=_D.MARKET_NAME,
            marker_color=clrs,
            opacity=0.55,
            showlegend=False,
            hovertemplate=(
                f"Date: %{{x|%Y-%m-%d}}<br>"
                f"Bank: {_D.MARKET_NAME}<br>"
                f"Value: %{{y:.4f}}<extra></extra>"
            ),
        ), row=2, col=1)
        fig.update_yaxes(title_text="Mkt Return", row=2, col=1, title_font_size=CHART_FONT_SIZE)

    if show_crises:
        add_crisis_overlays(fig, data_start, data_end, row=1 if rows == 2 else None)

    fig.update_layout(
        title=dict(text=title, font=dict(size=CHART_TITLE_SIZE)),
        yaxis_title=ylabel,
        showlegend=False,
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
        marker_color=ACCENT_BLUE,
        marker_line_width=0,
        text=text, textposition="auto",
        insidetextanchor="middle",
        textfont=dict(size=CHART_FONT_SIZE, color=CHART_TEXT_COLOR),
        insidetextfont=dict(size=CHART_FONT_SIZE, color=CHART_TEXT_COLOR),
        outsidetextfont=dict(size=CHART_FONT_SIZE, color=CHART_TEXT_COLOR),
        hovertemplate="%{y}: %{text}<extra></extra>",
    ))
    base = base_layout()
    base["margin"] = dict(l=10, r=90, t=45, b=30)
    fig.update_layout(
        title=dict(text=title, font=dict(size=CHART_TITLE_SIZE)),
        xaxis_title=xlabel,
        yaxis=dict(autorange="reversed", tickfont=dict(size=CHART_FONT_SIZE)),
        height=320,
        **base,
    )
    return fig


def srisk_pie(series: pd.Series, top_n: int = 5) -> go.Figure:
    """SRISK share pie: keep the top ``top_n`` banks individually and
    collapse the remaining positive-SRISK banks into a single 'Other' slice."""
    s = series.dropna()
    s = s[s > 0].sort_values(ascending=False)
    if s.empty:
        return go.Figure().update_layout(title="No positive SRISK", **base_layout())

    top   = s.head(top_n)
    other = s.iloc[top_n:]

    labels = [name_for(t) for t in top.index]
    values = list(top.values)
    colors = [color_for(t) for t in top.index]

    if not other.empty:
        labels.append(f"Other ({len(other)})")
        values.append(float(other.sum()))
        colors.append(NEUTRAL_GREY)

    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        marker_colors=colors,
        textinfo="label+percent",
        textfont=dict(size=CHART_FONT_SIZE, color=CHART_TEXT_COLOR),
        insidetextfont=dict(size=CHART_FONT_SIZE, color=CHART_TEXT_COLOR),
        outsidetextfont=dict(size=CHART_FONT_SIZE, color=CHART_TEXT_COLOR),
        hovertemplate="%{label}: %{value:.2e} (%{percent})<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=f"SRISK Share (%)",
                   font=dict(size=CHART_TITLE_SIZE)),
        height=320,
        **base_layout(),
    )
    return fig


def srisk_stacked_area(
    df: pd.DataFrame,
    y_unit: str = "bn",
    total_mc_ts: pd.Series | None = None,
    top_n: int = 10,
    show_crises: bool = False,
    data_start=None,
    data_end=None,
) -> go.Figure:
    """Per-bank stacked area of SRISK over time.

    ``y_unit`` controls the y-axis:
      - "bn"       → stacked SRISK in USD billions
      - "pct_agg"  → stacked share of aggregate SRISK (sums to 100%)
      - "pct_mc"   → each bank's SRISK divided by total market cap

    Keeps the top ``top_n`` banks (ranked by mean SRISK over the window)
    and rolls the remainder into an 'Other' series.
    """
    if df.empty:
        return go.Figure().update_layout(title="No SRISK data", **base_layout())

    # Rank by mean over the window — largest contributors surface first.
    means   = df.mean(axis=0).dropna()
    ordered = means.sort_values(ascending=False).index.tolist()
    top     = [t for t in ordered[:top_n] if means.get(t, 0) > 0]
    rest    = [t for t in ordered[top_n:] if means.get(t, 0) > 0]

    # Build display frame (USD)
    plot_df = df[top].copy().fillna(0.0)
    if rest:
        plot_df["__other__"] = df[rest].fillna(0.0).sum(axis=1)

    # Apply unit transform
    if y_unit == "pct_agg":
        denom = plot_df.sum(axis=1).replace(0, np.nan)
        plot_df = plot_df.div(denom, axis=0) * 100.0
        y_label = "SRISK (% of aggregate)"
        hover_fmt = "%{y:.2f}%"
        title_unit = "% of aggregate SRISK"
    elif y_unit == "pct_mc" and total_mc_ts is not None:
        denom = total_mc_ts.reindex(plot_df.index).ffill().replace(0, np.nan)
        plot_df = plot_df.div(denom, axis=0) * 100.0
        y_label = "SRISK (% of total market cap)"
        hover_fmt = "%{y:.3f}%"
        title_unit = "% of total market cap"
    else:  # "bn"
        plot_df = plot_df / 1e9
        y_label = "SRISK (bn USD)"
        hover_fmt = "%{y:.2f} bn"
        title_unit = "USD bn"

    fig = go.Figure()
    # Add in reverse rank order so largest ends up at the bottom of the stack.
    for col in reversed(plot_df.columns.tolist()):
        if col == "__other__":
            display_name = f"Other ({len(rest)})"
            line_color   = NEUTRAL_GREY
        else:
            display_name = name_for(col)
            line_color   = color_for(col)
        fig.add_trace(go.Scatter(
            x=plot_df.index, y=plot_df[col].values,
            name=display_name,
            mode="lines",
            line=dict(width=0.5, color=line_color),
            stackgroup="one",
            fillcolor=line_color,
            hovertemplate=(
                f"Date: %{{x|%Y-%m-%d}}<br>"
                f"Bank: {display_name}<br>"
                f"Value: {hover_fmt}<extra></extra>"
            ),
        ))

    if show_crises:
        add_crisis_overlays(fig, data_start, data_end)

    fig.update_layout(
        title=dict(text=f"SRISK over Time — Stacked by Bank ({title_unit})",
                   font=dict(size=CHART_TITLE_SIZE)),
        yaxis_title=y_label,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, font_size=CHART_FONT_SIZE),
        height=360,
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
                   font=dict(size=CHART_TITLE_SIZE)),
        yaxis_title="Index (start = 100)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, font_size=CHART_FONT_SIZE),
        height=380,
        **base_layout(),
    )
    return fig


def corr_heatmap(returns: pd.DataFrame) -> go.Figure:
    cols  = [c for c in returns.columns if c in _D.ALL_BANKS]
    if not cols:
        return go.Figure().update_layout(title="No data", **base_layout())
    corr  = returns[cols].corr()
    n     = len(cols)
    names = [name_for(t) for t in corr.columns]

    # Annotation text: show value for small matrices, hide for large ones to
    # avoid overprinting. When shown (n ≤ 12), the cells are large enough
    # to host the standard chart font without overlap.
    show_text = n <= 12
    text_fmt  = "%{text:.2f}" if show_text else ""

    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=names,
        y=names,
        colorscale="RdBu_r",
        zmin=-1, zmax=1,
        text=np.round(corr.values, 2),
        texttemplate=text_fmt,
        textfont=dict(size=CHART_FONT_SIZE, color=CHART_TEXT_COLOR),
        hovertemplate="<b>%{x}</b> / <b>%{y}</b><br>Correlation: %{z:.3f}<extra></extra>",
        colorbar=dict(
            title=dict(text="ρ", side="right", font=dict(size=CHART_FONT_SIZE)),
            thickness=14,
            len=0.85,
            tickvals=[-1, -0.5, 0, 0.5, 1],
            ticktext=["-1.0", "-0.5", "0", "+0.5", "+1.0"],
            tickfont=dict(size=CHART_FONT_SIZE),
            outlinewidth=0,
        ),
    ))

    fig.update_layout(
        title=dict(text="Return Correlation Matrix",
                   font=dict(size=CHART_TITLE_SIZE),
                   x=0.5, xanchor="center"),
        xaxis=dict(
            tickangle=-40,
            tickfont=dict(size=CHART_FONT_SIZE),
            side="bottom",
            showgrid=False,
        ),
        yaxis=dict(
            tickfont=dict(size=CHART_FONT_SIZE),
            autorange="reversed",
            showgrid=False,
        ),
        height=650,
        **base_layout(),
    )
    fig.update_layout(margin=dict(l=10, r=60, t=55, b=120))
    return fig


def market_dcc_chart(
    prices: pd.DataFrame,
    dcc_rho: pd.DataFrame,
    show_crises: bool = False,
    data_start=None,
    data_end=None,
) -> go.Figure:
    """Three-row subplot: rebased prices | DCC ρ per bank | mean DCC ρ.
    Shared x-axis and a single legend (rows 2-3 use legendgroup + showlegend=False)."""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.40, 0.35, 0.25],
        vertical_spacing=0.05,
        subplot_titles=[
            "Rebased Price (100 = period start)",
            f"DCC ρ(t) — Correlation with {_D.MARKET_NAME}",
            "Mean DCC ρ across selected banks",
        ],
    )

    # Row 1: rebased prices — per-bank legend entries hidden
    bank_tickers = [t for t in prices.columns if t in _D.ALL_BANKS]
    for ticker in bank_tickers:
        s = prices[ticker].dropna()
        if len(s) < 2:
            continue
        rebased = s / s.iloc[0] * 100
        fig.add_trace(go.Scatter(
            x=rebased.index, y=rebased.values,
            name=name_for(ticker),
            legendgroup=ticker,
            showlegend=False,
            line=dict(color=color_for(ticker), width=1.5),
            hovertemplate=(
                f"Date: %{{x|%Y-%m-%d}}<br>"
                f"Bank: {name_for(ticker)}<br>"
                f"Value: %{{y:.2f}}<extra></extra>"
            ),
        ), row=1, col=1)
    if _D.MARKET_NAME in prices.columns:
        s = prices[_D.MARKET_NAME].dropna()
        if len(s) >= 2:
            rebased = s / s.iloc[0] * 100
            # White halo below for contrast against the coloured bank lines.
            fig.add_trace(go.Scatter(
                x=rebased.index, y=rebased.values,
                name=_D.MARKET_NAME,
                legendgroup=_D.MARKET_NAME,
                showlegend=False,
                line=dict(color="#ffffff", width=7.0),
                opacity=0.9,
                hoverinfo="skip",
            ), row=1, col=1)
            # Prominent solid dark line on top.
            fig.add_trace(go.Scatter(
                x=rebased.index, y=rebased.values,
                name=_D.MARKET_NAME,
                legendgroup=_D.MARKET_NAME,
                showlegend=True,
                line=dict(color=ACCENT_BLUE_DARK, width=3.2),
                hovertemplate=(
                    f"Date: %{{x|%Y-%m-%d}}<br>"
                    f"<b>Bank: {_D.MARKET_NAME}</b><br>"
                    f"Value: %{{y:.2f}}<extra></extra>"
                ),
            ), row=1, col=1)

    # Row 2: DCC ρ per bank
    for ticker in dcc_rho.columns:
        s = dcc_rho[ticker].dropna()
        if s.empty:
            continue
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values,
            name=name_for(ticker),
            legendgroup=ticker,
            showlegend=False,
            line=dict(color=color_for(ticker), width=1.5),
            hovertemplate=(
                f"Date: %{{x|%Y-%m-%d}}<br>"
                f"Bank: {name_for(ticker)}<br>"
                f"Value: %{{y:.3f}}<extra></extra>"
            ),
        ), row=2, col=1)

    # Row 3: mean ρ
    if not dcc_rho.empty:
        avg = dcc_rho.mean(axis=1).dropna()
        if not avg.empty:
            fig.add_trace(go.Scatter(
                x=avg.index, y=avg.values,
                name="Mean ρ",
                legendgroup="__mean_rho__",
                showlegend=True,
                line=dict(color=ACCENT_BLUE_DARK, width=2),
                fill="tozeroy",
                fillcolor="rgba(94, 110, 237, 0.10)",
                hovertemplate=(
                    "Date: %{x|%Y-%m-%d}<br>"
                    "Bank: Mean ρ<br>"
                    "Value: %{y:.3f}<extra></extra>"
                ),
            ), row=3, col=1)
            fig.add_hline(
                y=float(avg.mean()), line_dash="dot", line_color="#555",
                annotation_text="Sample mean", annotation_position="top left",
                annotation_font_size=CHART_FONT_SIZE, row=3, col=1,
            )

    # Crisis overlays — annotated labels only on row 1, plain shading on rows 2-3
    if show_crises and data_start and data_end:
        add_crisis_overlays(fig, data_start, data_end, row=1)
        try:
            ds, de = pd.to_datetime(data_start), pd.to_datetime(data_end)
        except Exception:
            ds = de = None
        if ds is not None:
            for cs_str, ce_str, _, color in CRISIS_PERIODS:
                cs, ce = pd.to_datetime(cs_str), pd.to_datetime(ce_str)
                if ce < ds or cs > de:
                    continue
                for r in [2, 3]:
                    fig.add_vrect(
                        x0=max(cs, ds), x1=min(ce, de),
                        fillcolor=color, line_width=0, layer="below",
                        row=r, col=1,
                    )

    fig.update_yaxes(title_text="Index", title_font_size=CHART_FONT_SIZE, row=1, col=1)
    fig.update_yaxes(title_text="ρ(t)", range=[-0.2, 1.0], title_font_size=CHART_FONT_SIZE, row=2, col=1)
    fig.update_yaxes(title_text="Mean ρ", range=[-0.2, 1.0], title_font_size=CHART_FONT_SIZE, row=3, col=1)
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, font_size=CHART_FONT_SIZE),
        height=650,
        **base_layout(),
    )
    fig.update_layout(margin=dict(l=10, r=10, t=70, b=30))
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
        title=dict(text="Daily Return Distribution (%)",
                   font=dict(size=CHART_TITLE_SIZE)),
        xaxis_title="Daily Return (%)",
        barmode="overlay",
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, font_size=CHART_FONT_SIZE),
        height=300,
        **base_layout(),
    )
    return fig


# ── KPI card ──────────────────────────────────────────────────────────────────

def kpi_card(title: str, value: str, subtitle: str, accent: str,
             delta_text: str | None = None,
             delta_direction: str = "neutral",
             risk_level: str | None = None,
             risk_tooltip: str | None = None,
             info_content=None,
             info_id: str | None = None) -> dbc.Card:
    """KPI card with an optional 7-day delta badge.

    delta_direction ∈ {"up", "down", "neutral"} controls colour:
      up   → red    (worsening for risk measures)
      down → green  (improving)
      neutral → grey

    risk_level ∈ {"Low", "Medium", "High", None} renders a coloured pill
    inline next to the value, derived from the rolling-percentile classifier.

    info_content + info_id render a small circular "i" icon next to the
    title. On hover, a dbc.Popover containing ``info_content`` (a Dash
    component — typically the matching methodology card) is shown.
    """
    badge_colour = {"up": ACCENT_RED, "down": ACCENT_GREEN,
                    "neutral": "#6c757d"}.get(delta_direction, "#6c757d")
    badge_arrow  = {"up": "▲ ", "down": "▼ ", "neutral": "◆ "}.get(
        delta_direction, "")

    risk_colour = {"Low": ACCENT_GREEN, "Medium": ACCENT_AMBER,
                   "High": ACCENT_RED}.get(risk_level or "", "#6c757d")

    value_node = html.H4(value, style={"color": accent, "fontWeight": "700",
                                       "marginBottom": "0",
                                       "textAlign": "center"})
    if risk_level:
        risk_pill = html.Span(
            risk_level,
            title=risk_tooltip or "",
            style={
                "backgroundColor": risk_colour + "1a",
                "color": risk_colour,
                "border": f"1px solid {risk_colour}33",
                "borderRadius": "999px",
                "padding": "1px 8px",
                "fontSize": "0.7rem",
                "fontWeight": "700",
                "letterSpacing": "0.04em",
                "textTransform": "uppercase",
                "alignSelf": "center",
                "whiteSpace": "nowrap",
            },
        )
        value_row = html.Div(
            [value_node, risk_pill],
            style={"display": "flex", "alignItems": "center",
                   "justifyContent": "center",
                   "flexWrap": "wrap", "gap": "6px",
                   "marginBottom": "2px"},
        )
    else:
        value_row = value_node

    # Title row (optionally with a small circular "i" icon that triggers a
    # dbc.Popover containing the methodology card on hover).
    has_info = info_content is not None and info_id is not None
    title_children: list = [title]
    if has_info:
        title_children.extend([
            " ",
            html.Span(
                "i",
                id=info_id,
                className="kpi-info-icon",
                style={
                    "cursor": "help",
                    "color": ACCENT_BLUE,
                    "border": f"1.5px solid {ACCENT_BLUE}",
                    "borderRadius": "50%",
                    "width": "16px",
                    "height": "16px",
                    "display": "inline-flex",
                    "alignItems": "center",
                    "justifyContent": "center",
                    "fontSize": "0.66rem",
                    "fontStyle": "italic",
                    "fontWeight": "700",
                    "fontFamily": "Georgia, 'Times New Roman', serif",
                    "marginLeft": "6px",
                    "verticalAlign": "middle",
                    "lineHeight": "1",
                    "userSelect": "none",
                    "transition": "background-color 0.15s, color 0.15s",
                },
            ),
        ])

    body_children = [
        html.P(title_children, className="mb-1 text-muted text-center",
               style={"fontSize": "0.78rem", "fontWeight": "600",
                      "letterSpacing": "0.05em", "textTransform": "uppercase"}),
        value_row,
    ]
    if has_info:
        body_children.append(
            dbc.Popover(
                dbc.PopoverBody(info_content,
                                style={"padding": 0,
                                       "backgroundColor": "transparent"}),
                target=info_id,
                trigger="hover focus",
                placement="auto",
                className="kpi-info-popover",
            )
        )
    if delta_text:
        body_children.append(
            html.Div(
                html.Span(
                    f"{badge_arrow}{delta_text}",
                    style={
                        "display": "inline-block",
                        "backgroundColor": badge_colour + "1a",  # ~10% alpha
                        "color": badge_colour,
                        "borderRadius": "4px",
                        "padding": "1px 6px",
                        "fontSize": "0.72rem",
                        "fontWeight": "600",
                    },
                ),
                style={"textAlign": "center", "marginBottom": "2px"},
            )
        )
    body_children.append(
        html.P(subtitle, className="mb-0 text-muted text-center",
               style={"fontSize": "0.78rem"})
    )

    return dbc.Card([
        dbc.CardBody(body_children, style={"textAlign": "center"})
    ], style={"backgroundColor": BG_CARD, "border": f"1px solid {BORDER}",
              "borderTop": f"4px solid {accent}",
              "boxShadow": "0 1px 3px rgba(0,0,0,0.06)",
              "height": "100%"})
