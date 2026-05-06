# Systemic Risk Analyser

An interactive Dash dashboard for monitoring systemic risk in the U.S.
banking sector. The four standard academic measures — **MES**, **LRMES**,
**ΔCoVaR**, and **SRISK** — are computed from daily equity returns and
bank balance-sheet data using a DCC-GJR-GARCH model fit to each
bank-vs-market pair, then surfaced through five tabs (Start, Overview,
Time Series, SRISK, Market & Correlation, Methodology).

## Features

- **Risk indicators** — Low / Medium / High pill next to every aggregated
  KPI, derived from the rolling 500-observation percentile of the daily
  aggregate (`<70%` Low · `70–90%` Medium · `≥90%` High).
- **Live KPIs** — MES, LRMES, |ΔCoVaR|, total SRISK, and average leverage
  with 7-day delta badges.
- **Per-bank rankings** — top-10 ranking bars and 7-day shift charts on
  the Overview tab.
- **SRISK scenario tool** — `k` (capital ratio) and `d` (market-decline)
  sliders recompute SRISK live; per-bank breakdown, stacked area, and
  share pie.
- **Market & correlation** — rebased prices, return distributions, and
  DCC ρ(t) dynamics with crisis-window overlays.
- **Snapshot date** — pick any historical day to recompute the Overview
  risk-summary table.
- **Daily auto-refresh** — APScheduler job re-fetches prices and
  re-estimates the model at 06:00 UTC each day.

## Project layout

```
.
├── app.py                   # Dash layout, callbacks, and entry-point (exposes `app` + `server`)
├── charts.py                # Plotly chart builders, style tokens, KPI card
├── data_load.py             # Bank universe, price/balance-sheet ingestion, caching
├── systemic_measures.py     # DCC-GJR-GARCH fit + measure computation (MES/LRMES/CoVaR/SRISK)
├── assets/
│   └── custom.css           # Tab / slider / card styling overrides
├── cache/                   # Parquet snapshots (committed for fast cold-starts)
├── requirements.txt
├── runtime.txt              # python-3.10.13
├── render.yaml              # Render Blueprint definition
├── Procfile                 # gunicorn launcher (Heroku / local)
└── README.md
```

## Running locally

Requires Python 3.10.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python app.py
# open http://localhost:8050
```

The first cold start uses the parquet snapshots in `cache/` so the model
fit isn't repeated. Delete the `cache/` directory to force a full
re-estimation (≈ 10–20 minutes on a laptop).

## Deploying to Render

The repo ships a `render.yaml` blueprint:

1. Push the repo to GitHub (with `cache/` committed).
2. On Render: **New → Blueprint** → select the repo.
3. Render reads `render.yaml`, builds with `pip install -r requirements.txt`,
   and starts `gunicorn app:server --workers=1 --threads=4 --timeout 120`.

Single worker is intentional — the in-process APScheduler must run in
exactly one Python process to avoid concurrent daily refreshes.

## Methodology summary

| Measure | Formula | Source |
|---|---|---|
| **MES** | `E[r_i \| r_m ≤ VaR_α]` (one-day) | Acharya, Pedersen, Philippon & Richardson (2010, 2017) |
| **LRMES** | `1 − exp(log(1−d) · β_GARCH)` (closed-form approx.) | Brownlees & Engle (2017) |
| **ΔCoVaR** | `CoVaR(distress) − CoVaR(median)` via quantile regression | Adrian & Brunnermeier (2016) |
| **SRISK** | `max(0, k · (D + (1−LRMES)·MV) − (1−LRMES)·MV)` | Brownlees & Engle (2017) |

Defaults: `α = 5%`, `k = 8%`, `d = 40%`, rolling 5-year estimation
window. `α` is exposed in the topbar; `k` and `d` are sliders on the
SRISK tab.

## Stack

- **Dash 4** + **dash-bootstrap-components 2** (FLATLY theme)
- **Plotly 6** for all charts
- **arch 8** for GJR-GARCH; **statsmodels** for quantile regression
- **pandas / numpy / scipy** for data work
- **pyarrow** for parquet caching
- **APScheduler** for the daily refresh job
- **gunicorn** for production serving

## License / disclaimer

Educational and research tool — **not investment advice**. Data sourced
from Yahoo Finance and public bank filings.
