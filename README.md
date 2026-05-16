# Systemic Risk Analyser

A data-driven dashboard for exploring systemic risk dynamics in the U.S. banking sector through interactive visualisations, econometric risk models, and established systemic risk indicators (MES, ΔCoVaR, SRISK).


## Features

The dashboard features six tabs, allowing the analysis of systemic risk measurs over different dimensions.

- **Start** — Accessible entry point before engaging with the analytical tabs.
- **Overview** — Analytical entry point of the dashboard, presenting a real-time snapshot of systemic risk conditions across the full bank sample.
- **Time Series** — Dynamic of view of systemic risk measures over the full sample period.
- **SRISK** — Dedicated environment for capital shortfall analysis under customisable stress scenar-ios.
- **Market and Correlation** — Multi-panel view of price performance, dynamic correlations, and pairwise return co-movement across the full bank sample.
- **Methodology** — Built-in reference guide, consolidating all technical documentation in a single accessible location

## Local Setup

Requires Python 3.10.

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
python app.py
```

Open http://localhost:8050 in your browser.

The initial cold start loads the parquet snapshots stored in cache/,
preventing the model from being re-estimated.

Delete the cache/ directory to force a full re-estimation.

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

## Data
Prices & balance sheets: Yahoo Finance (yfinance).

Rates, yields, VIX: Yahoo Finance (ZQ=F, ^TNX, ^IRX, ^VIX).

Credit spread (BAA10YM): FRED.

## Stack

- **Dash 4** + **dash-bootstrap-components 2** (FLATLY theme)
- **Plotly 6** for all charts
- **arch 8** for GJR-GARCH; **statsmodels** for quantile regression
- **pandas / numpy / scipy** for data work
- **pyarrow** for parquet caching
- **APScheduler** for the daily refresh job
- **gunicorn** for production serving

## License / disclaimer

Educational and research tool — **not investment advice**.
