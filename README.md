# 📈 Portfolio Optimizer — Modern Portfolio Theory (Markowitz)

An interactive web app that builds the **efficient frontier**, finds the
**Minimum-Variance** and **Maximum-Sharpe (tangency)** portfolios, and shows how
your *current* allocation compares — all on an interactive Plotly chart, with
concrete reallocation suggestions underneath.

Built with [`yfinance`](https://github.com/ranaroussi/yfinance) (price data),
[`PyPortfolioOpt`](https://github.com/robertmartin8/PyPortfolioOpt)
(optimisation), [`Plotly`](https://plotly.com/python/) (charts) and
[`Streamlit`](https://streamlit.io/) (UI).

---

## Features

| Requirement | Where it lives |
|---|---|
| **Data input** — tickers + shares, capital to invest, risk-free rate | `app.py` (holdings editor + sidebar) |
| **Data fetching** — daily close prices over a 1–10y lookback | `src/data_loader.py` (yfinance) |
| **Expected returns, covariance & correlation (ρ)** | `src/optimizer.py` |
| **Minimum-Variance Portfolio (MVP)** | `optimizer.min_variance_portfolio` |
| **Efficient Frontier** | `optimizer.efficient_frontier_curve` |
| **Tangency / Max-Sharpe Portfolio** | `optimizer.max_sharpe_portfolio` |
| **Markowitz bullet** (Monte-Carlo feasible set) | `optimizer.random_portfolios` |
| **Capital Market Line** tangent at the max-Sharpe point | `src/visualization.py` |
| **Risk-free rate on the Y-axis** (x = 0) | `src/visualization.py` |
| **Current-portfolio marker** | `src/portfolio.py` + `visualization.py` |
| **Comparison & reallocation tables** | `app.py` + `portfolio.reallocation_plan` |

The chart shows the feasible-set cloud coloured by Sharpe ratio, the efficient
frontier, the CML, the risk-free point, and star/diamond markers for the MVP,
the tangency portfolio and your current portfolio. Hover any point to see its
return, volatility, Sharpe ratio and weights.

---

## Quick start

```bash
# 1. Install dependencies (a virtual environment is recommended)
pip install -r requirements.txt

# 2. Launch the app
streamlit run app.py
```

Then open the URL Streamlit prints (default <http://localhost:8501>).

1. Edit the **holdings table**: tickers + shares (leave shares at `0` to include
   an asset in the optimisation without treating it as a current holding).
2. Set the **risk-free rate**, **lookback window** and optional **capital to
   invest** in the sidebar.
3. Click **Optimize portfolio**.

---

## Data sources (and the network requirement)

Live prices are downloaded from **Yahoo Finance**, which requires outbound
HTTPS access to Yahoo's servers. Some networks (corporate egress policies,
sandboxes, CI) block Yahoo, and yfinance is occasionally rate-limited. To keep
the app usable everywhere, the **Data source** selector in the sidebar offers
three options:

- **Yahoo Finance (live)** — real historical prices (default).
- **Sample data (offline)** — a synthetic but realistically correlated price
  panel generated locally, so you can explore the app with **no network at
  all**. Clearly flagged as synthetic in the UI.
- **Upload CSV** — bring your own close prices (first column `Date`, one column
  of prices per ticker).

> If live fetching fails with a connection/403 error, that is a *network policy*
> issue, not a bug — switch to **Sample data (offline)** to try the app instantly.

---

## Project structure

```
portfolio-optimizer/
├── app.py                  # Streamlit UI (input, orchestration, results)
├── src/
│   ├── config.py           # Defaults & constants
│   ├── data_loader.py      # yfinance download + cleaning
│   ├── sample_data.py      # Offline synthetic generator + CSV loader
│   ├── optimizer.py        # MPT engine (PyPortfolioOpt): MVP, tangency, frontier, MC
│   ├── portfolio.py        # Current-holdings valuation & reallocation plan
│   └── visualization.py    # Plotly figures (bullet, frontier, CML, heatmap)
├── tests/
│   └── test_engine.py      # Offline unit tests (no network)
├── conftest.py             # Makes `src` importable in tests
├── requirements.txt
└── README.md
```

The financial engine (`src/`) has **no Streamlit dependency**, so it can be
unit-tested or reused in other front-ends.

---

## The math, briefly

- **Expected returns** `μ`: annualised mean historical return per asset.
- **Risk** `Σ`: annualised covariance matrix (sample or Ledoit-Wolf shrinkage).
- **Portfolio return / risk**: `μₚ = wᵀμ`, `σₚ = √(wᵀΣw)`.
- **Sharpe ratio**: `(μₚ − R_f) / σₚ`.
- **MVP**: minimise `wᵀΣw` s.t. `Σw = 1`, `w ≥ 0`.
- **Tangency portfolio**: maximise the Sharpe ratio — the point where a line
  from `R_f` on the Y-axis is tangent to the efficient frontier.
- **Capital Market Line**: `μ = R_f + Sharpe_tangency · σ`.

All optimisation is **long-only** (`0 ≤ w ≤ 1`, no shorting/leverage).

---

## Tests

```bash
pytest -q
```

The suite runs fully offline against a synthetic, correlated price series and
checks the core invariants (weights sum to one, the MVP is the lowest-variance
portfolio, the tangency portfolio has the highest Sharpe ratio, the frontier is
monotonic above the MVP, etc.).

---

## Disclaimer

This is an **educational tool**. Estimates are based on historical data, which
does not guarantee future results. Nothing here is investment advice.
