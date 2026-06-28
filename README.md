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
| **Live risk-free rate** — auto-fetch ^IRX (3M T-Bill) | `data_loader.fetch_risk_free_rate` |
| **IBKR position import** (Flex Query / activity CSV) | `portfolio.parse_ibkr_positions` |
| **Max-allocation-per-asset cap** | `optimizer` (`max_weight`) |
| **Whole-share rounding + residual cash + fees** | `portfolio.reallocation_plan` |
| **Market-benchmark reference point** | `optimizer.benchmark_point` + `visualization` |
| **Black-Litterman market-implied returns** | `optimizer.black_litterman_implied_returns` |
| **Hybrid holdings table** (IBKR + manual + purchase price / P&L) | `app.py` + `portfolio` |
| **Core global anchor** (VWCE.DE fixed/capped weight) | `optimizer.AnchorConstraint` |
| **Watchlist diversification scanner** | `optimizer.screen_watchlist` |
| **Asset typing** (Stock/ETF vs Bond/Fixed Income) | `portfolio.guess_asset_type` |
| **Currency normalisation** (EUR/USD via live FX) | `data_loader.convert_to_base_currency` |
| **Sector concentration guardrails** | `optimizer` (`add_sector_constraints`) |
| **Lump-sum vs DCA-tranche deployment** | `portfolio.deployment_plan` |
| **Rebalancing tolerance bands** | `portfolio.reallocation_plan` |
| **Italian capital-gains tax engine** (26% / 12.5%) | `portfolio.reallocation_plan` |
| **Behavioral volatility noise classifier** | `portfolio.classify_noise` |
| **Historical backtesting** (equity curves + Max DD) | `optimizer.backtest_equity_curves` |
| **GBM Monte-Carlo wealth projection** (fan chart) | `optimizer.simulate_gbm_wealth` |
| **IBKR TWS Basket-Trader export** | `portfolio.basket_trader_csv` |
| **Local persistence** (auto-save holdings/settings) | `portfolio.PortfolioStore` (SQLite) |
| **Premium dark-mode UI** + neon metric cards | `app.py` (CSS injection) |
| **Optional password gate** | `app.py` (`st.secrets`) |

The chart shows the feasible-set cloud coloured by Sharpe ratio, the efficient
frontier, the CML, the risk-free point, and star/diamond markers for the MVP,
the tangency portfolio, the **market benchmark** and your current portfolio.
Hover any point to see its return, volatility, Sharpe ratio and weights.

### Advanced controls (sidebar)

- **Fetch Live Risk-Free Rate (^IRX)** — toggled on by default; pulls the US
  3-Month Treasury Bill yield from Yahoo Finance and divides by 100. Toggle off
  (or on fetch failure) to enter Rf manually.
- **Market Benchmark Ticker** — defaults to `SPY`; plotted as a gold marker and
  used as the market proxy for Black-Litterman priors.
- **Expected-returns model** — *Historical mean* or *Black-Litterman
  (market-implied)*. BL derives the implied risk-aversion from the benchmark and
  returns the equilibrium prior `Π = δ·Σ·w_mkt + R_f` (equal-weight market
  portfolio); it falls back to historical returns if no benchmark is available.
- **Max Allocation per Asset (%)** — caps each weight so the tangency portfolio
  cannot over-concentrate in one high performer (clamped to ≥ 1/N to stay
  feasible).
- **Disable Fractional Shares** — rounds trades down to whole shares and reports
  the undeployed remainder as **Residual Cash**.
- **Estimated Transaction Cost per Trade ($)** — totalled across executed trades
  and subtracted from the rebalanced portfolio's expected return (net-of-fee).

### Hybrid holdings table (IBKR + manual + Trade Republic …)

Drag an Interactive Brokers **position report** or **activity statement** CSV
onto the uploader: the parser locates `Symbol`/`Ticker`, `Quantity`/`Shares` and
the cost price (from a cost-price or cost-basis column), auto-fills the table and
runs the optimisation. The table is fully editable — click **+** to add assets
held on other brokers (Trade Republic, Degiro, …). Each row has:

- **Ticker** — any Yahoo Finance symbol, stocks **and** bonds/bond ETFs
  (`AAPL`, `VWCE.DE`, `AGGH.MI`, `EMBE.MI`).
- **Shares** — units held (`0` = include in the optimisation as watch-only).
- **Purchase Price** — optional average buy price; drives the per-asset and
  total **unrealised P/L**.

### Core global anchor (VWCE.DE)

`VWCE.DE` (an all-world ETF) is auto-injected into the universe as a stabilising
core. The **Core Global Anchor Allocation (VWCE) %** slider (0–50%, default 20%)
either **pins** its weight exactly or **caps** it (toggle), enforced through
per-asset weight bounds. The remaining `(100 − slider)%` is optimised across your
own assets. The other assets' caps are automatically widened if needed to keep
the problem feasible.

### Watchlist diversification scanner

Type comma-separated tickers into *Enter Tickers to screen for Diversification*.
For each candidate the engine re-optimises the tangency portfolio over
`your pool + candidate` (under the same cap/anchor) and shows a recommendation
card: the suggested weight, the Max-Sharpe improvement (old → new) and the
candidate's average correlation to your pool. Enlarging the universe can never
lower the attainable Sharpe, so the cards highlight which candidates genuinely
shift your efficient frontier upwards.

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

All optimisation is **long-only** with an optional per-asset cap
(`0 ≤ wᵢ ≤ w_max`, no shorting/leverage). The Black-Litterman model replaces the
historical `μ` with the market-implied equilibrium prior anchored to the
benchmark.

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

## Local persistence

Your holdings, watchlist and every sidebar setting auto-save to a free, local
**SQLite** database (`.portfolio_state.db`, git-ignored) via
`portfolio.PortfolioStore`. On the next launch the table, watchlist and
parameters are restored automatically — close the browser without losing your
configuration. Corrupt database files are quarantined (renamed `*.corrupt`) and
re-created rather than crashing the app. Override the location with the
`PORTFOLIO_DB_PATH` environment variable.

## Deployment

The repo is ready for **free hosting on Streamlit Community Cloud**:

- `.streamlit/config.toml` pre-sets the dark theme and server options.
- `requirements.txt` lists all production dependencies.
- A root `Dockerfile` is provided for containerised/private hosting
  (`docker build -t portfolio-optimizer . && docker run -p 8501:8501 portfolio-optimizer`).
- **Optional password protection**: copy `.streamlit/secrets.toml.example` to
  `.streamlit/secrets.toml` (git-ignored) and set `APP_PASSWORD`, or paste the
  same key under *App settings → Secrets* on Community Cloud. Leave it unset to
  keep the app open.

## Disclaimer

This is an **educational tool**. Estimates are based on historical data, which
does not guarantee future results. Nothing here is investment advice.
