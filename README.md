# 📈 Portfolio Optimizer — from plan to orders

A practical rebalancing tool for a **buy-and-hold private investor** (built for
the Italian tax regime, EUR base, Trade Republic + IBKR Europe). You declare
your **own target allocation as buckets** (Core / Thematic / Satellite Quality /
Speculative / Gold), enter your holdings and the cash you want to invest, and
get an **Action Plan in whole shares and euros** — *"BUY 3 × VWCE.DE ≈ €390"* —
never abstract percentages. Markowitz analytics (efficient frontier, risk
clustering, backtests, Monte-Carlo) are attached as a **research report that
explains the plan but does not drive it**.

Built with [`yfinance`](https://github.com/ranaroussi/yfinance) (price data),
[`PyPortfolioOpt`](https://github.com/robertmartin8/PyPortfolioOpt)
(optimisation), [`Plotly`](https://plotly.com/python/) (charts) and
[`Streamlit`](https://streamlit.io/) (UI).

---

## How to use it (the real workflow)

1. **Enter your holdings** in the grid: Yahoo ticker (`VWCE.DE`, `EIMI.MI`,
   `ABT`…), shares, average purchase price (drives P/L and the tax estimate),
   asset type (sets 26% vs 12.5% tax) and — crucially — the **Bucket** each
   position belongs to. Or drag-and-drop an IBKR position CSV.
2. **Set your plan** in *🎯 My target allocation*: percent per bucket
   (default: Core 40 · Thematic 12 · Satellite 20 · Speculative 8 · Gold 20).
   Inside a bucket the target splits equally across its holdings.
3. **Enter the new cash** you want to invest (sidebar → *Additional capital*)
   and click **Build my Action Plan**.
4. **Read the verdict first**:
   - **✅ NO ACTION NEEDED** — every drift is under your materiality threshold
     (default 3%); trading would only burn fees and tax.
   - **🛒 Deploy €X** — a whole-share, buy-only order list (ticker, shares,
     euro cost, custodian) that pushes you back toward your plan **without
     selling anything** — so no capital-gains tax.
   - **🔁 Full rebalance** — only on explicit request, always shown *with* the
     estimated 26%/12.5% tax bill so you can weigh tax paid vs drift removed.
5. Every number carries a plain-language explanation, and the *Assumptions*
   panel lists every model input (risk-free rate, lookback, estimator…) with a
   one-line "why".

## How to interpret the outputs

- **Max drift** — the largest gap between a current and target weight, at
  asset *and* bucket level. It is the single number the verdict hangs on.
- **Buy-only plan** — allocates your cash to the most-underweight positions
  (whole shares, knapsack-rounded so residual cash is minimal). Selling is
  *never* proposed here.
- **Tax estimate** — Italian capital gains: 26% on equities/ETFs/ETCs, 12.5%
  on white-list government bonds, charged only on the *gain* portion of sells.
  Trade Republic withholds at source (sostituto d'imposta); on IBKR you settle
  through your tax return (regime dichiarativo).
- **Research report** — the Markowitz frontier, Sharpe ratios, risk clustering
  and backtests are *advisory context*: historical estimates are noisy, so the
  optimiser output is deliberately not used as your trading target.

## Known limitations (read before trusting a number)

- **Expected returns are historical means** — the least reliable estimator in
  finance. They colour the research tabs; they do not drive your plan.
- **Equal split inside buckets** — a deliberate simplification; adjust bucket
  membership to change effective per-asset targets.
- **One average cost per ticker** (≈ Italian *costo medio ponderato*): no
  per-lot accounting, no acquisition dates.
- **Tax simplifications**: the 26%/12.5% estimate ignores the minusvalenze
  compensation rules — in Italy **ETF losses cannot offset other gains**
  (redditi di capitale) while single-stock/ETC losses can (redditi diversi) —
  and assumes the white-list rate from the "Bond" asset type.
- **Short histories truncate the common window** — the app now warns loudly
  which ticker cut the window and when estimates rest on <1 year of data.
- **Decision-support aid, not financial advice.** Estimates come from past
  data and can be substantially wrong.

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
| **High-contrast light "research report" UI** (5 commented tabs) | `app.py` (CSS injection) |
| **Static, anchored Plotly charts** (zoom/pan locked, hover live) | `visualization.STATIC_PLOT_CONFIG` |
| **Executive Action Report** + Master Execution blueprint | `app._tab_executive` |
| **Execution custodian routing** (IBKR Live vs Trade Republic) | `portfolio.custodian_for` |
| **Optional password gate** | `app.py` (`st.secrets`) |
| **Buy-only cash allocation** (no-sell rebalancing) | `optimizer.buy_only_max_sharpe` |
| **Smart knapsack residual-cash rounding** | `portfolio.reallocation_plan` |
| **Macro-regime backtests** (2022/2008/2000 shocks) | `data_loader.apply_regime` |
| **IBKR suffix mapping** (`.DE`/`.MI` → clean root) | `portfolio.map_ibkr_symbol` |
| **HRP correlation cluster dendrogram** | `portfolio.hierarchical_clusters` |
| **Abstract DataProvider framework** | `src/providers.py` (`BaseDataProvider`) |
| **Live IBKR native engine** (`ib_insync`) | `providers.IBKRNativeProvider` |
| **Hybrid live + manual holdings merge** | `portfolio.merge_holdings` |
| **PAC ledger** (TWR + MWR/XIRR) | `portfolio.compute_pac_metrics` |
| **Cash-inflow drift efficiency monitor** | `optimizer.required_capital_to_align` |
| **Executive HTML report export** | `portfolio.build_executive_report_html` |
| **Live snapshot freeze + offline recovery** | `PortfolioStore.save_frozen_snapshot` |
| **Bucket target allocation** (your plan as the rebalancing target) | `portfolio.target_weights_from_buckets` |
| **Drift report + materiality verdict** ("no action needed") | `portfolio.compute_drift` |
| **Answer-first Action Plan** (whole shares, EUR, buy-only default) | `app._render_action_plan` |
| **Tax-aware full rebalance on request** (26%/12.5% bill + trade-off) | `app.py` + `portfolio.reallocation_plan` |
| **Short-history truncation warnings** | `data_loader.build_price_data` |
| **Black-Litterman market-view overrides** (per-ticker excess + confidence) | `optimizer.black_litterman_posterior_returns` |
| **Regime-shift correlation stress** (×1.4 convergence, vol-preserving) | `optimizer.stress_correlation_convergence` |
| **Dynamic adaptive rebalancing bands** (`base_factor × σᵢ`) | `portfolio.adaptive_tolerance_bands` |
| **Smart savings goal** (reverse Monte-Carlo + required-saving solve) | `optimizer.gbm_goal_analysis` |

The chart shows the feasible-set cloud coloured by Sharpe ratio, the efficient
frontier, the CML, the risk-free point, and star/diamond markers for the MVP,
the tangency portfolio, the **market benchmark** and your current portfolio.
Hover any point to see its return, volatility, Sharpe ratio and weights.

### Advanced controls (sidebar)

- **Risk-free rate** — defaults to a manual **2.5% EUR short-rate proxy**
  (consistent with a EUR portfolio). The live ^IRX fetch (US 3-month T-Bill, a
  *USD* rate) remains available behind a toggle.
- **Market Benchmark Ticker** — defaults to `VWCE.DE` (FTSE All-World UCITS);
  plotted as a gold marker and used as the market proxy for Black-Litterman
  priors.
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
  (`VWCE.DE`, `EIMI.MI`, `ABT`, `AGGH.MI`).
- **Shares** — units held (`0` = include in the optimisation as watch-only).
- **Purchase Price** — optional average buy price; drives the per-asset and
  total **unrealised P/L** and the capital-gains tax estimate.
- **Bucket** — which slice of *your* plan the position belongs to; the Action
  Plan rebalances toward the bucket targets.

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

- **Black-Litterman views**: the equilibrium prior `Π = δΣw_mkt + R_f` is
  blended with your absolute views (`benchmark return + excess`) under Idzorek
  confidences to yield the posterior `μ` fed to Max-Sharpe; no views → standard MPT.
- **Correlation stress**: decompose `Σ → (σ, ρ)`, inflate equity/risk
  off-diagonals `ρ ← min(1.4·ρ, 0.99)`, then recompose with the **original**
  variances — diversification collapses while per-asset vol is preserved.
- **Adaptive bands**: per-asset HOLD tolerance `θᵢ = base_factor · σᵢ`, so
  volatile assets churn less and low-vol bonds rebalance tighter.
- **Savings goal (reverse Monte-Carlo)**: `P(success)` is the share of 10 000 GBM
  paths crossing the target by the horizon; when it falls below 85% a bisection
  solves the monthly contribution whose **median** path lands on the goal.

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

## Data architecture (provider framework)

Price/metadata ingestion is abstracted behind `BaseDataProvider` (`src/providers.py`):

- **`YahooFinanceProvider`** — the robust, decoupled / offline-safe default.
- **`IBKRNativeProvider`** — a live engine that connects to a local **TWS
  (`7496`)** or **IB Gateway (`4001`)** socket via `ib_insync`, ingesting the
  account's cash, base currency and positions, and pulling daily bars from IBKR
  with a **per-ticker Yahoo fallback** for assets lacking a market-data
  subscription.

Pick the engine from **"Active Data Architecture Source"** in the sidebar. When
IBKR is selected the app connects, shows a green *"Connected to Account: Uxxxxxx"*
badge and auto-populates the holdings table from the live feed. If TWS isn't
running it cleanly falls back to safe layout mode. **Manual freedom is
preserved at all times** — regardless of engine, you can always click **+** to
add/edit/delete external assets (Trade Republic, bonds, funds, ETFs); live rows
are *merged* on top via `portfolio.merge_holdings` without ever locking the
table. `ib_insync` is optional: the app imports and unit-tests cleanly without
it, and its asyncio loop is isolated via `util.startLoop()`.

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
