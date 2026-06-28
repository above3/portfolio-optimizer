"""Portfolio Optimizer — institutional-grade Modern Portfolio Theory web app.

Run with::

    streamlit run app.py

Features: hybrid IBKR/manual holdings (with purchase price, asset type and P/L),
a VWCE.DE core anchor, sector guardrails, currency normalisation (EUR/USD),
lump-sum vs DCA-tranche deployment, rebalancing-tolerance bands, an Italian
capital-gains tax engine, a behavioral volatility classifier, historical
backtesting, a GBM Monte-Carlo wealth projection and IBKR Basket-Trader export.

The financial engines live in ``src/`` and are completely Streamlit-free.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src import config
from src.data_loader import PriceDataError, fetch_price_history, fetch_risk_free_rate
from src.optimizer import (
    AnchorConstraint,
    FrontierResult,
    PortfolioPoint,
    WatchlistRecommendation,
    analyse,
    backtest_equity_curves,
    screen_watchlist,
    simulate_gbm_wealth,
)
from src.portfolio import (
    CurrentPortfolio,
    NoiseReport,
    basket_trader_csv,
    build_current_portfolio,
    classify_noise,
    deployment_plan,
    parse_ibkr_positions,
    portfolio_config_csv,
    reallocation_plan,
)
from src.sample_data import generate_sample_prices, load_prices_from_csv
from src.visualization import (
    backtest_figure,
    correlation_heatmap,
    efficient_frontier_figure,
    gbm_fan_chart,
)

st.set_page_config(page_title="Portfolio Optimizer · MPT", page_icon="📈", layout="wide")


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #
def _sym(currency: str | None) -> str:
    return config.CURRENCY_SYMBOLS.get((currency or "USD").upper(), "$")


def _money(value: float, currency: str | None) -> str:
    return f"{_sym(currency)}{value:,.0f}"


# --------------------------------------------------------------------------- #
# Cached compute layer
# --------------------------------------------------------------------------- #
def _as_dict(data) -> dict:
    return {
        "prices": data.prices,
        "latest_prices": data.latest_prices,
        "valid_tickers": data.valid_tickers,
        "failed_tickers": data.failed_tickers,
        "start": data.start,
        "end": data.end,
        "benchmark_ticker": data.benchmark_ticker,
        "benchmark_prices": data.benchmark_prices,
        "currencies": data.currencies,
        "sectors": data.sectors,
        "base_currency": data.base_currency,
        "notes": data.notes,
    }


@st.cache_data(show_spinner=False, ttl=3600)
def _fetch_live_rf() -> float:
    return fetch_risk_free_rate()


@st.cache_data(show_spinner=False)
def _load_prices_live(tickers, benchmark, lookback, base_currency) -> dict:
    return _as_dict(fetch_price_history(list(tickers), lookback_years=lookback,
                                        benchmark=benchmark, base_currency=base_currency))


@st.cache_data(show_spinner=False)
def _load_prices_sample(tickers, benchmark, lookback, base_currency) -> dict:
    return _as_dict(generate_sample_prices(list(tickers), lookback_years=lookback,
                                           benchmark=benchmark, base_currency=base_currency))


def _make_anchor(ticker, weight, mode) -> AnchorConstraint | None:
    return AnchorConstraint(ticker=ticker, weight=weight, mode=mode) if ticker else None


@st.cache_data(show_spinner=False)
def _run_analysis(prices, risk_free_rate, cov_method, n_simulations, max_weight,
                  returns_model, benchmark_prices, benchmark_ticker,
                  anchor_ticker, anchor_weight, anchor_mode,
                  sector_items, sector_max) -> FrontierResult:
    return analyse(
        prices, risk_free_rate=risk_free_rate, cov_method=cov_method,
        n_simulations=n_simulations, max_weight=max_weight, returns_model=returns_model,
        benchmark_prices=benchmark_prices, benchmark_ticker=benchmark_ticker,
        anchor=_make_anchor(anchor_ticker, anchor_weight, anchor_mode),
        sector_map=dict(sector_items), sector_max=sector_max,
    )


@st.cache_data(show_spinner=False)
def _run_watchlist(prices, base_tickers, watchlist_tickers, risk_free_rate, cov_method,
                   max_weight, returns_model, benchmark_prices,
                   anchor_ticker, anchor_weight, anchor_mode,
                   sector_items, sector_max) -> tuple[float, list[WatchlistRecommendation]]:
    return screen_watchlist(
        prices, base_tickers=list(base_tickers), watchlist_tickers=list(watchlist_tickers),
        risk_free_rate=risk_free_rate, cov_method=cov_method, max_weight=max_weight,
        anchor=_make_anchor(anchor_ticker, anchor_weight, anchor_mode),
        returns_model=returns_model, benchmark_prices=benchmark_prices,
        sector_map=dict(sector_items), sector_max=sector_max,
    )


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
def _init_state() -> None:
    if "holdings_df" not in st.session_state:
        st.session_state.holdings_df = pd.DataFrame(config.DEMO_PORTFOLIO)
    st.session_state.setdefault("editor_version", 0)
    st.session_state.setdefault("ibkr_file_id", None)
    st.session_state.setdefault("auto_run", False)


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def _risk_free_controls() -> float:
    fetch_live = st.sidebar.toggle(
        f"Fetch Live Risk-Free Rate ({config.RISK_FREE_TICKER})", value=True,
        help="Use the US 3-Month T-Bill yield as Rf for the tangency portfolio.")
    if fetch_live:
        try:
            rf = _fetch_live_rf()
            st.sidebar.success(f"Live Rf ({config.RISK_FREE_TICKER}): **{rf * 100:.2f}%**")
            return rf
        except Exception as exc:  # noqa: BLE001 - blocked network/parse error
            st.sidebar.warning(f"Live rate unavailable ({type(exc).__name__}). Enter manually.")
    rf_pct = st.sidebar.number_input("Risk-free rate Rf (%)", 0.0, 25.0,
                                     config.DEFAULT_RISK_FREE_RATE * 100, 0.25)
    return rf_pct / 100.0


def render_sidebar() -> dict:
    st.sidebar.header("⚙️ Parameters")

    base_currency = st.sidebar.selectbox(
        "Base Display Currency", options=config.BASE_CURRENCIES,
        index=config.BASE_CURRENCIES.index(config.DEFAULT_BASE_CURRENCY),
        help="All prices, values and P/L are normalised into this currency via "
             "live FX rates (graceful fallback to native currency on failure).")

    data_source = st.sidebar.radio(
        "Data source", ["Yahoo Finance (live)", "Sample data (offline)", "Upload CSV"],
        index=0, help="Live needs Yahoo Finance access; otherwise use offline/CSV.")
    uploaded_csv = None
    if data_source == "Upload CSV":
        uploaded_csv = st.sidebar.file_uploader("Close-price CSV", type=["csv"])

    st.sidebar.markdown("### 💵 Risk-free rate")
    risk_free_rate = _risk_free_controls()

    st.sidebar.markdown("### 📊 Benchmark & lookback")
    benchmark = st.sidebar.text_input("Market Benchmark Ticker",
                                      value=config.DEFAULT_BENCHMARK).strip().upper() or None
    lookback = st.sidebar.selectbox("Historical lookback (years)", config.LOOKBACK_CHOICES,
                                    index=config.LOOKBACK_CHOICES.index(config.DEFAULT_LOOKBACK_YEARS))

    st.sidebar.markdown("### 🧮 Returns & risk model")
    returns_label = st.sidebar.radio("Expected-returns model",
                                     list(config.RETURNS_MODELS.keys()), index=0)
    cov_label = st.sidebar.selectbox("Covariance estimator",
                                     list(config.COVARIANCE_METHODS.keys()), index=0)

    st.sidebar.markdown("### 🌍 Core global anchor")
    anchor_pct = st.sidebar.slider(
        f"Core Global Anchor Allocation ({config.ANCHOR_TICKER}) %",
        0, config.ANCHOR_MAX_PCT, config.DEFAULT_ANCHOR_PCT, 1)
    anchor_exact = st.sidebar.toggle("Strict Target Match (else Upper Cap Bound)", value=True)

    st.sidebar.markdown("### 🛡️ Constraints & guardrails")
    max_weight_pct = st.sidebar.slider("Max Allocation per Asset (%)", 5, 100,
                                       int(config.DEFAULT_MAX_WEIGHT * 100), 5)
    sector_max_pct = st.sidebar.slider("Max Sector Exposure (%)", config.SECTOR_MIN_PCT,
                                       config.SECTOR_MAX_PCT, config.DEFAULT_SECTOR_MAX_PCT, 5)
    tolerance_pct = st.sidebar.slider("Rebalancing Tolerance Threshold (%)",
                                      config.TOLERANCE_MIN_PCT, config.TOLERANCE_MAX_PCT,
                                      config.DEFAULT_TOLERANCE_PCT, 1)

    st.sidebar.markdown("### 💸 Friction")
    disable_fractional = st.sidebar.checkbox("Disable Fractional Shares", value=False)
    transaction_cost = st.sidebar.number_input(
        f"Estimated Transaction Cost per Trade ({_sym(base_currency)})",
        min_value=0.0, value=config.DEFAULT_TRANSACTION_COST, step=0.5)

    st.sidebar.markdown("### 💰 Capital deployment")
    capital = st.sidebar.number_input(
        f"Additional capital to invest ({_sym(base_currency)})",
        min_value=0.0, value=0.0, step=1000.0)
    deployment_strategy = st.sidebar.selectbox("Capital Deployment Strategy",
                                               config.DEPLOYMENT_STRATEGIES, index=0)
    tranches = config.DEFAULT_TRANCHES
    if "Tranche" in deployment_strategy or "DCA" in deployment_strategy:
        tranches = st.sidebar.slider("Number of Active Tranches (Periods)",
                                     config.TRANCHE_MIN, config.TRANCHE_MAX,
                                     config.DEFAULT_TRANCHES, 1)

    st.sidebar.markdown("### 🎲 Projection & simulation")
    horizon_years = st.sidebar.slider("Investment Horizon (Years)",
                                      config.HORIZON_MIN_YEARS, config.HORIZON_MAX_YEARS,
                                      config.DEFAULT_HORIZON_YEARS, 1)
    n_sim = st.sidebar.slider("Simulated portfolios (Markowitz bullet)",
                              1000, 20000, config.DEFAULT_N_SIMULATIONS, 1000)

    st.sidebar.markdown("---")
    st.sidebar.caption("Educational tool. Past performance does not guarantee future "
                       "results. Long-only weights; simulated Italian taxes.")

    return {
        "base_currency": base_currency, "data_source": data_source,
        "uploaded_csv": uploaded_csv, "risk_free_rate": risk_free_rate,
        "benchmark": benchmark, "lookback": int(lookback),
        "returns_model": config.RETURNS_MODELS[returns_label],
        "cov_method": config.COVARIANCE_METHODS[cov_label],
        "anchor_ticker": config.ANCHOR_TICKER, "anchor_weight": anchor_pct / 100.0,
        "anchor_mode": "exact" if anchor_exact else "cap",
        "max_weight": max_weight_pct / 100.0, "sector_max": sector_max_pct / 100.0,
        "tolerance": tolerance_pct / 100.0, "allow_fractional": not disable_fractional,
        "transaction_cost": float(transaction_cost), "capital": float(capital),
        "deployment_strategy": deployment_strategy, "tranches": int(tranches),
        "horizon_years": int(horizon_years), "n_simulations": int(n_sim),
    }


# --------------------------------------------------------------------------- #
# Holdings input (IBKR import + editable hybrid table)
# --------------------------------------------------------------------------- #
def render_data_input() -> tuple[pd.DataFrame, list[str]]:
    st.subheader("1 · Your portfolio")
    uploaded = st.file_uploader(
        "📥 Drag and Drop your IBKR Position Report (CSV)", type=["csv"], key="ibkr_uploader",
        help="IBKR position/activity export. Detected positions (symbol, shares, "
             "purchase price, asset type) auto-fill the table and run the optimisation.")
    if uploaded is not None:
        file_id = getattr(uploaded, "file_id", f"{uploaded.name}:{uploaded.size}")
        if file_id != st.session_state.ibkr_file_id:
            try:
                parsed = parse_ibkr_positions(uploaded)
                st.session_state.holdings_df = parsed
                st.session_state.ibkr_file_id = file_id
                st.session_state.editor_version += 1
                st.session_state.auto_run = True
                st.success(f"✅ Imported {len(parsed)} positions from your IBKR report.")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not parse IBKR CSV: {exc}")

    st.caption("Add tickers from **any** broker (IBKR import, Trade Republic, …) — stocks "
               "**and** bonds/bond ETFs. Click **+** to add rows. *Purchase Price* drives "
               "P/L & tax; *Asset Type* sets the Italian tax rate (26% equity / 12.5% bond).")
    edited = st.data_editor(
        st.session_state.holdings_df, num_rows="dynamic", use_container_width=True,
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker", help="Yahoo symbol, e.g. AAPL, VWCE.DE, AGGH.MI."),
            "Shares": st.column_config.NumberColumn("Shares", help="Units held (0 = watch-only)."),
            "Purchase Price": st.column_config.NumberColumn("Purchase Price", min_value=0.0, format="%.2f"),
            "Asset Type": st.column_config.SelectboxColumn("Asset Type", options=list(config.ASSET_TYPES),
                                                           default=config.DEFAULT_ASSET_TYPE),
        },
        key=f"holdings_editor_{st.session_state.editor_version}")
    st.session_state.holdings_df = edited

    watchlist_raw = st.text_input(
        "🔭 Enter Tickers to screen for Diversification (comma-separated, e.g. GLD, NVDA, LVMH)",
        value=config.DEFAULT_WATCHLIST)
    watchlist = [t.strip().upper() for t in watchlist_raw.split(",") if t.strip()]
    return edited, list(dict.fromkeys(watchlist))


def _parse_holdings(edited: pd.DataFrame) -> tuple[dict, dict, dict]:
    """Return ``(shares, purchase_prices, asset_types)`` from the editor frame."""
    shares: dict[str, float] = {}
    purchase_prices: dict[str, float] = {}
    asset_types: dict[str, str] = {}
    for _, row in edited.iterrows():
        ticker = str(row.get("Ticker", "") or "").strip().upper()
        if not ticker:
            continue
        try:
            qty = float(row.get("Shares", 0) or 0)
        except (TypeError, ValueError):
            qty = 0.0
        shares[ticker] = shares.get(ticker, 0.0) + qty
        pp = row.get("Purchase Price", None)
        try:
            pp = float(pp) if pp is not None and str(pp) != "" else None
        except (TypeError, ValueError):
            pp = None
        if pp and pp > 0:
            purchase_prices[ticker] = pp
        atype = str(row.get("Asset Type", config.DEFAULT_ASSET_TYPE) or config.DEFAULT_ASSET_TYPE)
        asset_types[ticker] = atype
    return shares, purchase_prices, asset_types


# --------------------------------------------------------------------------- #
# Results: P&L ledger, noise classifier, comparison
# --------------------------------------------------------------------------- #
def _pnl_ledger(current: CurrentPortfolio | None, symbol_ccy: str) -> None:
    sym = _sym(symbol_ccy)
    c1, c2, c3 = st.columns(3)
    if current is None or not current.has_cost_data:
        c1.metric("Portfolio value", _money(current.total_value, symbol_ccy) if current else "—")
        c2.metric("Unrealised P/L", "n/a")
        c3.metric("Total return", "n/a")
        return
    pnl_pct = current.total_pnl / current.total_cost * 100 if current.total_cost > 0 else 0.0
    c1.metric("Portfolio value", _money(current.total_value, symbol_ccy))
    c2.metric("Unrealised P/L", f"{sym}{current.total_pnl:,.0f}", delta=f"{pnl_pct:+.1f}%")
    c3.metric("Invested cost basis", _money(current.total_cost, symbol_ccy))


def _noise_section(noise: NoiseReport) -> None:
    if not noise.has_data:
        st.info("ℹ️ " + noise.commentary)
        return
    if noise.verdict == "grave":
        st.error(noise.commentary)
    else:
        st.success(noise.commentary)
    with st.expander("🔬 Per-asset volatility noise classification"):
        rows = []
        for a in noise.assets:
            badge = "🟢 Normal Market Noise" if a.within_1sigma else "🔴 Statistical Outlier / Deep Drawdown"
            rows.append({"Ticker": a.ticker, "P/L %": round(a.pnl_pct * 100, 1),
                         "Annual σ %": round(a.annual_vol * 100, 1), "Classification": badge})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption("Within ±1σ of annualised volatility → expected, non-structural noise. "
                   "Beyond ±1σ → a statistical outlier worth a deliberate review.")


def _weights_table(point: PortfolioPoint) -> pd.DataFrame:
    return pd.DataFrame([
        {"Ticker": t, "Weight %": round(w * 100, 2)}
        for t, w in sorted(point.weights.items(), key=lambda kv: kv[1], reverse=True)
        if abs(w) > 1e-4])


def _metric_columns(result: FrontierResult, current: CurrentPortfolio | None, ccy: str) -> None:
    cols = st.columns(3)
    with cols[0]:
        st.markdown("#### 🧍 Current portfolio")
        if current is None:
            st.info("No current holdings (drafting a new portfolio).")
        else:
            st.metric("Expected return", f"{current.expected_return * 100:.2f}%")
            st.metric("Volatility", f"{current.volatility * 100:.2f}%")
            st.metric("Sharpe ratio", f"{current.sharpe:.2f}")
            st.metric("Total value", _money(current.total_value, ccy))
    with cols[1]:
        st.markdown("#### 🚀 Max Sharpe (Tangency)")
        t = result.tangency
        if t is None:
            st.warning("No long-only portfolio beats Rf at this rate.")
        else:
            st.metric("Expected return", f"{t.expected_return * 100:.2f}%",
                      delta=None if current is None else f"{(t.expected_return - current.expected_return) * 100:.2f}%")
            st.metric("Volatility", f"{t.volatility * 100:.2f}%",
                      delta=None if current is None else f"{(t.volatility - current.volatility) * 100:.2f}%",
                      delta_color="inverse")
            st.metric("Sharpe ratio", f"{t.sharpe:.2f}",
                      delta=None if current is None else f"{t.sharpe - current.sharpe:.2f}")
    with cols[2]:
        st.markdown("#### 🛡️ Minimum variance")
        m = result.mvp
        st.metric("Expected return", f"{m.expected_return * 100:.2f}%")
        st.metric("Volatility", f"{m.volatility * 100:.2f}%")
        st.metric("Sharpe ratio", f"{m.sharpe:.2f}")


def _benchmark_banner(result: FrontierResult) -> None:
    b = result.benchmark
    if b is None:
        return
    msg = (f"📌 **{b.name}** — return {b.expected_return * 100:.2f}%, vol "
           f"{b.volatility * 100:.2f}%, Sharpe **{b.sharpe:.2f}**.")
    if result.tangency is not None:
        if result.tangency.sharpe >= b.sharpe:
            msg += f" Your tangency Sharpe {result.tangency.sharpe:.2f} **beats** the benchmark."
        else:
            msg += f" The benchmark Sharpe still edges the tangency ({result.tangency.sharpe:.2f})."
    st.info(msg)


def _deployment_metrics(deployment, ccy: str) -> None:
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Capital Wealth Pool", _money(deployment.total_wealth, ccy),
              help="Current portfolio value + all additional cash.")
    c2.metric("Deployed in Current Tranche", _money(deployment.active_capital, ccy),
              help="Capital allocated by the target weights in this step.")
    c3.metric("Sidelined Cash Reserve", _money(deployment.sidelined_cash, ccy),
              help="Cash buffered for future DCA tranches.")


def _render_reallocation(target: PortfolioPoint, current, deployment, latest_prices,
                         mu, params, asset_types, sectors, ccy, key_prefix: str) -> None:
    sym = _sym(ccy)
    left, right = st.columns([1, 1.7])
    with left:
        st.markdown("**Target weights**")
        st.dataframe(_weights_table(target), hide_index=True, use_container_width=True)
    with right:
        st.markdown(f"**Reallocation plan** (active capital ≈ {_money(deployment.active_capital, ccy)})")
        res = reallocation_plan(
            current_values=current.values if current is not None else {},
            target_weights=target.weights, investable_capital=deployment.active_capital,
            latest_prices=latest_prices, mu=mu, risk_free_rate=params["risk_free_rate"],
            allow_fractional=params["allow_fractional"], transaction_cost=params["transaction_cost"],
            tolerance=params["tolerance"], purchase_prices=current.purchase_prices if current else {},
            asset_types=asset_types)
        st.dataframe(res.table, hide_index=True, use_container_width=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Trades", res.num_trades)
        m2.metric("Fees", f"{sym}{res.total_fees:,.2f}")
        m3.metric("Cap-gains tax", f"{sym}{res.total_tax:,.2f}")
        m4.metric("Residual cash", f"{sym}{res.residual_cash:,.2f}")

        if res.gross_return == res.gross_return:  # not NaN
            g1, g2 = st.columns(2)
            g1.metric("Gross Optimized Return", f"{res.gross_return * 100:.2f}%")
            net_delta = (res.net_return - res.gross_return) * 100
            g2.metric("Net-of-Tax-and-Fees Return", f"{res.net_return * 100:.2f}%",
                      delta=f"{net_delta:.2f}%" if net_delta else None, delta_color="inverse")

        # ---- Institutional execution exports ----
        st.markdown("**📤 Execution export**")
        e1, e2 = st.columns(2)
        e1.download_button("Export IBKR Basket Orders (.csv)", basket_trader_csv(res),
                           file_name=f"ibkr_basket_orders_{key_prefix}.csv", mime="text/csv",
                           use_container_width=True, key=f"basket_{key_prefix}")
        cfg_csv = portfolio_config_csv(
            current.values if current is not None else {}, target.weights,
            deployment.active_capital, sectors, ccy)
        e2.download_button("Download Portfolio Configuration (.csv)", cfg_csv,
                           file_name=f"portfolio_configuration_{key_prefix}.csv", mime="text/csv",
                           use_container_width=True, key=f"config_{key_prefix}")


def _render_watchlist(base_sharpe, recommendations, anchor_ticker) -> None:
    st.markdown("---")
    st.subheader("Watchlist — diversification scanner")
    if not recommendations:
        st.info("Add tickers to the watchlist box to screen candidates.")
        return
    st.caption(f"Baseline Max Sharpe{' (incl. anchor)' if anchor_ticker else ''}: "
               f"**{base_sharpe:.2f}**. Each card simulates adding one candidate.")
    for rec in recommendations:
        if not rec.feasible:
            st.warning(f"**{rec.ticker}** — {rec.note}")
            continue
        improves = rec.sharpe_delta > 1e-4
        with st.container(border=True):
            cols = st.columns([2, 1, 1, 1])
            cols[0].markdown(f"### {'✅' if improves else '➖'} {rec.ticker}")
            cols[1].metric("Suggested weight", f"{rec.suggested_weight * 100:.1f}%")
            cols[2].metric("Max Sharpe", f"{rec.new_sharpe:.2f}", delta=f"{rec.sharpe_delta:+.2f}")
            cols[3].metric("Avg |ρ| to pool", f"{rec.avg_correlation:.2f}")
            if improves:
                st.success(
                    f"Adding **{rec.ticker}** with a suggested weight of "
                    f"**{rec.suggested_weight * 100:.1f}%** would shift your Efficient "
                    f"Frontier upwards, improving your Max Sharpe Ratio from "
                    f"**{rec.base_sharpe:.2f}** to **{rec.new_sharpe:.2f}** "
                    f"(+{rec.sharpe_delta:.2f}) by lowering overall portfolio covariance "
                    f"(avg ρ ≈ {rec.avg_correlation:.2f}).")
            else:
                st.caption(f"Adding **{rec.ticker}** would not improve risk-adjusted return "
                           f"({rec.base_sharpe:.2f} → {rec.new_sharpe:.2f}).")


# --------------------------------------------------------------------------- #
# Backtest & Monte-Carlo tabs
# --------------------------------------------------------------------------- #
def _render_backtest(prices, base_pool, current, result, benchmark_prices,
                     benchmark_ticker, risk_free_rate) -> None:
    st.subheader("Historical Backtest Analysis")
    weight_sets: dict[str, dict] = {}
    if current is not None:
        weight_sets["Current Portfolio"] = current.weights
    if result.tangency is not None:
        weight_sets["Optimized Max Sharpe"] = result.tangency.weights
    if not weight_sets:
        st.info("Need a current and/or optimised portfolio to backtest.")
        return
    bt = backtest_equity_curves(prices[base_pool], weight_sets, risk_free_rate,
                                benchmark_prices=benchmark_prices, benchmark_name=benchmark_ticker)
    st.plotly_chart(backtest_figure(bt), use_container_width=True)
    st.markdown("**Diagnostics** (frozen weights over the lookback window)")
    st.dataframe(bt.diagnostics, use_container_width=True)


def _render_montecarlo(result, deployment, params, ccy) -> None:
    st.subheader("Monte Carlo Wealth Projection (GBM)")
    if result.tangency is None:
        st.warning("No tangency portfolio to project. Adjust Rf or constraints.")
        return
    monthly = deployment.per_tranche_cash if deployment.tranches > 1 else 0.0
    initial = deployment.active_capital if deployment.active_capital > 0 else 10000.0
    proj = simulate_gbm_wealth(
        initial_capital=initial, mu=result.tangency.expected_return,
        sigma=result.tangency.volatility, horizon_years=params["horizon_years"],
        monthly_contribution=monthly)
    st.caption(f"Compounding **{_money(initial, ccy)}** at the tangency portfolio's "
               f"μ={result.tangency.expected_return * 100:.1f}% / σ="
               f"{result.tangency.volatility * 100:.1f}% over {params['horizon_years']}y, "
               f"with **{_money(monthly, ccy)}/mo** contributions ({proj.n_paths:,} paths).")
    st.plotly_chart(gbm_fan_chart(proj, _sym(ccy)), use_container_width=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Pessimistic (P10)", _money(proj.final_p10, ccy))
    c2.metric("Median (P50)", _money(proj.final_p50, ccy))
    c3.metric("Optimistic (P90)", _money(proj.final_p90, ccy))
    st.caption(f"Total contributed over horizon: {_money(proj.total_contributed, ccy)}.")


# --------------------------------------------------------------------------- #
# Top-level results renderer
# --------------------------------------------------------------------------- #
def render_results(result, current, noise, deployment, data, params,
                   watchlist_result, asset_types) -> None:
    ccy = params["base_currency"]
    latest_prices = data["latest_prices"]

    # ---- Top P&L ledger + behavioral classifier ----
    st.subheader("💹 Real-time P&L & behavioral check")
    _pnl_ledger(current, ccy)
    _noise_section(noise)

    tab_opt, tab_bt, tab_mc = st.tabs(
        ["📊 Optimization", "📈 Historical Backtest Analysis", "🎲 Monte Carlo Projection"])

    with tab_opt:
        caption_bits = [f"Returns model: **{result.returns_model.replace('_', ' ').title()}**",
                        f"currency **{ccy}**"]
        if result.max_weight < 1.0:
            caption_bits.append(f"per-asset cap **{result.max_weight * 100:.0f}%**")
        if result.sector_max < 1.0:
            caption_bits.append(f"sector cap **{result.sector_max * 100:.0f}%**")
        if result.anchor is not None and result.anchor.weight > 0:
            verb = "fixed at" if result.anchor.mode == "exact" else "≤"
            caption_bits.append(f"anchor **{result.anchor.ticker}** {verb} **{result.anchor.weight * 100:.0f}%**")
        st.caption(" · ".join(caption_bits) + ".")
        for note in result.notes + data.get("notes", []):
            st.warning(note)

        st.plotly_chart(efficient_frontier_figure(result, current), use_container_width=True)

        st.subheader("Portfolio comparison")
        _benchmark_banner(result)
        _metric_columns(result, current, ccy)

        st.markdown("---")
        st.subheader("Capital deployment & reallocation")
        _deployment_metrics(deployment, ccy)
        tab_tan, tab_mvp = st.tabs(["🚀 Max Sharpe (Tangency)", "🛡️ Minimum variance"])
        with tab_tan:
            if result.tangency is None:
                st.warning("Tangency portfolio unavailable at the current Rf.")
            else:
                _render_reallocation(result.tangency, current, deployment, latest_prices,
                                     result.mu, params, asset_types, result.sector_map, ccy, "tan")
        with tab_mvp:
            _render_reallocation(result.mvp, current, deployment, latest_prices,
                                 result.mu, params, asset_types, result.sector_map, ccy, "mvp")

        if watchlist_result is not None:
            base_sharpe, recs = watchlist_result
            anchor_ticker = result.anchor.ticker if result.anchor is not None else None
            _render_watchlist(base_sharpe, recs, anchor_ticker)

        st.markdown("---")
        st.subheader("Diversification — correlation matrix")
        st.plotly_chart(correlation_heatmap(result.corr), use_container_width=True)

    with tab_bt:
        _render_backtest(data["prices"], list(result.mu.index), current, result,
                         data["benchmark_prices"], data["benchmark_ticker"],
                         params["risk_free_rate"])

    with tab_mc:
        _render_montecarlo(result, deployment, params, ccy)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    _init_state()
    st.title("📈 Portfolio Optimizer — Institutional MPT")
    st.markdown("Hybrid holdings · VWCE core anchor · sector guardrails · currency "
                "normalisation · DCA tranches · Italian tax · behavioral P&L · "
                "backtesting · GBM projection · IBKR export.")

    params = render_sidebar()
    edited, watchlist = render_data_input()

    run = st.button("🔍 Optimize portfolio", type="primary", use_container_width=True)
    auto_run = st.session_state.auto_run
    st.session_state.auto_run = False
    if not (run or auto_run):
        st.info("Edit or import your holdings, set parameters, then click **Optimize portfolio**.")
        return

    shares, purchase_prices, asset_types = _parse_holdings(edited)
    if len(shares) < 1:
        st.error("Please enter at least one ticker.")
        return

    anchor_ticker = params["anchor_ticker"]
    base_set = set(shares) | {anchor_ticker}
    watchlist_extra = [t for t in watchlist if t not in base_set]
    universe = tuple(sorted(base_set | set(watchlist_extra)))
    source = params["data_source"]
    benchmark = params["benchmark"]
    base_ccy = params["base_currency"]

    spinner = {"Yahoo Finance (live)": "Fetching prices & FX from Yahoo Finance…",
               "Sample data (offline)": "Generating offline sample prices…",
               "Upload CSV": "Reading uploaded prices…"}[source]
    with st.spinner(spinner):
        try:
            if source == "Yahoo Finance (live)":
                data = _load_prices_live(universe, benchmark, params["lookback"], base_ccy)
            elif source == "Sample data (offline)":
                data = _load_prices_sample(universe, benchmark, params["lookback"], base_ccy)
            else:
                if params["uploaded_csv"] is None:
                    st.error("Please upload a price CSV, or pick another data source.")
                    return
                data = _as_dict(load_prices_from_csv(params["uploaded_csv"],
                                                     tickers=list(universe), benchmark=benchmark,
                                                     base_currency=base_ccy))
        except PriceDataError as exc:
            st.error(f"Data error: {exc}")
            if source == "Yahoo Finance (live)":
                st.info("If Yahoo Finance is blocked/rate-limited, switch to *Sample data (offline)*.")
            return
        except Exception as exc:  # noqa: BLE001
            st.error(f"Unexpected error while loading prices: {exc}")
            return

    valid = set(data["valid_tickers"])
    base_pool = sorted(t for t in base_set if t in valid)
    watchlist_valid = [t for t in watchlist if t in valid and t not in base_set]
    anchor_loaded = anchor_ticker in valid

    if data["failed_tickers"]:
        st.warning("No data for: " + ", ".join(data["failed_tickers"]) + ". Excluded.")
    if len(base_pool) < 2:
        st.error("Fewer than two assets (holdings + anchor) returned usable data.")
        return

    st.subheader("2 · Data")
    bits = [f"Source: **{source}**", f"currency **{base_ccy}**",
            f"optimising **{len(base_pool)}** assets"]
    if anchor_loaded:
        bits.append(f"anchor **{anchor_ticker}**")
    if data.get("benchmark_ticker"):
        bits.append(f"benchmark **{data['benchmark_ticker']}**")
    if watchlist_valid:
        bits.append(f"watchlist **{len(watchlist_valid)}**")
    st.caption(" · ".join(bits) + f" · {data['start']} → {data['end']} "
               f"({len(data['prices'])} trading days).")
    for note in data.get("notes", []):
        st.warning(note)
    if benchmark and not data.get("benchmark_ticker"):
        st.warning(f"Benchmark **{benchmark}** could not be loaded; marker/BL proxy disabled.")
    if not anchor_loaded and params["anchor_weight"] > 0:
        st.warning(f"Anchor **{anchor_ticker}** could not be loaded; optimising without it.")
    if source == "Sample data (offline)":
        st.warning("⚠️ Using **synthetic** prices for demonstration — not real market data.",
                   icon="⚠️")

    eff_anchor = anchor_ticker if anchor_loaded else None
    base_prices = data["prices"][base_pool]
    sector_map = {t: data["sectors"].get(t, config.UNKNOWN_SECTOR) for t in base_pool}
    sector_items = tuple(sorted(sector_map.items()))

    with st.spinner("Computing efficient frontier…"):
        result = _run_analysis(
            base_prices, params["risk_free_rate"], params["cov_method"],
            params["n_simulations"], params["max_weight"], params["returns_model"],
            data["benchmark_prices"], data["benchmark_ticker"],
            eff_anchor, params["anchor_weight"], params["anchor_mode"],
            sector_items, params["sector_max"])
        current = build_current_portfolio(shares, data["latest_prices"], result.mu,
                                          result.cov, params["risk_free_rate"],
                                          purchase_prices=purchase_prices)
        noise = classify_noise(current, result.cov)
        base_value = current.total_value if current is not None else 0.0
        deployment = deployment_plan(base_value, params["capital"],
                                     params["deployment_strategy"], params["tranches"])

    watchlist_result = None
    if watchlist_valid:
        with st.spinner("Scanning watchlist for diversification…"):
            screen_pool = base_pool + watchlist_valid
            screen_sectors = tuple(sorted(
                {t: data["sectors"].get(t, config.UNKNOWN_SECTOR) for t in screen_pool}.items()))
            watchlist_result = _run_watchlist(
                data["prices"][screen_pool], tuple(base_pool), tuple(watchlist_valid),
                params["risk_free_rate"], params["cov_method"], params["max_weight"],
                params["returns_model"], data["benchmark_prices"],
                eff_anchor, params["anchor_weight"], params["anchor_mode"],
                screen_sectors, params["sector_max"])

    render_results(result, current, noise, deployment, data, params,
                   watchlist_result, asset_types)


if __name__ == "__main__":
    main()
