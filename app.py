"""Portfolio Optimizer — a Modern Portfolio Theory (Markowitz) web app.

Run with::

    streamlit run app.py

The app lets a user describe (or import) their current holdings, fetches
historical prices, and computes the Minimum-Variance and Maximum-Sharpe
(tangency) portfolios along with the efficient frontier and Capital Market Line.
It supports a live risk-free rate (^IRX), IBKR position import, a per-asset
weight cap, whole-share rounding with transaction costs, a market benchmark and
a Black-Litterman returns model. Results are shown on an interactive Plotly
chart plus reallocation tables.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src import config
from src.data_loader import PriceDataError, fetch_price_history, fetch_risk_free_rate
from src.optimizer import FrontierResult, PortfolioPoint, analyse
from src.portfolio import (
    CurrentPortfolio,
    ReallocationResult,
    build_current_portfolio,
    parse_ibkr_positions,
    reallocation_plan,
)
from src.sample_data import generate_sample_prices, load_prices_from_csv
from src.visualization import correlation_heatmap, efficient_frontier_figure

st.set_page_config(
    page_title="Portfolio Optimizer · MPT",
    page_icon="📈",
    layout="wide",
)


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
    }


@st.cache_data(show_spinner=False, ttl=3600)
def _fetch_live_rf() -> float:
    """Cached live risk-free rate (^IRX), refreshed hourly."""
    return fetch_risk_free_rate()


@st.cache_data(show_spinner=False)
def _load_prices_live(tickers: tuple[str, ...], benchmark: str | None, lookback: int) -> dict:
    return _as_dict(fetch_price_history(list(tickers), lookback_years=lookback,
                                        benchmark=benchmark))


@st.cache_data(show_spinner=False)
def _load_prices_sample(tickers: tuple[str, ...], benchmark: str | None, lookback: int) -> dict:
    return _as_dict(generate_sample_prices(list(tickers), lookback_years=lookback,
                                           benchmark=benchmark))


@st.cache_data(show_spinner=False)
def _run_analysis(
    prices: pd.DataFrame,
    risk_free_rate: float,
    cov_method: str,
    n_simulations: int,
    max_weight: float,
    returns_model: str,
    benchmark_prices: pd.Series | None,
    benchmark_ticker: str | None,
) -> FrontierResult:
    return analyse(
        prices,
        risk_free_rate=risk_free_rate,
        cov_method=cov_method,
        n_simulations=n_simulations,
        max_weight=max_weight,
        returns_model=returns_model,
        benchmark_prices=benchmark_prices,
        benchmark_ticker=benchmark_ticker,
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
# Sidebar inputs
# --------------------------------------------------------------------------- #
def _risk_free_controls() -> float:
    """Render the live/manual risk-free-rate controls and return the rate."""
    fetch_live = st.sidebar.toggle(
        f"Fetch Live Risk-Free Rate ({config.RISK_FREE_TICKER})",
        value=True,
        help="Automatically use the US 3-Month T-Bill yield as the risk-free "
             "rate for the tangency portfolio and CML.",
    )

    if fetch_live:
        try:
            rf = _fetch_live_rf()
            st.sidebar.success(
                f"Live Rf ({config.RISK_FREE_TICKER}): **{rf * 100:.2f}%**"
            )
            return rf
        except Exception as exc:  # noqa: BLE001 - network blocked / parse error
            st.sidebar.warning(
                f"Live rate unavailable ({type(exc).__name__}). Enter it manually."
            )

    rf_pct = st.sidebar.number_input(
        "Risk-free rate Rf (%)",
        min_value=0.0, max_value=25.0,
        value=config.DEFAULT_RISK_FREE_RATE * 100, step=0.25,
        help="Annual risk-free rate. Anchors the Capital Market Line.",
    )
    return rf_pct / 100.0


def render_sidebar() -> dict:
    st.sidebar.header("⚙️ Parameters")

    data_source = st.sidebar.radio(
        "Data source",
        options=["Yahoo Finance (live)", "Sample data (offline)", "Upload CSV"],
        index=0,
        help="Live prices need access to Yahoo Finance. If blocked or "
             "rate-limited, use the offline sample generator or a price CSV.",
    )

    uploaded_csv = None
    if data_source == "Upload CSV":
        uploaded_csv = st.sidebar.file_uploader(
            "Close-price CSV", type=["csv"],
            help="First column = Date; one column of close prices per ticker.",
        )

    st.sidebar.markdown("### 💵 Risk-free rate")
    risk_free_rate = _risk_free_controls()

    st.sidebar.markdown("### 📊 Benchmark & lookback")
    benchmark = st.sidebar.text_input(
        "Market Benchmark Ticker", value=config.DEFAULT_BENCHMARK,
        help="Plotted as a reference point and used as the Black-Litterman "
             "market proxy. E.g. SPY, VWCE, ACWI.",
    ).strip().upper() or None

    lookback = st.sidebar.selectbox(
        "Historical lookback (years)", options=config.LOOKBACK_CHOICES,
        index=config.LOOKBACK_CHOICES.index(config.DEFAULT_LOOKBACK_YEARS),
    )

    st.sidebar.markdown("### 🧮 Returns & risk model")
    returns_label = st.sidebar.radio(
        "Expected-returns model", options=list(config.RETURNS_MODELS.keys()),
        index=0,
        help="Black-Litterman anchors returns to market-implied equilibrium "
             "(using the benchmark as the market proxy).",
    )
    cov_label = st.sidebar.selectbox(
        "Covariance estimator", options=list(config.COVARIANCE_METHODS.keys()),
        index=0, help="Ledoit-Wolf shrinkage is more stable for short histories.",
    )

    st.sidebar.markdown("### 🛡️ Constraints & friction")
    max_weight_pct = st.sidebar.slider(
        "Max Allocation per Asset (%)", min_value=5, max_value=100,
        value=int(config.DEFAULT_MAX_WEIGHT * 100), step=5,
        help="Caps each asset's weight to prevent the tangency portfolio from "
             "over-concentrating in a single high performer.",
    )
    disable_fractional = st.sidebar.checkbox(
        "Disable Fractional Shares", value=False,
        help="Round trades down to whole shares; leftover becomes residual cash.",
    )
    transaction_cost = st.sidebar.number_input(
        "Estimated Transaction Cost per Trade ($)", min_value=0.0,
        value=config.DEFAULT_TRANSACTION_COST, step=0.5,
        help="Subtracted from the rebalanced portfolio's expected return.",
    )

    st.sidebar.markdown("### 💰 Capital & simulation")
    capital = st.sidebar.number_input(
        "Additional capital to invest ($)", min_value=0.0, value=0.0, step=1000.0,
        help="Optional new cash added to the current value for reallocation.",
    )
    n_sim = st.sidebar.slider(
        "Simulated portfolios (Markowitz bullet)",
        min_value=1000, max_value=20000, value=config.DEFAULT_N_SIMULATIONS, step=1000,
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Educational tool. Past performance does not guarantee future results. "
        "Long-only weights (no shorting/leverage)."
    )

    return {
        "data_source": data_source,
        "uploaded_csv": uploaded_csv,
        "risk_free_rate": risk_free_rate,
        "benchmark": benchmark,
        "lookback": int(lookback),
        "returns_model": config.RETURNS_MODELS[returns_label],
        "cov_method": config.COVARIANCE_METHODS[cov_label],
        "max_weight": max_weight_pct / 100.0,
        "allow_fractional": not disable_fractional,
        "transaction_cost": float(transaction_cost),
        "capital": float(capital),
        "n_simulations": int(n_sim),
    }


# --------------------------------------------------------------------------- #
# Holdings input (IBKR import + editable table)
# --------------------------------------------------------------------------- #
def render_data_input() -> pd.DataFrame:
    st.subheader("1 · Your portfolio")

    uploaded = st.file_uploader(
        "📥 Drag and Drop your IBKR Position Report (CSV)", type=["csv"],
        key="ibkr_uploader",
        help="Interactive Brokers position or activity export. Detected "
             "positions auto-fill the table below and run the optimisation.",
    )
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
            except Exception as exc:  # noqa: BLE001 - report any parse failure
                st.error(f"Could not parse IBKR CSV: {exc}")

    st.caption(
        "Enter the tickers you hold (or are considering) and the number of "
        "shares. Leave shares at 0 to include an asset in the optimisation "
        "without marking it as a current holding."
    )
    edited = st.data_editor(
        st.session_state.holdings_df,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Ticker": st.column_config.TextColumn(
                "Ticker", help="Yahoo Finance symbol, e.g. AAPL, MSFT, BTC-USD."),
            "Shares": st.column_config.NumberColumn(
                "Shares", help="Number of shares currently held."),
        },
        key=f"holdings_editor_{st.session_state.editor_version}",
    )
    st.session_state.holdings_df = edited
    return edited


def _parse_holdings(edited: pd.DataFrame) -> dict[str, float]:
    """Turn the editor frame into a {ticker: shares} mapping."""
    holdings: dict[str, float] = {}
    for _, row in edited.iterrows():
        ticker = str(row.get("Ticker", "") or "").strip().upper()
        if not ticker:
            continue
        try:
            shares = float(row.get("Shares", 0) or 0)
        except (TypeError, ValueError):
            shares = 0.0
        holdings[ticker] = holdings.get(ticker, 0.0) + shares
    return holdings


# --------------------------------------------------------------------------- #
# Results rendering
# --------------------------------------------------------------------------- #
def _weights_table(point: PortfolioPoint) -> pd.DataFrame:
    rows = [
        {"Ticker": t, "Weight %": round(w * 100, 2)}
        for t, w in sorted(point.weights.items(), key=lambda kv: kv[1], reverse=True)
        if abs(w) > 1e-4
    ]
    return pd.DataFrame(rows)


def _metric_columns(result: FrontierResult, current: CurrentPortfolio | None) -> None:
    cols = st.columns(3)

    with cols[0]:
        st.markdown("#### 🧍 Current portfolio")
        if current is None:
            st.info("No current holdings (drafting a new portfolio).")
        else:
            st.metric("Expected return", f"{current.expected_return * 100:.2f}%")
            st.metric("Volatility", f"{current.volatility * 100:.2f}%")
            st.metric("Sharpe ratio", f"{current.sharpe:.2f}")
            st.metric("Total value", f"${current.total_value:,.0f}")

    with cols[1]:
        st.markdown("#### 🚀 Max Sharpe (Tangency)")
        t = result.tangency
        if t is None:
            st.warning("No long-only portfolio beats Rf at this rate.")
        else:
            st.metric("Expected return", f"{t.expected_return * 100:.2f}%",
                      delta=None if current is None
                      else f"{(t.expected_return - current.expected_return) * 100:.2f}%")
            st.metric("Volatility", f"{t.volatility * 100:.2f}%",
                      delta=None if current is None
                      else f"{(t.volatility - current.volatility) * 100:.2f}%",
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
    msg = (f"📌 **Benchmark {b.name.split('(')[-1].rstrip(')')}** — "
           f"return {b.expected_return * 100:.2f}%, vol {b.volatility * 100:.2f}%, "
           f"Sharpe **{b.sharpe:.2f}**.")
    if result.tangency is not None:
        if result.tangency.sharpe >= b.sharpe:
            msg += (f" Your optimised (tangency) Sharpe {result.tangency.sharpe:.2f} "
                    "**beats** the benchmark on a risk-adjusted basis.")
        else:
            msg += (f" The benchmark's Sharpe still edges the tangency portfolio "
                    f"({result.tangency.sharpe:.2f}).")
    st.info(msg)


def _render_reallocation(
    target: PortfolioPoint,
    current: CurrentPortfolio | None,
    investable: float,
    latest_prices: pd.Series,
    mu: pd.Series,
    params: dict,
) -> None:
    left, right = st.columns([1, 1.6])
    with left:
        st.markdown("**Target weights**")
        st.dataframe(_weights_table(target), hide_index=True, use_container_width=True)

    with right:
        st.markdown(f"**Reallocation plan** (investable ≈ ${investable:,.0f})")
        res: ReallocationResult = reallocation_plan(
            current_values=current.values if current is not None else {},
            target_weights=target.weights,
            investable_capital=investable,
            latest_prices=latest_prices,
            mu=mu,
            risk_free_rate=params["risk_free_rate"],
            allow_fractional=params["allow_fractional"],
            transaction_cost=params["transaction_cost"],
        )
        st.dataframe(res.table, hide_index=True, use_container_width=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Trades", res.num_trades)
        m2.metric("Total fees", f"${res.total_fees:,.2f}")
        if not params["allow_fractional"]:
            m3.metric("Residual cash", f"${res.residual_cash:,.2f}")
        else:
            m3.metric("Residual cash", "$0.00")
        if res.net_return == res.net_return:  # not NaN
            fee_delta = (res.net_return - res.gross_return) * 100
            m4.metric("Net-of-fee return", f"{res.net_return * 100:.2f}%",
                      delta=f"{fee_delta:.2f}%" if fee_delta else None,
                      delta_color="inverse")


def render_results(
    result: FrontierResult,
    current: CurrentPortfolio | None,
    latest_prices: pd.Series,
    params: dict,
) -> None:
    st.subheader("3 · The Markowitz bullet")
    if result.max_weight < 1.0:
        st.caption(f"Per-asset weight capped at **{result.max_weight * 100:.0f}%**. "
                   f"Returns model: **{result.returns_model.replace('_', ' ').title()}**.")
    for note in result.notes:
        st.warning(note)
    st.plotly_chart(efficient_frontier_figure(result, current), use_container_width=True)

    st.subheader("4 · Portfolio comparison")
    _benchmark_banner(result)
    _metric_columns(result, current)

    st.markdown("---")
    st.subheader("5 · Suggested allocations & reallocation")

    base_value = current.total_value if current is not None else 0.0
    investable = base_value + params["capital"]
    if investable <= 0:
        investable = 10000.0  # nominal base so dollar targets are still shown

    tab_tan, tab_mvp = st.tabs(["🚀 Max Sharpe (Tangency)", "🛡️ Minimum variance"])
    with tab_tan:
        if result.tangency is None:
            st.warning("Tangency portfolio unavailable at the current risk-free rate.")
        else:
            _render_reallocation(result.tangency, current, investable,
                                 latest_prices, result.mu, params)
    with tab_mvp:
        _render_reallocation(result.mvp, current, investable,
                             latest_prices, result.mu, params)

    st.markdown("---")
    st.subheader("6 · Diversification — correlation matrix")
    st.caption(
        "Lower (or negative) correlations between assets reduce portfolio "
        "variance — the core benefit of diversification."
    )
    st.plotly_chart(correlation_heatmap(result.corr), use_container_width=True)

    with st.expander("Annualised expected returns (μ)"):
        st.dataframe(
            result.mu.mul(100).round(2).rename("Expected return %").to_frame(),
            use_container_width=True,
        )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    _init_state()
    st.title("📈 Portfolio Optimizer — Modern Portfolio Theory")
    st.markdown(
        "Build the **efficient frontier**, find the **Minimum-Variance** and "
        "**Maximum-Sharpe (tangency)** portfolios, compare against a **market "
        "benchmark**, and get a friction-aware **reallocation plan** — powered "
        "by `yfinance` + `PyPortfolioOpt`."
    )

    params = render_sidebar()
    edited = render_data_input()

    run = st.button("🔍 Optimize portfolio", type="primary", use_container_width=True)
    auto_run = st.session_state.auto_run
    st.session_state.auto_run = False  # consume one-shot trigger

    if not (run or auto_run):
        st.info("Edit or import your holdings above, set parameters in the "
                "sidebar, then click **Optimize portfolio**.")
        return

    holdings = _parse_holdings(edited)
    if len(holdings) < 2:
        st.error("Please enter at least **two** distinct tickers to optimise.")
        return

    # ---- Load prices (live / offline sample / CSV) ----
    tickers = tuple(sorted(holdings))
    source = params["data_source"]
    benchmark = params["benchmark"]
    spinner_msg = {
        "Yahoo Finance (live)": "Fetching historical prices from Yahoo Finance…",
        "Sample data (offline)": "Generating offline sample price history…",
        "Upload CSV": "Reading uploaded price history…",
    }[source]

    with st.spinner(spinner_msg):
        try:
            if source == "Yahoo Finance (live)":
                data = _load_prices_live(tickers, benchmark, params["lookback"])
            elif source == "Sample data (offline)":
                data = _load_prices_sample(tickers, benchmark, params["lookback"])
            else:  # Upload CSV
                if params["uploaded_csv"] is None:
                    st.error("Please upload a price CSV in the sidebar, or pick "
                             "another data source.")
                    return
                data = _as_dict(load_prices_from_csv(
                    params["uploaded_csv"], tickers=list(tickers), benchmark=benchmark))
        except PriceDataError as exc:
            st.error(f"Data error: {exc}")
            if source == "Yahoo Finance (live)":
                st.info("If Yahoo Finance is blocked or rate-limited, switch "
                        "**Data source** to *Sample data (offline)* to try the app.")
            return
        except Exception as exc:  # noqa: BLE001 - surface network/library errors
            st.error(f"Unexpected error while loading prices: {exc}")
            return

    if data["failed_tickers"]:
        st.warning("No data for: " + ", ".join(data["failed_tickers"]) + ". Excluded.")
    if len(data["valid_tickers"]) < 2:
        st.error("Fewer than two tickers returned usable data. Cannot optimise.")
        return

    st.subheader("2 · Data")
    bench_note = (f" · benchmark **{data['benchmark_ticker']}**"
                  if data.get("benchmark_ticker") else "")
    st.caption(
        f"Source: **{source}** · Loaded **{len(data['valid_tickers'])}** assets{bench_note} · "
        f"{data['start']} → {data['end']} ({len(data['prices'])} trading days)."
    )
    if benchmark and not data.get("benchmark_ticker"):
        st.warning(f"Benchmark **{benchmark}** could not be loaded from this data "
                   "source; benchmark marker and Black-Litterman proxy disabled.")
    if source == "Sample data (offline)":
        st.warning(
            "⚠️ Using **synthetic** prices for demonstration — not real market data.",
            icon="⚠️",
        )

    # ---- Run optimisation ----
    with st.spinner("Computing efficient frontier…"):
        result = _run_analysis(
            data["prices"],
            params["risk_free_rate"],
            params["cov_method"],
            params["n_simulations"],
            params["max_weight"],
            params["returns_model"],
            data["benchmark_prices"],
            data["benchmark_ticker"],
        )
        current = build_current_portfolio(
            holdings, data["latest_prices"], result.mu, result.cov,
            params["risk_free_rate"],
        )

    render_results(result, current, data["latest_prices"], params)


if __name__ == "__main__":
    main()
