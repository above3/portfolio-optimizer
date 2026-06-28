"""Portfolio Optimizer — a Modern Portfolio Theory (Markowitz) web app.

Run with::

    streamlit run app.py

The app lets a user describe their current holdings, fetches historical prices
from Yahoo Finance, and computes the Minimum-Variance and Maximum-Sharpe
(tangency) portfolios along with the full efficient frontier and Capital Market
Line. Results are shown on an interactive Plotly chart plus reallocation tables.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src import config
from src.data_loader import PriceDataError, fetch_price_history
from src.optimizer import FrontierResult, PortfolioPoint, analyse
from src.portfolio import (
    CurrentPortfolio,
    build_current_portfolio,
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
    }


@st.cache_data(show_spinner=False)
def _load_prices_live(tickers: tuple[str, ...], lookback_years: int) -> dict:
    """Cached wrapper around the live Yahoo Finance download."""
    return _as_dict(fetch_price_history(list(tickers), lookback_years=lookback_years))


@st.cache_data(show_spinner=False)
def _load_prices_sample(tickers: tuple[str, ...], lookback_years: int) -> dict:
    """Cached wrapper around the offline synthetic-price generator."""
    return _as_dict(generate_sample_prices(list(tickers), lookback_years=lookback_years))


@st.cache_data(show_spinner=False)
def _run_analysis(
    prices: pd.DataFrame,
    risk_free_rate: float,
    cov_method: str,
    n_simulations: int,
) -> FrontierResult:
    return analyse(
        prices,
        risk_free_rate=risk_free_rate,
        cov_method=cov_method,
        n_simulations=n_simulations,
    )


# --------------------------------------------------------------------------- #
# Sidebar inputs
# --------------------------------------------------------------------------- #
def render_sidebar() -> dict:
    st.sidebar.header("⚙️ Parameters")

    data_source = st.sidebar.radio(
        "Data source",
        options=["Yahoo Finance (live)", "Sample data (offline)", "Upload CSV"],
        index=0,
        help=(
            "Live prices need outbound access to Yahoo Finance. If that is "
            "blocked or rate-limited, use the offline sample generator or "
            "upload your own CSV of close prices."
        ),
    )

    uploaded_csv = None
    if data_source == "Upload CSV":
        uploaded_csv = st.sidebar.file_uploader(
            "Close-price CSV",
            type=["csv"],
            help="First column = Date; one column of close prices per ticker.",
        )

    lookback = st.sidebar.selectbox(
        "Historical lookback (years)",
        options=config.LOOKBACK_CHOICES,
        index=config.LOOKBACK_CHOICES.index(config.DEFAULT_LOOKBACK_YEARS),
        help="Length of the daily price window used to estimate returns and risk.",
    )

    rf_pct = st.sidebar.number_input(
        "Risk-free rate Rf (%)",
        min_value=0.0,
        max_value=25.0,
        value=config.DEFAULT_RISK_FREE_RATE * 100,
        step=0.25,
        help="Annual risk-free rate. Anchors the Capital Market Line on the Y-axis.",
    )

    capital = st.sidebar.number_input(
        "Additional capital to invest ($)",
        min_value=0.0,
        value=0.0,
        step=1000.0,
        help="Optional new cash. Added to the current portfolio value when "
             "computing the suggested reallocation.",
    )

    cov_label = st.sidebar.selectbox(
        "Covariance estimator",
        options=list(config.COVARIANCE_METHODS.keys()),
        index=0,
        help="Ledoit-Wolf shrinkage is more stable for short histories.",
    )

    n_sim = st.sidebar.slider(
        "Simulated portfolios (Markowitz bullet)",
        min_value=1000,
        max_value=20000,
        value=config.DEFAULT_N_SIMULATIONS,
        step=1000,
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Educational tool. Historical performance does not guarantee future "
        "results. Long-only weights (no shorting/leverage)."
    )

    return {
        "data_source": data_source,
        "uploaded_csv": uploaded_csv,
        "lookback": int(lookback),
        "risk_free_rate": rf_pct / 100.0,
        "capital": float(capital),
        "cov_method": config.COVARIANCE_METHODS[cov_label],
        "n_simulations": int(n_sim),
    }


# --------------------------------------------------------------------------- #
# Holdings input
# --------------------------------------------------------------------------- #
def render_holdings_editor() -> pd.DataFrame:
    st.subheader("1 · Your portfolio")
    st.caption(
        "Enter the tickers you hold (or are considering) and the number of "
        "shares. Leave shares at 0 to include an asset in the optimisation "
        "without marking it as a current holding."
    )
    default = pd.DataFrame(config.DEMO_PORTFOLIO)
    edited = st.data_editor(
        default,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Ticker": st.column_config.TextColumn(
                "Ticker", help="Yahoo Finance symbol, e.g. AAPL, MSFT, BTC-USD."
            ),
            "Shares": st.column_config.NumberColumn(
                "Shares", help="Number of shares currently held.", min_value=0.0
            ),
        },
        key="holdings_editor",
    )
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
        # Sum duplicate rows for the same ticker.
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


def _metric_columns(
    result: FrontierResult, current: CurrentPortfolio | None
) -> None:
    cols = st.columns(3)

    with cols[0]:
        st.markdown("#### 🧍 Current portfolio")
        if current is None:
            st.info("No current holdings entered (drafting a new portfolio).")
        else:
            st.metric("Expected return", f"{current.expected_return * 100:.2f}%")
            st.metric("Volatility", f"{current.volatility * 100:.2f}%")
            st.metric("Sharpe ratio", f"{current.sharpe:.2f}")
            st.metric("Total value", f"${current.total_value:,.0f}")

    with cols[1]:
        st.markdown("#### 🚀 Max Sharpe (Tangency)")
        t = result.tangency
        if t is None:
            st.warning(
                "No long-only portfolio beats the risk-free rate at this Rf. "
                "Lower Rf or add higher-return assets."
            )
        else:
            st.metric(
                "Expected return",
                f"{t.expected_return * 100:.2f}%",
                delta=None if current is None
                else f"{(t.expected_return - current.expected_return) * 100:.2f}%",
            )
            st.metric(
                "Volatility",
                f"{t.volatility * 100:.2f}%",
                delta=None if current is None
                else f"{(t.volatility - current.volatility) * 100:.2f}%",
                delta_color="inverse",
            )
            st.metric(
                "Sharpe ratio",
                f"{t.sharpe:.2f}",
                delta=None if current is None
                else f"{t.sharpe - current.sharpe:.2f}",
            )

    with cols[2]:
        st.markdown("#### 🛡️ Minimum variance")
        m = result.mvp
        st.metric("Expected return", f"{m.expected_return * 100:.2f}%")
        st.metric("Volatility", f"{m.volatility * 100:.2f}%")
        st.metric("Sharpe ratio", f"{m.sharpe:.2f}")


def render_results(
    result: FrontierResult,
    current: CurrentPortfolio | None,
    latest_prices: pd.Series,
    capital: float,
) -> None:
    st.subheader("3 · The Markowitz bullet")
    st.plotly_chart(
        efficient_frontier_figure(result, current),
        use_container_width=True,
    )

    st.subheader("4 · Portfolio comparison")
    _metric_columns(result, current)

    st.markdown("---")
    st.subheader("5 · Suggested allocations & reallocation")

    # Investable capital: current value (if any) plus new cash.
    base_value = current.total_value if current is not None else max(capital, 0.0)
    investable = base_value + (capital if current is not None else 0.0)
    if investable <= 0:
        investable = 10000.0  # nominal base so dollar targets are still shown

    tab_tan, tab_mvp = st.tabs(["🚀 Max Sharpe (Tangency)", "🛡️ Minimum variance"])

    current_weights = current.weights if current is not None else {}

    with tab_tan:
        if result.tangency is None:
            st.warning("Tangency portfolio unavailable at the current risk-free rate.")
        else:
            left, right = st.columns([1, 1.4])
            with left:
                st.markdown("**Target weights**")
                st.dataframe(_weights_table(result.tangency),
                            hide_index=True, use_container_width=True)
            with right:
                st.markdown(f"**Reallocation plan** (investable ≈ ${investable:,.0f})")
                st.dataframe(
                    reallocation_plan(current_weights, result.tangency.weights,
                                      investable, latest_prices),
                    hide_index=True, use_container_width=True,
                )

    with tab_mvp:
        left, right = st.columns([1, 1.4])
        with left:
            st.markdown("**Target weights**")
            st.dataframe(_weights_table(result.mvp),
                        hide_index=True, use_container_width=True)
        with right:
            st.markdown(f"**Reallocation plan** (investable ≈ ${investable:,.0f})")
            st.dataframe(
                reallocation_plan(current_weights, result.mvp.weights,
                                  investable, latest_prices),
                hide_index=True, use_container_width=True,
            )

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
    st.title("📈 Portfolio Optimizer — Modern Portfolio Theory")
    st.markdown(
        "Build the **efficient frontier**, find the **Minimum-Variance** and "
        "**Maximum-Sharpe (tangency)** portfolios, and see how your current "
        "allocation compares — powered by `yfinance` + `PyPortfolioOpt`."
    )

    params = render_sidebar()
    edited = render_holdings_editor()

    run = st.button("🔍 Optimize portfolio", type="primary", use_container_width=True)
    if not run:
        st.info("Edit your holdings above, set parameters in the sidebar, then "
                "click **Optimize portfolio**.")
        return

    holdings = _parse_holdings(edited)
    if len(holdings) < 2:
        st.error("Please enter at least **two** distinct tickers to optimise a portfolio.")
        return

    # ---- Fetch prices (live / offline sample / CSV) ----
    tickers = tuple(sorted(holdings))
    source = params["data_source"]
    spinner_msg = {
        "Yahoo Finance (live)": "Fetching historical prices from Yahoo Finance…",
        "Sample data (offline)": "Generating offline sample price history…",
        "Upload CSV": "Reading uploaded price history…",
    }[source]

    with st.spinner(spinner_msg):
        try:
            if source == "Yahoo Finance (live)":
                data = _load_prices_live(tickers, params["lookback"])
            elif source == "Sample data (offline)":
                data = _load_prices_sample(tickers, params["lookback"])
            else:  # Upload CSV
                if params["uploaded_csv"] is None:
                    st.error("Please upload a CSV file in the sidebar, or pick "
                             "another data source.")
                    return
                data = _as_dict(
                    load_prices_from_csv(params["uploaded_csv"], tickers=list(tickers))
                )
        except PriceDataError as exc:
            st.error(f"Data error: {exc}")
            if source == "Yahoo Finance (live)":
                st.info("If Yahoo Finance is blocked or rate-limited on your "
                        "network, switch **Data source** to *Sample data "
                        "(offline)* in the sidebar to try the app instantly.")
            return
        except Exception as exc:  # noqa: BLE001 - surface network/library errors
            st.error(f"Unexpected error while loading prices: {exc}")
            return

    if data["failed_tickers"]:
        st.warning("No data for: " + ", ".join(data["failed_tickers"]) +
                   ". These were excluded.")
    if len(data["valid_tickers"]) < 2:
        st.error("Fewer than two tickers returned usable data. Cannot optimise.")
        return

    st.subheader("2 · Data")
    st.caption(
        f"Source: **{source}** · "
        f"Loaded **{len(data['valid_tickers'])}** assets · "
        f"{data['start']} → {data['end']} "
        f"({len(data['prices'])} trading days)."
    )
    if source == "Sample data (offline)":
        st.warning(
            "⚠️ Using **synthetic** prices for demonstration — these are not real "
            "market data. Switch to *Yahoo Finance (live)* for actual prices.",
            icon="⚠️",
        )

    # ---- Run optimisation ----
    with st.spinner("Computing efficient frontier…"):
        result = _run_analysis(
            data["prices"],
            params["risk_free_rate"],
            params["cov_method"],
            params["n_simulations"],
        )
        current = build_current_portfolio(
            holdings,
            data["latest_prices"],
            result.mu,
            result.cov,
            params["risk_free_rate"],
        )

    render_results(result, current, data["latest_prices"], params["capital"])


if __name__ == "__main__":
    main()
