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
from src.data_loader import (
    PriceDataError,
    apply_regime,
    fetch_price_history,
    fetch_risk_free_rate,
)
from src.data_loader import iso_timestamp
from src.optimizer import (
    AnchorConstraint,
    FrontierResult,
    PortfolioPoint,
    WatchlistRecommendation,
    analyse,
    backtest_equity_curves,
    buy_only_max_sharpe,
    required_capital_to_align,
    screen_watchlist,
    simulate_gbm_wealth,
    weight_tracking_error,
)
from src.portfolio import (
    CurrentPortfolio,
    NoiseReport,
    PortfolioStore,
    basket_trader_csv,
    build_current_portfolio,
    build_executive_report_html,
    classify_noise,
    compute_pac_metrics,
    deployment_plan,
    hierarchical_clusters,
    merge_holdings,
    normalise_holdings_df,
    parse_ibkr_positions,
    portfolio_config_csv,
    positions_to_holdings_df,
    reallocation_plan,
)
from src.providers import DEFAULT_IBKR_CLIENT_ID, DEFAULT_IBKR_HOST, IBKRNativeProvider, ProviderError
from src.sample_data import generate_sample_prices, load_prices_from_csv
from src.visualization import (
    backtest_figure,
    correlation_heatmap,
    custodian_donut,
    dendrogram_figure,
    efficient_frontier_figure,
    gbm_fan_chart,
)

st.set_page_config(page_title="Portfolio Optimizer · MPT", page_icon="📈", layout="wide")


# --------------------------------------------------------------------------- #
# Premium institutional dark-mode theme (CSS injection)
# --------------------------------------------------------------------------- #
_DARK_CSS = """
<style>
:root { --pc-bg:#0e1117; --pc-card:#1a1d24; --pc-border:#2a2e39;
        --pc-text:#e6e9ef; --pc-muted:#8b95a5; --pc-green:#2ecc71; --pc-red:#e74c3c; }
.stApp { background-color: var(--pc-bg); color: var(--pc-text); }
section[data-testid="stSidebar"] { background-color: #14171d; border-right: 1px solid var(--pc-border); }
h1, h2, h3 { color: var(--pc-text); letter-spacing: .2px; }
/* Native metric cards */
div[data-testid="stMetric"] {
    background: var(--pc-card); border: 1px solid var(--pc-border);
    border-radius: 12px; padding: 14px 16px; }
div[data-testid="stMetricValue"] { font-weight: 700; }
/* Custom financial cards */
.pc-card { background: var(--pc-card); border: 1px solid var(--pc-border);
    border-radius: 14px; padding: 16px 18px; margin-bottom: 10px; }
.pc-card .pc-label { color: var(--pc-muted); font-size: .72rem; letter-spacing: .09em;
    text-transform: uppercase; margin-bottom: 6px; }
.pc-card .pc-value { font-size: 1.55rem; font-weight: 700; color: var(--pc-text); line-height: 1.15; }
.pc-card .pc-sub { font-size: .82rem; margin-top: 4px; color: var(--pc-muted); }
.pc-card.pc-pos { border-left: 4px solid var(--pc-green); box-shadow: 0 0 18px rgba(46,204,113,.10); }
.pc-card.pc-pos .pc-value { color: var(--pc-green); }
.pc-card.pc-neg { border-left: 4px solid var(--pc-red); box-shadow: 0 0 18px rgba(231,76,60,.10); }
.pc-card.pc-neg .pc-value { color: var(--pc-red); }
.stButton>button, .stDownloadButton>button { border-radius: 10px; border: 1px solid var(--pc-border); font-weight: 600; }
.stTabs [data-baseweb="tab-list"] { gap: 4px; }
.stTabs [data-baseweb="tab"] { background: var(--pc-card); border-radius: 8px 8px 0 0; }
/* Tranche drift efficiency bar */
.pc-drift-track { background:#11141b; border:1px solid var(--pc-border); border-radius:999px;
    height:22px; width:100%; overflow:hidden; margin:6px 0; }
.pc-drift-fill { height:100%; border-radius:999px; text-align:right; color:#08130c;
    font-weight:700; font-size:.74rem; line-height:22px; padding-right:10px;
    background:linear-gradient(90deg,#27ae60,#2ecc71); }
.pc-drift-low .pc-drift-fill { background:linear-gradient(90deg,#c0392b,#e74c3c); color:#fff; }
</style>
"""


def _inject_theme() -> None:
    st.markdown(_DARK_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Formatting helpers & custom metric cards
# --------------------------------------------------------------------------- #
def _sym(currency: str | None) -> str:
    return config.CURRENCY_SYMBOLS.get((currency or "USD").upper(), "$")


def _money(value: float, currency: str | None) -> str:
    return f"{_sym(currency)}{value:,.0f}"


def _card(label: str, value: str, sub: str = "", tone: str = "neutral") -> None:
    """Render a styled financial metric card (tone: positive/negative/neutral)."""
    cls = {"positive": "pc-pos", "negative": "pc-neg"}.get(tone, "")
    sub_html = f'<div class="pc-sub">{sub}</div>' if sub else ""
    st.markdown(
        f'<div class="pc-card {cls}"><div class="pc-label">{label}</div>'
        f'<div class="pc-value">{value}</div>{sub_html}</div>',
        unsafe_allow_html=True)


def _tone_for(value: float) -> str:
    return "positive" if value >= 0 else "negative"


# --------------------------------------------------------------------------- #
# Optional password gate (Streamlit secrets)
# --------------------------------------------------------------------------- #
def _check_password() -> bool:
    """Optional login gate. Enabled only when ``APP_PASSWORD`` is set in
    ``.streamlit/secrets.toml`` (or the Community Cloud secrets manager)."""
    try:
        expected = st.secrets.get("APP_PASSWORD", None)
    except Exception:  # noqa: BLE001 - no secrets file present
        expected = None
    if not expected:
        return True  # authentication disabled
    if st.session_state.get("_authenticated"):
        return True
    st.markdown("### 🔒 Protected app")
    pwd = st.text_input("Password", type="password", key="_pwd_input")
    if pwd and pwd == expected:
        st.session_state["_authenticated"] = True
        return True
    if pwd:
        st.error("Incorrect password.")
    return False


# --------------------------------------------------------------------------- #
# Local persistence: store accessor + settings seeding
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def _get_store() -> PortfolioStore:
    return PortfolioStore()


_SETTING_DEFAULTS: dict[str, object] = {
    "pc_architecture": config.ARCH_YAHOO,
    "pc_ibkr_port": next(iter(config.IBKR_PORT_CHOICES)),
    "pc_currency": config.DEFAULT_BASE_CURRENCY,
    "pc_data_source": "Yahoo Finance (live)",
    "pc_rf_live": True,
    "pc_rf_manual": config.DEFAULT_RISK_FREE_RATE * 100,
    "pc_benchmark": config.DEFAULT_BENCHMARK,
    "pc_lookback": config.DEFAULT_LOOKBACK_YEARS,
    "pc_returns": next(iter(config.RETURNS_MODELS)),
    "pc_cov": next(iter(config.COVARIANCE_METHODS)),
    "pc_anchor_pct": config.DEFAULT_ANCHOR_PCT,
    "pc_anchor_exact": True,
    "pc_max_weight": int(config.DEFAULT_MAX_WEIGHT * 100),
    "pc_sector_max": config.DEFAULT_SECTOR_MAX_PCT,
    "pc_tolerance": config.DEFAULT_TOLERANCE_PCT,
    "pc_disable_fractional": False,
    "pc_transaction_cost": config.DEFAULT_TRANSACTION_COST,
    "pc_capital": 0.0,
    "pc_deployment": config.DEPLOYMENT_STRATEGIES[0],
    "pc_tranches": config.DEFAULT_TRANCHES,
    "pc_buy_only": False,
    "pc_horizon": config.DEFAULT_HORIZON_YEARS,
    "pc_nsim": config.DEFAULT_N_SIMULATIONS,
    "pc_watchlist": config.DEFAULT_WATCHLIST,
    "pc_regime": config.MACRO_REGIMES[0],
}
_SETTING_OPTIONS: dict[str, tuple] = {
    "pc_architecture": tuple(config.DATA_ARCHITECTURES),
    "pc_ibkr_port": tuple(config.IBKR_PORT_CHOICES),
    "pc_currency": tuple(config.BASE_CURRENCIES),
    "pc_data_source": ("Yahoo Finance (live)", "Sample data (offline)", "Upload CSV"),
    "pc_lookback": tuple(config.LOOKBACK_CHOICES),
    "pc_returns": tuple(config.RETURNS_MODELS),
    "pc_cov": tuple(config.COVARIANCE_METHODS),
    "pc_deployment": tuple(config.DEPLOYMENT_STRATEGIES),
    "pc_regime": tuple(config.MACRO_REGIMES),
}


def _seed_settings(store: PortfolioStore) -> None:
    """Seed widget session_state from the persisted settings (or defaults)."""
    saved = store.load_settings()
    saved = saved if isinstance(saved, dict) else {}
    for key, default in _SETTING_DEFAULTS.items():
        if key in st.session_state:
            continue
        value = saved.get(key, default)
        options = _SETTING_OPTIONS.get(key)
        if options is not None and value not in options:
            value = default
        st.session_state[key] = value


def _persist_state(store: PortfolioStore, holdings: pd.DataFrame) -> None:
    """Auto-commit holdings + settings to the local DB (best-effort)."""
    try:
        store.save_holdings(holdings)
        store.save_settings({k: st.session_state.get(k) for k in _SETTING_DEFAULTS})
    except Exception:  # noqa: BLE001 - persistence must never break the UI
        pass


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


@st.cache_resource(show_spinner=False)
def _connect_ibkr(host: str, port: int, client_id: int) -> IBKRNativeProvider:
    """Cached live IBKR connection. Raises ProviderError if TWS/Gateway is down
    or ib_insync is unavailable (cache_resource does not cache the exception, so
    each rerun cleanly retries)."""
    provider = IBKRNativeProvider(host=host, port=port, client_id=client_id)
    provider.connect()
    return provider


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
    store = _get_store()
    if "holdings_df" not in st.session_state:
        loaded = store.load_holdings()
        st.session_state.holdings_df = (
            loaded if loaded is not None and not loaded.empty
            else normalise_holdings_df(pd.DataFrame(config.DEMO_PORTFOLIO)))
    st.session_state.setdefault("editor_version", 0)
    st.session_state.setdefault("ibkr_file_id", None)
    st.session_state.setdefault("ibkr_ingest_id", None)
    st.session_state.setdefault("auto_run", False)
    st.session_state.setdefault("has_run", False)
    _seed_settings(store)


def _ingest_ibkr_positions(snapshot) -> None:
    """Hybrid append-merge of live IBKR positions into the editable table.

    Live rows are injected on top while existing manual rows are preserved
    (never locked or discarded). Runs once per (account, position-count) so the
    user can keep editing freely without being overwritten every rerun.
    """
    guard = f"{snapshot.account_id}:{len(snapshot.positions)}"
    if st.session_state.get("ibkr_ingest_id") == guard:
        return
    live = positions_to_holdings_df(snapshot.positions)
    if live.empty:
        return
    st.session_state.holdings_df = merge_holdings(live, st.session_state.holdings_df)
    st.session_state.ibkr_ingest_id = guard
    st.session_state.editor_version += 1
    st.session_state.auto_run = True


def _freeze_live_snapshot(provider, snapshot) -> None:
    """Thread-safely freeze the live account to the local DB and go offline."""
    store = _get_store()
    payload = {
        "account_id": snapshot.account_id,
        "base_currency": snapshot.base_currency,
        "cash": snapshot.cash,
        "timestamp": iso_timestamp(),
        "positions": [
            {"symbol": p.symbol, "shares": p.shares, "avg_cost": p.avg_cost,
             "currency": p.currency, "sec_type": p.sec_type, "asset_type": p.asset_type}
            for p in snapshot.positions],
    }
    store.save_frozen_snapshot(payload)
    try:
        provider.disconnect()  # terminate the ib_insync loop
    except Exception:  # noqa: BLE001
        pass
    _connect_ibkr.clear()
    st.session_state["force_offline"] = True


def _load_frozen_into_table(snapshot: dict) -> None:
    """Merge a frozen snapshot's positions into the editable table (once)."""
    guard = f"frozen:{snapshot.get('timestamp')}"
    if st.session_state.get("ibkr_ingest_id") == guard:
        return
    live = positions_to_holdings_df(snapshot.get("positions", []))
    if not live.empty:
        st.session_state.holdings_df = merge_holdings(live, st.session_state.holdings_df)
    st.session_state.ibkr_ingest_id = guard
    st.session_state.editor_version += 1


def _drift_efficiency_bar(efficiency: float, required_cash: float,
                          alignable: bool, ccy: str) -> None:
    """Custom CSS progress bar for the cash-inflow drift efficiency monitor."""
    pct = max(0.0, min(efficiency, 1.0)) * 100.0
    low = " pc-drift-low" if pct < 60 else ""
    st.markdown("**🎯 Tranche Drift Efficiency**")
    st.markdown(
        f'<div class="pc-drift-track{low}"><div class="pc-drift-fill" '
        f'style="width:{pct:.1f}%">{pct:.0f}%</div></div>', unsafe_allow_html=True)
    if pct >= 99.5 and alignable:
        st.caption("✅ This tranche fully aligns the portfolio to the Max-Sharpe node "
                   "with no sell liquidation.")
    elif not alignable:
        st.caption(f"Current tranche efficiency is **{pct:.0f}%**. Some held assets sit "
                   "above their target weight — perfect alignment would require a "
                   "sale (blocked in buy-only mode).")
    else:
        st.caption(f"Current tranche efficiency is **{pct:.0f}%**. To perfectly align to "
                   f"the Max Sharpe node without triggering sell-liquidation taxes, your "
                   f"required tranche capital injection is **{_money(required_cash, ccy)}**.")


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def _risk_free_controls() -> float:
    fetch_live = st.sidebar.toggle(
        f"Fetch Live Risk-Free Rate ({config.RISK_FREE_TICKER})", key="pc_rf_live",
        help="Use the US 3-Month T-Bill yield as Rf for the tangency portfolio.")
    if fetch_live:
        try:
            rf = _fetch_live_rf()
            st.sidebar.success(f"Live Rf ({config.RISK_FREE_TICKER}): **{rf * 100:.2f}%**")
            return rf
        except Exception as exc:  # noqa: BLE001 - blocked network/parse error
            st.sidebar.warning(f"Live rate unavailable ({type(exc).__name__}). Enter manually.")
    rf_pct = st.sidebar.number_input("Risk-free rate Rf (%)", 0.0, 25.0, step=0.25,
                                     key="pc_rf_manual")
    return rf_pct / 100.0


def _ibkr_connection_panel() -> tuple[bool, int]:
    """Render the IBKR socket selector + live connection status.

    Returns ``(ibkr_active, port)``. On failure (or a frozen/offline state) it
    cleanly falls back to safe layout mode (``ibkr_active=False``) while keeping
    the app fully operational on the last frozen snapshot.
    """
    store = _get_store()
    port_label = st.sidebar.selectbox("Native socket", list(config.IBKR_PORT_CHOICES),
                                      key="pc_ibkr_port")
    port = config.IBKR_PORT_CHOICES[port_label]
    st.session_state.pop("_ibkr_provider", None)

    # ---- Frozen / offline mode (after a "Freeze Live Snapshot") ----
    if st.session_state.get("force_offline"):
        snap = store.load_frozen_snapshot()
        if snap:
            _load_frozen_into_table(snap)
            st.session_state["frozen_banner"] = snap.get("timestamp")
            st.sidebar.warning(f"❄️ Frozen snapshot loaded · {snap.get('timestamp')}")
            if st.sidebar.button("🔌 Reconnect Live", use_container_width=True):
                st.session_state["force_offline"] = False
                _connect_ibkr.clear()
                st.rerun()
            return False, port
        st.session_state["force_offline"] = False  # nothing frozen -> normal flow

    with st.spinner("Connecting to Local TWS/Gateway Socket…"):
        try:
            provider = _connect_ibkr(DEFAULT_IBKR_HOST, port, DEFAULT_IBKR_CLIENT_ID)
            snapshot = provider.fetch_account_snapshot()
        except ProviderError as exc:
            st.sidebar.info(f"ℹ️ TWS not detected on {DEFAULT_IBKR_HOST}:{port}. "
                            "Falling back to safe layout mode.")
            st.sidebar.caption(f"({exc})")
            frozen = store.load_frozen_snapshot()
            if frozen:
                _load_frozen_into_table(frozen)
                st.session_state["frozen_banner"] = frozen.get("timestamp")
                st.sidebar.warning(f"❄️ Loaded last frozen snapshot · {frozen.get('timestamp')}")
            return False, port

    st.session_state["_ibkr_provider"] = provider
    st.sidebar.success(f"🟢 Connected to Account: **{snapshot.account_id}** · "
                       f"cash {_money(snapshot.cash, st.session_state.get('pc_currency'))}")
    _ingest_ibkr_positions(snapshot)
    if st.sidebar.button("🧊 Freeze Live Snapshot & Go Offline", use_container_width=True):
        _freeze_live_snapshot(provider, snapshot)
        st.rerun()
    return True, port


def render_sidebar() -> dict:
    st.sidebar.header("⚙️ Parameters")
    st.sidebar.caption("💾 Settings & holdings auto-save to a local database.")
    st.session_state["frozen_banner"] = None  # recomputed by the IBKR panel

    st.sidebar.markdown("### 🛰️ Active Data Architecture Source")
    architecture = st.sidebar.radio("Engine", config.DATA_ARCHITECTURES,
                                    key="pc_architecture", label_visibility="collapsed")
    ibkr_active, ibkr_port = False, config.IBKR_PORT_CHOICES[next(iter(config.IBKR_PORT_CHOICES))]
    if architecture == config.ARCH_IBKR:
        ibkr_active, ibkr_port = _ibkr_connection_panel()

    base_currency = st.sidebar.selectbox(
        "Base Display Currency", options=config.BASE_CURRENCIES, key="pc_currency",
        help="All prices, values and P/L are normalised into this currency via "
             "live FX rates (graceful fallback to native currency on failure).")

    uploaded_csv = None
    if ibkr_active:
        data_source = "IBKR Live API"
        st.sidebar.caption("📡 Prices stream from IBKR (per-ticker Yahoo fallback).")
    else:
        data_source = st.sidebar.radio(
            "Price source", ["Yahoo Finance (live)", "Sample data (offline)", "Upload CSV"],
            key="pc_data_source", help="Live needs Yahoo access; otherwise use offline/CSV.")
        if data_source == "Upload CSV":
            uploaded_csv = st.sidebar.file_uploader("Close-price CSV", type=["csv"])

    st.sidebar.markdown("### 💵 Risk-free rate")
    risk_free_rate = _risk_free_controls()

    st.sidebar.markdown("### 📊 Benchmark & lookback")
    benchmark = st.sidebar.text_input("Market Benchmark Ticker",
                                      key="pc_benchmark").strip().upper() or None
    lookback = st.sidebar.selectbox("Historical lookback (years)", config.LOOKBACK_CHOICES,
                                    key="pc_lookback")

    st.sidebar.markdown("### 🧮 Returns & risk model")
    returns_label = st.sidebar.radio("Expected-returns model",
                                     list(config.RETURNS_MODELS.keys()), key="pc_returns")
    cov_label = st.sidebar.selectbox("Covariance estimator",
                                     list(config.COVARIANCE_METHODS.keys()), key="pc_cov")

    st.sidebar.markdown("### 🌍 Core global anchor")
    anchor_pct = st.sidebar.slider(
        f"Core Global Anchor Allocation ({config.ANCHOR_TICKER}) %",
        0, config.ANCHOR_MAX_PCT, step=1, key="pc_anchor_pct")
    anchor_exact = st.sidebar.toggle("Strict Target Match (else Upper Cap Bound)",
                                     key="pc_anchor_exact")

    st.sidebar.markdown("### 🛡️ Constraints & guardrails")
    max_weight_pct = st.sidebar.slider("Max Allocation per Asset (%)", 5, 100, step=5,
                                       key="pc_max_weight")
    sector_max_pct = st.sidebar.slider("Max Sector Exposure (%)", config.SECTOR_MIN_PCT,
                                       config.SECTOR_MAX_PCT, step=5, key="pc_sector_max")
    tolerance_pct = st.sidebar.slider("Rebalancing Tolerance Threshold (%)",
                                      config.TOLERANCE_MIN_PCT, config.TOLERANCE_MAX_PCT,
                                      step=1, key="pc_tolerance")

    st.sidebar.markdown("### 💸 Friction")
    disable_fractional = st.sidebar.checkbox("Disable Fractional Shares",
                                             key="pc_disable_fractional")
    transaction_cost = st.sidebar.number_input(
        f"Estimated Transaction Cost per Trade ({_sym(base_currency)})",
        min_value=0.0, step=0.5, key="pc_transaction_cost")

    st.sidebar.markdown("### 💰 Capital deployment")
    capital = st.sidebar.number_input(
        f"Additional capital to invest ({_sym(base_currency)})",
        min_value=0.0, step=1000.0, key="pc_capital")
    deployment_strategy = st.sidebar.selectbox("Capital Deployment Strategy",
                                               config.DEPLOYMENT_STRATEGIES, key="pc_deployment")
    if "Tranche" in deployment_strategy or "DCA" in deployment_strategy:
        tranches = st.sidebar.slider("Number of Active Tranches (Periods)",
                                     config.TRANCHE_MIN, config.TRANCHE_MAX, step=1,
                                     key="pc_tranches")
    else:
        tranches = int(st.session_state.get("pc_tranches", config.DEFAULT_TRANCHES))
    buy_only = st.sidebar.toggle(
        "Optimize Using New Capital Only (No Sells)", key="pc_buy_only",
        help="Deploy only the new tranche cash to drift toward the Max-Sharpe "
             "target without selling — eliminates capital-gains tax and sell fees.")

    st.sidebar.markdown("### 🎲 Projection & simulation")
    horizon_years = st.sidebar.slider("Investment Horizon (Years)",
                                      config.HORIZON_MIN_YEARS, config.HORIZON_MAX_YEARS,
                                      step=1, key="pc_horizon")
    n_sim = st.sidebar.slider("Simulated portfolios (Markowitz bullet)",
                              1000, 20000, step=1000, key="pc_nsim")

    st.sidebar.markdown("---")
    st.sidebar.caption("Educational tool. Past performance does not guarantee future "
                       "results. Long-only weights; simulated Italian taxes.")

    return {
        "architecture": architecture, "ibkr_active": ibkr_active, "ibkr_port": ibkr_port,
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
        "buy_only": bool(buy_only),
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
        key="pc_watchlist")
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
    with c1:
        _card("Portfolio value",
              _money(current.total_value, symbol_ccy) if current else "—")
    if current is None or not current.has_cost_data:
        with c2:
            _card("Unrealised P/L", "n/a", "Add purchase prices to enable")
        with c3:
            _card("Total return", "n/a")
        return
    pnl_pct = current.total_pnl / current.total_cost * 100 if current.total_cost > 0 else 0.0
    tone = _tone_for(current.total_pnl)
    with c2:
        _card("Unrealised P/L", f"{sym}{current.total_pnl:,.0f}",
              f"{pnl_pct:+.1f}% vs cost basis", tone)
    with c3:
        _card("Invested cost basis", _money(current.total_cost, symbol_ccy))


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
    with c1:
        _card("Total Capital Wealth Pool", _money(deployment.total_wealth, ccy),
              "Portfolio value + all additional cash")
    with c2:
        _card("Deployed in Current Tranche", _money(deployment.active_capital, ccy),
              f"{deployment.tranches} tranche(s)")
    with c3:
        tone = "positive" if deployment.sidelined_cash > 0 else "neutral"
        _card("Sidelined Cash Reserve", _money(deployment.sidelined_cash, ccy),
              "Buffered for future DCA steps", tone)


def _render_reallocation(target: PortfolioPoint, current, deployment, latest_prices,
                         mu, params, asset_types, sectors, ccy, key_prefix: str,
                         buy_only: bool = False, reference_weights: dict | None = None) -> None:
    sym = _sym(ccy)
    if buy_only:
        st.success("🛒 **Buy-Only mode** — deploying new cash without selling "
                   "(zero capital-gains tax, zero sell-side fees).")
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
            asset_types=asset_types, buy_only=buy_only, smart_rounding=True)
        st.dataframe(res.table, hide_index=True, use_container_width=True)
        if not params["allow_fractional"]:
            st.caption(f"🧩 Smart knapsack rounding · residual cash "
                       f"{sym}{res.residual_cash:,.2f} · integer tracking error "
                       f"{res.tracking_error:.4f}.")

        # ---- Cash-inflow drift efficiency monitor (buy-only) ----
        if buy_only and reference_weights:
            current_vals = current.values if current is not None else {}
            current_w = current.weights if current is not None else {}
            te_target = weight_tracking_error(res.achieved_weights, reference_weights)
            te_base = weight_tracking_error(current_w, reference_weights)
            efficiency = 1.0 if te_base <= 1e-9 else max(0.0, (te_base - te_target) / te_base)
            required_cash, alignable = required_capital_to_align(current_vals, reference_weights)
            _drift_efficiency_bar(efficiency, required_cash, alignable, ccy)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Trades", res.num_trades)
        m2.metric("Fees", f"{sym}{res.total_fees:,.2f}")
        m3.metric("Cap-gains tax", f"{sym}{res.total_tax:,.2f}")
        m4.metric("Residual cash", f"{sym}{res.residual_cash:,.2f}")

        if res.gross_return == res.gross_return:  # not NaN
            g1, g2 = st.columns(2)
            with g1:
                _card("Gross Optimized Return", f"{res.gross_return * 100:.2f}%",
                      "Before friction", _tone_for(res.gross_return))
            with g2:
                friction_pct = (res.net_return - res.gross_return) * 100
                _card("Net-of-Tax-and-Fees Return", f"{res.net_return * 100:.2f}%",
                      f"{friction_pct:.2f}% friction drag", _tone_for(res.net_return))

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

        # ---- Live IBKR execution staging (placeholder for future routing) ----
        if params.get("ibkr_active"):
            st.checkbox("Stage Orders Directly to TWS Monitor", value=False, disabled=True,
                        key=f"stage_tws_{key_prefix}",
                        help="🔒 Reserved for live automated execution. Connected to "
                             "TWS — direct order staging will be enabled in a future release.")


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
                     benchmark_ticker, risk_free_rate, sectors) -> None:
    st.subheader("Historical Backtest Analysis")
    regime = st.selectbox("Historical Macro Regime Backtest Simulation",
                          config.MACRO_REGIMES, key="pc_regime",
                          help="Stress-test current vs optimised allocations against "
                               "synthetic shock regimes scaled from historical proxies.")
    weight_sets: dict[str, dict] = {}
    if current is not None:
        weight_sets["Current Portfolio"] = current.weights
    if result.tangency is not None:
        weight_sets["Optimized Max Sharpe"] = result.tangency.weights
    if not weight_sets:
        st.info("Need a current and/or optimised portfolio to backtest.")
        return

    # Apply the regime shock (graceful fallback to live lookback on failure).
    panel = prices[base_pool]
    bt_prices = apply_regime(panel, sectors, regime)
    bench = benchmark_prices
    if benchmark_prices is not None and benchmark_ticker:
        bench_sectors = {benchmark_ticker: sectors.get(benchmark_ticker, "Diversified ETF")}
        bench = apply_regime(benchmark_prices.to_frame(name=benchmark_ticker),
                             bench_sectors, regime)[benchmark_ticker]
    if not regime.startswith("None"):
        if bt_prices is panel:
            st.warning("⚠️ Regime transform failed (singular/degenerate) — "
                       "reverted to the live lookback.")
        else:
            st.info(f"🌐 Simulating **{regime}** — synthetic shock scaled from "
                    "historical proxy covariance & drift shifts.")

    bt = backtest_equity_curves(bt_prices, weight_sets, risk_free_rate,
                                benchmark_prices=bench, benchmark_name=benchmark_ticker)
    st.plotly_chart(backtest_figure(bt), use_container_width=True)
    st.markdown("**Diagnostics** (frozen weights over the window)")
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


def _record_tranche_controls(current, deployment, ccy: str) -> None:
    """Button to log the active deployment into the PAC tranche ledger."""
    store = _get_store()
    pv = current.total_value if current is not None else 0.0
    c1, c2 = st.columns([1, 2])
    if c1.button("📌 Record this tranche deployment", use_container_width=True,
                 help="Logs the active capital + portfolio value to the PAC ledger "
                      "for TWR/MWR tracking over time."):
        ok = store.record_tranche(deployment.active_capital, deployment.strategy, pv)
        if ok:
            st.success(f"Recorded tranche of {_money(deployment.active_capital, ccy)} "
                       f"at portfolio value {_money(pv, ccy)}.")
        else:
            st.error("Could not write to the local ledger.")
    c2.caption("Build a Piano di Accumulo (PAC) history to evaluate TWR vs MWR over time.")


def _render_pac_ledger(current, ccy: str) -> None:
    st.subheader("PAC Performance Ledger — TWR vs MWR")
    store = _get_store()
    history = store.load_tranche_history()
    if history is None or history.empty:
        st.info("No tranche deployments recorded yet. Use **📌 Record this tranche "
                "deployment** under *Capital deployment & reallocation* to build the "
                "savings-plan history, then revisit this tab.")
        return

    current_value = current.total_value if current is not None else 0.0
    metrics = compute_pac_metrics(history, current_value)
    if metrics is None:
        st.warning("Not enough valid ledger data (or zero current value) to compute "
                   "TWR / MWR yet.")
        st.dataframe(history, hide_index=True, use_container_width=True)
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        _card("Time-Weighted Return (TWR, ann.)", f"{metrics.twr_annual * 100:.2f}%",
              "Allocation-engine quality (cash-timing neutral)", _tone_for(metrics.twr_annual))
    with c2:
        mwr_txt = "n/a" if metrics.mwr_annual != metrics.mwr_annual else f"{metrics.mwr_annual * 100:.2f}%"
        _card("Money-Weighted Return (MWR / IRR)", mwr_txt,
              "Your actual €/$ compounding velocity",
              _tone_for(metrics.mwr_annual if metrics.mwr_annual == metrics.mwr_annual else 0))
    with c3:
        _card("Total Gain", _money(metrics.total_gain, ccy),
              f"on {_money(metrics.total_contributed, ccy)} contributed over "
              f"{metrics.period_years:.1f}y", _tone_for(metrics.total_gain))

    st.markdown("**Capital-inflow history**")
    st.dataframe(history, hide_index=True, use_container_width=True)
    if st.button("🗑️ Clear PAC ledger"):
        store.clear_tranche_history()
        st.rerun()


def _render_executive_report(result, current, deployment, latest_prices, params,
                             asset_types) -> None:
    ccy = params["base_currency"]
    with st.expander("🧾 Executive Investment Report"):
        st.caption("Compile a printable, dark-mode HTML report (P&L, efficient "
                   "frontier, custodian donut, IBKR basket matrix).")
        if st.button("Generate Executive Investment Report", type="secondary"):
            try:
                ef_html = efficient_frontier_figure(result, current).to_html(
                    full_html=False, include_plotlyjs="cdn")
                donut_html = custodian_donut(
                    current.values if current is not None else {}, _sym(ccy)).to_html(
                    full_html=False, include_plotlyjs=False)
                target = result.tangency or result.mvp
                res = reallocation_plan(
                    current.values if current is not None else {}, target.weights,
                    deployment.active_capital, latest_prices, mu=result.mu,
                    allow_fractional=params["allow_fractional"],
                    purchase_prices=current.purchase_prices if current else {},
                    asset_types=asset_types)
                from io import StringIO
                basket_df = pd.read_csv(StringIO(basket_trader_csv(res))) \
                    if basket_trader_csv(res).strip().count("\n") >= 1 else pd.DataFrame()
                pnl = {
                    "total_value": current.total_value if current else 0.0,
                    "total_pnl": current.total_pnl if current and current.has_cost_data else 0.0,
                    "pnl_pct": (current.total_pnl / current.total_cost * 100)
                    if current and current.has_cost_data and current.total_cost > 0 else 0.0,
                    "total_cost": current.total_cost if current and current.has_cost_data else 0.0,
                }
                st.session_state["exec_report_html"] = build_executive_report_html(
                    title="Executive Investment Report",
                    base_currency=ccy, generated_at=iso_timestamp(), pnl=pnl,
                    ef_chart_html=ef_html, donut_chart_html=donut_html,
                    basket_df=basket_df,
                    extra_note="Max-Sharpe tangency allocation")
            except Exception as exc:  # noqa: BLE001 - never crash the dashboard
                st.session_state["exec_report_html"] = None
                st.error(f"Could not generate the report: {exc}")
        if st.session_state.get("exec_report_html"):
            st.success("Report ready — download below (print to PDF from your browser).")
            st.download_button("⬇️ Download Report (.html)",
                               st.session_state["exec_report_html"],
                               file_name="executive_investment_report.html",
                               mime="text/html", use_container_width=True,
                               key="dl_exec_report")


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
    _render_executive_report(result, current, deployment, latest_prices, params, asset_types)

    tab_opt, tab_bt, tab_mc, tab_pac = st.tabs(
        ["📊 Optimization", "📈 Historical Backtest Analysis",
         "🎲 Monte Carlo Projection", "🧾 PAC Performance Ledger"])

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
        _record_tranche_controls(current, deployment, ccy)
        tab_tan, tab_mvp = st.tabs(["🚀 Max Sharpe (Tangency)", "🛡️ Minimum variance"])
        with tab_tan:
            if result.tangency is None:
                st.warning("Tangency portfolio unavailable at the current Rf.")
            else:
                # Buy-Only: re-solve the no-sell target reachable with new cash.
                use_buy_only = (params["buy_only"] and current is not None
                                and deployment.additional_cash > 0)
                target = result.tangency
                if use_buy_only:
                    bo = buy_only_max_sharpe(
                        result.mu, result.cov, current.values, deployment.active_capital,
                        params["risk_free_rate"], result.max_weight,
                        result.sector_map, result.sector_max)
                    if bo is not None:
                        target = bo
                    else:
                        use_buy_only = False
                        st.warning("No feasible buy-only target; showing standard rebalance.")
                _render_reallocation(target, current, deployment, latest_prices,
                                     result.mu, params, asset_types, result.sector_map,
                                     ccy, "tan", buy_only=use_buy_only,
                                     reference_weights=result.tangency.weights)
        with tab_mvp:
            _render_reallocation(result.mvp, current, deployment, latest_prices,
                                 result.mu, params, asset_types, result.sector_map, ccy, "mvp")

        if watchlist_result is not None:
            base_sharpe, recs = watchlist_result
            anchor_ticker = result.anchor.ticker if result.anchor is not None else None
            _render_watchlist(base_sharpe, recs, anchor_ticker)

        st.markdown("---")
        st.subheader("Diversification — correlation & risk clustering")
        col_heat, col_tree = st.columns(2)
        with col_heat:
            st.plotly_chart(correlation_heatmap(result.corr), use_container_width=True)
        with col_tree:
            tree = hierarchical_clusters(result.corr)
            if tree is None:
                st.warning("⚠️ Hierarchical clustering needs ≥3 assets (or the matrix "
                           "was singular). Showing the heatmap only.")
            else:
                st.plotly_chart(dendrogram_figure(tree), use_container_width=True)

    with tab_bt:
        _render_backtest(data["prices"], list(result.mu.index), current, result,
                         data["benchmark_prices"], data["benchmark_ticker"],
                         params["risk_free_rate"], data.get("sectors", {}))

    with tab_mc:
        _render_montecarlo(result, deployment, params, ccy)

    with tab_pac:
        _render_pac_ledger(current, ccy)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    _inject_theme()
    if not _check_password():
        return
    _init_state()
    store = _get_store()

    st.title("📈 Portfolio Optimizer — Institutional MPT")
    st.markdown("Hybrid holdings · VWCE core anchor · sector guardrails · currency "
                "normalisation · DCA tranches · Italian tax · behavioral P&L · "
                "backtesting · GBM projection · IBKR export.")

    params = render_sidebar()

    frozen_ts = st.session_state.get("frozen_banner")
    if frozen_ts:
        st.warning(f"❄️ Displaying Frozen Portfolio Snapshot from **{frozen_ts}**. "
                   "(TWS Offline Mode) — backtester & Monte-Carlo remain fully operational.")
    edited, watchlist = render_data_input()

    # Auto-commit holdings + settings to the local DB on every interaction.
    _persist_state(store, edited)

    run = st.button("🔍 Optimize portfolio", type="primary", use_container_width=True)
    auto_run = st.session_state.auto_run
    st.session_state.auto_run = False
    if run or auto_run:
        st.session_state["has_run"] = True
    # Results stay sticky once computed, so in-panel buttons (record tranche,
    # report, freeze) re-render without forcing another Optimize click.
    if not st.session_state.get("has_run"):
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

    spinners = {"Yahoo Finance (live)": "Fetching prices & FX from Yahoo Finance…",
                "Sample data (offline)": "Generating offline sample prices…",
                "Upload CSV": "Reading uploaded prices…",
                "IBKR Live API": "Streaming daily bars from Interactive Brokers…"}
    with st.spinner(spinners.get(source, "Loading prices…")):
        try:
            if params["ibkr_active"] and st.session_state.get("_ibkr_provider") is not None:
                provider = st.session_state["_ibkr_provider"]
                data = _as_dict(provider.fetch_historical_prices(
                    list(universe), lookback_years=params["lookback"],
                    benchmark=benchmark, base_currency=base_ccy))
            elif source == "Sample data (offline)":
                data = _load_prices_sample(universe, benchmark, params["lookback"], base_ccy)
            elif source == "Upload CSV":
                if params["uploaded_csv"] is None:
                    st.error("Please upload a price CSV, or pick another data source.")
                    return
                data = _as_dict(load_prices_from_csv(params["uploaded_csv"],
                                                     tickers=list(universe), benchmark=benchmark,
                                                     base_currency=base_ccy))
            else:  # Yahoo Finance (live)
                data = _load_prices_live(universe, benchmark, params["lookback"], base_ccy)
        except PriceDataError as exc:
            st.error(f"Data error: {exc}")
            if source in ("Yahoo Finance (live)", "IBKR Live API"):
                st.info("If the data feed is blocked/rate-limited, switch to *Sample data (offline)*.")
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
