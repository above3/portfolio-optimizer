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
    MarketView,
    PortfolioPoint,
    WatchlistRecommendation,
    analyse,
    backtest_equity_curves,
    buy_only_max_sharpe,
    gbm_goal_analysis,
    required_capital_to_align,
    screen_watchlist,
    simulate_gbm_wealth,
    weight_tracking_error,
)
from src.portfolio import (
    CurrentPortfolio,
    NoiseReport,
    PortfolioStore,
    adaptive_tolerance_bands,
    basket_trader_csv,
    build_current_portfolio,
    build_executive_report_html,
    classify_noise,
    compute_pac_metrics,
    custodian_for,
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
    STATIC_PLOT_CONFIG,
    backtest_figure,
    correlation_heatmap,
    custodian_donut,
    dendrogram_figure,
    efficient_frontier_figure,
    gbm_fan_chart,
)

st.set_page_config(page_title="Portfolio Optimizer · MPT", page_icon="📈",
                   layout="wide", initial_sidebar_state="collapsed")


# --------------------------------------------------------------------------- #
# High-contrast light "research report" theme (CSS injection)
# --------------------------------------------------------------------------- #
_LIGHT_CSS = """
<style>
:root { --pc-bg:#ffffff; --pc-card:#ffffff; --pc-panel:#f6f8fa; --pc-border:#d7dce3;
        --pc-text:#111111; --pc-muted:#55606e; --pc-green:#157347; --pc-red:#b02a37;
        --pc-accent:#0b5cad; }
.stApp { background-color: var(--pc-bg); color: var(--pc-text); }
section[data-testid="stSidebar"] { background-color: var(--pc-panel);
    border-right: 1px solid var(--pc-border); }
h1, h2, h3, h4 { color: var(--pc-text); letter-spacing: .2px; }
p, li, label, span, div[data-testid="stMarkdownContainer"] { color: var(--pc-text); }
/* Native metric cards */
div[data-testid="stMetric"] {
    background: var(--pc-card); border: 1px solid var(--pc-border);
    border-radius: 12px; padding: 14px 16px; box-shadow: 0 1px 3px rgba(16,24,40,.06); }
div[data-testid="stMetricValue"] { font-weight: 700; color: var(--pc-text); }
/* Custom financial cards */
.pc-card { background: var(--pc-card); border: 1px solid var(--pc-border);
    border-radius: 14px; padding: 16px 18px; margin-bottom: 10px;
    box-shadow: 0 1px 3px rgba(16,24,40,.06); }
.pc-card .pc-label { color: var(--pc-muted); font-size: .72rem; letter-spacing: .09em;
    text-transform: uppercase; margin-bottom: 6px; }
.pc-card .pc-value { font-size: 1.55rem; font-weight: 700; color: var(--pc-text); line-height: 1.15; }
.pc-card .pc-sub { font-size: .82rem; margin-top: 4px; color: var(--pc-muted); }
.pc-card.pc-pos { border-left: 4px solid var(--pc-green); }
.pc-card.pc-pos .pc-value { color: var(--pc-green); }
.pc-card.pc-neg { border-left: 4px solid var(--pc-red); }
.pc-card.pc-neg .pc-value { color: var(--pc-red); }
.stButton>button, .stDownloadButton>button { border-radius: 10px;
    border: 1px solid var(--pc-border); font-weight: 600; }
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] { background: var(--pc-panel); border-radius: 8px 8px 0 0;
    padding: 10px 16px; font-weight: 600; }
.stTabs [aria-selected="true"] { background: var(--pc-card);
    border-bottom: 3px solid var(--pc-accent); }
/* Per-tab generated description banner */
.pc-intro { background:#eef4fb; border:1px solid #cfe0f2; border-left:5px solid var(--pc-accent);
    border-radius:10px; padding:15px 20px; margin:2px 0 22px 0; color:#1c2b3a;
    font-size:.96rem; line-height:1.6; }
.pc-intro b { color:#0b3b66; }
/* Textbook commentary callout */
.pc-note { background: var(--pc-panel); border: 1px solid var(--pc-border);
    border-left: 5px solid var(--pc-accent); border-radius: 10px;
    padding: 14px 18px; margin: 8px 0 20px 0; }
.pc-note h4 { margin: 0 0 6px 0; color: var(--pc-accent); }
.pc-note p, .pc-note li { color: #24303f; font-size: .92rem; line-height: 1.55; }
/* Master execution schema directions */
.pc-buy { color: var(--pc-green); font-weight: 800; }
.pc-sell { color: var(--pc-red); font-weight: 800; }
/* Tranche drift efficiency bar */
.pc-drift-track { background:#eef1f5; border:1px solid var(--pc-border); border-radius:999px;
    height:22px; width:100%; overflow:hidden; margin:6px 0; }
.pc-drift-fill { height:100%; border-radius:999px; text-align:right; color:#ffffff;
    font-weight:700; font-size:.74rem; line-height:22px; padding-right:10px;
    background:linear-gradient(90deg,#157347,#1a9950); }
.pc-drift-low .pc-drift-fill { background:linear-gradient(90deg,#b02a37,#d1434f); color:#fff; }
</style>
"""


def _inject_theme() -> None:
    st.markdown(_LIGHT_CSS, unsafe_allow_html=True)


def _commentary(title: str, body_md: str, formulas: list[str] | None = None) -> None:
    """Render a student-friendly textbook callout beneath a chart or table.

    ``formulas`` are optional KaTeX strings rendered with ``st.latex`` *after*
    the prose, isolated from any literal ``$`` currency symbols in the body.
    """
    with st.container(border=True):
        st.markdown(f"#### 📘 {title}")
        st.markdown(body_md)
        for tex in (formulas or []):
            st.latex(tex)


def _tab_intro(title: str, description_html: str) -> None:
    """Render a tab's title plus a generated 'what this screen does' description."""
    st.markdown(f"## {title}")
    st.markdown(f'<div class="pc-intro">📖 <b>What this screen shows.</b> '
                f'{description_html}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Formatting helpers & custom metric cards
# --------------------------------------------------------------------------- #
def _sym(currency: str | None) -> str:
    return config.CURRENCY_SYMBOLS.get((currency or "USD").upper(), "$")


def _money(value: float, currency: str | None) -> str:
    return f"{_sym(currency)}{value:,.0f}"


def _money_md(value: float, currency: str | None) -> str:
    """``_money`` with ``$`` escaped so it is safe inside ``st.markdown`` prose
    (a bare ``$`` there can be mis-parsed as a LaTeX delimiter)."""
    return _money(value, currency).replace("$", r"\$")


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
    "pc_stress_corr": False,
    "pc_dynamic_bands": True,
    "pc_band_factor": config.DEFAULT_BAND_FACTOR,
    "pc_horizon": config.DEFAULT_HORIZON_YEARS,
    "pc_nsim": config.DEFAULT_N_SIMULATIONS,
    "pc_goal_wealth": config.DEFAULT_GOAL_WEALTH,
    "pc_goal_horizon": config.DEFAULT_GOAL_HORIZON_YEARS,
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
                  sector_items, sector_max,
                  view_items=(), stress_correlation=False,
                  asset_type_items=()) -> FrontierResult:
    views = [MarketView(t, x, c) for t, x, c in view_items]
    return analyse(
        prices, risk_free_rate=risk_free_rate, cov_method=cov_method,
        n_simulations=n_simulations, max_weight=max_weight, returns_model=returns_model,
        benchmark_prices=benchmark_prices, benchmark_ticker=benchmark_ticker,
        anchor=_make_anchor(anchor_ticker, anchor_weight, anchor_mode),
        sector_map=dict(sector_items), sector_max=sector_max,
        views=views, stress_correlation=stress_correlation,
        asset_types=dict(asset_type_items),
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


def _bl_views_editor() -> list[tuple[str, float, float]]:
    """Editable Black-Litterman market-views table -> ``[(ticker, excess, conf)]``."""
    views: list[tuple[str, float, float]] = []
    with st.sidebar.expander("🧠 Black-Litterman Market Views Overrides"):
        st.caption("State how each ticker will outperform the benchmark "
                   "(annualised), with your confidence. Active views engage the "
                   "Black-Litterman posterior; none → standard MPT.")
        if "bl_views_df" not in st.session_state:
            st.session_state.bl_views_df = pd.DataFrame(
                [{"Ticker": "", "Outperformance %/yr": 0.0, "Confidence %": 50}])
        edited = st.data_editor(
            st.session_state.bl_views_df, num_rows="dynamic", use_container_width=True,
            key="bl_views_editor",
            column_config={
                "Ticker": st.column_config.TextColumn("Ticker"),
                "Outperformance %/yr": st.column_config.NumberColumn(
                    "Outperf %/yr", format="%.1f",
                    help="Annualised excess return vs the benchmark."),
                "Confidence %": st.column_config.NumberColumn(
                    "Confidence %", min_value=10, max_value=100, step=5),
            })
        st.session_state.bl_views_df = edited
        for _, row in edited.iterrows():
            ticker = str(row.get("Ticker", "") or "").strip().upper()
            try:
                excess = float(row.get("Outperformance %/yr", 0) or 0) / 100.0
            except (TypeError, ValueError):
                excess = 0.0
            try:
                conf = float(row.get("Confidence %", 50) or 50) / 100.0
            except (TypeError, ValueError):
                conf = 0.5
            if ticker and abs(excess) > 0:
                views.append((ticker, excess, max(0.1, min(conf, 1.0))))
        if views:
            st.success(f"✅ {len(views)} active view(s) — Black-Litterman posterior engaged.")
    return views


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
    bl_views = _bl_views_editor()

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

    dynamic_bands = st.sidebar.toggle(
        "Dynamic Adaptive Rebalancing Bands", key="pc_dynamic_bands",
        help="Per-asset tolerance Threshold_i = Base Factor × σ_i (annualised "
             "vol): volatile assets get wider bands (less churn), low-vol bonds "
             "tighter ones. Off → a single static threshold for every asset.")
    if dynamic_bands:
        band_factor = st.sidebar.slider(
            "Adaptive Band Base Factor (× annualised σ)", 0.05, 1.0, step=0.05,
            key="pc_band_factor")
        tolerance_pct = config.DEFAULT_TOLERANCE_PCT
        st.sidebar.caption("ℹ️ Each asset's HOLD band scales with its own volatility.")
    else:
        band_factor = float(st.session_state.get("pc_band_factor", config.DEFAULT_BAND_FACTOR))
        tolerance_pct = st.sidebar.slider("Rebalancing Tolerance Threshold (%)",
                                          config.TOLERANCE_MIN_PCT, config.TOLERANCE_MAX_PCT,
                                          step=1, key="pc_tolerance")

    st.sidebar.markdown("### 🧨 Stress testing")
    stress_correlation = st.sidebar.toggle(
        "Systemic Liquidity Crisis Shock (Correlation Convergence)",
        key="pc_stress_corr",
        help="Inflate every equity/risk off-diagonal correlation ×1.4 (capped "
             "0.99) while preserving volatilities — diversification collapses as "
             "in a liquidity crisis. Re-runs volatilities, drawdowns & Monte "
             "Carlo under the stressed regime. Bonds are left unstressed.")

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
        "dynamic_bands": bool(dynamic_bands), "band_factor": float(band_factor),
        "stress_correlation": bool(stress_correlation), "views": bl_views,
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
            asset_types=asset_types, buy_only=buy_only, smart_rounding=True,
            asset_tolerances=params.get("asset_tolerances"))
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
    st.plotly_chart(backtest_figure(bt), use_container_width=True,
                    config=STATIC_PLOT_CONFIG)
    st.markdown("**Diagnostics** (frozen weights over the window)")
    st.dataframe(bt.diagnostics, use_container_width=True)


def _render_montecarlo(result, deployment, params, ccy) -> None:
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
    st.plotly_chart(gbm_fan_chart(proj, _sym(ccy)), use_container_width=True,
                    config=STATIC_PLOT_CONFIG)
    c1, c2, c3 = st.columns(3)
    c1.metric("Pessimistic (P10)", _money(proj.final_p10, ccy))
    c2.metric("Median (P50)", _money(proj.final_p50, ccy))
    c3.metric("Optimistic (P90)", _money(proj.final_p90, ccy))
    st.caption(f"Total contributed over horizon: {_money(proj.total_contributed, ccy)}.")

    _render_goal_targeting(result, initial, monthly, params, ccy)


def _render_goal_targeting(result, initial: float, monthly: float, params, ccy: str) -> None:
    """Reverse Monte-Carlo: probability of hitting a wealth goal + savings gap."""
    sym = _sym(ccy)
    with st.container(border=True):
        st.markdown("### 🎯 Financial Goal Targeting")
        st.caption("Reverse Monte-Carlo: across 10,000 GBM paths, the probability "
                   "your wealth crosses a target by a chosen horizon — and, if "
                   "unlikely, the extra monthly saving needed to get there.")
        g1, g2 = st.columns(2)
        target_wealth = g1.number_input(
            f"Target Wealth Goal ({sym})", min_value=0.0,
            step=10_000.0, key="pc_goal_wealth")
        target_horizon = g2.number_input(
            "Target Horizon (Years)", min_value=1, max_value=config.HORIZON_MAX_YEARS,
            step=1, key="pc_goal_horizon")
        if target_wealth <= 0:
            st.info("Enter a target wealth above zero to compute the success probability.")
            return

        goal = gbm_goal_analysis(
            initial_capital=initial, mu=result.tangency.expected_return,
            sigma=result.tangency.volatility, horizon_years=int(target_horizon),
            monthly_contribution=monthly, target_wealth=float(target_wealth),
            success_threshold=config.GOAL_SUCCESS_THRESHOLD)

        if goal.prob_success != goal.prob_success:  # NaN -> degraded gracefully
            st.warning("⚠️ Goal simulation could not run on the current inputs.")
            return

        pct = goal.prob_success * 100.0
        threshold_pct = config.GOAL_SUCCESS_THRESHOLD * 100.0
        m1, m2 = st.columns(2)
        tone = "positive" if goal.prob_success >= config.GOAL_SUCCESS_THRESHOLD else "negative"
        with m1:
            _card("Probability of Success", f"{pct:.1f}%",
                  f"of paths reach {_money(target_wealth, ccy)} "
                  f"within {int(target_horizon)}y", tone)
        with m2:
            _card("Median Projected Wealth", _money(goal.median_terminal, ccy),
                  f"current saving {_money(monthly, ccy)}/mo")

        if goal.prob_success >= config.GOAL_SUCCESS_THRESHOLD:
            st.success(f"✅ At **{pct:.1f}%** you clear the {threshold_pct:.0f}% confidence "
                       "bar — your current plan is on track for this goal.")
        elif goal.additional_monthly > 0:
            st.error(
                f"⚠️ Only **{pct:.1f}%** of paths reach the goal (below the "
                f"{threshold_pct:.0f}% bar). Increase contributions by "
                f"**{_money(goal.additional_monthly, ccy)}/mo** "
                f"(to {_money(goal.required_monthly, ccy)}/mo total) to lift the "
                "median path onto the target.")
        else:
            st.warning(f"Only **{pct:.1f}%** of paths reach the goal. The median path "
                       "already clears it — the shortfall is dispersion/volatility, "
                       "not the savings rate; consider a longer horizon.")


def _record_tranche_button(current, deployment, ccy: str,
                           key: str = "record_tranche") -> None:
    """Single button that logs the active deployment into the PAC tranche ledger."""
    store = _get_store()
    pv = current.total_value if current is not None else 0.0
    if st.button("📌 Record Tranche Deployment", use_container_width=True, key=key,
                 help="Logs the active capital + portfolio value to the PAC ledger "
                      "for TWR/MWR tracking over time."):
        ok = store.record_tranche(deployment.active_capital, deployment.strategy, pv)
        if ok:
            st.success(f"Recorded tranche of {_money(deployment.active_capital, ccy)} "
                       f"at portfolio value {_money(pv, ccy)}.")
        else:
            st.error("Could not write to the local ledger.")


def _render_pac_ledger(current, ccy: str) -> None:
    store = _get_store()
    history = store.load_tranche_history()
    if history is None or history.empty:
        st.info("No tranche deployments recorded yet. Use **📌 Record Tranche "
                "Deployment** on the *🏁 Executive Action Report* tab to build the "
                "savings-plan history, then revisit this tab.")
        _commentary(
            "Time-Weighted vs Money-Weighted Return (why both matter)",
            "This ledger will contrast **two very different truths** about your "
            "returns once you log a few tranches:\n\n"
            "- **Time-Weighted Return (TWR)** strips out *when* you added cash. It "
            "chains each sub-period's growth so a large deposit right before a rally "
            "cannot flatter the number. TWR answers: *how good is the allocation "
            "engine itself?* — it is the fair way to grade the strategy.\n"
            "- **Money-Weighted Return (MWR / IRR)** is the single rate that sets the "
            "present value of every cash inflow equal to today's portfolio value "
            "(solved here by **bisection**). It answers: *how fast is my actual money "
            "compounding?* — it rewards good timing and penalises bad timing.\n\n"
            "A disciplined **Piano di Accumulo (PAC)** where TWR ≈ MWR means your "
            "cash-injection timing is neither helping nor hurting the underlying "
            "engine — the sign of a healthy, unemotional savings plan.")
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
    twr_pct = f"{metrics.twr_annual * 100:.2f}%"
    mwr_pct = ("n/a" if metrics.mwr_annual != metrics.mwr_annual
               else f"{metrics.mwr_annual * 100:.2f}%")
    _commentary(
        "Time-Weighted Return (TWR) — grading the strategy",
        f"Your annualised **TWR is {twr_pct}**. TWR deliberately **neutralises the "
        "timing of your deposits**. It chops the timeline at every cash-flow date, "
        "computes the pure investment growth *within* each sub-period, and chain-links "
        "them by multiplication. Because a fresh deposit lands at the *start* of a new "
        "link rather than inflating an existing one, a large contribution made right "
        "before a rally cannot flatter the result. TWR therefore answers one question "
        "only: **how good is the allocation engine itself?** It is the fair, "
        "cash-timing-neutral way to grade the strategy — and the standard by which "
        "professional funds are legally required to advertise performance.",
        formulas=[r"1 + \text{TWR} = \prod_{k=1}^{n}\big(1 + r_k\big)"])
    _commentary(
        "Money-Weighted Return (MWR / IRR) — grading your timing",
        f"Your annualised **MWR is {mwr_pct}**. MWR is the single discount rate `r` "
        "that makes the present value of *every* cash flow — each deposit as a "
        "negative, today's portfolio value as the final positive — sum to exactly "
        "zero. That is the definition of an **Internal Rate of Return (IRR)**, and "
        "because the equation has no closed-form solution the app finds it by "
        "**bisection**: bracket the rate between two bounds, test the midpoint, keep "
        "the half that changes sign, and repeat until it converges. Unlike TWR, MWR "
        "*does* reward timing — it measures the true compounding velocity of the actual "
        "euros/dollars you put in, when you put them in.\n\n"
        "**Reading the two together:** when **MWR > TWR** your tranche timing *added* "
        "value (you happened to deploy more before up-moves); when **MWR < TWR** your "
        "timing *cost* you versus simply drip-feeding on schedule. A disciplined PAC "
        "where the two sit close together is the sign of an unemotional saver whose "
        "timing is neither helping nor hurting the underlying engine.",
        formulas=[r"\sum_{t}\frac{CF_t}{(1+\text{IRR})^{\,t}} = 0"])
    if st.button("🗑️ Clear PAC ledger"):
        store.clear_tranche_history()
        st.rerun()


def _render_executive_report(result, current, deployment, latest_prices, params,
                             asset_types) -> None:
    ccy = params["base_currency"]
    with st.expander("🧾 Printable HTML committee report (download / print to PDF)"):
        st.caption("Compile a self-contained, printable HTML report (P&L, efficient "
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
def _tab_composition(current, noise, asset_types: dict, ccy: str) -> None:
    """Tab 1 — current positions grid, custodian donut, P&L and behavioral check."""
    _tab_intro(
        "📋 Portfolio Composition & P&L",
        "This is your <b>live balance sheet</b>. Every position is priced in real time "
        "and converted to your base currency, so you can see exactly what you own, its "
        "portfolio weight, its Italian tax treatment and where it is custodied — "
        "followed by your unrealised profit &amp; loss and a behavioral volatility "
        "check that flags any holding moving abnormally far from its own history.")
    if current is None or not current.values:
        st.info("No priced holdings detected — you're drafting a fresh portfolio. "
                "Add positions in the **1 · Your portfolio** grid at the top of the page.")
    else:
        rows = []
        for t, val in sorted(current.values.items(), key=lambda kv: kv[1], reverse=True):
            atype = asset_types.get(t, config.DEFAULT_ASSET_TYPE)
            is_bond = "bond" in atype.lower()
            rows.append({
                "Ticker": t,
                "Shares": round(float(current.shares.get(t, 0.0)), 4),
                "Market Value": _money(val, ccy),
                "Weight %": round(current.weights.get(t, 0.0) * 100, 2),
                "Asset Class": atype,
                "Italian Tax": "12.5%" if is_bond else "26%",
                "Custodian": custodian_for(t),
            })
        left, right = st.columns([1.4, 1])
        with left:
            st.markdown("**Current positions ledger**")
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        with right:
            st.plotly_chart(custodian_donut(current.values, _sym(ccy)),
                            use_container_width=True, config=STATIC_PLOT_CONFIG)
        _commentary(
            "The positions ledger, column by column",
            "This table is your **balance sheet**, and every column is a deliberate "
            "piece of the risk picture:\n\n"
            "- **Shares** — the raw quantity you hold. On its own it tells you almost "
            "nothing about risk (10 shares of a €5 bond ETF is tiny; 10 shares of a "
            "€900 stock is not), which is exactly why the next columns exist.\n"
            "- **Market Value** — `Shares × latest price`, then translated into your "
            "base display currency using **live foreign-exchange rates**. This currency "
            "normalisation is what lets a US-dollar stock, a euro Xetra ETF and an "
            "Italian bond sit in one comparable column; without it you would be adding "
            "dollars to euros, a classic beginner error.\n"
            "- **Weight %** — each holding's market value divided by the *total* "
            "portfolio value. Weights are the language Modern Portfolio Theory actually "
            "speaks: the optimiser never asks 'how many shares?', it asks 'what "
            "fraction of the pie?'. The weights always sum to 100%.\n"
            "- **Asset Class** — *Stock/ETF* or *Bond/Fixed Income*. Beyond taxes "
            "(below), this tag also decides whether a holding is inflated during the "
            "correlation-stress test and how tightly its rebalancing band is drawn.\n"
            "- **Custodian** — inferred from the listing suffix: European lines "
            "(`.DE` = Xetra, `.MI` = Borsa Italiana, …) are assumed to sit at "
            "**Trade Republic**, while plain US/SMART-routed tickers execute on the "
            "**IBKR Live** desk. This is what the execution blueprint uses to tell you "
            "*where* to place each order.",
            formulas=[r"w_i \;=\; \frac{\text{Shares}_i \times P_i^{\text{FX}}}"
                      r"{\sum_j \text{Shares}_j \times P_j^{\text{FX}}}, "
                      r"\qquad \sum_i w_i = 1"])
        _commentary(
            "Custodian donut — concentration at a glance",
            "The donut re-plots the **Weight %** column as area. Its real job is to "
            "expose **concentration**: one slice swallowing half the ring is a "
            "single-name risk that the flat table can hide. A healthy long-term "
            "portfolio usually shows several comparably-sized slices rather than one "
            "dominant wedge — the visual echo of diversification you will quantify "
            "properly in the *Risk Clustering* tab.")

    st.markdown("#### 💹 Real-time profit & loss")
    _pnl_ledger(current, ccy)
    _commentary(
        "How your profit & loss is computed",
        "P&L here is strictly **unrealised** (nothing is sold), and it is built from "
        "two numbers per holding:\n\n"
        "- **Cost basis** = `shares × purchase price` — what you originally paid. If "
        "you never entered a purchase price the app cannot compute P&L and shows "
        "*n/a* rather than guessing.\n"
        "- **Market value** = `shares × current price` — what it is worth today.\n\n"
        "The **unrealised P/L** is simply the gap between them, and the **total return "
        "%** expresses that gap relative to what you invested — the honest measure of "
        "performance, because a €1,000 gain means something very different on a €2,000 "
        "stake than on a €200,000 one. Note this figure is **gross**: the Italian "
        "capital-gains tax (26% / 12.5%) only bites when you actually *sell*, which is "
        "why the reallocation and executive tabs re-compute a separate net-of-tax "
        "number.",
        formulas=[r"\text{P/L} = \underbrace{(P_{\text{now}}-P_{\text{buy}})\times "
                  r"\text{shares}}_{\text{gain per holding}}, \qquad "
                  r"\text{Total Return \%} = \frac{\text{P/L}}"
                  r"{\text{cost basis}}\times 100"])
    st.markdown("#### 🧠 Behavioral volatility check")
    _noise_section(noise)
    _commentary(
        "The behavioral volatility check (±1σ) — fighting panic with statistics",
        "The single most expensive investing mistake is selling a *normal* wobble as "
        "if it were a *structural* break. This check draws a statistical fence around "
        "each holding to tell the two apart.\n\n"
        "- **σ (sigma)** is the asset's **annualised volatility** — the standard "
        "deviation of its daily returns scaled up by `√252` (there are ≈252 trading "
        "days a year, and because independent random shocks add up in *variance*, "
        "volatility grows with the **square root** of time, not linearly).\n"
        "- Each position's current drawdown is compared to a **±1σ band**. In a roughly "
        "bell-shaped return distribution about **68%** of all outcomes fall inside ±1σ, "
        "so a move *within* the band is statistically ordinary — expected noise, not a "
        "signal.\n"
        "- A move **beyond ±1σ** is flagged as a statistical **outlier**: it lands in "
        "the ~32% tail and genuinely deserves a calm, deliberate review of the thesis "
        "— which is very different from a reflexive panic sell.\n\n"
        "In short, the badge converts a gut feeling ('this is crashing!') into a "
        "probability statement ('this is a 1-in-3 wobble' vs 'this is a real tail "
        "event').",
        formulas=[r"\sigma_{\text{annual}} = \sigma_{\text{daily}}\times\sqrt{252}, "
                  r"\qquad \Pr\big(|r-\mu|<\sigma\big)\approx 68\%"])


def _tab_frontier(result, current, deployment, latest_prices, params,
                  asset_types: dict, watchlist_result, data: dict, ccy: str) -> None:
    """Tab 2 — efficient frontier, metric cards and reallocation working detail."""
    _tab_intro(
        "📈 Markowitz Efficient Frontier",
        "The <b>heart of Modern Portfolio Theory</b>. This plots every possible "
        "risk/return trade-off as a cloud, bends the optimal <b>frontier</b> and "
        "<b>Capital Market Line</b> through it, and marks the two special portfolios — "
        "<b>Max Sharpe</b> (best reward per unit of risk) and <b>Minimum Variance</b> "
        "(calmest mix) — then shows exactly how to rebalance your holdings toward them.")
    caption_bits = [f"Returns model: **{result.returns_model.replace('_', ' ').title()}**",
                    f"currency **{ccy}**"]
    if result.max_weight < 1.0:
        caption_bits.append(f"per-asset cap **{result.max_weight * 100:.0f}%**")
    if result.sector_max < 1.0:
        caption_bits.append(f"sector cap **{result.sector_max * 100:.0f}%**")
    if result.anchor is not None and result.anchor.weight > 0:
        verb = "fixed at" if result.anchor.mode == "exact" else "≤"
        caption_bits.append(f"anchor **{result.anchor.ticker}** {verb} "
                            f"**{result.anchor.weight * 100:.0f}%**")
    st.caption(" · ".join(caption_bits) + ".")
    for note in result.notes + data.get("notes", []):
        st.warning(note)

    st.plotly_chart(efficient_frontier_figure(result, current),
                    use_container_width=True, config=STATIC_PLOT_CONFIG)

    tan_txt = (f"a Sharpe ratio of **{result.tangency.sharpe:.2f}** at "
               f"{result.tangency.volatility * 100:.1f}% volatility"
               if result.tangency is not None else "the highest attainable Sharpe ratio")
    cur_txt = (
        f"Your **current portfolio** (purple diamond) sits at "
        f"{current.volatility * 100:.1f}% volatility for a "
        f"{current.expected_return * 100:.1f}% return (Sharpe "
        f"{current.sharpe:.2f}). The vertical/horizontal gap to the frontier curve is "
        "your *inefficiency*: risk you are taking without being paid for it."
        if current is not None else
        "Once you hold priced positions, your current portfolio is plotted as a purple "
        "diamond so you can see how far it sits below the optimal frontier.")
    _commentary(
        "Modern Portfolio Theory, from the ground up",
        "In 1952 Harry Markowitz won a Nobel Prize for one counter-intuitive idea: an "
        "asset's risk is **not** a property of the asset itself but of how it moves "
        "*with everything else you own*. Two risky assets that zig and zag at "
        "different times partly cancel, so the portfolio is calmer than either piece — "
        "diversification is, in his phrase, 'the only free lunch in finance'.\n\n"
        "Reading the chart from the raw ingredients:\n"
        "- **The dots** are thousands of random portfolios (the *feasible set*). Their "
        "up-down position is **expected return** `μₚ = wᵀμ`; their left-right position "
        "is **volatility** `σₚ = √(wᵀΣw)`, where `Σ` is the covariance matrix that "
        "encodes every pairwise co-movement. The colour is each dot's Sharpe ratio.\n"
        "- **The frontier** (blue curve) is the *upper-left envelope* of that cloud: "
        "for any given risk it is the highest return reachable. Everything strictly "
        "*below* it is **dominated** — some other portfolio gives more return for the "
        "same risk, or the same return for less risk, so no rational investor should "
        "sit there.\n"
        f"- **🚀 Max Sharpe (Tangency)** — the red star, offering {tan_txt}. Draw a "
        "straight line from the risk-free rate `R_f` on the vertical axis; the point "
        "where it just *touches* the frontier is the tangency portfolio, and that line "
        "is the **Capital Market Line**. Its slope is the Sharpe ratio — reward per "
        "unit of risk — and no portfolio of risky assets can beat it. This is the "
        "theoretical optimum for most investors.\n"
        "- **🛡️ Minimum Variance** — the orange star at the extreme left tip of the "
        "frontier: the single calmest mix that exists, accepting lower return for the "
        "smallest achievable volatility. It is the natural home for a conservative or "
        "near-retirement investor who prizes stability over growth.\n"
        f"- **📍 Current portfolio.** {cur_txt}\n\n"
        "All weights are **long-only** (`0 ≤ wᵢ ≤ w_max`, no shorting or leverage), so "
        "the frontier respects the same real-world constraints your broker does.",
        formulas=[r"\mu_p = \mathbf{w}^\top\boldsymbol{\mu}, \qquad "
                  r"\sigma_p = \sqrt{\mathbf{w}^\top\Sigma\,\mathbf{w}}, \qquad "
                  r"\text{Sharpe} = \frac{\mu_p - R_f}{\sigma_p}"])

    st.markdown("### 🧭 Portfolio comparison")
    _benchmark_banner(result)
    _metric_columns(result, current, ccy)
    _commentary(
        "Reading the three metric cards",
        "The cards line up your **Current** portfolio against the two optimised "
        "targets so the trade-off is explicit. Three numbers matter:\n\n"
        "- **Expected return** — the weighted average of each holding's forecast "
        "annual return. It is a *forecast*, not a guarantee, derived from the returns "
        "model named in the caption above.\n"
        "- **Volatility** — annualised standard deviation of the portfolio, the "
        "single-number summary of how bumpy the ride is. Lower is calmer.\n"
        "- **Sharpe ratio** — `(return − R_f) / volatility`, the *efficiency* score. "
        "It answers the only question that really matters: **how much extra return am "
        "I paid for each unit of risk I take?** A portfolio with lower raw return but "
        "a higher Sharpe is usually the smarter hold, because you can always dial risk "
        "up or down along the Capital Market Line but you can rarely buy efficiency.\n\n"
        "The small **coloured deltas** show the change versus your current portfolio — "
        "green where the optimised target improves the metric, and note that a *lower* "
        "volatility is the improvement there (hence its inverted colour).")

    st.markdown("---")
    st.subheader("💰 Capital deployment & reallocation (working detail)")
    _deployment_metrics(deployment, ccy)
    _commentary(
        "Capital deployment — lump sum vs DCA tranches",
        "Before any trade, the engine decides *how much* cash is in play right now:\n\n"
        "- **Total capital wealth pool** = current portfolio value + all the new cash "
        "you entered. It is the full pie the target weights are applied to.\n"
        "- **Deployed in current tranche** = the slice actually invested this round. "
        "Under **Lump Sum** this is everything at once; under **DCA / tranches** the "
        "new cash is sliced into equal periodic instalments.\n"
        "- **Sidelined cash reserve** = the instalments held back for future tranches.\n\n"
        "**Why tranche at all?** Dollar-Cost Averaging spreads the entry over time, so "
        "you buy more units when prices are low and fewer when high, mechanically "
        "lowering your average cost and removing the emotional agony of trying to time "
        "a single perfect entry. The trade-off is that, on average, markets rise, so "
        "sidelined cash gives up some expected return — DCA buys *behavioural "
        "smoothness and regret-minimisation*, not higher expected wealth.")
    tab_tan, tab_mvp = st.tabs(["🚀 Max Sharpe (Tangency)", "🛡️ Minimum variance"])
    with tab_tan:
        if result.tangency is None:
            st.warning("Tangency portfolio unavailable at the current Rf.")
        else:
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
    _commentary(
        "How to read the reallocation plan",
        "Each row turns the abstract target weight into a concrete order, and the "
        "engine is deliberately **friction-aware** because trading is not free:\n\n"
        "- **Δ Weight %** is the drift between where a holding *is* and where it "
        "*should be*. A holding is only traded when that drift breaks its **rebalancing "
        "band**; anything inside is marked *HOLD*, because churning a 0.3% drift burns "
        "fees and tax for no real benefit. With **dynamic bands** on, each asset's "
        "threshold is `base_factor × σᵢ`, so a jumpy equity gets a wide band (fewer "
        "needless trades) while a placid bond gets a tight one.\n"
        "- **Fees** accrue per executed trade; **Cap-gains tax** is charged only on the "
        "*gain* portion of a **sell** (26% equity / 12.5% bond), never on a buy.\n"
        "- **Gross vs Net return** — the gross figure is the target's expected return; "
        "the net figure subtracts the tax-and-fee **friction drag**, showing what you "
        "actually keep. The gap is the real cost of rebalancing.\n"
        "- **Buy-Only mode** (if enabled) forbids every sell, so the plan drifts you "
        "toward the target using *new cash only* — zero realised gains, zero "
        "capital-gains tax. The **drift efficiency** bar then scores how much of the "
        "remaining misalignment this cash-only tranche removes.\n"
        "- In whole-share mode a **knapsack** routine spends leftover cash on the "
        "shares that most reduce tracking error, so residual cash is driven toward "
        "zero rather than left idle.")

    if watchlist_result is not None:
        base_sharpe, recs = watchlist_result
        anchor_ticker = result.anchor.ticker if result.anchor is not None else None
        _render_watchlist(base_sharpe, recs, anchor_ticker)
        _commentary(
            "The diversification scanner",
            "Each watchlist candidate is added to your optimised portfolio one at a "
            "time and the frontier is re-solved. The verdict is **not** whether the "
            "candidate has a high return on its own, but whether it lifts the whole "
            "portfolio's **Max Sharpe** — which it does mainly by having *low "
            "correlation* to what you already own. A mediocre asset that zigs when your "
            "book zags can raise risk-adjusted return more than a brilliant asset that "
            "simply doubles an existing bet. The **avg |ρ|** column is the tell: the "
            "lower it is, the more genuine diversification the name brings.")


def _tab_risk(result) -> None:
    """Tab 3 — correlation heatmap and HRP risk-clustering dendrogram."""
    _tab_intro(
        "🧬 Risk Clustering & Correlations",
        "A <b>diversification x-ray</b>. The heatmap shows how tightly each pair of "
        "assets moves together, and the dendrogram groups them into a <b>family tree "
        "of shared risk</b> — revealing hidden concentration that a simple count of "
        "tickers would completely miss.")
    col_heat, col_tree = st.columns(2)
    with col_heat:
        st.plotly_chart(correlation_heatmap(result.corr),
                        use_container_width=True, config=STATIC_PLOT_CONFIG)
    with col_tree:
        tree = hierarchical_clusters(result.corr)
        if tree is None:
            st.warning("⚠️ Hierarchical clustering needs ≥3 assets (or the matrix "
                       "was singular). Showing the heatmap only.")
        else:
            st.plotly_chart(dendrogram_figure(tree),
                            use_container_width=True, config=STATIC_PLOT_CONFIG)
    _commentary(
        "The correlation coefficient ρ, explained properly",
        "Correlation `ρ` measures how two assets move *relative to each other*, on a "
        "fixed scale from **−1 to +1**. It is the covariance between them normalised by "
        "the product of their volatilities, which strips out the size of each asset and "
        "leaves only the *direction and tightness* of their co-movement:\n\n"
        "- **ρ ≈ +1.0** (deep blue) — the two move in near-lockstep. Think two mega-cap "
        "tech names or two S&P 500 ETFs: holding both is essentially **the same bet "
        "twice**, adding position count but almost no diversification.\n"
        "- **ρ ≈ 0** — the two are statistically unrelated; each carries its own "
        "independent risk, which is already helpful.\n"
        "- **ρ < 0** (deep red) — the two lean in *opposite* directions, the classic "
        "Gold-versus-Equities relationship. These are the structural **shock "
        "absorbers**: when one leg falls the other tends to hold or rise, damping the "
        "swings of the whole portfolio.\n\n"
        "Why it matters mathematically: portfolio variance is `wᵀΣw`, and every "
        "off-diagonal term carries a `ρᵢⱼ`. Lowering those correlations directly "
        "shrinks portfolio volatility for the *same* holdings — the precise mechanism "
        "behind Markowitz's 'free lunch'.",
        formulas=[r"\rho_{ij} = \frac{\operatorname{Cov}(r_i, r_j)}"
                  r"{\sigma_i\,\sigma_j} \in [-1,\,+1]"])
    _commentary(
        "The dendrogram — an 'ancestral tree of risk'",
        "Where the heatmap shows every pair in isolation, the dendrogram reveals the "
        "*group* structure. It is built by **hierarchical clustering**: convert each "
        "correlation into a distance, repeatedly marry the two closest (most "
        "correlated) assets into a cluster, then the next, until a single family tree "
        "remains.\n\n"
        "- **The distance metric** is `d = √(½(1−ρ))`. Two assets with `ρ = 1` sit at "
        "distance 0 (identical risk); with `ρ = 0` at ≈0.71; with `ρ = −1` at the "
        "maximum distance 1. So *short links = high correlation*.\n"
        "- **Branches that merge low (near the floor)** are siblings driven by the same "
        "systemic force — a hidden concentration even when the tickers look unrelated "
        "(e.g. three different funds that all secretly track US large-cap tech).\n"
        "- **Branches that only join high, near the root**, are the genuine "
        "diversifiers — the assets that march to their own drum.\n\n"
        "Read it as a warning system: a portfolio that looks spread across ten tickers "
        "but collapses into one low cluster is, in risk terms, barely diversified at "
        "all.",
        formulas=[r"d_{ij} = \sqrt{\tfrac{1}{2}\,(1-\rho_{ij})}"])


def _tab_backtest(result, data: dict, current, params) -> None:
    """Tab 4 — historical / stress-regime backtest of current vs optimised weights."""
    _tab_intro(
        "📉 Historical Backtest & Stress",
        "A <b>look backwards</b>. This replays your <b>current</b> versus the "
        "<b>optimised</b> weights across real history — or a synthetic crisis regime — "
        "as the growth of a single base-100 unit, so you can judge how the allocation "
        "would actually have behaved through past shocks before you commit to it.")
    _render_backtest(data["prices"], list(result.mu.index), current, result,
                     data["benchmark_prices"], data["benchmark_ticker"],
                     params["risk_free_rate"], data.get("sectors", {}))
    _commentary(
        "Reading the equity curves",
        "A backtest freezes today's weights and asks *'what if I had held exactly this "
        "mix across the window?'*. Every strategy is **rebased to 100** on day one and "
        "its daily returns are **compounded** (multiplied, not added) forward, so the "
        "lines are directly comparable regardless of starting capital. A line ending at "
        "150 means +50% cumulative growth. Crucially, watch the *shape*, not just the "
        "endpoint: a curve that reaches the same place via a terrifying mid-window "
        "plunge is a very different lived experience from a smooth climb — and most "
        "investors sell at the bottom of the plunge, never realising the final return.",
        formulas=[r"V_t = 100 \times \prod_{k=1}^{t}\big(1 + r_k\big)"])
    _commentary(
        "The three diagnostics: return, Sharpe, and Max Drawdown",
        "- **Cumulative Return %** — total growth over the whole window; the headline "
        "but the *least* informative number, because it ignores the ride.\n"
        "- **Annualised Sharpe** — the same efficiency ratio as elsewhere, computed on "
        "realised daily returns and annualised by `√252`. It rewards *smooth* growth "
        "and penalises volatility, so it is a far better grade than raw return.\n"
        "- **Max Drawdown** — the single most important risk number here: the deepest "
        "peak-to-trough fall inside the window, i.e. the worst loss you would have had "
        "to endure *without capitulating*. A portfolio with a −55% max drawdown demands "
        "an iron stomach; one with −20% is genuinely holdable. Two allocations can post "
        "the identical return yet have wildly different drawdowns — and the calmer one "
        "is almost always the one you can actually stick with.\n\n"
        "**Under the *Systemic Liquidity Crisis* stress**, off-diagonal correlations are "
        "forced toward +1 (capped 0.99) while volatilities are preserved: "
        "diversification evaporates exactly when you need it, so the stressed curves "
        "expose the true black-swan drawdown that a calm-market frontier hides.",
        formulas=[r"\text{MaxDD} = \min_t\!\left(\frac{V_t}"
                  r"{\max_{s\le t} V_s} - 1\right)"])


def _tab_montecarlo(result, deployment, params, ccy: str) -> None:
    """Tab 5 — forward GBM Monte-Carlo wealth projection + reverse goal targeting."""
    _tab_intro(
        "🎲 Monte Carlo Projection",
        "A <b>look forwards</b>. Ten thousand randomised market futures are simulated "
        "with Geometric Brownian Motion to turn a single 'expected return' into an "
        "honest <b>cone of outcomes</b> (pessimistic → median → optimistic). The "
        "goal-targeting engine then works in reverse to tell you the probability of "
        "hitting a wealth target — and the extra monthly saving needed if you fall "
        "short.")
    _render_montecarlo(result, deployment, params, ccy)
    _commentary(
        "Geometric Brownian Motion — the engine under the fan",
        "Each of the 10,000 paths is a simulated future generated by **Geometric "
        "Brownian Motion (GBM)**, the same equation that underpins the Nobel-winning "
        "Black–Scholes model. Every month the wealth is nudged by two forces: a steady "
        "**drift** (`μ`, the tangency portfolio's expected return) that pushes it "
        "upward, and a random **diffusion** shock (`σ`, its volatility) drawn from a "
        "bell curve that can push either way.\n\n"
        "- Because returns *multiply* rather than add, terminal wealth follows a "
        "**log-normal** distribution — it cannot go below zero but has a long upper "
        "tail, which is why the fan is asymmetric.\n"
        "- The band spans the **10th percentile (pessimistic)** to the **90th "
        "(optimistic)**, and the solid line is the **median (P50)**. The fan *widens "
        "with time* because volatility accumulates with `√t` — the further out you "
        "look, the less certain any single number is. This is the honest antidote to a "
        "single 'your money will be worth X' promise.\n"
        "- **Monthly contributions** are injected along every path, so the cone blends "
        "market compounding with your disciplined savings.",
        formulas=[r"W_{t+\Delta t} = W_t\,\exp\!\Big[\big(\mu - \tfrac{1}{2}\sigma^2"
                  r"\big)\Delta t + \sigma\sqrt{\Delta t}\; Z\Big] + C, "
                  r"\quad Z\sim\mathcal{N}(0,1)"])
    _commentary(
        "Financial Goal Targeting — running the simulation in reverse",
        "Forward Monte-Carlo asks *'where might I end up?'*. Goal targeting flips the "
        "question to *'what do I need to do to get there?'*:\n\n"
        "- **Probability of success** = the fraction of the 10,000 paths that touch or "
        "cross your target wealth by the chosen horizon. 62% means 6,200 of the "
        "simulated futures made it.\n"
        "- If that probability sits below the **85%** confidence bar, the engine solves "
        "for the **extra monthly saving** required to lift the *median* path onto the "
        "goal. It does this by **bisection** — repeatedly halving a search interval, "
        "testing whether that contribution reaches target, and narrowing until it "
        "converges — the same robust root-finding idea used for the IRR in the PAC "
        "ledger. The output is a single, concrete number: 'save €X more per month'.\n\n"
        "It converts a vague anxiety ('am I saving enough?') into an actionable, "
        "quantified plan.")


def _execution_target(result, current, deployment, params):
    """Resolve the target the blueprint should reach (buy-only aware)."""
    target = result.tangency or result.mvp
    use_buy_only = (params["buy_only"] and current is not None
                    and deployment.additional_cash > 0)
    if use_buy_only and result.tangency is not None:
        bo = buy_only_max_sharpe(
            result.mu, result.cov, current.values, deployment.active_capital,
            params["risk_free_rate"], result.max_weight,
            result.sector_map, result.sector_max)
        if bo is not None:
            target = bo
        else:
            use_buy_only = False
    return target, use_buy_only


def _execution_blueprint_df(res, asset_types: dict, ccy: str) -> pd.DataFrame:
    """Filter a reallocation plan down to only the actionable BUY/SELL trades."""
    rows = []
    for _, r in res.table.iterrows():
        action = str(r.get("Action", ""))
        if action not in ("BUY", "SELL"):
            continue  # drop HOLD / within-tolerance rows entirely
        shares = r.get("Shares to trade")
        try:
            qty = int(round(abs(float(shares))))
        except (TypeError, ValueError):
            qty = 0
        if qty <= 0:
            continue
        try:
            cash = abs(float(r.get("Trade Value")))
        except (TypeError, ValueError):
            cash = 0.0
        t = str(r.get("Ticker"))
        rows.append({
            "Ticker": t,
            "Asset Class": asset_types.get(t, config.DEFAULT_ASSET_TYPE),
            "Custodian": custodian_for(t),
            "Direction": ("🟢 BUY" if action == "BUY" else "🔴 SELL"),
            "Exact Target Shares": qty,
            "Estimated Cash Required": cash,
        })
    return pd.DataFrame(rows, columns=[
        "Ticker", "Asset Class", "Custodian", "Direction",
        "Exact Target Shares", "Estimated Cash Required"])


def _style_blueprint(blueprint: pd.DataFrame, ccy: str):
    """High-contrast styling: bold green BUY, bold red SELL, localized cash."""
    def _dir_color(col):
        return ["color:#157347;font-weight:800" if "BUY" in str(v)
                else "color:#b02a37;font-weight:800" if "SELL" in str(v)
                else "" for v in col]
    return (blueprint.style
            .apply(_dir_color, subset=["Direction"])
            .format({"Estimated Cash Required": lambda v: _money(v, ccy)}))


def _tab_executive(result, current, deployment, latest_prices, params,
                   asset_types: dict, ccy: str) -> None:
    """Tab 7 — formal Investment-Committee summary + master execution blueprint."""
    _tab_intro(
        "🏁 Executive Action Report",
        "The <b>action page</b>. A formal, unpadded investment-committee summary of the "
        "portfolio's health and drift, followed by a single <b>execution blueprint</b> "
        "listing only the trades you need to place right now — with a one-click basket "
        "export and tranche logging as the final steps of the cycle.")

    target, use_buy_only = _execution_target(result, current, deployment, params)

    # ---- 1. Written executive summary ----
    n_pos = len([v for v in (current.values.values() if current else []) if v > 0])
    total_value = current.total_value if current is not None else 0.0
    lines: list[str] = []
    if current is not None and result.tangency is not None:
        te = weight_tracking_error(current.weights, result.tangency.weights)
        sharpe_gap = result.tangency.sharpe - current.sharpe
        drift_word = ("well-aligned with" if te < 0.05 else
                      "moderately drifted from" if te < 0.15 else
                      "materially drifted from")
        lines.append(
            f"The portfolio currently holds **{_money_md(total_value, ccy)}** across "
            f"**{n_pos}** priced position(s), and is **{drift_word}** the Max-Sharpe "
            f"optimum — a weight-space tracking error of **{te * 100:.1f}%**.")
        if sharpe_gap > 0.01:
            lines.append(
                f"Rebalancing toward the tangency node would improve the risk-adjusted "
                f"Sharpe ratio by **{sharpe_gap:+.2f}** (from {current.sharpe:.2f} to "
                f"{result.tangency.sharpe:.2f}).")
        else:
            lines.append("The current allocation already tracks the optimal frontier "
                         "closely; only marginal Sharpe improvement is available.")
    elif current is None:
        lines.append(
            f"No existing holdings — this is a clean deployment of "
            f"**{_money_md(deployment.active_capital, ccy)}** into the optimised target.")
    lines.append(
        f"This cycle deploys **{_money_md(deployment.active_capital, ccy)}** of active "
        f"capital under the *{deployment.strategy}* strategy"
        + (f", holding **{_money_md(deployment.sidelined_cash, ccy)}** in reserve for "
           "future tranches." if deployment.sidelined_cash > 0 else "."))
    if use_buy_only:
        lines.append("Executed in **buy-only** mode: no sells, so zero realised "
                     "capital-gains tax and zero sell-side fees.")
    with st.container(border=True):
        st.markdown("#### 🧾 Written Executive Summary")
        for ln in lines:
            st.markdown("- " + ln)
    _commentary(
        "What the summary is measuring",
        "This paragraph is generated from two diagnostics:\n\n"
        "- **Tracking error** — the root-mean-square distance, in weight space, between "
        "your current allocation and the Max-Sharpe target. Think of it as a single "
        "percentage that answers *'how far off-optimal am I?'*: 2% is a nudge, 15%+ is a "
        "different portfolio wearing your name. It drives the *well-aligned / "
        "moderately drifted / materially drifted* verdict.\n"
        "- **Sharpe gap** — how much risk-adjusted efficiency you would *gain* by "
        "rebalancing to the tangency node. A large gap justifies paying the fees and "
        "tax to trade; a near-zero gap means you should probably leave things alone.\n\n"
        "The final line confirms the **capital being deployed** this cycle and the "
        "strategy (lump-sum or DCA), so the committee sees exactly how much money is "
        "moving before approving a single order.",
        formulas=[r"\text{TE} = \sqrt{\sum_i \big(w_i^{\text{current}} "
                  r"- w_i^{\text{target}}\big)^2}"])

    # ---- 2. Master Execution Schema (blueprint) ----
    st.markdown("#### 🧮 The Master Execution Schema")
    res = reallocation_plan(
        current_values=current.values if current is not None else {},
        target_weights=target.weights, investable_capital=deployment.active_capital,
        latest_prices=latest_prices, mu=result.mu, risk_free_rate=params["risk_free_rate"],
        allow_fractional=False, transaction_cost=params["transaction_cost"],
        tolerance=params["tolerance"],
        purchase_prices=current.purchase_prices if current else {},
        asset_types=asset_types, buy_only=use_buy_only, smart_rounding=True,
        asset_tolerances=params.get("asset_tolerances"))
    blueprint = _execution_blueprint_df(res, asset_types, ccy)
    if blueprint.empty:
        st.success("✅ No trades required — every holding sits within its rebalancing "
                   "band. The portfolio is already aligned to target.")
    else:
        st.dataframe(_style_blueprint(blueprint, ccy), hide_index=True,
                     use_container_width=True)
        st.caption("Only assets requiring immediate market interaction are listed — "
                   "within-tolerance holds are filtered out.")
    _commentary(
        "How to read the execution blueprint",
        "This is intentionally **stripped to only the actionable trades** — every "
        "*HOLD* row is filtered out so nothing distracts from what you must actually "
        "do. Each column is an instruction:\n\n"
        "- **Direction** — a bold **🟢 BUY** (green) or **🔴 SELL** (red); the colour is "
        "the fastest possible read of intent.\n"
        "- **Exact Target Shares** — whole-share, integer-rounded quantities, because "
        "you cannot place an order for 12.7 shares on most venues.\n"
        "- **Estimated Cash Required** — the trade's value in your base currency, so "
        "you can confirm the buys fit the cash you are deploying.\n"
        "- **Custodian** — *where* to route each order (IBKR Live vs Trade Republic), "
        "removing the last piece of ambiguity between decision and execution.\n\n"
        "It is the same friction- and tax-aware plan from the Frontier tab, but "
        "distilled to a committee-ready order ticket.")

    # ---- 3. Operational bottom bar (definitive final steps) ----
    st.markdown("#### ⚙️ Execute")
    b1, b2 = st.columns(2)
    with b1:
        st.download_button("📤 Export Basket CSV", basket_trader_csv(res),
                           file_name="executive_basket_orders.csv", mime="text/csv",
                           use_container_width=True, key="exec_basket_csv")
    with b2:
        _record_tranche_button(current, deployment, ccy, key="exec_record_tranche")
    st.caption("**Export Basket CSV** hands the whole-share orders to IBKR's Basket "
               "Trader in one file; **Record Tranche Deployment** logs this round's "
               "capital and portfolio value to the PAC ledger so the TWR/MWR history "
               "keeps building. Together they close the loop: decide → execute → "
               "record → measure.")

    # ---- 4. Printable HTML committee report ----
    _render_executive_report(result, current, deployment, latest_prices, params, asset_types)


def render_results(result, current, noise, deployment, data, params,
                   watchlist_result, asset_types) -> None:
    ccy = params["base_currency"]
    latest_prices = data["latest_prices"]

    # Per-asset adaptive rebalancing bands (Threshold_i = base_factor × σ_i).
    if params.get("dynamic_bands"):
        vols = {t: float(max(result.cov.loc[t, t], 0.0)) ** 0.5 for t in result.cov.index}
        params["asset_tolerances"] = adaptive_tolerance_bands(vols, params["band_factor"])
    else:
        params["asset_tolerances"] = None

    # Clean narrative order: what you hold → optimise → risk structure →
    # past behaviour → future projection → savings history → act.
    (tab_comp, tab_frontier, tab_risk, tab_bt,
     tab_mc, tab_pac, tab_exec) = st.tabs([
        "📋 Portfolio Composition & P&L",
        "📈 Markowitz Efficient Frontier",
        "🧬 Risk Clustering & Correlations",
        "📉 Historical Backtest",
        "🎲 Monte Carlo Projection",
        "🧾 Historical PAC Ledger",
        "🏁 Executive Action Report",
    ])

    with tab_comp:
        _tab_composition(current, noise, asset_types, ccy)

    with tab_frontier:
        _tab_frontier(result, current, deployment, latest_prices, params,
                      asset_types, watchlist_result, data, ccy)

    with tab_risk:
        _tab_risk(result)

    with tab_bt:
        _tab_backtest(result, data, current, params)

    with tab_mc:
        _tab_montecarlo(result, deployment, params, ccy)

    with tab_pac:
        _tab_intro(
            "🧾 Historical PAC Ledger",
            "Your real <b>savings-plan track record</b>. Once you log capital tranches, "
            "this contrasts <b>Time-Weighted Return</b> (the quality of the allocation "
            "engine, stripped of cash-timing) with <b>Money-Weighted Return</b> (the "
            "true compounding velocity of your own money) to grade both the strategy "
            "and your timing.")
        _render_pac_ledger(current, ccy)

    with tab_exec:
        _tab_executive(result, current, deployment, latest_prices, params,
                       asset_types, ccy)


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
    st.markdown("A spacious **Financial Research Report** in seven screens: "
                "composition & P&L · efficient frontier · risk clustering · historical "
                "backtest · Monte-Carlo projection · PAC ledger · executive action "
                "blueprint. *Open the sidebar (top-left ›) to adjust parameters.*")

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
            sector_items, params["sector_max"],
            view_items=tuple(params["views"]),
            stress_correlation=params["stress_correlation"],
            asset_type_items=tuple(sorted(asset_types.items())))
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
