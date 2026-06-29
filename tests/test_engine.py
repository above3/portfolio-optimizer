"""Unit tests for the MPT engine.

These run fully offline against a synthetic, correlated price series, so they
never touch the network. Run with::

    pytest -q
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

from src import optimizer
from src.data_loader import PriceDataError, apply_regime, convert_to_base_currency
from src.optimizer import (
    AnchorConstraint,
    backtest_equity_curves,
    benchmark_point,
    black_litterman_implied_returns,
    buy_only_max_sharpe,
    screen_watchlist,
    simulate_gbm_wealth,
)
from src.portfolio import (
    PortfolioStore,
    basket_trader_csv,
    build_current_portfolio,
    classify_noise,
    clean_ib_symbol,
    deployment_plan,
    guess_asset_type,
    hierarchical_clusters,
    ibkr_exchange_for,
    map_ibkr_symbol,
    normalise_holdings_df,
    parse_ibkr_positions,
    portfolio_config_csv,
    reallocation_plan,
)
from src.sample_data import generate_sample_prices


def _synthetic_prices(n_days: int = 1000, seed: int = 3) -> pd.DataFrame:
    """Three assets with distinct positive drifts and a controlled correlation.

    Parameters are chosen so every asset has a clearly positive realised return
    (above any sensible risk-free rate) and AAA/BBB are strongly correlated
    while CCC is a near-independent diversifier — exactly the setup that makes a
    tangency portfolio well defined.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n_days)

    # Two correlated assets + one weakly correlated diversifier.
    z = rng.standard_normal((n_days, 3))
    a = 0.0010 + 0.011 * z[:, 0]
    b = 0.0007 + 0.009 * (0.7 * z[:, 0] + 0.7141 * z[:, 1])
    c = 0.0005 + 0.007 * (0.1 * z[:, 0] + 0.9 * z[:, 2])
    rets = np.column_stack([a, b, c])

    prices = 100 * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(prices, index=dates, columns=["AAA", "BBB", "CCC"])


@pytest.fixture(scope="module")
def prices() -> pd.DataFrame:
    return _synthetic_prices()


@pytest.fixture(scope="module")
def inputs(prices):
    return optimizer.estimate_inputs(prices)


def test_estimate_inputs_shapes(prices, inputs):
    mu, cov = inputs
    assert list(mu.index) == ["AAA", "BBB", "CCC"]
    assert cov.shape == (3, 3)
    # Covariance must be symmetric.
    assert np.allclose(cov.values, cov.values.T)


def test_correlation_diagonal_is_one(inputs):
    _, cov = inputs
    corr = optimizer.correlation_from_covariance(cov)
    assert np.allclose(np.diag(corr.values), 1.0, atol=1e-8)
    assert (corr.values <= 1.0 + 1e-9).all()
    assert (corr.values >= -1.0 - 1e-9).all()


def test_weights_sum_to_one(inputs):
    mu, cov = inputs
    mvp = optimizer.min_variance_portfolio(mu, cov, 0.02)
    tan = optimizer.max_sharpe_portfolio(mu, cov, 0.02)
    # clean_weights() rounds to 5 dp, so allow a small rounding tolerance.
    assert mvp.weights and abs(sum(mvp.weights.values()) - 1.0) < 1e-4
    assert tan is not None
    assert abs(sum(tan.weights.values()) - 1.0) < 1e-4
    # Long-only: no negative weights.
    assert all(w >= -1e-9 for w in mvp.weights.values())
    assert all(w >= -1e-9 for w in tan.weights.values())


def test_mvp_is_lowest_variance(inputs):
    """The MVP volatility should not exceed any single-asset volatility."""
    mu, cov = inputs
    mvp = optimizer.min_variance_portfolio(mu, cov, 0.02)
    single_asset_vols = np.sqrt(np.diag(cov.values))
    assert mvp.volatility <= single_asset_vols.min() + 1e-6


def test_tangency_has_max_sharpe(inputs):
    """Tangency Sharpe should beat a large sample of random portfolios."""
    mu, cov = inputs
    rf = 0.02
    tan = optimizer.max_sharpe_portfolio(mu, cov, rf)
    _, _, sharpes = optimizer.random_portfolios(mu, cov, rf, n_samples=20000)
    assert tan is not None
    assert tan.sharpe >= np.nanmax(sharpes) - 1e-3


def test_frontier_is_monotonic_and_above_mvp(inputs):
    mu, cov = inputs
    vol, ret = optimizer.efficient_frontier_curve(mu, cov, n_points=40)
    mvp = optimizer.min_variance_portfolio(mu, cov, 0.0)
    assert len(vol) == len(ret) > 1
    # Returns sorted ascending; every frontier return >= MVP return.
    assert np.all(np.diff(ret) >= -1e-9)
    assert ret.min() >= mvp.expected_return - 1e-6


def test_analyse_endtoend(prices):
    result = optimizer.analyse(prices, risk_free_rate=0.02, n_simulations=2000)
    assert result.tangency is not None
    assert result.random_vol.shape == result.random_ret.shape == (2000,)
    assert result.corr.shape == (3, 3)


def test_current_portfolio_and_reallocation(prices, inputs):
    mu, cov = inputs
    latest = prices.iloc[-1]
    shares = {"AAA": 10, "BBB": 5, "CCC": 0}
    current = build_current_portfolio(shares, latest, mu, cov, 0.02)
    assert current is not None
    # CCC has 0 shares -> excluded from current weights.
    assert "CCC" not in current.weights
    assert abs(sum(current.weights.values()) - 1.0) < 1e-9

    tan = optimizer.max_sharpe_portfolio(mu, cov, 0.02)
    res = reallocation_plan(current.values, tan.weights, current.total_value,
                            latest, mu=mu)
    assert {"Ticker", "Current %", "Target %", "Action"} <= set(res.table.columns)
    assert len(res.table) == 3
    # Fractional rebalance reaches the target -> gross return ~ tangency return
    # (tiny difference comes from clean_weights() rounding to 5 dp).
    assert res.gross_return == pytest.approx(tan.expected_return, abs=1e-3)


def test_no_current_portfolio_when_empty(inputs):
    mu, cov = inputs
    latest = pd.Series({"AAA": 100, "BBB": 100, "CCC": 100})
    assert build_current_portfolio({"AAA": 0, "BBB": 0}, latest, mu, cov, 0.02) is None


# --------------------------------------------------------------------------- #
# Maximum-weight constraint
# --------------------------------------------------------------------------- #
def test_max_weight_cap_is_respected(inputs):
    mu, cov = inputs
    cap = 0.4
    tan = optimizer.max_sharpe_portfolio(mu, cov, 0.02, max_weight=cap)
    mvp = optimizer.min_variance_portfolio(mu, cov, 0.02, max_weight=cap)
    assert tan is not None
    assert max(tan.weights.values()) <= cap + 1e-4
    assert max(mvp.weights.values()) <= cap + 1e-4


def test_max_weight_clamped_when_infeasible(inputs):
    """A cap below 1/n is impossible; it should be clamped, not crash."""
    mu, cov = inputs  # 3 assets -> need >= 1/3 each
    mvp = optimizer.min_variance_portfolio(mu, cov, 0.02, max_weight=0.1)
    assert abs(sum(mvp.weights.values()) - 1.0) < 1e-4


# --------------------------------------------------------------------------- #
# Fractional shares, residual cash & transaction costs
# --------------------------------------------------------------------------- #
def test_whole_share_rounding_and_residual_cash(inputs):
    mu, cov = inputs
    latest = pd.Series({"AAA": 97.3, "BBB": 41.7, "CCC": 28.1})
    target = {"AAA": 0.4, "BBB": 0.35, "CCC": 0.25}
    res = reallocation_plan({}, target, 10000.0, latest, mu=mu,
                            allow_fractional=False)
    # Every traded quantity is a whole integer.
    traded = [r for r in res.table["Shares to trade"] if r is not None]
    assert all(float(x).is_integer() for x in traded)
    # Rounding down leaves a non-negative residual smaller than the sum of all
    # share prices (each asset under-deploys by less than one share).
    assert 0.0 <= res.residual_cash < latest.sum()


def test_transaction_costs_reduce_net_return(inputs):
    mu, cov = inputs
    latest = pd.Series({"AAA": 100.0, "BBB": 50.0, "CCC": 25.0})
    target = {"AAA": 0.4, "BBB": 0.35, "CCC": 0.25}
    res = reallocation_plan({}, target, 10000.0, latest, mu=mu,
                            transaction_cost=5.0)
    assert res.num_trades == 3
    assert res.total_fees == pytest.approx(15.0)
    assert res.net_return == pytest.approx(res.gross_return - 15.0 / 10000.0)


# --------------------------------------------------------------------------- #
# Black-Litterman & benchmark
# --------------------------------------------------------------------------- #
def test_black_litterman_returns_shape(prices, inputs):
    _, cov = inputs
    benchmark = prices.mean(axis=1)  # equal-weight market proxy
    bl = black_litterman_implied_returns(cov, benchmark, risk_free_rate=0.02)
    assert list(bl.index) == list(cov.index)
    assert np.isfinite(bl.to_numpy()).all()


def test_analyse_with_black_litterman_and_benchmark(prices):
    benchmark = prices.mean(axis=1).rename("MKT")
    result = optimizer.analyse(
        prices, risk_free_rate=0.02, n_simulations=1000,
        returns_model="black_litterman",
        benchmark_prices=benchmark, benchmark_ticker="MKT",
    )
    assert result.returns_model == "black_litterman"
    assert result.benchmark is not None
    assert result.benchmark.weights == {"MKT": 1.0}


def test_black_litterman_falls_back_without_benchmark(prices):
    result = optimizer.analyse(
        prices, risk_free_rate=0.02, n_simulations=500,
        returns_model="black_litterman", benchmark_prices=None,
    )
    assert result.returns_model == "historical"
    assert result.notes  # a fallback note was recorded


def test_benchmark_point(prices):
    bench = prices["AAA"]
    point = benchmark_point(bench, "AAA", risk_free_rate=0.02)
    assert point is not None
    assert point.volatility > 0


# --------------------------------------------------------------------------- #
# IBKR CSV import
# --------------------------------------------------------------------------- #
def test_parse_ibkr_flat_flex_csv():
    csv_text = (
        "Symbol,Quantity,Cost Price,Value\n"
        "AAPL,10,150,1500\n"
        "MSFT,5,300,1500\n"
        "AAPL,4,160,640\n"  # second lot -> should be summed
    )
    df = parse_ibkr_positions(io.StringIO(csv_text))
    holdings = dict(zip(df["Ticker"], df["Shares"]))
    assert holdings["AAPL"] == 14  # 10 + 4 summed
    assert holdings["MSFT"] == 5


def test_parse_ibkr_activity_statement():
    csv_text = (
        "Statement,Header,Field Name,Field Value\n"
        "Statement,Data,BrokerName,Interactive Brokers\n"
        "Positions,Header,DataDiscriminator,Asset Category,Currency,Symbol,Quantity,Cost Basis\n"
        "Positions,Data,Summary,Stocks,USD,AAPL,100,15000\n"
        "Positions,Data,Summary,Stocks,USD,MSFT,50,15000\n"
        "Positions,Total,,Stocks,USD,,,30000\n"
    )
    df = parse_ibkr_positions(io.StringIO(csv_text))
    holdings = dict(zip(df["Ticker"], df["Shares"]))
    assert holdings == {"AAPL": 100, "MSFT": 50}


def test_parse_ibkr_missing_columns_raises():
    with pytest.raises(PriceDataError):
        parse_ibkr_positions(io.StringIO("Foo,Bar\n1,2\n3,4\n"))


def test_parse_ibkr_with_cost_price():
    csv_text = (
        "Symbol,Quantity,Cost Price\n"
        "AAPL,10,150.0\n"
        "AAPL,10,170.0\n"  # second lot -> share-weighted avg = 160
        "MSFT,5,300.0\n"
    )
    df = parse_ibkr_positions(io.StringIO(csv_text)).set_index("Ticker")
    assert df.loc["AAPL", "Shares"] == 20
    assert df.loc["AAPL", "Purchase Price"] == pytest.approx(160.0)
    assert df.loc["MSFT", "Purchase Price"] == pytest.approx(300.0)


def test_parse_ibkr_section_derives_price_from_cost_basis():
    csv_text = (
        "Statement,Header,Field,Value\n"
        "Statement,Data,Broker,IBKR\n"
        "Positions,Header,DataDiscriminator,Symbol,Quantity,Cost Basis\n"
        "Positions,Data,Summary,AAPL,100,15000\n"
        "Positions,Data,Summary,MSFT,50,15000\n"
    )
    df = parse_ibkr_positions(io.StringIO(csv_text)).set_index("Ticker")
    assert df.loc["AAPL", "Purchase Price"] == pytest.approx(150.0)  # 15000 / 100
    assert df.loc["MSFT", "Purchase Price"] == pytest.approx(300.0)  # 15000 / 50


def test_current_portfolio_unrealized_pnl(inputs):
    mu, cov = inputs
    latest = pd.Series({"AAA": 120.0, "BBB": 90.0, "CCC": 100.0})
    shares = {"AAA": 10, "BBB": 5}
    pp = {"AAA": 100.0, "BBB": 100.0}
    cur = build_current_portfolio(shares, latest, mu, cov, 0.02, purchase_prices=pp)
    assert cur.has_cost_data
    assert cur.unrealized_pnl["AAA"] == pytest.approx(200.0)   # (120-100)*10
    assert cur.unrealized_pnl["BBB"] == pytest.approx(-50.0)   # (90-100)*5
    assert cur.total_pnl == pytest.approx(150.0)
    assert cur.total_cost == pytest.approx(1500.0)


# --------------------------------------------------------------------------- #
# Core global anchor constraint
# --------------------------------------------------------------------------- #
def test_anchor_exact_pins_weight(inputs):
    mu, cov = inputs
    anchor = AnchorConstraint("CCC", 0.30, "exact")
    tan = optimizer.max_sharpe_portfolio(mu, cov, 0.02, anchor=anchor)
    mvp = optimizer.min_variance_portfolio(mu, cov, 0.02, anchor=anchor)
    assert tan is not None
    assert tan.weights["CCC"] == pytest.approx(0.30, abs=1e-3)
    assert mvp.weights["CCC"] == pytest.approx(0.30, abs=1e-3)


def test_anchor_cap_limits_weight(inputs):
    mu, cov = inputs
    anchor = AnchorConstraint("CCC", 0.10, "cap")
    tan = optimizer.max_sharpe_portfolio(mu, cov, 0.02, anchor=anchor)
    assert tan is not None
    assert tan.weights["CCC"] <= 0.10 + 1e-4


def test_anchor_remains_feasible_under_tight_cap(inputs):
    """Anchor above the per-asset cap must widen the other assets' caps."""
    mu, cov = inputs
    anchor = AnchorConstraint("CCC", 0.40, "exact")
    mvp = optimizer.min_variance_portfolio(mu, cov, 0.02, max_weight=0.05, anchor=anchor)
    assert mvp.weights["CCC"] == pytest.approx(0.40, abs=1e-3)
    assert abs(sum(mvp.weights.values()) - 1.0) < 1e-4


# --------------------------------------------------------------------------- #
# Watchlist diversification screening
# --------------------------------------------------------------------------- #
def test_screen_watchlist_improves_or_holds_sharpe(prices):
    base = ["AAA", "BBB"]
    base_sharpe, recs = screen_watchlist(prices, base, ["CCC"], risk_free_rate=0.02)
    assert len(recs) == 1
    rec = recs[0]
    assert rec.ticker == "CCC" and rec.feasible
    assert rec.base_sharpe == pytest.approx(base_sharpe)
    # Enlarging the asset universe cannot lower the attainable max Sharpe.
    assert rec.new_sharpe >= base_sharpe - 1e-6
    assert rec.sharpe_delta == pytest.approx(rec.new_sharpe - base_sharpe)
    assert 0.0 <= rec.avg_correlation <= 1.0


def test_screen_watchlist_skips_unknown_and_duplicates(prices):
    base = ["AAA", "BBB"]
    # "AAA" already in base, "ZZZ" not in the price panel -> both skipped.
    base_sharpe, recs = screen_watchlist(prices, base, ["AAA", "ZZZ", "CCC"], 0.02)
    assert [r.ticker for r in recs] == ["CCC"]


# --------------------------------------------------------------------------- #
# Sector concentration guardrails
# --------------------------------------------------------------------------- #
def test_sector_cap_is_enforced(inputs):
    mu, cov = inputs
    sector_map = {"AAA": "Tech", "BBB": "Tech", "CCC": "Bond"}
    tan = optimizer.max_sharpe_portfolio(mu, cov, 0.02, sector_map=sector_map, sector_max=0.6)
    assert tan is not None
    tech = tan.weights["AAA"] + tan.weights["BBB"]
    assert tech <= 0.6 + 1e-4


def test_unknown_sector_is_unconstrained(inputs):
    mu, cov = inputs
    # All 'Unknown' -> no sector constraint added, weights still sum to one.
    sector_map = {"AAA": "Unknown", "BBB": "Unknown", "CCC": "Unknown"}
    mvp = optimizer.min_variance_portfolio(mu, cov, 0.02, sector_map=sector_map, sector_max=0.1)
    assert abs(sum(mvp.weights.values()) - 1.0) < 1e-4


# --------------------------------------------------------------------------- #
# GBM Monte-Carlo wealth projection
# --------------------------------------------------------------------------- #
def test_gbm_percentiles_ordered_and_contributions():
    proj = simulate_gbm_wealth(10_000, mu=0.07, sigma=0.15, horizon_years=10,
                               monthly_contribution=100, n_paths=4000, seed=1)
    assert len(proj.times_years) == 10 * 12 + 1
    assert proj.final_p10 <= proj.final_p50 <= proj.final_p90
    assert proj.total_contributed == pytest.approx(10_000 + 100 * 120)
    # Zero volatility -> deterministic compounding (all percentiles equal).
    det = simulate_gbm_wealth(1000, mu=0.05, sigma=0.0, horizon_years=5,
                              n_paths=500, seed=2)
    assert det.final_p10 == pytest.approx(det.final_p90, rel=1e-9)


# --------------------------------------------------------------------------- #
# Historical backtest
# --------------------------------------------------------------------------- #
def test_backtest_curves_and_diagnostics(prices):
    weight_sets = {"Single": {"AAA": 1.0},
                   "Equal": {"AAA": 1 / 3, "BBB": 1 / 3, "CCC": 1 / 3}}
    bt = backtest_equity_curves(prices, weight_sets, 0.02,
                                benchmark_prices=prices["CCC"], benchmark_name="CCC")
    assert set(bt.curves.columns) == {"Single", "Equal", "CCC"}
    assert set(bt.diagnostics.columns) == {
        "Cumulative Return %", "Annualised Sharpe", "Max Drawdown %"}
    # A 100% AAA portfolio must match AAA's own total return.
    aaa_total = prices["AAA"].iloc[-1] / prices["AAA"].iloc[0] - 1
    assert bt.diagnostics.loc["Single", "Cumulative Return %"] == pytest.approx(
        aaa_total * 100, abs=1.0)


# --------------------------------------------------------------------------- #
# Deployment plan (lump sum vs tranches)
# --------------------------------------------------------------------------- #
def test_deployment_lump_sum():
    dp = deployment_plan(10_000, 12_000, "Lump Sum (Deploy All Now)")
    assert dp.active_capital == pytest.approx(22_000)
    assert dp.sidelined_cash == pytest.approx(0.0)
    assert dp.total_wealth == pytest.approx(22_000)


def test_deployment_tranches():
    dp = deployment_plan(10_000, 12_000, "Tranches / DCA (Gradual Deployment)", tranches=4)
    assert dp.per_tranche_cash == pytest.approx(3_000)
    assert dp.active_capital == pytest.approx(13_000)
    assert dp.sidelined_cash == pytest.approx(9_000)


# --------------------------------------------------------------------------- #
# Rebalancing tolerance + Italian tax
# --------------------------------------------------------------------------- #
def test_tolerance_flags_hold(prices, inputs):
    mu, _ = inputs
    latest = pd.Series({"AAA": 100.0, "BBB": 100.0})
    current_values = {"AAA": 5_000.0, "BBB": 5_000.0}
    target = {"AAA": 0.51, "BBB": 0.49}  # ±1% drift, below a 3% tolerance
    res = reallocation_plan(current_values, target, 10_000.0, latest, mu=mu,
                            tolerance=0.03)
    assert res.num_trades == 0
    assert list(res.table["Action"]) == ["HOLD / Within Tolerance"] * 2


def test_italian_capital_gains_tax_on_sale(inputs):
    mu, _ = inputs
    latest = pd.Series({"AAA": 100.0, "BBB": 100.0})
    current_values = {"AAA": 6_000.0, "BBB": 4_000.0}
    target = {"AAA": 0.30, "BBB": 0.70}     # sell AAA, buy BBB
    res = reallocation_plan(
        current_values, target, 10_000.0, latest, mu=mu, tolerance=0.0,
        purchase_prices={"AAA": 50.0}, asset_types={"AAA": "Stock/ETF"})
    # Sell 30 shares of AAA at 100 bought at 50 -> gain 1500 -> tax 26% = 390.
    assert res.total_tax == pytest.approx(390.0, abs=1e-6)
    assert res.net_return <= res.gross_return


def test_bond_tax_rate_is_lower(inputs):
    mu, _ = inputs
    latest = pd.Series({"AAA": 100.0, "BBB": 100.0})
    res = reallocation_plan(
        {"AAA": 6_000.0, "BBB": 4_000.0}, {"AAA": 0.30, "BBB": 0.70}, 10_000.0,
        latest, mu=mu, purchase_prices={"AAA": 50.0},
        asset_types={"AAA": "Bond/Fixed Income"})
    assert res.total_tax == pytest.approx(0.125 * 1500.0)  # 12.5% white-list rate


# --------------------------------------------------------------------------- #
# Behavioral noise classifier
# --------------------------------------------------------------------------- #
def test_noise_classifier_with_and_without_costs(prices, inputs):
    mu, cov = inputs
    latest = prices.iloc[-1]
    cur = build_current_portfolio({"AAA": 10, "BBB": 5}, latest, mu, cov, 0.02,
                                  purchase_prices={"AAA": float(latest["AAA"]) * 0.9,
                                                   "BBB": float(latest["BBB"]) * 1.5})
    report = classify_noise(cur, cov)
    assert report.has_data and report.verdict in {"normale", "grave"}
    flags = {a.ticker: a.within_1sigma for a in report.assets}
    assert set(flags) == {"AAA", "BBB"}

    no_cost = build_current_portfolio({"AAA": 10}, latest, mu, cov, 0.02)
    assert classify_noise(no_cost, cov).has_data is False


# --------------------------------------------------------------------------- #
# Currency normalisation
# --------------------------------------------------------------------------- #
def test_convert_to_base_currency():
    idx = pd.bdate_range("2022-01-01", periods=4)
    prices = pd.DataFrame({"US": [100, 101, 102, 103], "EU": [10, 11, 12, 13]}, index=idx)
    fx = pd.Series([1.1, 1.1, 1.1, 1.1], index=idx)  # USD per 1 EUR
    out = convert_to_base_currency(prices, {"US": "USD", "EU": "EUR"}, "USD", {"EUR": fx})
    assert list(out["US"]) == [100, 101, 102, 103]          # base unchanged
    assert out["EU"].tolist() == pytest.approx([11.0, 12.1, 13.2, 14.3])


def test_sample_currency_conversion_scales_foreign_assets():
    eur = generate_sample_prices(["AAPL", "VWCE.DE"], lookback_years=2, base_currency="EUR")
    assert eur.base_currency == "EUR"
    assert all(c == "EUR" for c in eur.currencies.values())


# --------------------------------------------------------------------------- #
# Execution exports & asset typing
# --------------------------------------------------------------------------- #
def test_basket_trader_csv_format(inputs):
    mu, _ = inputs
    latest = pd.Series({"AAA": 100.0, "VWCE.DE": 100.0})
    res = reallocation_plan({"AAA": 6_000.0}, {"AAA": 0.3, "VWCE.DE": 0.7}, 10_000.0,
                            latest, mu=mu, allow_fractional=False)
    csv_text = basket_trader_csv(res)
    header = csv_text.splitlines()[0]
    assert header == "Action,Quantity,Symbol,OrderType"
    assert "VWCE," in csv_text          # .DE suffix stripped
    assert "SELL" in csv_text and "BUY" in csv_text


def test_clean_ib_symbol():
    assert clean_ib_symbol("VWCE.DE") == "VWCE"
    assert clean_ib_symbol("AGGH.MI") == "AGGH"
    assert clean_ib_symbol("AAPL") == "AAPL"


def test_portfolio_config_csv_columns():
    csv_text = portfolio_config_csv({"AAA": 5_000.0}, {"AAA": 0.4, "BBB": 0.6}, 10_000.0,
                                    {"AAA": "Tech", "BBB": "Bond"}, "EUR")
    header = csv_text.splitlines()[0]
    assert "Current Weight %" in header and "Target Weight %" in header
    assert "Sector" in header and "(EUR)" in header


def test_guess_asset_type():
    assert guess_asset_type("AGGH.MI") == "Bond/Fixed Income"
    assert guess_asset_type("TLT") == "Bond/Fixed Income"
    assert guess_asset_type("AAPL") == "Stock/ETF"
    assert guess_asset_type("VWCE.DE") == "Stock/ETF"


# --------------------------------------------------------------------------- #
# Local persistence (PortfolioStore)
# --------------------------------------------------------------------------- #
def test_store_roundtrip(tmp_path):
    db = tmp_path / "state.db"
    store = PortfolioStore(db)
    holdings = pd.DataFrame({
        "Ticker": ["AAPL", "VWCE.DE"], "Shares": [10, 5],
        "Purchase Price": [175.0, None], "Asset Type": ["Stock/ETF", "Stock/ETF"]})
    store.save_session(holdings, ["GLD", "NVDA"], {"pc_currency": "EUR", "pc_tranches": 6})

    reopened = PortfolioStore(db)  # simulate a fresh session
    loaded = reopened.load_holdings()
    assert list(loaded["Ticker"]) == ["AAPL", "VWCE.DE"]
    assert loaded.loc[0, "Shares"] == 10
    assert reopened.load_watchlist() == ["GLD", "NVDA"]
    assert reopened.load_settings()["pc_currency"] == "EUR"
    assert reopened.load_settings()["pc_tranches"] == 6


def test_store_missing_key_default(tmp_path):
    store = PortfolioStore(tmp_path / "s.db")
    assert store.get("nope", "fallback") == "fallback"
    assert store.load_holdings() is None
    assert store.load_watchlist(default=["X"]) == ["X"]
    assert store.load_settings() == {}


def test_store_recovers_from_corrupt_db(tmp_path):
    db = tmp_path / "corrupt.db"
    db.write_bytes(b"this is definitely not a sqlite database file")
    # Construction must not raise; the corrupt file is quarantined.
    store = PortfolioStore(db)
    assert store.set("k", {"a": 1}) is True
    assert store.get("k") == {"a": 1}
    assert db.with_suffix(".db.corrupt").exists()


def test_normalise_holdings_df_fills_schema():
    df = pd.DataFrame({"Ticker": ["AAPL"], "Shares": [3]})  # missing 2 columns
    out = normalise_holdings_df(df)
    assert list(out.columns) == ["Ticker", "Shares", "Purchase Price", "Asset Type"]
    assert out.loc[0, "Asset Type"] == "Stock/ETF"


# --------------------------------------------------------------------------- #
# Buy-only cash allocation (no sells)
# --------------------------------------------------------------------------- #
def test_buy_only_never_requires_a_sell(inputs):
    mu, cov = inputs
    current_values = {"AAA": 6_000.0, "BBB": 3_000.0}  # CCC unheld; +1k new cash
    total = 10_000.0
    point = buy_only_max_sharpe(mu, cov, current_values, total, 0.02, max_weight=1.0)
    assert point is not None
    # Every existing holding's target value must be >= its current value.
    for ticker, cur_val in current_values.items():
        assert point.weights.get(ticker, 0.0) * total >= cur_val - 1e-6


def test_buy_only_reallocation_has_no_sells(inputs):
    mu, _ = inputs
    latest = pd.Series({"AAA": 100.0, "BBB": 100.0, "CCC": 50.0})
    current_values = {"AAA": 8_000.0, "BBB": 1_000.0}  # AAA over-weight
    target = {"AAA": 0.30, "BBB": 0.30, "CCC": 0.40}   # would normally sell AAA
    res = reallocation_plan(current_values, target, 12_000.0, latest, mu=mu,
                            allow_fractional=False, buy_only=True)
    assert "SELL" not in set(res.table["Action"])
    assert res.total_tax == 0.0


# --------------------------------------------------------------------------- #
# Smart knapsack residual-cash rounding
# --------------------------------------------------------------------------- #
def test_knapsack_reduces_residual_and_tracking_error():
    latest = pd.Series({"A": 10.0, "B": 12.0, "C": 8.0})
    target = {"A": 1 / 3, "B": 1 / 3, "C": 1 / 3}
    naive = reallocation_plan({}, target, 10_000.0, latest, allow_fractional=False,
                              smart_rounding=False)
    smart = reallocation_plan({}, target, 10_000.0, latest, allow_fractional=False,
                              smart_rounding=True)
    assert smart.residual_cash < naive.residual_cash
    assert smart.tracking_error <= naive.tracking_error + 1e-12


# --------------------------------------------------------------------------- #
# Macro-regime backtest transforms
# --------------------------------------------------------------------------- #
def test_regime_none_is_passthrough(prices):
    assert apply_regime(prices, {}, "None / Live Lookback") is prices
    assert apply_regime(prices, {}, None) is prices


def test_gfc_regime_inflates_volatility(prices):
    sectors = {"AAA": "Technology", "BBB": "Financial Services", "CCC": "Fixed Income"}
    gfc = apply_regime(prices, sectors, "2008 Great Financial Crisis (GFC)")
    assert gfc.shape == prices.shape
    assert (gfc.to_numpy() > 0).all()
    assert gfc.pct_change().std().mean() > prices.pct_change().std().mean()


# --------------------------------------------------------------------------- #
# IBKR symbol mapping
# --------------------------------------------------------------------------- #
def test_ibkr_symbol_mapping():
    assert map_ibkr_symbol("VWCE.DE") == "VWCE"
    assert map_ibkr_symbol("AGGH.MI") == "AGGH"
    assert map_ibkr_symbol("AAPL") == "AAPL"
    assert clean_ib_symbol("EMBE.MI") == "EMBE"
    assert ibkr_exchange_for("VWCE.DE") == "IBIS"
    assert ibkr_exchange_for("AGGH.MI") == "BVME"
    assert ibkr_exchange_for("AAPL") == "SMART"


# --------------------------------------------------------------------------- #
# Hierarchical Risk Parity clustering
# --------------------------------------------------------------------------- #
def test_hierarchical_clusters(inputs):
    _, cov = inputs
    corr = optimizer.correlation_from_covariance(cov)
    tree = hierarchical_clusters(corr)
    assert tree is not None
    assert set(tree.leaves) == {"AAA", "BBB", "CCC"}
    assert len(tree.icoord) == len(tree.dcoord) == 2  # n-1 merges for 3 assets
    # Fewer than 3 assets -> graceful None.
    assert hierarchical_clusters(corr.iloc[:2, :2]) is None
