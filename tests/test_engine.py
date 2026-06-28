"""Unit tests for the MPT engine.

These run fully offline against a synthetic, correlated price series, so they
never touch the network. Run with::

    pytest -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import optimizer
from src.portfolio import build_current_portfolio, reallocation_plan


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
    plan = reallocation_plan(current.weights, tan.weights,
                             current.total_value, latest)
    assert {"Ticker", "Current %", "Target %", "Action"} <= set(plan.columns)
    assert len(plan) == 3


def test_no_current_portfolio_when_empty(inputs):
    mu, cov = inputs
    latest = pd.Series({"AAA": 100, "BBB": 100, "CCC": 100})
    assert build_current_portfolio({"AAA": 0, "BBB": 0}, latest, mu, cov, 0.02) is None
