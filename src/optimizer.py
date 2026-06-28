"""Modern Portfolio Theory engine built on top of PyPortfolioOpt.

This module is responsible for turning a price history into the building blocks
of a Markowitz analysis:

* annualised expected returns (``mu``) and the covariance matrix (``S``);
* the Minimum-Variance Portfolio (MVP);
* the Maximum-Sharpe / Tangency portfolio;
* a sampled efficient-frontier curve;
* a Monte-Carlo cloud of random portfolios (the "Markowitz bullet").

Everything is plain pandas/numpy in, dataclasses out, so it is trivial to test
and is fully decoupled from the Streamlit UI.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier, expected_returns, risk_models

from .config import (
    DEFAULT_N_SIMULATIONS,
    N_FRONTIER_POINTS,
    TRADING_DAYS_PER_YEAR,
)


@dataclass
class PortfolioPoint:
    """A single named portfolio and its annualised performance."""

    name: str
    weights: dict[str, float]
    expected_return: float
    volatility: float
    sharpe: float


@dataclass
class FrontierResult:
    """Full output of an MPT analysis, ready for plotting and reporting."""

    mu: pd.Series
    cov: pd.DataFrame
    corr: pd.DataFrame
    risk_free_rate: float
    frontier_vol: np.ndarray
    frontier_ret: np.ndarray
    random_vol: np.ndarray
    random_ret: np.ndarray
    random_sharpe: np.ndarray
    mvp: PortfolioPoint
    tangency: PortfolioPoint | None


# --------------------------------------------------------------------------- #
# Inputs: expected returns & covariance
# --------------------------------------------------------------------------- #
def estimate_inputs(
    prices: pd.DataFrame,
    cov_method: str = "sample_cov",
    frequency: int = TRADING_DAYS_PER_YEAR,
) -> tuple[pd.Series, pd.DataFrame]:
    """Estimate annualised expected returns and the covariance matrix.

    Parameters
    ----------
    prices:
        DataFrame of daily close prices (one column per asset).
    cov_method:
        ``"sample_cov"`` for the plain sample covariance or ``"ledoit_wolf"``
        for the Ledoit-Wolf shrinkage estimator (more stable for short series).
    frequency:
        Trading days per year used for annualisation.
    """
    mu = expected_returns.mean_historical_return(prices, frequency=frequency)

    if cov_method == "ledoit_wolf":
        cov = risk_models.CovarianceShrinkage(prices, frequency=frequency).ledoit_wolf()
    else:
        cov = risk_models.sample_cov(prices, frequency=frequency)

    return mu, cov


def correlation_from_covariance(cov: pd.DataFrame) -> pd.DataFrame:
    """Convert a covariance matrix into a correlation matrix."""
    std = np.sqrt(np.diag(cov.values))
    denom = np.outer(std, std)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where(denom > 0, cov.values / denom, 0.0)
    return pd.DataFrame(corr, index=cov.index, columns=cov.columns)


# --------------------------------------------------------------------------- #
# Performance of an arbitrary weight vector
# --------------------------------------------------------------------------- #
def performance(
    weights: dict[str, float] | pd.Series,
    mu: pd.Series,
    cov: pd.DataFrame,
    risk_free_rate: float,
) -> tuple[float, float, float]:
    """Return ``(expected_return, volatility, sharpe)`` for ``weights``.

    Weights are aligned to the order of ``mu``'s index, so callers do not need
    to worry about column ordering.
    """
    if isinstance(weights, dict):
        w = np.array([float(weights.get(asset, 0.0)) for asset in mu.index])
    else:
        w = weights.reindex(mu.index).fillna(0.0).to_numpy(dtype=float)

    ret = float(w @ mu.to_numpy())
    var = float(w @ cov.to_numpy() @ w)
    vol = float(np.sqrt(max(var, 0.0)))
    sharpe = (ret - risk_free_rate) / vol if vol > 0 else float("nan")
    return ret, vol, sharpe


def _make_point(
    name: str,
    weights: dict[str, float],
    mu: pd.Series,
    cov: pd.DataFrame,
    risk_free_rate: float,
) -> PortfolioPoint:
    ret, vol, sharpe = performance(weights, mu, cov, risk_free_rate)
    return PortfolioPoint(name=name, weights=weights, expected_return=ret,
                          volatility=vol, sharpe=sharpe)


# --------------------------------------------------------------------------- #
# Named optimal portfolios
# --------------------------------------------------------------------------- #
def _fresh_ef(mu: pd.Series, cov: pd.DataFrame) -> EfficientFrontier:
    """A new EfficientFrontier (long-only). PyPortfolioOpt mutates these in
    place, so a fresh instance is required for every optimisation."""
    return EfficientFrontier(mu, cov, weight_bounds=(0, 1))


def min_variance_portfolio(
    mu: pd.Series, cov: pd.DataFrame, risk_free_rate: float
) -> PortfolioPoint:
    """The long-only Minimum-Variance Portfolio."""
    ef = _fresh_ef(mu, cov)
    ef.min_volatility()
    weights = dict(ef.clean_weights())
    return _make_point("Minimum Variance", weights, mu, cov, risk_free_rate)


def max_sharpe_portfolio(
    mu: pd.Series, cov: pd.DataFrame, risk_free_rate: float
) -> PortfolioPoint | None:
    """The Tangency (max-Sharpe) portfolio for the given risk-free rate.

    Returns ``None`` if no long-only portfolio can beat the risk-free rate
    (PyPortfolioOpt raises in that situation).
    """
    ef = _fresh_ef(mu, cov)
    try:
        ef.max_sharpe(risk_free_rate=risk_free_rate)
    except Exception:  # noqa: BLE001 - surface as "no tangency" to the caller
        return None
    weights = dict(ef.clean_weights())
    return _make_point("Maximum Sharpe (Tangency)", weights, mu, cov, risk_free_rate)


# --------------------------------------------------------------------------- #
# Efficient frontier curve
# --------------------------------------------------------------------------- #
def efficient_frontier_curve(
    mu: pd.Series,
    cov: pd.DataFrame,
    n_points: int = N_FRONTIER_POINTS,
) -> tuple[np.ndarray, np.ndarray]:
    """Trace the efficient frontier as ``(volatilities, returns)`` arrays.

    The curve is sampled between the MVP return and the maximum attainable
    long-only return (100% in the highest-return asset).
    """
    mvp = min_variance_portfolio(mu, cov, 0.0)
    r_min = mvp.expected_return
    r_max = float(mu.max())

    if r_max <= r_min:
        return np.array([mvp.volatility]), np.array([mvp.expected_return])

    # Shrink the top very slightly to stay inside the feasible region.
    targets = np.linspace(r_min, r_max - 1e-6 * abs(r_max), n_points)
    vols: list[float] = []
    rets: list[float] = []
    for target in targets:
        ef = _fresh_ef(mu, cov)
        try:
            ef.efficient_return(target_return=float(target))
        except Exception:  # noqa: BLE001 - skip infeasible targets
            continue
        ret, vol, _ = ef.portfolio_performance(risk_free_rate=0.0)
        vols.append(vol)
        rets.append(ret)

    if not vols:
        return np.array([mvp.volatility]), np.array([mvp.expected_return])

    order = np.argsort(rets)
    return np.array(vols)[order], np.array(rets)[order]


# --------------------------------------------------------------------------- #
# Monte-Carlo cloud (Markowitz bullet)
# --------------------------------------------------------------------------- #
def random_portfolios(
    mu: pd.Series,
    cov: pd.DataFrame,
    risk_free_rate: float,
    n_samples: int = DEFAULT_N_SIMULATIONS,
    seed: int | None = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample ``n_samples`` long-only portfolios to draw the feasible set.

    Weights are drawn from a Dirichlet distribution whose concentration is
    itself randomised, which spreads samples from the centre of the simplex out
    towards the corners and fills in the classic bullet shape.

    Returns ``(volatilities, returns, sharpe_ratios)``.
    """
    rng = np.random.default_rng(seed)
    n_assets = len(mu)
    mu_v = mu.to_numpy()
    cov_v = cov.to_numpy()

    if n_assets == 1:
        weights = np.ones((n_samples, 1))
    else:
        # Vary the Dirichlet concentration per sample so draws spread from the
        # centre of the simplex out toward the corners, filling the bullet.
        concentration = rng.uniform(0.1, 2.5, size=n_samples)
        weights = np.array(
            [rng.dirichlet(np.full(n_assets, c)) for c in concentration]
        )

    rets = weights @ mu_v
    # Quadratic form w S w' evaluated row-wise.
    vars_ = np.einsum("ij,jk,ik->i", weights, cov_v, weights)
    vols = np.sqrt(np.clip(vars_, 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        sharpes = np.where(vols > 0, (rets - risk_free_rate) / vols, np.nan)
    return vols, rets, sharpes


# --------------------------------------------------------------------------- #
# Top-level orchestration
# --------------------------------------------------------------------------- #
def analyse(
    prices: pd.DataFrame,
    risk_free_rate: float,
    cov_method: str = "sample_cov",
    n_simulations: int = DEFAULT_N_SIMULATIONS,
    n_frontier_points: int = N_FRONTIER_POINTS,
) -> FrontierResult:
    """Run a full MPT analysis on a price history.

    This is the single entry point the UI calls; it wires together the input
    estimation, the named optimal portfolios, the frontier curve and the
    Monte-Carlo cloud into one :class:`FrontierResult`.
    """
    mu, cov = estimate_inputs(prices, cov_method=cov_method)
    corr = correlation_from_covariance(cov)

    mvp = min_variance_portfolio(mu, cov, risk_free_rate)
    tangency = max_sharpe_portfolio(mu, cov, risk_free_rate)

    frontier_vol, frontier_ret = efficient_frontier_curve(mu, cov, n_frontier_points)
    random_vol, random_ret, random_sharpe = random_portfolios(
        mu, cov, risk_free_rate, n_samples=n_simulations
    )

    return FrontierResult(
        mu=mu,
        cov=cov,
        corr=corr,
        risk_free_rate=risk_free_rate,
        frontier_vol=frontier_vol,
        frontier_ret=frontier_ret,
        random_vol=random_vol,
        random_ret=random_ret,
        random_sharpe=random_sharpe,
        mvp=mvp,
        tangency=tangency,
    )
