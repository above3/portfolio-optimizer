"""Modern Portfolio Theory engine built on top of PyPortfolioOpt.

This module turns a price history into the building blocks of a Markowitz
analysis:

* annualised expected returns (``mu``) and the covariance matrix (``S``);
* the Minimum-Variance Portfolio (MVP);
* the Maximum-Sharpe / Tangency portfolio;
* a sampled efficient-frontier curve;
* a Monte-Carlo cloud of random portfolios (the "Markowitz bullet").

It also supports two expected-return models (historical mean and a
market-implied Black-Litterman prior), a per-asset maximum-weight constraint,
and a market-benchmark reference point. Everything is plain pandas/numpy in,
dataclasses out, so it is trivial to test and fully decoupled from the UI.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier, expected_returns, risk_models
from pypfopt.black_litterman import (
    market_implied_prior_returns,
    market_implied_risk_aversion,
)

from .config import (
    DEFAULT_MAX_WEIGHT,
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
    benchmark: PortfolioPoint | None = None
    max_weight: float = DEFAULT_MAX_WEIGHT
    returns_model: str = "historical"
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Inputs: expected returns & covariance
# --------------------------------------------------------------------------- #
def estimate_covariance(
    prices: pd.DataFrame,
    cov_method: str = "sample_cov",
    frequency: int = TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame:
    """Estimate the annualised covariance matrix.

    ``cov_method`` is ``"sample_cov"`` (plain sample covariance) or
    ``"ledoit_wolf"`` (shrinkage estimator, more stable for short series).
    """
    if cov_method == "ledoit_wolf":
        return risk_models.CovarianceShrinkage(prices, frequency=frequency).ledoit_wolf()
    return risk_models.sample_cov(prices, frequency=frequency)


def black_litterman_implied_returns(
    cov: pd.DataFrame,
    benchmark_prices: pd.Series,
    risk_free_rate: float,
    frequency: int = TRADING_DAYS_PER_YEAR,
    market_caps: pd.Series | dict | None = None,
) -> pd.Series:
    """Market-implied (Black-Litterman equilibrium) expected returns.

    The benchmark is used as the market proxy: its excess return and variance
    give the implied risk-aversion ``δ``, and the equilibrium prior is
    ``Π = δ · Σ · w_mkt + r_f``. Without external market-cap data, an
    equal-weight market portfolio is assumed for ``w_mkt``.
    """
    delta = market_implied_risk_aversion(
        benchmark_prices, frequency=frequency, risk_free_rate=risk_free_rate
    )
    delta = float(delta)
    if not np.isfinite(delta) or delta <= 0:
        # Benchmark did not earn a positive risk premium; fall back to a small
        # positive risk-aversion so the prior stays economically sensible.
        delta = 1.0

    if market_caps is None:
        weights = pd.Series(1.0 / len(cov.index), index=cov.index)
    else:
        weights = pd.Series(market_caps).reindex(cov.index).fillna(0.0)
        total = weights.sum()
        weights = weights / total if total > 0 else pd.Series(
            1.0 / len(cov.index), index=cov.index
        )

    prior = market_implied_prior_returns(
        weights, delta, cov, risk_free_rate=risk_free_rate
    )
    return prior.reindex(cov.index)


def compute_expected_returns(
    prices: pd.DataFrame,
    cov: pd.DataFrame,
    returns_model: str = "historical",
    benchmark_prices: pd.Series | None = None,
    risk_free_rate: float = 0.0,
    frequency: int = TRADING_DAYS_PER_YEAR,
    market_caps: pd.Series | dict | None = None,
) -> tuple[pd.Series, str, str | None]:
    """Compute expected returns under the selected model.

    Returns ``(mu, model_used, note)`` where ``model_used`` may differ from the
    request (e.g. Black-Litterman falls back to historical when no benchmark is
    available or the estimation fails), and ``note`` explains any fallback.
    """
    historical = expected_returns.mean_historical_return(prices, frequency=frequency)

    if returns_model != "black_litterman":
        return historical, "historical", None

    if benchmark_prices is None or len(benchmark_prices) < 2:
        return (historical, "historical",
                "Black-Litterman needs a benchmark; used historical returns instead.")

    try:
        mu = black_litterman_implied_returns(
            cov, benchmark_prices, risk_free_rate, frequency, market_caps
        )
        return mu, "black_litterman", None
    except Exception as exc:  # noqa: BLE001 - degrade gracefully
        return (historical, "historical",
                f"Black-Litterman failed ({exc}); used historical returns instead.")


def estimate_inputs(
    prices: pd.DataFrame,
    cov_method: str = "sample_cov",
    frequency: int = TRADING_DAYS_PER_YEAR,
    *,
    returns_model: str = "historical",
    benchmark_prices: pd.Series | None = None,
    risk_free_rate: float = 0.0,
    market_caps: pd.Series | dict | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    """Estimate annualised expected returns and the covariance matrix.

    Thin convenience wrapper around :func:`estimate_covariance` and
    :func:`compute_expected_returns` for callers (and tests) that just want the
    ``(mu, cov)`` pair under the default historical model.
    """
    cov = estimate_covariance(prices, cov_method=cov_method, frequency=frequency)
    mu, _, _ = compute_expected_returns(
        prices, cov, returns_model=returns_model,
        benchmark_prices=benchmark_prices, risk_free_rate=risk_free_rate,
        frequency=frequency, market_caps=market_caps,
    )
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
def _clamp_max_weight(max_weight: float, n_assets: int) -> float:
    """Clamp the per-asset cap so the weights can still sum to one."""
    if n_assets <= 0:
        return 1.0
    lower = 1.0 / n_assets  # below this, the simplex is infeasible
    return float(min(1.0, max(max_weight, lower)))


def _fresh_ef(
    mu: pd.Series, cov: pd.DataFrame, max_weight: float = DEFAULT_MAX_WEIGHT
) -> EfficientFrontier:
    """A new long-only EfficientFrontier with an optional per-asset cap.

    PyPortfolioOpt mutates these in place, so a fresh instance is required for
    every optimisation.
    """
    upper = _clamp_max_weight(max_weight, len(mu))
    return EfficientFrontier(mu, cov, weight_bounds=(0, upper))


def min_variance_portfolio(
    mu: pd.Series, cov: pd.DataFrame, risk_free_rate: float,
    max_weight: float = DEFAULT_MAX_WEIGHT,
) -> PortfolioPoint:
    """The long-only Minimum-Variance Portfolio."""
    ef = _fresh_ef(mu, cov, max_weight)
    ef.min_volatility()
    weights = dict(ef.clean_weights())
    return _make_point("Minimum Variance", weights, mu, cov, risk_free_rate)


def max_sharpe_portfolio(
    mu: pd.Series, cov: pd.DataFrame, risk_free_rate: float,
    max_weight: float = DEFAULT_MAX_WEIGHT,
) -> PortfolioPoint | None:
    """The Tangency (max-Sharpe) portfolio for the given risk-free rate.

    Returns ``None`` if no long-only portfolio can beat the risk-free rate
    (PyPortfolioOpt raises in that situation).
    """
    ef = _fresh_ef(mu, cov, max_weight)
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
    max_weight: float = DEFAULT_MAX_WEIGHT,
) -> tuple[np.ndarray, np.ndarray]:
    """Trace the efficient frontier as ``(volatilities, returns)`` arrays.

    The curve is sampled between the MVP return and the maximum attainable
    return given the per-asset weight cap.
    """
    mvp = min_variance_portfolio(mu, cov, 0.0, max_weight)
    r_min = mvp.expected_return

    # Maximum attainable return under the cap: fill the top assets up to `upper`.
    upper = _clamp_max_weight(max_weight, len(mu))
    sorted_mu = np.sort(mu.to_numpy())[::-1]
    capacity = np.full(len(sorted_mu), upper)
    cum = np.cumsum(capacity)
    fill = np.minimum(capacity, np.maximum(0.0, 1.0 - np.concatenate([[0.0], cum[:-1]])))
    r_max = float(np.dot(sorted_mu, fill))

    if r_max <= r_min:
        return np.array([mvp.volatility]), np.array([mvp.expected_return])

    targets = np.linspace(r_min, r_max - 1e-6 * abs(r_max), n_points)
    vols: list[float] = []
    rets: list[float] = []
    for target in targets:
        ef = _fresh_ef(mu, cov, max_weight)
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
# Benchmark reference point
# --------------------------------------------------------------------------- #
def benchmark_point(
    benchmark_prices: pd.Series,
    ticker: str,
    risk_free_rate: float,
    frequency: int = TRADING_DAYS_PER_YEAR,
) -> PortfolioPoint | None:
    """Return/vol/Sharpe of the market benchmark as a standalone point."""
    if benchmark_prices is None or len(benchmark_prices) < 2:
        return None
    frame = benchmark_prices.to_frame(name=ticker)
    ret = float(expected_returns.mean_historical_return(frame, frequency=frequency).iloc[0])
    vol = float(np.sqrt(risk_models.sample_cov(frame, frequency=frequency).iloc[0, 0]))
    sharpe = (ret - risk_free_rate) / vol if vol > 0 else float("nan")
    return PortfolioPoint(
        name=f"Benchmark ({ticker})", weights={ticker: 1.0},
        expected_return=ret, volatility=vol, sharpe=sharpe,
    )


# --------------------------------------------------------------------------- #
# Top-level orchestration
# --------------------------------------------------------------------------- #
def analyse(
    prices: pd.DataFrame,
    risk_free_rate: float,
    cov_method: str = "sample_cov",
    n_simulations: int = DEFAULT_N_SIMULATIONS,
    n_frontier_points: int = N_FRONTIER_POINTS,
    *,
    max_weight: float = DEFAULT_MAX_WEIGHT,
    returns_model: str = "historical",
    benchmark_prices: pd.Series | None = None,
    benchmark_ticker: str | None = None,
    market_caps: pd.Series | dict | None = None,
) -> FrontierResult:
    """Run a full MPT analysis on a price history.

    Single entry point the UI calls; it wires together input estimation, the
    named optimal portfolios (subject to the per-asset cap), the frontier curve,
    the Monte-Carlo cloud, the benchmark point and any model-fallback notes.
    """
    cov = estimate_covariance(prices, cov_method=cov_method)
    mu, model_used, note = compute_expected_returns(
        prices, cov, returns_model=returns_model,
        benchmark_prices=benchmark_prices, risk_free_rate=risk_free_rate,
        market_caps=market_caps,
    )
    corr = correlation_from_covariance(cov)

    mvp = min_variance_portfolio(mu, cov, risk_free_rate, max_weight)
    tangency = max_sharpe_portfolio(mu, cov, risk_free_rate, max_weight)

    frontier_vol, frontier_ret = efficient_frontier_curve(
        mu, cov, n_frontier_points, max_weight
    )
    random_vol, random_ret, random_sharpe = random_portfolios(
        mu, cov, risk_free_rate, n_samples=n_simulations
    )

    bench = benchmark_point(benchmark_prices, benchmark_ticker, risk_free_rate) \
        if benchmark_prices is not None and benchmark_ticker else None

    notes = [note] if note else []

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
        benchmark=bench,
        max_weight=_clamp_max_weight(max_weight, len(mu)),
        returns_model=model_used,
        notes=notes,
    )
