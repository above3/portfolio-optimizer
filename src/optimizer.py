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
    GBM_N_PATHS,
    GBM_STEPS_PER_YEAR,
    N_FRONTIER_POINTS,
    TRADING_DAYS_PER_YEAR,
    UNKNOWN_SECTOR,
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
class AnchorConstraint:
    """A forced/maximum allocation to a single 'core anchor' asset.

    ``mode == "exact"`` pins the weight to exactly ``weight``; ``mode == "cap"``
    constrains it to at most ``weight``. Enforced via per-asset weight bounds,
    which PyPortfolioOpt transforms correctly even under ``max_sharpe``.
    """

    ticker: str
    weight: float
    mode: str = "exact"  # "exact" | "cap"


@dataclass
class WatchlistRecommendation:
    """Diversification impact of adding one watchlist asset to the pool."""

    ticker: str
    suggested_weight: float
    base_sharpe: float
    new_sharpe: float
    sharpe_delta: float
    avg_correlation: float
    feasible: bool
    note: str = ""


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
    anchor: AnchorConstraint | None = None
    sector_map: dict[str, str] = field(default_factory=dict)
    sector_max: float = 1.0
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
# Weight bounds (per-asset cap + optional core anchor)
# --------------------------------------------------------------------------- #
def _clamp_max_weight(max_weight: float, n_assets: int) -> float:
    """Clamp the per-asset cap so the weights can still sum to one."""
    if n_assets <= 0:
        return 1.0
    lower = 1.0 / n_assets  # below this, the simplex is infeasible
    return float(min(1.0, max(max_weight, lower)))


def _build_bounds(
    assets: list[str],
    max_weight: float,
    anchor: AnchorConstraint | None = None,
) -> list[tuple[float, float]]:
    """Per-asset ``(lower, upper)`` weight bounds.

    Every asset is capped at ``max_weight``. A core ``anchor`` either pins its
    weight exactly (``lower == upper == weight``) or caps it (``0, weight``).
    The cap on the *other* assets is widened if necessary so the remaining
    ``1 − anchor`` weight is feasible (i.e. the upper bounds can still sum to 1).
    """
    n = len(assets)
    if n == 0:
        return []
    base_cap = _clamp_max_weight(max_weight, n)

    if anchor is None or anchor.ticker not in assets:
        return [(0.0, base_cap)] * n

    a = float(min(max(anchor.weight, 0.0), 1.0))
    others = n - 1
    if others == 0:
        return [(1.0, 1.0)]  # the anchor is the only asset

    # Other assets must be able to absorb the remaining (1 - a) of weight.
    needed = (1.0 - a) / others
    other_cap = min(1.0, max(base_cap, needed))

    bounds: list[tuple[float, float]] = []
    for asset in assets:
        if asset == anchor.ticker:
            bounds.append((a, a) if anchor.mode == "exact" else (0.0, a))
        else:
            bounds.append((0.0, other_cap))
    return bounds


def _apply_sector_constraints(
    ef: EfficientFrontier,
    assets: list[str],
    sector_map: dict[str, str] | None,
    sector_max: float | None,
) -> None:
    """Add ``sum(w in sector) <= sector_max`` constraints, skipping 'Unknown'.

    PyPortfolioOpt transforms these linear constraints correctly even under the
    ``max_sharpe`` reformulation (verified empirically).
    """
    if not sector_map or sector_max is None or sector_max >= 1.0:
        return
    mapper = {a: str(sector_map.get(a, UNKNOWN_SECTOR) or UNKNOWN_SECTOR) for a in assets}
    constrained = {s for s in mapper.values() if s and s != UNKNOWN_SECTOR}
    if not constrained:
        return
    upper = {s: float(sector_max) for s in constrained}
    ef.add_sector_constraints(mapper, {}, upper)


def _ef(
    mu: pd.Series, cov: pd.DataFrame, bounds: list[tuple[float, float]],
    sector_map: dict[str, str] | None = None, sector_max: float | None = None,
) -> EfficientFrontier:
    """A fresh EfficientFrontier with per-asset bounds + optional sector caps.

    PyPortfolioOpt mutates instances in place, so a new one is built per solve.
    """
    ef = EfficientFrontier(mu, cov, weight_bounds=bounds)
    _apply_sector_constraints(ef, list(mu.index), sector_map, sector_max)
    return ef


def _max_return_given_bounds(
    mu: pd.Series, bounds: list[tuple[float, float]]
) -> float:
    """Greedy maximum attainable return subject to the weight bounds."""
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    w = lo.astype(float).copy()
    remaining = 1.0 - w.sum()
    for idx in np.argsort(mu.to_numpy())[::-1]:
        if remaining <= 1e-12:
            break
        add = min(remaining, hi[idx] - w[idx])
        w[idx] += add
        remaining -= add
    return float(np.dot(w, mu.to_numpy()))


# --------------------------------------------------------------------------- #
# Named optimal portfolios
# --------------------------------------------------------------------------- #
def min_variance_portfolio(
    mu: pd.Series, cov: pd.DataFrame, risk_free_rate: float,
    max_weight: float = DEFAULT_MAX_WEIGHT,
    anchor: AnchorConstraint | None = None,
    sector_map: dict[str, str] | None = None,
    sector_max: float = 1.0,
) -> PortfolioPoint:
    """The long-only Minimum-Variance Portfolio.

    If sector constraints make the problem infeasible, they are dropped (so the
    app never crashes) and the unconstrained MVP is returned.
    """
    bounds = _build_bounds(list(mu.index), max_weight, anchor)
    try:
        ef = _ef(mu, cov, bounds, sector_map, sector_max)
        ef.min_volatility()
        weights = dict(ef.clean_weights())
    except Exception:  # noqa: BLE001 - sector caps infeasible -> drop them
        ef = _ef(mu, cov, bounds)
        ef.min_volatility()
        weights = dict(ef.clean_weights())
    return _make_point("Minimum Variance", weights, mu, cov, risk_free_rate)


def max_sharpe_portfolio(
    mu: pd.Series, cov: pd.DataFrame, risk_free_rate: float,
    max_weight: float = DEFAULT_MAX_WEIGHT,
    anchor: AnchorConstraint | None = None,
    sector_map: dict[str, str] | None = None,
    sector_max: float = 1.0,
) -> PortfolioPoint | None:
    """The Tangency (max-Sharpe) portfolio for the given risk-free rate.

    Returns ``None`` if no long-only portfolio can beat the risk-free rate. If
    sector caps make it infeasible, they are dropped before giving up.
    """
    bounds = _build_bounds(list(mu.index), max_weight, anchor)
    for sm, smax in ((sector_map, sector_max), (None, 1.0)):
        ef = _ef(mu, cov, bounds, sm, smax)
        try:
            ef.max_sharpe(risk_free_rate=risk_free_rate)
        except Exception:  # noqa: BLE001 - retry without sector caps, then give up
            continue
        weights = dict(ef.clean_weights())
        return _make_point("Maximum Sharpe (Tangency)", weights, mu, cov, risk_free_rate)
    return None


def buy_only_max_sharpe(
    mu: pd.Series,
    cov: pd.DataFrame,
    current_values: dict[str, float],
    total_capital: float,
    risk_free_rate: float,
    max_weight: float = DEFAULT_MAX_WEIGHT,
    sector_map: dict[str, str] | None = None,
    sector_max: float = 1.0,
) -> PortfolioPoint | None:
    """Max-Sharpe target reachable by **buying only** (no sells).

    Each asset's weight is lower-bounded by its *current* value fraction of the
    post-injection ``total_capital``, so the solver can only deploy the new
    tranche cash to raise allocations toward the optimum — never reduce an
    existing position. This eliminates capital-gains tax and sell-side fees.

    Returns ``None`` if no feasible no-sell tangency portfolio exists.
    """
    assets = list(mu.index)
    n = len(assets)
    if n == 0:
        return None
    if total_capital <= 0:
        return max_sharpe_portfolio(mu, cov, risk_free_rate, max_weight, None,
                                    sector_map, sector_max)

    cap = _clamp_max_weight(max_weight, n)
    lowers = {a: min(max(float(current_values.get(a, 0.0) or 0.0) / total_capital, 0.0), 1.0)
              for a in assets}
    total_lower = sum(lowers.values())
    if total_lower > 1.0:  # current holdings already exceed the budget -> scale
        lowers = {a: v / total_lower for a, v in lowers.items()}

    bounds = [(lowers[a], max(cap, lowers[a])) for a in assets]
    for sm, smax in ((sector_map, sector_max), (None, 1.0)):
        ef = _ef(mu, cov, bounds, sm, smax)
        try:
            ef.max_sharpe(risk_free_rate=risk_free_rate)
        except Exception:  # noqa: BLE001 - drop sector caps, then give up
            continue
        return _make_point("Buy-Only Target", dict(ef.clean_weights()),
                           mu, cov, risk_free_rate)
    return None


# --------------------------------------------------------------------------- #
# Efficient frontier curve
# --------------------------------------------------------------------------- #
def efficient_frontier_curve(
    mu: pd.Series,
    cov: pd.DataFrame,
    n_points: int = N_FRONTIER_POINTS,
    max_weight: float = DEFAULT_MAX_WEIGHT,
    anchor: AnchorConstraint | None = None,
    sector_map: dict[str, str] | None = None,
    sector_max: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Trace the efficient frontier as ``(volatilities, returns)`` arrays.

    The curve is sampled between the MVP return and the maximum attainable
    return given the per-asset cap, the core-anchor and the sector guardrails.
    """
    bounds = _build_bounds(list(mu.index), max_weight, anchor)
    mvp = min_variance_portfolio(mu, cov, 0.0, max_weight, anchor, sector_map, sector_max)
    r_min = mvp.expected_return
    r_max = _max_return_given_bounds(mu, bounds)

    if r_max <= r_min:
        return np.array([mvp.volatility]), np.array([mvp.expected_return])

    targets = np.linspace(r_min, r_max - 1e-6 * abs(r_max), n_points)
    vols: list[float] = []
    rets: list[float] = []
    for target in targets:
        ef = _ef(mu, cov, bounds, sector_map, sector_max)
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
    anchor: AnchorConstraint | None = None,
    sector_map: dict[str, str] | None = None,
    sector_max: float = 1.0,
) -> FrontierResult:
    """Run a full MPT analysis on a price history.

    Single entry point the UI calls; it wires together input estimation, the
    named optimal portfolios (subject to the per-asset cap, the core anchor and
    the sector guardrails), the frontier curve, the Monte-Carlo cloud, the
    benchmark point and any model-fallback notes.
    """
    cov = estimate_covariance(prices, cov_method=cov_method)
    mu, model_used, note = compute_expected_returns(
        prices, cov, returns_model=returns_model,
        benchmark_prices=benchmark_prices, risk_free_rate=risk_free_rate,
        market_caps=market_caps,
    )
    corr = correlation_from_covariance(cov)

    mvp = min_variance_portfolio(mu, cov, risk_free_rate, max_weight, anchor,
                                 sector_map, sector_max)
    tangency = max_sharpe_portfolio(mu, cov, risk_free_rate, max_weight, anchor,
                                    sector_map, sector_max)

    frontier_vol, frontier_ret = efficient_frontier_curve(
        mu, cov, n_frontier_points, max_weight, anchor, sector_map, sector_max
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
        anchor=anchor,
        sector_map=dict(sector_map or {}),
        sector_max=sector_max,
        notes=notes,
    )


# --------------------------------------------------------------------------- #
# Watchlist diversification screening
# --------------------------------------------------------------------------- #
def screen_watchlist(
    prices: pd.DataFrame,
    base_tickers: list[str],
    watchlist_tickers: list[str],
    risk_free_rate: float,
    cov_method: str = "sample_cov",
    max_weight: float = DEFAULT_MAX_WEIGHT,
    anchor: AnchorConstraint | None = None,
    returns_model: str = "historical",
    benchmark_prices: pd.Series | None = None,
    sector_map: dict[str, str] | None = None,
    sector_max: float = 1.0,
) -> tuple[float, list[WatchlistRecommendation]]:
    """Quantify the diversification impact of each watchlist candidate.

    For every candidate, the tangency portfolio is re-optimised over
    ``base_tickers + [candidate]`` (under the same cap/anchor/sector caps) and
    compared to the baseline tangency over ``base_tickers`` alone.

    Returns ``(base_sharpe, recommendations sorted by Sharpe improvement)``.
    """
    base = [t for t in base_tickers if t in prices.columns]
    if len(base) < 2:
        return float("nan"), []

    base_cov = estimate_covariance(prices[base], cov_method=cov_method)
    base_mu, _, _ = compute_expected_returns(
        prices[base], base_cov, returns_model=returns_model,
        benchmark_prices=benchmark_prices, risk_free_rate=risk_free_rate,
    )
    base_tan = max_sharpe_portfolio(base_mu, base_cov, risk_free_rate,
                                    max_weight, anchor, sector_map, sector_max)
    base_sharpe = base_tan.sharpe if base_tan is not None else float("nan")

    recs: list[WatchlistRecommendation] = []
    seen = set(base)
    for candidate in dict.fromkeys(watchlist_tickers):  # de-dupe, keep order
        if candidate in seen or candidate not in prices.columns:
            continue
        seen.add(candidate)

        pool = base + [candidate]
        cov = estimate_covariance(prices[pool], cov_method=cov_method)
        mu, _, _ = compute_expected_returns(
            prices[pool], cov, returns_model=returns_model,
            benchmark_prices=benchmark_prices, risk_free_rate=risk_free_rate,
        )
        tan = max_sharpe_portfolio(mu, cov, risk_free_rate, max_weight, anchor,
                                   sector_map, sector_max)

        corr = correlation_from_covariance(cov)
        avg_corr = float(corr.loc[candidate, base].abs().mean())

        if tan is None:
            recs.append(WatchlistRecommendation(
                ticker=candidate, suggested_weight=0.0, base_sharpe=base_sharpe,
                new_sharpe=float("nan"), sharpe_delta=float("nan"),
                avg_correlation=avg_corr, feasible=False,
                note="No tangency portfolio with this candidate at the current Rf.",
            ))
            continue

        recs.append(WatchlistRecommendation(
            ticker=candidate,
            suggested_weight=float(tan.weights.get(candidate, 0.0)),
            base_sharpe=base_sharpe,
            new_sharpe=tan.sharpe,
            sharpe_delta=tan.sharpe - base_sharpe,
            avg_correlation=avg_corr,
            feasible=True,
        ))

    recs.sort(key=lambda r: (r.sharpe_delta if r.feasible else float("-inf")),
              reverse=True)
    return base_sharpe, recs


# --------------------------------------------------------------------------- #
# Geometric Brownian Motion wealth projection
# --------------------------------------------------------------------------- #
@dataclass
class WealthProjection:
    """Output of a GBM Monte-Carlo wealth simulation (monthly steps)."""

    times_years: np.ndarray
    p10: np.ndarray
    p50: np.ndarray
    p90: np.ndarray
    mean_path: np.ndarray
    initial_capital: float
    monthly_contribution: float
    horizon_years: float
    mu: float
    sigma: float
    n_paths: int
    total_contributed: float

    @property
    def final_p10(self) -> float:
        return float(self.p10[-1])

    @property
    def final_p50(self) -> float:
        return float(self.p50[-1])

    @property
    def final_p90(self) -> float:
        return float(self.p90[-1])


def simulate_gbm_wealth(
    initial_capital: float,
    mu: float,
    sigma: float,
    horizon_years: float,
    monthly_contribution: float = 0.0,
    n_paths: int = GBM_N_PATHS,
    steps_per_year: int = GBM_STEPS_PER_YEAR,
    seed: int | None = 42,
) -> WealthProjection:
    """Project wealth under Geometric Brownian Motion ``dS = μ S dt + σ S dW``.

    The portfolio's annualised ``mu``/``sigma`` (from the tangency portfolio)
    drive the drift and diffusion; ``monthly_contribution`` is injected at the
    end of every step. Returns the 10th/50th/90th percentile bands plus the mean
    path across ``n_paths`` simulations.
    """
    steps = max(int(round(horizon_years * steps_per_year)), 1)
    dt = 1.0 / steps_per_year
    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * float(np.sqrt(dt))

    rng = np.random.default_rng(seed)
    wealth = np.full(n_paths, float(initial_capital), dtype=float)
    paths = np.empty((steps + 1, n_paths), dtype=float)
    paths[0] = wealth
    for s in range(1, steps + 1):
        z = rng.standard_normal(n_paths)
        wealth = wealth * np.exp(drift + diffusion * z) + monthly_contribution
        paths[s] = wealth

    return WealthProjection(
        times_years=np.arange(steps + 1) / steps_per_year,
        p10=np.percentile(paths, 10, axis=1),
        p50=np.percentile(paths, 50, axis=1),
        p90=np.percentile(paths, 90, axis=1),
        mean_path=paths.mean(axis=1),
        initial_capital=float(initial_capital),
        monthly_contribution=float(monthly_contribution),
        horizon_years=float(horizon_years),
        mu=float(mu),
        sigma=float(sigma),
        n_paths=int(n_paths),
        total_contributed=float(initial_capital) + float(monthly_contribution) * steps,
    )


# --------------------------------------------------------------------------- #
# Historical backtesting (equity curves + diagnostics)
# --------------------------------------------------------------------------- #
@dataclass
class BacktestResult:
    """Normalised equity curves and risk diagnostics for competing allocations."""

    curves: pd.DataFrame        # base-100 curves, one column per strategy
    diagnostics: pd.DataFrame   # index=strategy; Cumulative Return %, Sharpe, Max DD %


def _diagnostics_from_returns(
    daily_returns: pd.Series, risk_free_rate: float, frequency: int
) -> tuple[pd.Series, float, float, float]:
    """Return ``(curve, cumulative_return, ann_sharpe, max_drawdown)``."""
    daily_returns = daily_returns.dropna()
    curve = 100.0 * (1.0 + daily_returns).cumprod()
    if curve.empty:
        return curve, float("nan"), float("nan"), float("nan")
    cumulative = float(curve.iloc[-1] / 100.0 - 1.0)
    ann_ret = float((1.0 + daily_returns.mean()) ** frequency - 1.0)
    ann_vol = float(daily_returns.std(ddof=1) * np.sqrt(frequency))
    sharpe = (ann_ret - risk_free_rate) / ann_vol if ann_vol > 0 else float("nan")
    drawdown = curve / curve.cummax() - 1.0
    max_dd = float(drawdown.min())
    return curve, cumulative, sharpe, max_dd


def backtest_equity_curves(
    prices: pd.DataFrame,
    weight_sets: dict[str, dict[str, float]],
    risk_free_rate: float,
    benchmark_prices: pd.Series | None = None,
    benchmark_name: str | None = None,
    frequency: int = TRADING_DAYS_PER_YEAR,
) -> BacktestResult:
    """Reconstruct base-100 equity curves for fixed-weight allocations.

    Each entry in ``weight_sets`` (e.g. the current and optimised portfolios) is
    held at constant weights (daily-rebalanced) over the price history. The
    benchmark, if supplied, is added as a buy-and-hold single-asset curve.
    """
    daily = prices.pct_change()
    curves: dict[str, pd.Series] = {}
    diag_rows: dict[str, dict[str, float]] = {}

    for name, weights in weight_sets.items():
        w = pd.Series(weights, dtype=float).reindex(prices.columns).fillna(0.0)
        total = w.sum()
        if total <= 0:
            continue
        w = w / total
        port_ret = daily.mul(w, axis=1).sum(axis=1)
        curve, cum, sharpe, max_dd = _diagnostics_from_returns(
            port_ret, risk_free_rate, frequency)
        curves[name] = curve
        diag_rows[name] = {
            "Cumulative Return %": round(cum * 100, 2),
            "Annualised Sharpe": round(sharpe, 2),
            "Max Drawdown %": round(max_dd * 100, 2),
        }

    if benchmark_prices is not None and benchmark_name:
        bench_ret = benchmark_prices.reindex(prices.index).ffill().pct_change()
        curve, cum, sharpe, max_dd = _diagnostics_from_returns(
            bench_ret, risk_free_rate, frequency)
        curves[benchmark_name] = curve
        diag_rows[benchmark_name] = {
            "Cumulative Return %": round(cum * 100, 2),
            "Annualised Sharpe": round(sharpe, 2),
            "Max Drawdown %": round(max_dd * 100, 2),
        }

    curves_df = pd.DataFrame(curves)
    diagnostics_df = pd.DataFrame(diag_rows).T
    return BacktestResult(curves=curves_df, diagnostics=diagnostics_df)
