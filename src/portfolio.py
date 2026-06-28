"""Helpers for valuing and scoring the user's *current* portfolio.

These turn a table of holdings (ticker + shares) into portfolio weights, and
translate a set of target weights back into concrete buy/sell actions so the
user can see how to reallocate toward an optimal portfolio.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .optimizer import performance


@dataclass
class CurrentPortfolio:
    """The user's existing allocation, valued at the latest close prices."""

    weights: dict[str, float]
    values: dict[str, float]
    shares: dict[str, float]
    total_value: float
    expected_return: float
    volatility: float
    sharpe: float


def build_current_portfolio(
    shares: dict[str, float],
    latest_prices: pd.Series,
    mu: pd.Series,
    cov: pd.DataFrame,
    risk_free_rate: float,
) -> CurrentPortfolio | None:
    """Value a holdings dict and compute its MPT performance.

    Parameters
    ----------
    shares:
        Mapping of ticker -> number of shares held.
    latest_prices:
        Latest close price per ticker.
    mu, cov:
        Annualised expected returns and covariance (from the optimizer).
    risk_free_rate:
        Annual risk-free rate as a decimal.

    Returns
    -------
    CurrentPortfolio or None
        ``None`` when no positive holdings are supplied (e.g. the user is only
        drafting a brand-new portfolio).
    """
    values: dict[str, float] = {}
    for ticker in mu.index:
        qty = float(shares.get(ticker, 0.0) or 0.0)
        price = float(latest_prices.get(ticker, 0.0) or 0.0)
        value = qty * price
        if value > 0:
            values[ticker] = value

    total = sum(values.values())
    if total <= 0:
        return None

    weights = {ticker: value / total for ticker, value in values.items()}
    ret, vol, sharpe = performance(weights, mu, cov, risk_free_rate)

    return CurrentPortfolio(
        weights=weights,
        values=values,
        shares={t: float(shares.get(t, 0.0) or 0.0) for t in values},
        total_value=total,
        expected_return=ret,
        volatility=vol,
        sharpe=sharpe,
    )


def reallocation_plan(
    current_weights: dict[str, float],
    target_weights: dict[str, float],
    investable_capital: float,
    latest_prices: pd.Series,
) -> pd.DataFrame:
    """Build a per-asset reallocation table from current to target weights.

    Parameters
    ----------
    current_weights:
        Current portfolio weights (may be empty for a new portfolio).
    target_weights:
        Target weights (e.g. the tangency portfolio).
    investable_capital:
        Total capital the target weights are applied to. For an existing
        portfolio this is typically the current value plus any new cash.
    latest_prices:
        Latest close price per ticker, used to translate dollar deltas into
        an approximate number of shares to trade.

    Returns
    -------
    pandas.DataFrame
        Columns: Ticker, Current %, Target %, Δ Weight, Target $, and the
        approximate share delta (positive = buy, negative = sell).
    """
    tickers = sorted(set(current_weights) | set(target_weights))
    rows = []
    for ticker in tickers:
        cur_w = float(current_weights.get(ticker, 0.0))
        tgt_w = float(target_weights.get(ticker, 0.0))
        target_value = tgt_w * investable_capital
        price = float(latest_prices.get(ticker, 0.0) or 0.0)
        share_delta = (target_value - cur_w * investable_capital) / price if price > 0 else float("nan")
        rows.append(
            {
                "Ticker": ticker,
                "Current %": round(cur_w * 100, 2),
                "Target %": round(tgt_w * 100, 2),
                "Δ Weight %": round((tgt_w - cur_w) * 100, 2),
                "Target $": round(target_value, 2),
                "Shares to trade": round(share_delta, 2) if price > 0 else None,
                "Action": _action_label(tgt_w - cur_w),
            }
        )
    return pd.DataFrame(rows)


def _action_label(delta_weight: float, tol: float = 1e-4) -> str:
    if delta_weight > tol:
        return "Buy"
    if delta_weight < -tol:
        return "Sell"
    return "Hold"
