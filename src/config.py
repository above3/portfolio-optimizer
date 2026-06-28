"""Shared defaults and constants for the portfolio optimizer."""

from __future__ import annotations

# Number of trading days used to annualise daily statistics.
TRADING_DAYS_PER_YEAR: int = 252

# Default lookback window (in years) for the historical price download.
DEFAULT_LOOKBACK_YEARS: int = 5
LOOKBACK_CHOICES: tuple[int, ...] = (1, 2, 3, 5, 10)

# Default annual risk-free rate (as a decimal, i.e. 0.02 == 2%).
DEFAULT_RISK_FREE_RATE: float = 0.02

# Default number of Monte-Carlo portfolios used to draw the feasible-set cloud.
DEFAULT_N_SIMULATIONS: int = 5000

# Number of points used to trace the efficient-frontier curve.
N_FRONTIER_POINTS: int = 60

# Covariance estimators exposed in the UI.
COVARIANCE_METHODS: dict[str, str] = {
    "Sample covariance": "sample_cov",
    "Ledoit-Wolf shrinkage": "ledoit_wolf",
}

# A small, diversified starter basket so the app is usable out of the box.
DEMO_PORTFOLIO: list[dict[str, object]] = [
    {"Ticker": "AAPL", "Shares": 10},
    {"Ticker": "MSFT", "Shares": 8},
    {"Ticker": "JNJ", "Shares": 12},
    {"Ticker": "JPM", "Shares": 15},
    {"Ticker": "XOM", "Shares": 20},
]
