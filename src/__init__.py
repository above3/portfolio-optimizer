"""Portfolio optimization engine package.

Modules
-------
config        : Default parameters and shared constants.
data_loader   : Historical price fetching via yfinance (with caching helpers).
optimizer     : Modern Portfolio Theory engine built on PyPortfolioOpt.
portfolio     : Helpers for valuing and scoring a user's current holdings.
visualization : Plotly figures (Markowitz bullet, efficient frontier, CML).
"""

from . import config, data_loader, optimizer, portfolio, visualization

__all__ = ["config", "data_loader", "optimizer", "portfolio", "visualization"]
