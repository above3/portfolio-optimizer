"""Historical price downloading and cleaning via yfinance.

The functions here are deliberately framework-agnostic (no Streamlit imports) so
the financial engine can be unit-tested or reused outside the web app.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import pandas as pd
import yfinance as yf

from .config import DEFAULT_LOOKBACK_YEARS


class PriceDataError(RuntimeError):
    """Raised when usable price data cannot be obtained for the request."""


@dataclass
class PriceData:
    """Container for a cleaned price history download.

    Attributes
    ----------
    prices:
        DataFrame of daily close prices, one column per *valid* ticker.
    latest_prices:
        Series of the most recent close price per ticker.
    valid_tickers:
        Tickers that returned usable data.
    failed_tickers:
        Requested tickers that were dropped (no data / all NaN).
    start, end:
        The actual date bounds of the returned series.
    """

    prices: pd.DataFrame
    latest_prices: pd.Series
    valid_tickers: list[str]
    failed_tickers: list[str]
    start: dt.date
    end: dt.date


def _normalise_tickers(tickers) -> list[str]:
    """Upper-case, strip and de-duplicate a list of ticker symbols."""
    seen: dict[str, None] = {}
    for raw in tickers:
        if raw is None:
            continue
        symbol = str(raw).strip().upper()
        if symbol:
            seen.setdefault(symbol, None)
    return list(seen.keys())


def _extract_close(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Pull the close-price frame out of a yfinance download.

    yfinance returns a MultiIndex column frame for multiple tickers and a flat
    frame for a single ticker, so both shapes are handled here.
    """
    if raw is None or raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        # auto_adjust=True puts adjusted values under the "Close" level.
        field = "Close" if "Close" in raw.columns.get_level_values(0) else raw.columns.levels[0][0]
        close = raw[field].copy()
    else:
        field = "Close" if "Close" in raw.columns else raw.columns[0]
        close = raw[[field]].copy()
        close.columns = [tickers[0]]
    return close


def fetch_price_history(
    tickers,
    lookback_years: float = DEFAULT_LOOKBACK_YEARS,
    end: dt.date | None = None,
) -> PriceData:
    """Download and clean daily close prices for ``tickers``.

    Parameters
    ----------
    tickers:
        Iterable of ticker symbols (case-insensitive).
    lookback_years:
        Size of the historical window in years.
    end:
        Optional end date (defaults to today).

    Returns
    -------
    PriceData

    Raises
    ------
    PriceDataError
        If no tickers are supplied or none return usable data.
    """
    symbols = _normalise_tickers(tickers)
    if not symbols:
        raise PriceDataError("No valid tickers were provided.")

    end = end or dt.date.today()
    start = end - dt.timedelta(days=int(round(lookback_years * 365.25)))

    raw = yf.download(
        symbols,
        start=start.isoformat(),
        end=end.isoformat(),
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    close = _extract_close(raw, symbols)
    if close.empty:
        raise PriceDataError(
            "Could not download any price data. Check the ticker symbols "
            "and your network connection."
        )

    # Forward-fill small gaps, then drop tickers/rows that are still empty.
    close = close.sort_index().ffill()
    close = close.dropna(axis=1, how="all")
    valid = [t for t in symbols if t in close.columns]
    failed = [t for t in symbols if t not in valid]

    # Keep only rows where every remaining asset has a price so the
    # covariance estimate is computed on a common, aligned window.
    prices = close[valid].dropna(how="any")
    if prices.empty or prices.shape[0] < 2:
        raise PriceDataError(
            "Not enough overlapping price history to compute statistics. "
            "Try a longer lookback or different tickers."
        )

    return PriceData(
        prices=prices,
        latest_prices=prices.iloc[-1],
        valid_tickers=valid,
        failed_tickers=failed,
        start=prices.index[0].date(),
        end=prices.index[-1].date(),
    )
