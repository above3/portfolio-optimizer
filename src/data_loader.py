"""Historical price downloading and cleaning via yfinance.

The functions here are deliberately framework-agnostic (no Streamlit imports) so
the financial engine can be unit-tested or reused outside the web app.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import pandas as pd
import yfinance as yf

from .config import DEFAULT_LOOKBACK_YEARS, RISK_FREE_TICKER, UNKNOWN_SECTOR


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
    benchmark_ticker:
        Symbol of the market benchmark, if one was requested.
    benchmark_prices:
        Close-price series of the benchmark aligned to ``prices`` index, or
        ``None`` when no benchmark was requested / available.
    currencies:
        Native currency per ticker (after any normalisation this equals the
        base currency for converted assets).
    sectors:
        Sector group per ticker (``"Unknown"`` when unavailable).
    base_currency:
        The currency all prices/values are expressed in, or ``None``.
    notes:
        Human-readable notes about fallbacks (failed FX / metadata fetches).
    """

    prices: pd.DataFrame
    latest_prices: pd.Series
    valid_tickers: list[str]
    failed_tickers: list[str]
    start: dt.date
    end: dt.date
    benchmark_ticker: str | None = None
    benchmark_prices: pd.Series | None = None
    currencies: dict[str, str] = field(default_factory=dict)
    sectors: dict[str, str] = field(default_factory=dict)
    base_currency: str | None = None
    notes: list[str] = field(default_factory=list)


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


def fetch_risk_free_rate(
    ticker: str = RISK_FREE_TICKER,
    lookback_days: int = 10,
) -> float:
    """Fetch the latest US 3-Month Treasury Bill yield as a decimal.

    Yahoo Finance quotes ``^IRX`` as an annualised percentage (e.g. ``5.21``
    means 5.21%), so the value is divided by 100 to return a decimal rate
    suitable for the tangency-portfolio / CML calculations.

    Raises
    ------
    PriceDataError
        If the yield cannot be downloaded or parsed.
    """
    end = dt.date.today()
    start = end - dt.timedelta(days=max(lookback_days, 5))
    try:
        raw = yf.download(
            ticker,
            start=start.isoformat(),
            end=end.isoformat(),
            auto_adjust=False,
            progress=False,
            threads=False,
        )
    except Exception as exc:  # noqa: BLE001 - network/library errors
        raise PriceDataError(f"Could not fetch risk-free rate ({ticker}): {exc}")

    close = _extract_close(raw, [ticker])
    series = close.dropna(how="all").ffill().dropna(how="any")
    if series.empty:
        raise PriceDataError(f"No data returned for risk-free ticker {ticker}.")

    latest_yield_pct = float(series.iloc[-1, 0])
    if not (-1.0 < latest_yield_pct < 100.0):
        raise PriceDataError(f"Implausible risk-free yield parsed: {latest_yield_pct}.")
    return latest_yield_pct / 100.0


# --------------------------------------------------------------------------- #
# Asset metadata (currency + sector) and FX normalisation
# --------------------------------------------------------------------------- #
def fetch_asset_metadata(
    tickers,
    default_currency: str = "USD",
) -> dict[str, dict[str, str]]:
    """Best-effort ``{ticker: {"currency": .., "sector": ..}}`` via yfinance.

    Every lookup is wrapped in try/except so a blocked network or a missing
    ``.info`` payload degrades to ``(default_currency, "Unknown")`` instead of
    raising.
    """
    meta: dict[str, dict[str, str]] = {}
    for ticker in _normalise_tickers(tickers):
        currency, sector = default_currency, UNKNOWN_SECTOR
        try:
            info = yf.Ticker(ticker).info or {}
            currency = str(info.get("currency", default_currency) or default_currency).upper()
            sector = str(info.get("sector", UNKNOWN_SECTOR) or UNKNOWN_SECTOR)
        except Exception:  # noqa: BLE001 - degrade gracefully
            pass
        meta[ticker] = {"currency": currency, "sector": sector}
    return meta


def fx_pair_ticker(from_ccy: str, to_ccy: str) -> str:
    """Yahoo FX symbol giving ``to_ccy`` per 1 ``from_ccy`` (e.g. ``USDEUR=X``)."""
    return f"{from_ccy.upper()}{to_ccy.upper()}=X"


def fetch_fx_series(
    from_ccy: str,
    to_ccy: str,
    start: dt.date,
    end: dt.date,
) -> pd.Series:
    """Daily FX conversion factor series (``to_ccy`` per 1 ``from_ccy``).

    Raises ``PriceDataError`` on failure so callers can fall back gracefully.
    """
    if from_ccy.upper() == to_ccy.upper():
        raise PriceDataError("No FX conversion needed for identical currencies.")
    pair = fx_pair_ticker(from_ccy, to_ccy)
    try:
        raw = yf.download(pair, start=start.isoformat(), end=end.isoformat(),
                          auto_adjust=True, progress=False, threads=False)
    except Exception as exc:  # noqa: BLE001
        raise PriceDataError(f"Could not fetch FX rate {pair}: {exc}")
    close = _extract_close(raw, [pair]).dropna(how="all").ffill().dropna(how="any")
    if close.empty:
        raise PriceDataError(f"No FX data returned for {pair}.")
    series = close.iloc[:, 0]
    series.name = pair
    return series


def convert_to_base_currency(
    prices: pd.DataFrame,
    currency_map: dict[str, str],
    base_currency: str,
    fx_series_map: dict[str, pd.Series],
) -> pd.DataFrame:
    """Convert every column of ``prices`` into ``base_currency``.

    Pure function (no network): ``fx_series_map`` maps a *source* currency to a
    factor series giving ``base`` per 1 unit of that currency. Columns already
    in the base currency (or without a usable factor) are left unchanged.
    """
    base = base_currency.upper()
    converted = prices.copy()
    for ticker in prices.columns:
        ccy = str(currency_map.get(ticker, base)).upper()
        if ccy == base or ccy not in fx_series_map:
            continue
        factor = fx_series_map[ccy].reindex(prices.index).ffill().bfill()
        converted[ticker] = prices[ticker].to_numpy() * factor.to_numpy()
    return converted


def fetch_price_history(
    tickers,
    lookback_years: float = DEFAULT_LOOKBACK_YEARS,
    end: dt.date | None = None,
    benchmark: str | None = None,
    base_currency: str | None = None,
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
    benchmark:
        Optional market-benchmark symbol fetched alongside the assets and
        returned separately (aligned to the asset window).
    base_currency:
        If set (``"EUR"``/``"USD"``), every asset's native currency is detected
        and foreign series are converted into this base via live FX rates, with
        graceful fallback (left native + a note) when FX can't be fetched.

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

    bench = str(benchmark).strip().upper() if benchmark else None
    download_symbols = list(symbols)
    if bench and bench not in download_symbols:
        download_symbols.append(bench)

    end = end or dt.date.today()
    start = end - dt.timedelta(days=int(round(lookback_years * 365.25)))

    raw = yf.download(
        download_symbols,
        start=start.isoformat(),
        end=end.isoformat(),
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    close = _extract_close(raw, download_symbols)
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

    # Align the benchmark (if available) to the cleaned asset window.
    benchmark_prices = None
    if bench and bench in close.columns:
        benchmark_prices = close[bench].reindex(prices.index).ffill().dropna()
        if benchmark_prices.empty:
            benchmark_prices = None

    # ---- Metadata (currency + sector) and FX normalisation ----
    notes: list[str] = []
    meta_symbols = list(valid) + ([bench] if benchmark_prices is not None else [])
    metadata = fetch_asset_metadata(meta_symbols, default_currency=base_currency or "USD")
    currencies = {t: metadata.get(t, {}).get("currency", base_currency or "USD")
                  for t in meta_symbols}
    sectors = {t: metadata.get(t, {}).get("sector", UNKNOWN_SECTOR) for t in valid}

    if base_currency:
        base = base_currency.upper()
        foreign = {c for c in currencies.values() if c and c != base}
        fx_series_map: dict[str, pd.Series] = {}
        for ccy in foreign:
            try:
                fx_series_map[ccy] = fetch_fx_series(ccy, base, start, end)
            except PriceDataError as exc:
                notes.append(f"FX {ccy}->{base} unavailable; left in native currency. ({exc})")
        if fx_series_map:
            prices = convert_to_base_currency(prices, currencies, base, fx_series_map)
            if benchmark_prices is not None and bench in currencies:
                bench_frame = convert_to_base_currency(
                    benchmark_prices.to_frame(), {bench: currencies[bench]},
                    base, fx_series_map)
                benchmark_prices = bench_frame.iloc[:, 0]
        # After conversion, every asset is reported in the base currency.
        currencies = {t: base for t in currencies}

    return PriceData(
        prices=prices,
        latest_prices=prices.iloc[-1],
        valid_tickers=valid,
        failed_tickers=failed,
        start=prices.index[0].date(),
        end=prices.index[-1].date(),
        benchmark_ticker=bench if benchmark_prices is not None else None,
        benchmark_prices=benchmark_prices,
        currencies={t: currencies.get(t, base_currency or "USD") for t in valid},
        sectors=sectors,
        base_currency=base_currency,
        notes=notes,
    )
