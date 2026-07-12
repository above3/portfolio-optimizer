"""Offline sample-data generator and CSV loader.

Live prices come from Yahoo Finance via :mod:`src.data_loader`, but that
requires outbound network access to Yahoo. To keep the app demonstrable when
the network is unavailable (corporate egress policies, yfinance rate limits,
air-gapped demos), this module can:

* synthesise a realistic, correlated multi-asset price history, and
* load a user-supplied CSV of close prices.

The synthetic series uses a shared market factor plus per-sector factors so the
correlation structure (and therefore the diversification benefit) is realistic,
with per-asset drift and volatility calibrated to plausible annual figures.
"""

from __future__ import annotations

import datetime as dt
import hashlib
from io import StringIO

import numpy as np
import pandas as pd

from .config import DEFAULT_LOOKBACK_YEARS, TRADING_DAYS_PER_YEAR, UNKNOWN_SECTOR
from .data_loader import PriceData, PriceDataError

# Human-readable sector labels for the internal factor-model sector codes.
_SECTOR_LABELS: dict[str, str] = {
    "tech": "Technology",
    "health": "Healthcare",
    "financial": "Financial Services",
    "energy": "Energy",
    "consumer": "Consumer Defensive",
    "auto": "Automotive",
    "market": "Diversified ETF",
    "bond": "Fixed Income",
    "gold": "Commodities",
    "industrial": "Industrials",
}

# Native currency inferred from the Yahoo exchange suffix (offline heuristic).
_SUFFIX_CURRENCY: dict[str, str] = {
    ".DE": "EUR", ".MI": "EUR", ".AS": "EUR", ".PA": "EUR", ".BD": "EUR",
    ".L": "GBP", ".SW": "CHF", ".TO": "CAD", ".T": "JPY", ".HK": "HKD",
}

# Deterministic synthetic FX factors (base per 1 unit of source) for the
# offline path, so currency normalisation can be demonstrated without network.
_SYNTH_FX: dict[tuple[str, str], float] = {
    ("USD", "EUR"): 0.92, ("EUR", "USD"): 1.087,
    ("GBP", "USD"): 1.27, ("USD", "GBP"): 0.787,
    ("GBP", "EUR"): 1.17, ("EUR", "GBP"): 0.855,
    ("CHF", "USD"): 1.12, ("USD", "CHF"): 0.893,
    ("CHF", "EUR"): 1.03, ("EUR", "CHF"): 0.971,
}


def _native_currency(ticker: str) -> str:
    """Infer an asset's native currency from its Yahoo exchange suffix."""
    upper = ticker.upper()
    for suffix, ccy in _SUFFIX_CURRENCY.items():
        if upper.endswith(suffix):
            return ccy
    return "USD"


def _synthetic_fx_factor(from_ccy: str, to_ccy: str) -> float:
    """Constant offline FX factor (``to_ccy`` per 1 ``from_ccy``)."""
    if from_ccy == to_ccy:
        return 1.0
    return _SYNTH_FX.get((from_ccy, to_ccy), 1.0)


def sample_metadata(tickers, base_currency: str | None = None) -> dict[str, dict[str, str]]:
    """Offline ``{ticker: {"currency", "sector"}}`` mirroring fetch_asset_metadata."""
    meta: dict[str, dict[str, str]] = {}
    for raw in tickers:
        ticker = str(raw).strip().upper()
        if not ticker:
            continue
        _, _, sector_code = _profile_for(ticker)
        meta[ticker] = {
            "currency": base_currency.upper() if base_currency else _native_currency(ticker),
            "sector": _SECTOR_LABELS.get(sector_code, UNKNOWN_SECTOR),
        }
    return meta

# (annual_return, annual_volatility, sector) for a set of well-known tickers.
_KNOWN_PROFILES: dict[str, tuple[float, float, str]] = {
    "AAPL": (0.22, 0.30, "tech"),
    "MSFT": (0.20, 0.28, "tech"),
    "GOOGL": (0.18, 0.30, "tech"),
    "AMZN": (0.20, 0.34, "tech"),
    "META": (0.19, 0.36, "tech"),
    "NVDA": (0.35, 0.45, "tech"),
    "JNJ": (0.08, 0.16, "health"),
    "PFE": (0.06, 0.20, "health"),
    "UNH": (0.13, 0.22, "health"),
    "JPM": (0.12, 0.24, "financial"),
    "BAC": (0.11, 0.27, "financial"),
    "GS": (0.12, 0.26, "financial"),
    "XOM": (0.09, 0.24, "energy"),
    "CVX": (0.09, 0.23, "energy"),
    "KO": (0.07, 0.14, "consumer"),
    "PG": (0.08, 0.15, "consumer"),
    "WMT": (0.10, 0.18, "consumer"),
    "TSLA": (0.28, 0.55, "auto"),
    "SPY": (0.10, 0.16, "market"),
    "QQQ": (0.13, 0.20, "market"),
    # All-world / broad-market ETFs (the "global anchor").
    "VWCE.DE": (0.09, 0.15, "market"),
    "IWDA.AS": (0.09, 0.15, "market"),
    "ACWI": (0.09, 0.16, "market"),
    # The Italian demo portfolio: UCITS ETFs, quality names, gold ETC, spec.
    "EIMI.MI": (0.07, 0.18, "market"),      # iShares Core EM IMI (Borsa Italiana)
    "DFND.MI": (0.14, 0.24, "industrial"),  # VanEck Defense UCITS
    "XDWH.MI": (0.10, 0.14, "health"),      # Xtrackers MSCI World Health Care
    "UNA.AS": (0.07, 0.16, "consumer"),     # Unilever (Amsterdam, EUR)
    "ABT": (0.10, 0.18, "health"),          # Abbott
    "SGLD.MI": (0.06, 0.14, "gold"),        # Invesco Physical Gold ETC
    "NKE": (0.08, 0.26, "consumer"),        # Nike (concentrated spec position)
    "SOFI": (0.15, 0.55, "financial"),      # SoFi (high-vol speculative)
    "MC.PA": (0.11, 0.27, "consumer"),      # LVMH (Paris, EUR)
    "NVO": (0.12, 0.24, "health"),          # Novo Nordisk ADR
    # Bonds / bond ETFs — low return, low volatility, weakly correlated.
    "AGGH.MI": (0.030, 0.05, "bond"),
    "EMBE.MI": (0.045, 0.09, "bond"),
    "AGG": (0.030, 0.05, "bond"),
    "TLT": (0.035, 0.14, "bond"),
    # Commodities / gold — a classic diversifier.
    "GLD": (0.06, 0.15, "gold"),
    "LVMH": (0.11, 0.27, "consumer"),
}

# Factor loadings: how much of each asset's return is market vs sector vs idio.
_RHO_MARKET = 0.5
_RHO_SECTOR = 0.35
_RHO_IDIO = float(np.sqrt(max(1.0 - _RHO_MARKET**2 - _RHO_SECTOR**2, 0.0)))


def _profile_for(ticker: str) -> tuple[float, float, str]:
    """Return a (return, vol, sector) profile, deriving a stable one for
    unknown tickers from a hash of the symbol so results are reproducible."""
    if ticker in _KNOWN_PROFILES:
        return _KNOWN_PROFILES[ticker]
    digest = int(hashlib.sha256(ticker.encode()).hexdigest(), 16)
    annual_return = 0.04 + (digest % 1000) / 1000 * 0.20          # 4%–24%
    annual_vol = 0.15 + ((digest // 1000) % 1000) / 1000 * 0.30   # 15%–45%
    sector = f"sector_{digest % 6}"
    return annual_return, annual_vol, sector


def generate_sample_prices(
    tickers,
    lookback_years: float = DEFAULT_LOOKBACK_YEARS,
    end: dt.date | None = None,
    seed: int = 27,
    benchmark: str | None = None,
    base_currency: str | None = None,
) -> PriceData:
    """Synthesise a correlated daily close-price panel for ``tickers``.

    The output mirrors :class:`~src.data_loader.PriceData` so it is a drop-in
    replacement for the live download path. When ``benchmark`` is supplied it is
    generated with the same factor model and returned separately. When
    ``base_currency`` is given, foreign-currency assets are scaled by a constant
    synthetic FX factor so the normalisation pipeline is exercised offline.
    """
    symbols = [str(t).strip().upper() for t in tickers if str(t).strip()]
    symbols = list(dict.fromkeys(symbols))  # de-dupe, preserve order
    if not symbols:
        raise PriceDataError("No valid tickers were provided.")

    bench = str(benchmark).strip().upper() if benchmark else None
    all_symbols = list(symbols)
    if bench and bench not in all_symbols:
        all_symbols.append(bench)

    end = end or dt.date.today()
    n_days = max(int(round(lookback_years * TRADING_DAYS_PER_YEAR)), 60)
    dates = pd.bdate_range(end=end, periods=n_days)

    rng = np.random.default_rng(seed)
    market = rng.standard_normal(n_days)

    # One shared factor per distinct sector among the requested tickers.
    sectors = {sec for _, _, sec in (_profile_for(s) for s in all_symbols)}
    sector_factors = {sec: rng.standard_normal(n_days) for sec in sectors}

    sqrt_days = float(np.sqrt(TRADING_DAYS_PER_YEAR))
    columns: dict[str, np.ndarray] = {}
    for symbol in all_symbols:
        annual_return, annual_vol, sector = _profile_for(symbol)
        mu_daily = np.log1p(annual_return) / TRADING_DAYS_PER_YEAR
        sigma_daily = annual_vol / sqrt_days
        idio = rng.standard_normal(n_days)
        shock = (
            _RHO_MARKET * market
            + _RHO_SECTOR * sector_factors[sector]
            + _RHO_IDIO * idio
        )
        log_rets = mu_daily + sigma_daily * shock
        columns[symbol] = 100.0 * np.exp(np.cumsum(log_rets))

    panel = pd.DataFrame(columns, index=dates)

    # ---- Metadata + offline currency normalisation ----
    meta = sample_metadata(all_symbols)
    native_ccy = {t: meta[t]["currency"] for t in all_symbols}
    base = base_currency.upper() if base_currency else None
    if base:
        for symbol in all_symbols:
            factor = _synthetic_fx_factor(native_ccy[symbol], base)
            if factor != 1.0:
                panel[symbol] = panel[symbol] * factor
        currencies = {t: base for t in symbols}
    else:
        currencies = {t: native_ccy[t] for t in symbols}

    prices = panel[symbols]
    benchmark_prices = panel[bench] if bench else None
    sectors = {t: meta[t]["sector"] for t in symbols}

    return PriceData(
        prices=prices,
        latest_prices=prices.iloc[-1],
        valid_tickers=symbols,
        failed_tickers=[],
        start=prices.index[0].date(),
        end=prices.index[-1].date(),
        benchmark_ticker=bench,
        benchmark_prices=benchmark_prices,
        currencies=currencies,
        sectors=sectors,
        base_currency=base_currency,
        notes=[],
    )


def load_prices_from_csv(file, tickers=None, benchmark: str | None = None,
                         base_currency: str | None = None) -> PriceData:
    """Load a close-price panel from an uploaded CSV.

    The CSV must have a date column (first column or named ``Date``) and one
    column of close prices per ticker. If ``tickers`` is given, only those
    columns are kept (others are reported as failed). If ``benchmark`` is given
    and present as a column, it is returned separately.
    """
    if hasattr(file, "read"):
        raw = file.read()
        buffer = StringIO(raw.decode() if isinstance(raw, (bytes, bytearray)) else raw)
    else:
        buffer = file

    df = pd.read_csv(buffer)
    if df.empty:
        raise PriceDataError("The uploaded CSV is empty.")

    # Identify the date column.
    date_col = "Date" if "Date" in df.columns else df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()

    # Keep numeric price columns only.
    prices = df.apply(pd.to_numeric, errors="coerce")
    prices = prices.dropna(axis=1, how="all").ffill()
    prices.columns = [str(c).strip().upper() for c in prices.columns]

    bench = str(benchmark).strip().upper() if benchmark else None

    failed: list[str] = []
    if tickers:
        requested = [str(t).strip().upper() for t in tickers if str(t).strip()]
        available = [t for t in requested if t in prices.columns]
        failed = [t for t in requested if t not in prices.columns]
        keep = available + ([bench] if bench and bench in prices.columns else [])
        if keep:
            prices = prices[list(dict.fromkeys(keep))]

    prices = prices.dropna(how="any")

    # Split out the benchmark column if present.
    benchmark_prices = None
    if bench and bench in prices.columns:
        benchmark_prices = prices[bench]
        asset_cols = [c for c in prices.columns if c != bench or (tickers and bench in
                      [str(t).strip().upper() for t in tickers])]
        prices = prices[asset_cols]

    if prices.shape[0] < 2 or prices.shape[1] < 1:
        raise PriceDataError(
            "Not enough usable price data in the CSV (need a date column and at "
            "least one fully-populated price column)."
        )

    # CSV prices are assumed already in the base currency; sectors are inferred
    # from the offline metadata heuristic so the guardrails still function.
    asset_meta = sample_metadata(list(prices.columns), base_currency=base_currency)
    return PriceData(
        prices=prices,
        latest_prices=prices.iloc[-1],
        valid_tickers=list(prices.columns),
        failed_tickers=failed,
        start=prices.index[0].date(),
        end=prices.index[-1].date(),
        benchmark_ticker=bench if benchmark_prices is not None else None,
        benchmark_prices=benchmark_prices,
        currencies={t: (base_currency or asset_meta[t]["currency"])
                    for t in prices.columns},
        sectors={t: asset_meta[t]["sector"] for t in prices.columns},
        base_currency=base_currency,
        notes=[],
    )
