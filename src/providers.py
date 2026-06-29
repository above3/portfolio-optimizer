"""Abstract data-provider framework.

A :class:`BaseDataProvider` defines a single, narrow interface for asset
tracking. Two concrete engines implement it:

* :class:`YahooFinanceProvider` — the robust, decoupled / offline-safe engine
  built on ``yfinance`` (the long-standing default).
* :class:`IBKRNativeProvider` — a live engine that talks to a local Interactive
  Brokers TWS / IB Gateway socket via ``ib_insync``, ingesting the account's
  cash, base currency and positions, and pulling daily bars directly from IBKR
  with a *per-ticker* graceful fallback to Yahoo when a market-data subscription
  is missing.

``ib_insync`` is imported lazily inside the IBKR methods, so this module (and
the whole app) imports and unit-tests cleanly without it installed. Tests mock
:class:`IBKRNativeProvider`; its ``connect`` additionally degrades to a clean
:class:`ProviderError` when ``ib_insync`` is absent.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import pandas as pd

from .data_loader import (
    PriceData,
    build_price_data,
    fetch_asset_metadata,
    fetch_close_prices,
    fetch_price_history,
)
from .portfolio import guess_asset_type, ibkr_exchange_for, map_ibkr_symbol

# Standard native sockets: TWS on 7496, IB Gateway on 4001 (localhost).
DEFAULT_IBKR_HOST: str = "127.0.0.1"
IBKR_TWS_PORT: int = 7496
IBKR_GATEWAY_PORT: int = 4001
DEFAULT_IBKR_CLIENT_ID: int = 17


class ProviderError(RuntimeError):
    """Raised when a data provider cannot fulfil a request (connection, etc.)."""


@dataclass
class Position:
    """A single live broker position."""

    symbol: str
    shares: float
    avg_cost: float = 0.0
    currency: str = "USD"
    sec_type: str = "STK"

    @property
    def asset_type(self) -> str:
        """Map an IB security type onto the app's Asset Type vocabulary."""
        if str(self.sec_type).upper() in {"BOND", "BILL", "FIXED"}:
            return "Bond/Fixed Income"
        return guess_asset_type(self.symbol)


@dataclass
class AccountSnapshot:
    """Live account summary ingested from the broker."""

    account_id: str
    base_currency: str
    cash: float
    positions: list[Position] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Abstract interface
# --------------------------------------------------------------------------- #
class BaseDataProvider(ABC):
    """Common interface every data architecture engine must implement."""

    name: str = "base"
    supports_account_ingestion: bool = False

    @abstractmethod
    def fetch_historical_prices(
        self,
        tickers,
        lookback_years: float = 5,
        benchmark: str | None = None,
        base_currency: str | None = None,
    ) -> PriceData:
        """Return a clean daily price panel normalised in ``base_currency``."""

    @abstractmethod
    def fetch_asset_metadata(self, tickers) -> dict[str, dict[str, str]]:
        """Return ``{ticker: {"currency", "sector", "asset_type"}}``."""


# --------------------------------------------------------------------------- #
# Yahoo Finance engine (decoupled / offline-safe default)
# --------------------------------------------------------------------------- #
class YahooFinanceProvider(BaseDataProvider):
    """Robust fallback engine built on ``yfinance`` (see :mod:`src.data_loader`)."""

    name = "Yahoo Finance"
    supports_account_ingestion = False

    def fetch_historical_prices(
        self, tickers, lookback_years: float = 5,
        benchmark: str | None = None, base_currency: str | None = None,
    ) -> PriceData:
        return fetch_price_history(
            tickers, lookback_years=lookback_years,
            benchmark=benchmark, base_currency=base_currency)

    def fetch_asset_metadata(self, tickers) -> dict[str, dict[str, str]]:
        meta = fetch_asset_metadata(tickers)
        for ticker, info in meta.items():
            info.setdefault("asset_type", guess_asset_type(ticker))
        return meta


# --------------------------------------------------------------------------- #
# Interactive Brokers native engine (ib_insync)
# --------------------------------------------------------------------------- #
class IBKRNativeProvider(BaseDataProvider):
    """Live engine talking to a local TWS / IB Gateway socket via ``ib_insync``.

    Network access (and ``ib_insync``) is required only at call time; the class
    imports cleanly without either. A :class:`YahooFinanceProvider` is held as a
    per-ticker fallback for assets lacking an IBKR market-data subscription.
    """

    name = "Interactive Brokers Live API"
    supports_account_ingestion = True

    def __init__(
        self,
        host: str = DEFAULT_IBKR_HOST,
        port: int = IBKR_TWS_PORT,
        client_id: int = DEFAULT_IBKR_CLIENT_ID,
        timeout: float = 6.0,
        fallback: BaseDataProvider | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.client_id = client_id
        self.timeout = timeout
        self.fallback = fallback or YahooFinanceProvider()
        self._ib = None
        self.account_id: str | None = None

    # -- connection lifecycle --------------------------------------------- #
    @staticmethod
    def _import_ib():
        try:
            from ib_insync import IB, util  # type: ignore
        except ImportError as exc:  # pragma: no cover - exercised when missing
            raise ProviderError(
                "ib_insync is not installed. Run `pip install ib_insync` to "
                "enable the Interactive Brokers Live API engine."
            ) from exc
        return IB, util

    def connect(self) -> "IBKRNativeProvider":
        """Open a socket to TWS/Gateway. Raises :class:`ProviderError` on failure.

        ``util.startLoop()`` keeps the ib_insync asyncio loop thread-safe and
        isolated from Streamlit's main execution thread.
        """
        IB, util = self._import_ib()
        try:
            util.startLoop()
        except Exception:  # noqa: BLE001 - loop already running / not applicable
            pass
        ib = IB()
        try:
            ib.connect(self.host, self.port, clientId=self.client_id, timeout=self.timeout)
        except Exception as exc:  # noqa: BLE001 - socket refused / TWS closed
            raise ProviderError(
                f"Could not connect to TWS/Gateway on {self.host}:{self.port} "
                f"({type(exc).__name__}: {exc})."
            ) from exc
        self._ib = ib
        try:
            accounts = ib.managedAccounts()
            self.account_id = accounts[0] if accounts else "UNKNOWN"
        except Exception:  # noqa: BLE001
            self.account_id = "UNKNOWN"
        return self

    def is_connected(self) -> bool:
        return self._ib is not None and bool(getattr(self._ib, "isConnected", lambda: False)())

    def disconnect(self) -> None:
        try:
            if self._ib is not None:
                self._ib.disconnect()
        except Exception:  # noqa: BLE001
            pass
        self._ib = None

    # -- account ingestion ------------------------------------------------ #
    def fetch_account_snapshot(self) -> AccountSnapshot:
        """Download live cash, base currency and the positions matrix."""
        if not self.is_connected():
            raise ProviderError("Not connected to IBKR; call connect() first.")
        ib = self._ib

        base_currency, cash = "USD", 0.0
        try:
            for row in ib.accountSummary():
                if row.tag == "TotalCashValue":
                    cash = float(row.value)
                    if row.currency:
                        base_currency = row.currency
                elif row.tag in ("AvailableFunds",) and not cash:
                    cash = float(row.value)
        except Exception:  # noqa: BLE001 - degrade to zero cash
            pass

        positions: list[Position] = []
        try:
            for pos in ib.positions():
                contract = pos.contract
                positions.append(Position(
                    symbol=str(getattr(contract, "symbol", "")).upper(),
                    shares=float(getattr(pos, "position", 0.0) or 0.0),
                    avg_cost=float(getattr(pos, "avgCost", 0.0) or 0.0),
                    currency=str(getattr(contract, "currency", "USD") or "USD"),
                    sec_type=str(getattr(contract, "secType", "STK") or "STK"),
                ))
        except Exception:  # noqa: BLE001
            pass

        return AccountSnapshot(account_id=self.account_id or "UNKNOWN",
                               base_currency=base_currency, cash=cash,
                               positions=positions)

    # -- price history (IBKR bars + per-ticker Yahoo fallback) ------------ #
    def _request_bars(self, ticker: str, lookback_years: float,
                      currency: str) -> pd.Series | None:
        from ib_insync import Stock, util  # type: ignore
        exchange = ibkr_exchange_for(ticker)
        contract = Stock(map_ibkr_symbol(ticker),
                         exchange if exchange != "SMART" else "SMART", currency)
        try:
            self._ib.qualifyContracts(contract)
            years = max(1, min(int(round(lookback_years)), 15))
            bars = self._ib.reqHistoricalData(
                contract, endDateTime="", durationStr=f"{years} Y",
                barSizeSetting="1 day", whatToShow="ADJUSTED_LAST",
                useRTH=True, formatDate=1)
            if not bars:
                return None
            df = util.df(bars)
            series = pd.Series(df["close"].to_numpy(),
                               index=pd.to_datetime(df["date"]), name=ticker)
            return series.dropna()
        except Exception:  # noqa: BLE001 - subscription / contract error
            return None

    def fetch_historical_prices(
        self, tickers, lookback_years: float = 5,
        benchmark: str | None = None, base_currency: str | None = None,
    ) -> PriceData:
        symbols = [str(t).strip().upper() for t in tickers if str(t).strip()]
        symbols = list(dict.fromkeys(symbols))
        bench = str(benchmark).strip().upper() if benchmark else None
        wanted = symbols + ([bench] if bench and bench not in symbols else [])
        if not self.is_connected() or not wanted:
            return self.fallback.fetch_historical_prices(
                tickers, lookback_years, benchmark, base_currency)

        currencies: dict[str, str] = {}
        series_map: dict[str, pd.Series] = {}
        missing: list[str] = []
        for ticker in wanted:
            series = self._request_bars(ticker, lookback_years, "USD")
            if series is None or series.empty:
                missing.append(ticker)
            else:
                series_map[ticker] = series
                currencies[ticker] = "USD"

        # Route subscription gaps through the Yahoo fallback (per ticker).
        if missing:
            yahoo = fetch_close_prices(missing, lookback_years)
            for ticker in missing:
                if ticker in getattr(yahoo, "columns", []):
                    series_map[ticker] = yahoo[ticker].dropna()
                    currencies[ticker] = "USD"

        if not series_map:
            return self.fallback.fetch_historical_prices(
                tickers, lookback_years, benchmark, base_currency)

        close = pd.DataFrame(series_map).sort_index()
        # Sectors are not exposed by IBKR — borrow them from the Yahoo metadata.
        try:
            sectors = {t: info.get("sector", "Unknown")
                       for t, info in self.fallback.fetch_asset_metadata(symbols).items()}
        except Exception:  # noqa: BLE001
            sectors = {t: "Unknown" for t in symbols}

        return build_price_data(
            close, symbols, bench, base_currency,
            currencies=currencies, sectors=sectors)

    def fetch_asset_metadata(self, tickers) -> dict[str, dict[str, str]]:
        # Sector/asset-type come from the Yahoo fallback; currency from IBKR.
        return self.fallback.fetch_asset_metadata(tickers)


def get_provider(architecture: str, **kwargs) -> BaseDataProvider:
    """Factory: build a provider from the sidebar architecture label."""
    if "Interactive Brokers" in architecture or architecture.upper() == "IBKR":
        return IBKRNativeProvider(**kwargs)
    return YahooFinanceProvider()
