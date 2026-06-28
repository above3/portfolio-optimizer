"""Helpers for the user's *current* portfolio and rebalancing.

This module:

* values a holdings table (ticker + shares) and scores it under MPT;
* parses Interactive Brokers (IBKR) position / activity CSV exports into a
  clean holdings table, so positions can be imported without manual entry;
* turns a set of target weights into a concrete reallocation plan, with optional
  whole-share rounding (residual cash) and per-trade transaction-cost friction.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    DEFAULT_ASSET_TYPE,
    TAX_RATE_BOND,
    TAX_RATE_EQUITY,
    UNKNOWN_SECTOR,
)
from .data_loader import PriceDataError
from .optimizer import performance

# Substrings hinting that a ticker is a bond / fixed-income instrument.
_BOND_HINTS: tuple[str, ...] = (
    "AGG", "BND", "BOND", "TLT", "IEF", "SHY", "GOVT", "EMB", "EMBE",
    "BTP", "BUND", "TIP", "VGEA", "IBGL", "XGLE", "GLAB",
)


def guess_asset_type(ticker: str) -> str:
    """Heuristically classify a ticker as a bond or an equity/ETF."""
    upper = str(ticker).upper()
    root = upper.split(".")[0]
    if any(hint in root for hint in _BOND_HINTS):
        return "Bond/Fixed Income"
    return DEFAULT_ASSET_TYPE


def tax_rate_for(asset_type: str | None) -> float:
    """Italian capital-gains rate for an asset type (26% equity, 12.5% bond)."""
    if asset_type and "bond" in str(asset_type).lower():
        return TAX_RATE_BOND
    return TAX_RATE_EQUITY


@dataclass
class CurrentPortfolio:
    """The user's existing allocation, valued at the latest close prices.

    When purchase prices are supplied the unrealised profit/loss is computed per
    asset and in aggregate (only over assets that have a purchase price).
    """

    weights: dict[str, float]
    values: dict[str, float]
    shares: dict[str, float]
    total_value: float
    expected_return: float
    volatility: float
    sharpe: float
    purchase_prices: dict[str, float]
    cost_basis: dict[str, float]
    unrealized_pnl: dict[str, float]
    total_cost: float
    total_pnl: float
    has_cost_data: bool


def build_current_portfolio(
    shares: dict[str, float],
    latest_prices: pd.Series,
    mu: pd.Series,
    cov: pd.DataFrame,
    risk_free_rate: float,
    purchase_prices: dict[str, float] | None = None,
) -> CurrentPortfolio | None:
    """Value a holdings dict and compute its MPT performance and P/L.

    Returns ``None`` when no positive holdings are supplied (e.g. the user is
    only drafting a brand-new portfolio).
    """
    purchase_prices = purchase_prices or {}
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

    cost_basis: dict[str, float] = {}
    unrealized_pnl: dict[str, float] = {}
    for ticker in values:
        pp = purchase_prices.get(ticker)
        if pp is None or float(pp) <= 0:
            continue
        qty = float(shares.get(ticker, 0.0) or 0.0)
        cost = float(pp) * qty
        cost_basis[ticker] = cost
        unrealized_pnl[ticker] = values[ticker] - cost

    total_cost = sum(cost_basis.values())
    total_pnl = sum(unrealized_pnl.values())

    return CurrentPortfolio(
        weights=weights,
        values=values,
        shares={t: float(shares.get(t, 0.0) or 0.0) for t in values},
        total_value=total,
        expected_return=ret,
        volatility=vol,
        sharpe=sharpe,
        purchase_prices={t: float(purchase_prices[t]) for t in cost_basis},
        cost_basis=cost_basis,
        unrealized_pnl=unrealized_pnl,
        total_cost=total_cost,
        total_pnl=total_pnl,
        has_cost_data=bool(cost_basis),
    )


# --------------------------------------------------------------------------- #
# IBKR position / activity CSV import
# --------------------------------------------------------------------------- #
_SYMBOL_KEYS = {
    "symbol", "ticker", "symbolticker", "underlyingsymbol",
    "financialinstrument", "instrument", "contract",
}
_QTY_KEYS = {
    "quantity", "shares", "position", "qty", "quantityshares", "positionqty",
}
_PRICE_KEYS = {
    "costprice", "purchaseprice", "avgprice", "averageprice", "averagecost",
    "avgcost", "tradeprice", "openprice", "price",
}
_COSTBASIS_KEYS = {"costbasis", "cost", "basis", "totalcost"}
_SKIP_SYMBOLS = {"total", "subtotal", "", "cash", "totals"}


def _norm(value) -> str:
    """Normalise a header cell to lowercase alphanumerics for matching."""
    return re.sub(r"[^a-z0-9]", "", str(value).strip().lower())


def _to_float(value) -> float | None:
    """Parse a possibly comma-formatted numeric cell, or return None."""
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _find_key_index(norm_cells: list[str], keys: set[str]) -> int | None:
    for idx, cell in enumerate(norm_cells):
        if cell in keys:
            return idx
    return None


def _lot_price(cells, price_idx: int | None, cb_idx: int | None,
               qty: float) -> float | None:
    """Per-share purchase price from a row: prefer an explicit price column,
    otherwise derive it from a (total) cost-basis column divided by quantity."""
    if price_idx is not None and price_idx < len(cells):
        price = _to_float(cells[price_idx])
        if price is not None and price > 0:
            return price
    if cb_idx is not None and cb_idx < len(cells) and qty:
        basis = _to_float(cells[cb_idx])
        if basis is not None and qty != 0:
            return abs(basis / qty)
    return None


def _aggregate_positions(
    lots: list[tuple[str, float, float | None]]
) -> pd.DataFrame:
    """Aggregate (symbol, qty, price) lots into a Ticker/Shares/Purchase Price
    frame, summing quantities and share-weighting the purchase price."""
    totals: dict[str, list[float]] = {}  # sym -> [qty, priced_value, priced_qty]
    for symbol, qty, price in lots:
        sym = str(symbol).strip().upper()
        if _norm(sym) in _SKIP_SYMBOLS:
            continue
        rec = totals.setdefault(sym, [0.0, 0.0, 0.0])
        rec[0] += qty
        if price is not None:
            rec[1] += abs(qty) * price
            rec[2] += abs(qty)

    rows = []
    for sym, (qty, priced_value, priced_qty) in totals.items():
        if qty == 0:
            continue
        avg_price = round(priced_value / priced_qty, 4) if priced_qty > 0 else None
        rows.append({
            "Ticker": sym,
            "Shares": qty,
            "Purchase Price": avg_price,
            "Asset Type": guess_asset_type(sym),
        })
    if not rows:
        raise PriceDataError("No non-zero positions found in the IBKR file.")
    return pd.DataFrame(rows)


def _positions_from_dataframe(df: pd.DataFrame) -> pd.DataFrame | None:
    """Extract positions from a flat, single-table CSV (e.g. a Flex Query)."""
    norm_cols = [_norm(c) for c in df.columns]
    sym_idx = _find_key_index(norm_cols, _SYMBOL_KEYS)
    qty_idx = _find_key_index(norm_cols, _QTY_KEYS)
    if sym_idx is None or qty_idx is None:
        return None
    price_idx = _find_key_index(norm_cols, _PRICE_KEYS)
    cb_idx = _find_key_index(norm_cols, _COSTBASIS_KEYS)

    lots: list[tuple[str, float, float | None]] = []
    for _, row in df.iterrows():
        cells = list(row)
        qty = _to_float(cells[qty_idx])
        sym = str(cells[sym_idx]).strip()
        if sym and qty is not None:
            lots.append((sym, qty, _lot_price(cells, price_idx, cb_idx, qty)))
    if not lots:
        return None
    return _aggregate_positions(lots)


def _positions_from_sections(text: str) -> pd.DataFrame:
    """Extract positions from a multi-section IBKR *activity statement*.

    These statements repeat a section label in column 0 and tag each row as a
    ``Header`` or ``Data`` row in column 1, e.g.::

        Positions,Header,...,Symbol,Quantity,...
        Positions,Data,...,AAPL,100,...
    """
    rows = list(csv.reader(io.StringIO(text)))
    for i, row in enumerate(rows):
        norm = [_norm(c) for c in row]
        if "header" not in norm:
            continue
        sym_idx = _find_key_index(norm, _SYMBOL_KEYS)
        qty_idx = _find_key_index(norm, _QTY_KEYS)
        if sym_idx is None or qty_idx is None:
            continue
        price_idx = _find_key_index(norm, _PRICE_KEYS)
        cb_idx = _find_key_index(norm, _COSTBASIS_KEYS)

        section = _norm(row[0])
        lots: list[tuple[str, float, float | None]] = []
        for data_row in rows[i + 1:]:
            if len(data_row) <= max(sym_idx, qty_idx):
                continue
            if _norm(data_row[0]) != section or _norm(data_row[1]) != "data":
                continue
            qty = _to_float(data_row[qty_idx])
            sym = str(data_row[sym_idx]).strip()
            if sym and qty is not None:
                lots.append((sym, qty, _lot_price(data_row, price_idx, cb_idx, qty)))
        if lots:
            return _aggregate_positions(lots)

    raise PriceDataError(
        "Could not locate 'Symbol'/'Ticker' and 'Quantity'/'Shares' columns in "
        "the uploaded IBKR CSV."
    )


def parse_ibkr_positions(file) -> pd.DataFrame:
    """Parse an IBKR CSV into a ``Ticker``/``Shares``/``Purchase Price`` frame.

    Handles both the simple flat Flex-Query export (a single table with
    ``Symbol`` and ``Quantity`` columns) and the multi-section activity
    statement. Duplicate lots for the same symbol are summed, and the purchase
    price (from a cost-price or cost-basis column, if present) is share-weighted.

    Raises
    ------
    PriceDataError
        If the file cannot be read or no positions can be found.
    """
    if hasattr(file, "read"):
        raw = file.read()
        text = raw.decode("utf-8", errors="replace") if isinstance(
            raw, (bytes, bytearray)) else str(raw)
    else:
        text = str(file)

    if not text.strip():
        raise PriceDataError("The uploaded IBKR CSV is empty.")

    # Strategy 1: flat single-table CSV.
    try:
        flat = pd.read_csv(io.StringIO(text))
        result = _positions_from_dataframe(flat)
        if result is not None and not result.empty:
            return result
    except Exception:  # noqa: BLE001 - fall through to the section scanner
        pass

    # Strategy 2: multi-section IBKR activity statement.
    return _positions_from_sections(text)


# --------------------------------------------------------------------------- #
# Reallocation plan (whole shares, residual cash, transaction costs)
# --------------------------------------------------------------------------- #
@dataclass
class ReallocationResult:
    """A reallocation plan plus the friction-aware summary metrics."""

    table: pd.DataFrame
    achieved_weights: dict[str, float]
    residual_cash: float
    total_fees: float
    total_tax: float
    total_friction: float
    num_trades: int
    gross_return: float
    net_return: float
    allow_fractional: bool
    transaction_cost: float
    tolerance: float
    investable_capital: float


_HOLD_TOLERANCE = "HOLD / Within Tolerance"


def reallocation_plan(
    current_values: dict[str, float],
    target_weights: dict[str, float],
    investable_capital: float,
    latest_prices: pd.Series,
    mu: pd.Series | None = None,
    risk_free_rate: float = 0.0,
    allow_fractional: bool = True,
    transaction_cost: float = 0.0,
    tolerance: float = 0.0,
    purchase_prices: dict[str, float] | None = None,
    asset_types: dict[str, str] | None = None,
) -> ReallocationResult:
    """Build a friction- and tax-aware reallocation plan.

    Parameters
    ----------
    current_values:
        Ticker -> current value held (empty for a new portfolio).
    target_weights:
        Target weights (e.g. the tangency portfolio).
    investable_capital:
        Capital the target weights apply to (active tranche capital).
    latest_prices:
        Latest close price per ticker, used to size trades in shares.
    mu:
        Optional expected returns; enables gross/net achieved-return metrics.
    allow_fractional:
        If ``False``, trades are truncated to whole shares (residual cash).
    transaction_cost:
        Flat cost per executed trade.
    tolerance:
        If ``|w_current - w_target| < tolerance`` the asset is flagged
        ``"HOLD / Within Tolerance"`` and not traded (avoids needless friction).
    purchase_prices:
        Average cost basis per ticker; drives realised-gain tax on SELLs.
    asset_types:
        Ticker -> 'Stock/ETF' | 'Bond/Fixed Income' for the Italian tax rate.
    """
    purchase_prices = purchase_prices or {}
    asset_types = asset_types or {}
    tickers = sorted(set(current_values) | set(target_weights))
    rows = []
    new_values: dict[str, float] = {}
    num_trades = 0
    total_tax = 0.0

    for ticker in tickers:
        cur_val = float(current_values.get(ticker, 0.0) or 0.0)
        tgt_w = float(target_weights.get(ticker, 0.0))
        cur_w = cur_val / investable_capital if investable_capital > 0 else 0.0
        delta_w = tgt_w - cur_w
        price = float(latest_prices.get(ticker, 0.0) or 0.0)

        within_tol = abs(delta_w) < tolerance
        if within_tol:
            traded_shares: float = 0.0
        else:
            raw_shares = (tgt_w * investable_capital - cur_val) / price if price > 0 \
                else float("nan")
            traded_shares = math.trunc(raw_shares) if (not allow_fractional and price > 0) \
                else raw_shares

        executed_value = traded_shares * price if price > 0 else 0.0
        new_value = cur_val + executed_value
        new_values[ticker] = new_value

        traded = (not within_tol) and price > 0 and abs(traded_shares) >= (
            1 if not allow_fractional else 1e-9)
        if traded:
            num_trades += 1

        # Capital-gains tax only on the SOLD portion of a profitable position.
        tax = 0.0
        if traded and traded_shares < 0:
            pp = purchase_prices.get(ticker)
            if pp and price > 0:
                realised_gain = abs(traded_shares) * (price - float(pp))
                tax = tax_rate_for(asset_types.get(ticker)) * max(realised_gain, 0.0)
                total_tax += tax

        if within_tol:
            action = _HOLD_TOLERANCE
        elif traded_shares > (0 if allow_fractional else 0.0) and traded:
            action = "BUY"
        elif traded_shares < 0 and traded:
            action = "SELL"
        else:
            action = "HOLD"

        rows.append({
            "Ticker": ticker,
            "Current %": round(cur_w * 100, 2),
            "Target %": round(tgt_w * 100, 2),
            "Δ Weight %": round(delta_w * 100, 2),
            "Price": round(price, 2) if price > 0 else None,
            "Shares to trade": (
                int(traded_shares) if (not allow_fractional and price > 0)
                else (round(traded_shares, 4) if price > 0 else None)),
            "Trade Value": round(executed_value, 2) if price > 0 else None,
            "Cap-Gains Tax": round(tax, 2),
            "Action": action,
        })

    table = pd.DataFrame(rows)

    deployed = sum(new_values.values())
    residual_cash = max(investable_capital - deployed, 0.0)
    total_fees = transaction_cost * num_trades
    total_friction = total_fees + total_tax

    achieved_weights = {
        t: (v / deployed) for t, v in new_values.items() if deployed > 0 and v > 0
    }
    if mu is not None and achieved_weights:
        gross_return = float(sum(
            w * float(mu.get(t, 0.0)) for t, w in achieved_weights.items()))
    else:
        gross_return = float("nan")

    friction_drag = (total_friction / investable_capital) if investable_capital > 0 else 0.0
    net_return = gross_return - friction_drag if not math.isnan(gross_return) else float("nan")

    return ReallocationResult(
        table=table,
        achieved_weights=achieved_weights,
        residual_cash=residual_cash,
        total_fees=total_fees,
        total_tax=total_tax,
        total_friction=total_friction,
        num_trades=num_trades,
        gross_return=gross_return,
        net_return=net_return,
        allow_fractional=allow_fractional,
        transaction_cost=transaction_cost,
        tolerance=tolerance,
        investable_capital=investable_capital,
    )


# --------------------------------------------------------------------------- #
# Tactical liquidity deployment (lump sum vs DCA tranches)
# --------------------------------------------------------------------------- #
@dataclass
class DeploymentPlan:
    """Capital split for the current rebalancing step."""

    strategy: str
    tranches: int
    current_value: float
    additional_cash: float
    total_wealth: float       # current portfolio value + ALL additional cash
    active_capital: float     # capital deployed in THIS step
    sidelined_cash: float     # cash buffered for future tranches
    per_tranche_cash: float


def deployment_plan(
    current_value: float,
    additional_cash: float,
    strategy: str,
    tranches: int = 1,
) -> DeploymentPlan:
    """Compute the active vs sidelined capital for a deployment strategy.

    *Lump sum* deploys ``current_value + additional_cash`` now. *Tranches/DCA*
    deploys ``current_value + additional_cash / tranches`` now and sidelines the
    rest for future periods.
    """
    current_value = max(float(current_value), 0.0)
    additional_cash = max(float(additional_cash), 0.0)
    total_wealth = current_value + additional_cash
    is_tranche = ("tranche" in strategy.lower()) or ("dca" in strategy.lower())

    if is_tranche and tranches > 1:
        per_tranche = additional_cash / tranches
        active_capital = current_value + per_tranche
        sidelined = additional_cash - per_tranche
    else:
        tranches = 1
        per_tranche = additional_cash
        active_capital = total_wealth
        sidelined = 0.0

    return DeploymentPlan(
        strategy=strategy,
        tranches=tranches,
        current_value=current_value,
        additional_cash=additional_cash,
        total_wealth=total_wealth,
        active_capital=active_capital,
        sidelined_cash=sidelined,
        per_tranche_cash=per_tranche,
    )


# --------------------------------------------------------------------------- #
# Behavioral volatility noise classifier
# --------------------------------------------------------------------------- #
@dataclass
class AssetNoise:
    """Per-asset drawdown classified against its annualised volatility."""

    ticker: str
    pnl_pct: float
    annual_vol: float
    within_1sigma: bool


@dataclass
class NoiseReport:
    """Portfolio-level behavioral classification of current P/L vs volatility."""

    assets: list[AssetNoise]
    portfolio_pnl_pct: float
    portfolio_vol: float
    portfolio_within_1sigma: bool
    verdict: str            # "normale" | "grave" | "n/a"
    commentary: str
    has_data: bool


def classify_noise(current: "CurrentPortfolio", cov: pd.DataFrame) -> NoiseReport:
    """Classify each asset's P/L drawdown against its annualised volatility (σ).

    Moves within ±1σ are flagged as expected "market noise"; moves beyond ±1σ as
    statistical outliers. A portfolio-level verdict ("normale"/"grave") supports
    behavioural discipline. Degrades gracefully when no purchase prices exist.
    """
    if current is None or not current.has_cost_data:
        return NoiseReport(
            assets=[], portfolio_pnl_pct=float("nan"), portfolio_vol=float("nan"),
            portfolio_within_1sigma=True, verdict="n/a",
            commentary=("Add purchase prices to enable the behavioral volatility "
                        "classifier and P/L drawdown analysis."),
            has_data=False,
        )

    assets: list[AssetNoise] = []
    for ticker, cost in current.cost_basis.items():
        if cost <= 0:
            continue
        pnl_pct = current.unrealized_pnl[ticker] / cost
        annual_vol = float(np.sqrt(cov.loc[ticker, ticker])) if ticker in cov.index \
            else float("nan")
        within = (not np.isnan(annual_vol)) and abs(pnl_pct) <= annual_vol
        assets.append(AssetNoise(ticker, pnl_pct, annual_vol, within))

    port_pnl_pct = current.total_pnl / current.total_cost if current.total_cost > 0 else 0.0
    port_vol = float(current.volatility)
    within_port = abs(port_pnl_pct) <= port_vol if port_vol > 0 else True

    if within_port or port_pnl_pct >= 0:
        verdict = "normale"
        commentary = (
            "🟢 **Normale.** The portfolio's current swing is within ±1σ of its "
            f"annualised volatility ({port_vol * 100:.1f}%). This is standard "
            "market fluctuation — statistically expected noise, not a structural "
            "break. Stay disciplined and avoid emotional, fee-incurring trades."
        )
    else:
        verdict = "grave"
        commentary = (
            "🔴 **Grave.** The drawdown exceeds the portfolio's annualised "
            f"volatility ({port_vol * 100:.1f}%), i.e. beyond 1σ. This may signal "
            "a structural covariance break rather than normal noise — a deliberate "
            "portfolio review (not panic selling) is warranted."
        )

    return NoiseReport(
        assets=assets, portfolio_pnl_pct=port_pnl_pct, portfolio_vol=port_vol,
        portfolio_within_1sigma=within_port, verdict=verdict,
        commentary=commentary, has_data=True,
    )


# --------------------------------------------------------------------------- #
# Institutional execution export (IBKR TWS Basket Trader + config sheet)
# --------------------------------------------------------------------------- #
def clean_ib_symbol(ticker: str) -> str:
    """Strip the Yahoo exchange suffix (``.DE``/``.MI`` …) for IB compatibility."""
    return str(ticker).upper().split(".")[0]


def basket_trader_csv(realloc: ReallocationResult) -> str:
    """Render the reallocation plan as an IBKR TWS Basket Trader CSV string.

    Columns are exactly ``Action, Quantity, Symbol, OrderType`` (MKT). Only
    whole-share, non-zero trades are exported; within-tolerance holds are skipped.
    """
    rows = []
    for _, row in realloc.table.iterrows():
        shares = row.get("Shares to trade")
        if shares is None or (isinstance(shares, float) and math.isnan(shares)):
            continue
        qty = int(round(abs(float(shares))))
        if qty <= 0:
            continue
        action = "BUY" if float(shares) > 0 else "SELL"
        rows.append({
            "Action": action,
            "Quantity": qty,
            "Symbol": clean_ib_symbol(row["Ticker"]),
            "OrderType": "MKT",
        })
    frame = pd.DataFrame(rows, columns=["Action", "Quantity", "Symbol", "OrderType"])
    return frame.to_csv(index=False)


def portfolio_config_csv(
    current_values: dict[str, float],
    target_weights: dict[str, float],
    investable_capital: float,
    sector_map: dict[str, str] | None = None,
    base_currency: str = "USD",
) -> str:
    """Render a current-vs-target configuration sheet (weights, values, sectors)."""
    sector_map = sector_map or {}
    total_current = sum(current_values.values()) or 1.0
    tickers = sorted(set(current_values) | set(target_weights))
    rows = []
    for ticker in tickers:
        cur_val = float(current_values.get(ticker, 0.0))
        tgt_w = float(target_weights.get(ticker, 0.0))
        rows.append({
            "Ticker": ticker,
            "Sector": sector_map.get(ticker, UNKNOWN_SECTOR),
            "Current Weight %": round(cur_val / total_current * 100, 2),
            "Target Weight %": round(tgt_w * 100, 2),
            f"Current Value ({base_currency})": round(cur_val, 2),
            f"Target Value ({base_currency})": round(tgt_w * investable_capital, 2),
        })
    return pd.DataFrame(rows).to_csv(index=False)


# --------------------------------------------------------------------------- #
# Local persistence layer (free, file-based SQLite)
# --------------------------------------------------------------------------- #
_DEFAULT_DB_PATH = Path(
    os.environ.get(
        "PORTFOLIO_DB_PATH",
        Path(__file__).resolve().parent.parent / ".portfolio_state.db",
    )
)

# Columns the holdings table is normalised to on load.
HOLDINGS_COLUMNS: tuple[str, ...] = ("Ticker", "Shares", "Purchase Price", "Asset Type")


class PortfolioStore:
    """Free, local SQLite key/value store for holdings, watchlist and settings.

    Values are JSON-encoded in a single ``app_state`` table. Every operation is
    defensive: a corrupt or unreadable database file is quarantined (renamed
    with a ``.corrupt`` suffix) and re-created rather than crashing the app. The
    class has **no Streamlit dependency** so it is fully unit-testable offline.
    """

    def __init__(self, path: str | os.PathLike | None = None) -> None:
        self.path = Path(path) if path is not None else _DEFAULT_DB_PATH
        self._init_schema()

    # -- low-level helpers ------------------------------------------------- #
    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.path), timeout=5.0)

    def _create_table(self) -> None:
        with self._connect() as con:
            con.execute(
                "CREATE TABLE IF NOT EXISTS app_state "
                "(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )

    def _init_schema(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._create_table()
        except sqlite3.DatabaseError:
            self._quarantine_and_reset()
        except OSError:
            pass

    def _quarantine_and_reset(self) -> None:
        """Move a corrupt DB aside and start fresh; never raise."""
        try:
            if self.path.exists():
                self.path.replace(self.path.with_suffix(self.path.suffix + ".corrupt"))
        except OSError:
            try:
                self.path.unlink()
            except OSError:
                pass
        try:
            self._create_table()
        except sqlite3.DatabaseError:
            pass

    # -- key/value API ----------------------------------------------------- #
    def set(self, key: str, value: Any) -> bool:
        """JSON-encode and upsert ``value`` under ``key``. Returns success."""
        try:
            payload = json.dumps(value, default=str)
        except (TypeError, ValueError):
            return False
        try:
            with self._connect() as con:
                con.execute(
                    "INSERT OR REPLACE INTO app_state(key, value) VALUES (?, ?)",
                    (key, payload),
                )
            return True
        except sqlite3.DatabaseError:
            self._quarantine_and_reset()
            return False

    def get(self, key: str, default: Any = None) -> Any:
        """Return the JSON-decoded value for ``key`` (or ``default``)."""
        try:
            with self._connect() as con:
                row = con.execute(
                    "SELECT value FROM app_state WHERE key = ?", (key,)
                ).fetchone()
        except sqlite3.DatabaseError:
            self._quarantine_and_reset()
            return default
        if row is None:
            return default
        try:
            return json.loads(row[0])
        except (json.JSONDecodeError, TypeError):
            return default

    def clear(self) -> bool:
        try:
            with self._connect() as con:
                con.execute("DELETE FROM app_state")
            return True
        except sqlite3.DatabaseError:
            return False

    # -- convenience wrappers --------------------------------------------- #
    def save_holdings(self, holdings: pd.DataFrame) -> bool:
        try:
            records = holdings.where(pd.notna(holdings), None).to_dict("records")
        except Exception:  # noqa: BLE001 - never let persistence crash the app
            return False
        return self.set("holdings", records)

    def load_holdings(self) -> pd.DataFrame | None:
        records = self.get("holdings")
        if not records:
            return None
        try:
            return normalise_holdings_df(pd.DataFrame(records))
        except Exception:  # noqa: BLE001
            return None

    def save_watchlist(self, watchlist) -> bool:
        return self.set("watchlist", list(watchlist))

    def load_watchlist(self, default=None):
        value = self.get("watchlist", default)
        return value if isinstance(value, list) else default

    def save_settings(self, settings: dict) -> bool:
        return self.set("settings", dict(settings))

    def load_settings(self) -> dict:
        value = self.get("settings", {})
        return value if isinstance(value, dict) else {}

    def save_session(self, holdings: pd.DataFrame, watchlist, settings: dict) -> None:
        """Persist the full editable state in one go (best-effort)."""
        self.save_holdings(holdings)
        self.save_watchlist(watchlist)
        self.save_settings(settings)


def normalise_holdings_df(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the holdings frame has exactly the expected schema/columns."""
    out = df.copy()
    if "Asset Type" not in out.columns:
        out["Asset Type"] = DEFAULT_ASSET_TYPE
    if "Purchase Price" not in out.columns:
        out["Purchase Price"] = None
    for col in ("Ticker", "Shares"):
        if col not in out.columns:
            out[col] = "" if col == "Ticker" else 0.0
    return out[list(HOLDINGS_COLUMNS)]
