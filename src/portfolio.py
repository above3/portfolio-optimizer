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
import math
import re
from dataclasses import dataclass

import pandas as pd

from .data_loader import PriceDataError
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

    Returns ``None`` when no positive holdings are supplied (e.g. the user is
    only drafting a brand-new portfolio).
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


def _aggregate_positions(pairs: list[tuple[str, float]]) -> pd.DataFrame:
    """Sum duplicate symbols and return a clean Ticker/Shares frame."""
    totals: dict[str, float] = {}
    for symbol, qty in pairs:
        sym = str(symbol).strip().upper()
        if _norm(sym) in _SKIP_SYMBOLS:
            continue
        totals[sym] = totals.get(sym, 0.0) + qty
    rows = [{"Ticker": s, "Shares": q} for s, q in totals.items() if q != 0]
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

    sym_col = df.columns[sym_idx]
    qty_col = df.columns[qty_idx]
    pairs: list[tuple[str, float]] = []
    for _, row in df.iterrows():
        qty = _to_float(row[qty_col])
        sym = str(row[sym_col]).strip()
        if sym and qty is not None:
            pairs.append((sym, qty))
    if not pairs:
        return None
    return _aggregate_positions(pairs)


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

        section = _norm(row[0])
        pairs: list[tuple[str, float]] = []
        for data_row in rows[i + 1:]:
            if len(data_row) <= max(sym_idx, qty_idx):
                continue
            if _norm(data_row[0]) != section or _norm(data_row[1]) != "data":
                continue
            qty = _to_float(data_row[qty_idx])
            sym = str(data_row[sym_idx]).strip()
            if sym and qty is not None:
                pairs.append((sym, qty))
        if pairs:
            return _aggregate_positions(pairs)

    raise PriceDataError(
        "Could not locate 'Symbol'/'Ticker' and 'Quantity'/'Shares' columns in "
        "the uploaded IBKR CSV."
    )


def parse_ibkr_positions(file) -> pd.DataFrame:
    """Parse an IBKR position/activity CSV into a ``Ticker``/``Shares`` frame.

    Handles both the simple flat Flex-Query export (a single table with
    ``Symbol`` and ``Quantity`` columns) and the multi-section activity
    statement. Duplicate lots for the same symbol are summed.

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
    num_trades: int
    gross_return: float
    net_return: float
    allow_fractional: bool
    transaction_cost: float
    investable_capital: float


def reallocation_plan(
    current_values: dict[str, float],
    target_weights: dict[str, float],
    investable_capital: float,
    latest_prices: pd.Series,
    mu: pd.Series | None = None,
    risk_free_rate: float = 0.0,
    allow_fractional: bool = True,
    transaction_cost: float = 0.0,
) -> ReallocationResult:
    """Build a reallocation plan from current holdings to target weights.

    Parameters
    ----------
    current_values:
        Mapping of ticker -> current dollar value held (empty for a new
        portfolio).
    target_weights:
        Target weights (e.g. the tangency portfolio).
    investable_capital:
        Capital the target weights are applied to (current value + new cash).
    latest_prices:
        Latest close price per ticker, used to size trades in shares.
    mu:
        Optional expected-return series; when provided, the achieved (post-trade)
        gross and net-of-fee expected returns are computed.
    risk_free_rate:
        Reserved for callers that want to score residual cash (unused here).
    allow_fractional:
        If ``False``, share trades are truncated to whole integers and the
        undeployed remainder is reported as residual cash.
    transaction_cost:
        Flat simulated cost charged per executed (non-zero) trade. Its total is
        reported and subtracted from the expected return as a one-period drag.
    """
    tickers = sorted(set(current_values) | set(target_weights))
    rows = []
    new_values: dict[str, float] = {}
    num_trades = 0

    for ticker in tickers:
        cur_val = float(current_values.get(ticker, 0.0) or 0.0)
        tgt_w = float(target_weights.get(ticker, 0.0))
        target_value = tgt_w * investable_capital
        price = float(latest_prices.get(ticker, 0.0) or 0.0)

        dollar_delta = target_value - cur_val
        raw_shares = dollar_delta / price if price > 0 else float("nan")

        if not allow_fractional and price > 0:
            # Round toward zero so we never overshoot the available capital.
            traded_shares = math.trunc(raw_shares)
        else:
            traded_shares = raw_shares

        executed_value = traded_shares * price if price > 0 else 0.0
        new_value = cur_val + executed_value
        new_values[ticker] = new_value

        traded = price > 0 and abs(traded_shares) >= (1 if not allow_fractional else 1e-9)
        if traded:
            num_trades += 1

        rows.append(
            {
                "Ticker": ticker,
                "Current %": round(cur_val / investable_capital * 100, 2)
                if investable_capital > 0 else 0.0,
                "Target %": round(tgt_w * 100, 2),
                "Δ Weight %": round(
                    (tgt_w - (cur_val / investable_capital if investable_capital > 0 else 0.0))
                    * 100, 2),
                "Price $": round(price, 2) if price > 0 else None,
                "Shares to trade": (
                    int(traded_shares) if not allow_fractional and price > 0
                    else (round(traded_shares, 4) if price > 0 else None)
                ),
                "Trade $": round(executed_value, 2) if price > 0 else None,
                "Action": _action_label(tgt_w * investable_capital - cur_val),
            }
        )

    table = pd.DataFrame(rows)

    deployed = sum(new_values.values())
    residual_cash = max(investable_capital - deployed, 0.0)
    total_fees = transaction_cost * num_trades

    achieved_weights = {
        t: (v / deployed) for t, v in new_values.items() if deployed > 0 and v > 0
    }

    if mu is not None and achieved_weights:
        gross_return = float(sum(
            w * float(mu.get(t, 0.0)) for t, w in achieved_weights.items()
        ))
    else:
        gross_return = float("nan")

    fee_drag = (total_fees / investable_capital) if investable_capital > 0 else 0.0
    net_return = gross_return - fee_drag if not math.isnan(gross_return) else float("nan")

    return ReallocationResult(
        table=table,
        achieved_weights=achieved_weights,
        residual_cash=residual_cash,
        total_fees=total_fees,
        num_trades=num_trades,
        gross_return=gross_return,
        net_return=net_return,
        allow_fractional=allow_fractional,
        transaction_cost=transaction_cost,
        investable_capital=investable_capital,
    )


def _action_label(delta_value: float, tol: float = 1e-2) -> str:
    if delta_value > tol:
        return "Buy"
    if delta_value < -tol:
        return "Sell"
    return "Hold"
