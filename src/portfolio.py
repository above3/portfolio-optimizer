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
from .data_loader import PriceDataError, iso_timestamp
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
    buy_only: bool = False
    tracking_error: float = float("nan")


_HOLD_TOLERANCE = "HOLD / Within Tolerance"


def adaptive_tolerance_bands(
    volatilities: dict[str, float], base_factor: float = 0.2
) -> dict[str, float]:
    """Per-asset rebalancing tolerance band ``= base_factor × σ_i``.

    Volatile assets get wider bands (less churn); low-vol bonds get tighter ones.
    """
    return {ticker: max(0.0, float(base_factor) * float(sigma))
            for ticker, sigma in (volatilities or {}).items()}


def _knapsack_topup(
    traded: dict[str, float],
    cur: dict[str, float],
    price: dict[str, float],
    target_value: dict[str, float],
    investable: float,
    within_tol: dict[str, bool],
    max_iter: int = 20_000,
) -> None:
    """Greedily deploy residual cash to minimise integer tracking error.

    After whole-share rounding leaves uninvested cash, repeatedly buy one extra
    share of whichever (affordable, non-locked) asset most reduces the squared
    value-space tracking error to the continuous target — driving residual cash
    toward zero. Mutates ``traded`` in place. Only ever *buys*.
    """
    new_val = {t: cur[t] + traded[t] * price[t] for t in traded}
    residual = investable - sum(new_val.values())
    if residual <= 0 or not any(price[t] > 0 for t in traded):
        return

    for _ in range(max_iter):
        best, best_gain = None, 1e-12
        for t in traded:
            p = price[t]
            if p <= 0 or p > residual + 1e-9 or within_tol[t]:
                continue
            before = (new_val[t] - target_value[t]) ** 2
            after = (new_val[t] + p - target_value[t]) ** 2
            gain = before - after
            if gain > best_gain:
                best_gain, best = gain, t
        if best is None:
            break
        traded[best] += 1
        new_val[best] += price[best]
        residual -= price[best]


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
    buy_only: bool = False,
    smart_rounding: bool = True,
    asset_tolerances: dict[str, float] | None = None,
) -> ReallocationResult:
    """Build a friction- and tax-aware reallocation plan.

    Parameters
    ----------
    current_values, target_weights, investable_capital, latest_prices:
        See the module docs. ``current_values`` may be empty (new portfolio).
    mu:
        Optional expected returns; enables gross/net achieved-return metrics.
    allow_fractional:
        If ``False``, trades are whole shares; with ``smart_rounding`` the
        residual cash is then knapsack-deployed to cut integer tracking error.
    transaction_cost, tolerance, purchase_prices, asset_types:
        Friction, the rebalancing tolerance band and Italian-tax inputs.
    buy_only:
        If ``True``, any computed SELL is suppressed (held) so the plan deploys
        new cash without realising gains — pairs with ``buy_only_max_sharpe``.
    smart_rounding:
        Enable the knapsack residual-cash top-up (whole-share mode only).
    """
    purchase_prices = purchase_prices or {}
    asset_types = asset_types or {}
    tickers = sorted(set(current_values) | set(target_weights))

    cur = {t: float(current_values.get(t, 0.0) or 0.0) for t in tickers}
    price = {t: float(latest_prices.get(t, 0.0) or 0.0) for t in tickers}
    tgt_w = {t: float(target_weights.get(t, 0.0)) for t in tickers}
    cur_w = {t: (cur[t] / investable_capital if investable_capital > 0 else 0.0)
             for t in tickers}
    target_value = {t: tgt_w[t] * investable_capital for t in tickers}
    # Per-asset adaptive bands override the scalar tolerance when supplied.
    asset_tolerances = asset_tolerances or {}
    within_tol = {t: abs(tgt_w[t] - cur_w[t]) < float(asset_tolerances.get(t, tolerance))
                  for t in tickers}

    # ---- 1. Base integer/fractional trades ----
    traded: dict[str, float] = {}
    for t in tickers:
        if within_tol[t] or price[t] <= 0:
            traded[t] = 0.0
            continue
        raw = (target_value[t] - cur[t]) / price[t]
        if buy_only and raw < 0:          # suppress sells entirely
            traded[t] = 0.0
            continue
        traded[t] = math.trunc(raw) if not allow_fractional else raw

    # ---- 2. Smart knapsack residual-cash deployment (whole-share mode) ----
    if not allow_fractional and smart_rounding:
        _knapsack_topup(traded, cur, price, target_value, investable_capital, within_tol)

    # ---- 3. Build the table + friction ----
    rows = []
    new_values: dict[str, float] = {}
    num_trades = 0
    total_tax = 0.0
    for t in tickers:
        shares = traded[t]
        executed_value = shares * price[t] if price[t] > 0 else 0.0
        new_values[t] = cur[t] + executed_value
        is_trade = price[t] > 0 and abs(shares) >= (1 if not allow_fractional else 1e-9)
        if is_trade:
            num_trades += 1

        tax = 0.0
        if is_trade and shares < 0:        # tax the sold portion of a gain
            pp = purchase_prices.get(t)
            if pp and price[t] > 0:
                realised_gain = abs(shares) * (price[t] - float(pp))
                tax = tax_rate_for(asset_types.get(t)) * max(realised_gain, 0.0)
                total_tax += tax

        if within_tol[t]:
            action = _HOLD_TOLERANCE
        elif is_trade and shares > 0:
            action = "BUY"
        elif is_trade and shares < 0:
            action = "SELL"
        else:
            action = "HOLD"

        rows.append({
            "Ticker": t,
            "Current %": round(cur_w[t] * 100, 2),
            "Target %": round(tgt_w[t] * 100, 2),
            "Δ Weight %": round((tgt_w[t] - cur_w[t]) * 100, 2),
            "Price": round(price[t], 2) if price[t] > 0 else None,
            "Shares to trade": (
                int(shares) if (not allow_fractional and price[t] > 0)
                else (round(shares, 4) if price[t] > 0 else None)),
            "Trade Value": round(executed_value, 2) if price[t] > 0 else None,
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

    # Tracking error in value-weight space (consistent with the knapsack
    # objective): how far the deployed value vector sits from the target weights.
    tracking_error = float(np.sqrt(sum(
        (new_values[t] / investable_capital - tgt_w[t]) ** 2 for t in tickers))) \
        if investable_capital > 0 else float("nan")

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
        buy_only=buy_only,
        tracking_error=tracking_error,
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
# Yahoo exchange suffix -> Interactive Brokers exchange (context, not exported).
_IBKR_EXCHANGE_BY_SUFFIX: dict[str, str] = {
    "DE": "IBIS",   # Xetra
    "MI": "BVME",   # Borsa Italiana
    "AS": "AEB",    # Euronext Amsterdam
    "PA": "SBF",    # Euronext Paris
    "L": "LSE",     # London Stock Exchange
    "SW": "EBS",    # SIX Swiss
    "TO": "TSE",    # Toronto
    "HK": "SEHK",   # Hong Kong
    "T": "TSEJ",    # Tokyo
}
# Explicit overrides for symbols whose IB root differs from the Yahoo base.
_IBKR_SYMBOL_OVERRIDES: dict[str, str] = {}


def map_ibkr_symbol(ticker: str) -> str:
    """Translate a Yahoo ticker to the clean IB Basket-Trader ``Symbol`` root.

    ``VWCE.DE`` -> ``VWCE`` (Xetra), ``AGGH.MI`` -> ``AGGH`` (Borsa Italiana),
    plain US tickers pass through unchanged. Explicit overrides take precedence.
    """
    sym = str(ticker).strip().upper()
    if sym in _IBKR_SYMBOL_OVERRIDES:
        return _IBKR_SYMBOL_OVERRIDES[sym]
    return sym.split(".")[0]


def ibkr_exchange_for(ticker: str) -> str:
    """Return the IB exchange code implied by a Yahoo suffix (``"SMART"`` if US)."""
    parts = str(ticker).strip().upper().rsplit(".", 1)
    if len(parts) == 2:
        return _IBKR_EXCHANGE_BY_SUFFIX.get(parts[1], "SMART")
    return "SMART"


def clean_ib_symbol(ticker: str) -> str:
    """Backwards-compatible alias for :func:`map_ibkr_symbol`."""
    return map_ibkr_symbol(ticker)


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
            "Symbol": map_ibkr_symbol(row["Ticker"]),
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
# Hierarchical Risk Parity — correlation cluster tree
# --------------------------------------------------------------------------- #
@dataclass
class ClusterTree:
    """Dendrogram plotting data from agglomerative clustering of a correlation
    matrix (Hierarchical Risk Parity distance)."""

    icoord: list[list[float]]
    dcoord: list[list[float]]
    leaves: list[str]
    labels: list[str]
    linkage_matrix: np.ndarray


def hierarchical_clusters(corr: pd.DataFrame, method: str = "single") -> ClusterTree | None:
    """Agglomerative clustering of the correlation matrix for an HRP dendrogram.

    Uses the standard HRP distance ``d = sqrt(0.5 * (1 - ρ))``. Returns ``None``
    (graceful degradation) for fewer than three assets or any numerical failure
    (singular matrix, scipy import error, …).
    """
    try:
        from scipy.cluster.hierarchy import dendrogram, linkage
        from scipy.spatial.distance import squareform

        labels = [str(c) for c in corr.columns]
        if len(labels) < 3:
            return None
        rho = np.clip(corr.to_numpy(dtype=float), -1.0, 1.0)
        dist = np.sqrt(np.clip(0.5 * (1.0 - rho), 0.0, None))
        dist = (dist + dist.T) / 2.0
        np.fill_diagonal(dist, 0.0)
        condensed = squareform(dist, checks=False)
        link = linkage(condensed, method=method)
        ddata = dendrogram(link, labels=labels, no_plot=True)
        return ClusterTree(
            icoord=ddata["icoord"], dcoord=ddata["dcoord"],
            leaves=list(ddata["ivl"]), labels=labels, linkage_matrix=link)
    except Exception:  # noqa: BLE001 - degrade gracefully on any clustering failure
        return None


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
            con.execute(
                "CREATE TABLE IF NOT EXISTS tranche_history ("
                "timestamp TEXT NOT NULL, "
                "active_capital_deployed REAL NOT NULL, "
                "strategy_used TEXT, "
                "portfolio_value_at_deployment REAL NOT NULL)"
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

    # -- tranche history (PAC ledger) ------------------------------------- #
    def record_tranche(
        self,
        active_capital_deployed: float,
        strategy_used: str,
        portfolio_value_at_deployment: float,
        timestamp: str | None = None,
    ) -> bool:
        """Append a capital-deployment event to the ``tranche_history`` table."""
        timestamp = timestamp or iso_timestamp()
        try:
            with self._connect() as con:
                con.execute(
                    "INSERT INTO tranche_history(timestamp, active_capital_deployed, "
                    "strategy_used, portfolio_value_at_deployment) VALUES (?, ?, ?, ?)",
                    (timestamp, float(active_capital_deployed), str(strategy_used),
                     float(portfolio_value_at_deployment)))
            return True
        except sqlite3.DatabaseError:
            return False

    def load_tranche_history(self) -> pd.DataFrame:
        """Return the recorded deployment events ordered by timestamp."""
        cols = ["timestamp", "active_capital_deployed", "strategy_used",
                "portfolio_value_at_deployment"]
        try:
            with self._connect() as con:
                rows = con.execute(
                    "SELECT timestamp, active_capital_deployed, strategy_used, "
                    "portfolio_value_at_deployment FROM tranche_history "
                    "ORDER BY timestamp ASC").fetchall()
        except sqlite3.DatabaseError:
            return pd.DataFrame(columns=cols)
        return pd.DataFrame(rows, columns=cols)

    def clear_tranche_history(self) -> bool:
        try:
            with self._connect() as con:
                con.execute("DELETE FROM tranche_history")
            return True
        except sqlite3.DatabaseError:
            return False

    # -- frozen live snapshot (offline recovery) -------------------------- #
    def save_frozen_snapshot(self, snapshot: dict) -> bool:
        """Persist a frozen live snapshot (``is_frozen=True``) for offline use."""
        payload = dict(snapshot)
        payload["is_frozen"] = True
        payload.setdefault("timestamp", iso_timestamp())
        return self.set("frozen_snapshot", payload)

    def load_frozen_snapshot(self) -> dict | None:
        value = self.get("frozen_snapshot")
        return value if isinstance(value, dict) and value.get("is_frozen") else None

    def clear_frozen_snapshot(self) -> bool:
        return self.set("frozen_snapshot", None)


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


# --------------------------------------------------------------------------- #
# Live-broker positions -> editable holdings (hybrid append-merge)
# --------------------------------------------------------------------------- #
def positions_to_holdings_df(positions) -> pd.DataFrame:
    """Convert live broker positions into the editable holdings schema.

    ``positions`` is any iterable of objects exposing ``symbol``/``shares``/
    ``avg_cost``/``asset_type`` attributes (e.g. ``src.providers.Position``) or
    equivalent dicts. Zero-share lots are dropped.
    """
    rows = []
    for pos in positions or []:
        if isinstance(pos, dict):
            sym = pos.get("symbol", pos.get("Ticker", ""))
            shares = pos.get("shares", pos.get("Shares", 0))
            avg = pos.get("avg_cost", pos.get("Purchase Price"))
            atype = pos.get("asset_type", pos.get("Asset Type"))
        else:
            sym = getattr(pos, "symbol", "")
            shares = getattr(pos, "shares", 0)
            avg = getattr(pos, "avg_cost", None)
            atype = getattr(pos, "asset_type", None)
        sym = str(sym or "").strip().upper()
        if not sym:
            continue
        try:
            shares = float(shares or 0)
        except (TypeError, ValueError):
            shares = 0.0
        if shares == 0:
            continue
        try:
            avg = float(avg) if avg not in (None, "") else None
        except (TypeError, ValueError):
            avg = None
        rows.append({
            "Ticker": sym,
            "Shares": shares,
            "Purchase Price": round(avg, 4) if avg else None,
            "Asset Type": atype or guess_asset_type(sym),
        })
    return normalise_holdings_df(pd.DataFrame(rows, columns=list(HOLDINGS_COLUMNS)))


def merge_holdings(primary: pd.DataFrame, secondary: pd.DataFrame) -> pd.DataFrame:
    """Hybrid append-merge: ``primary`` rows first (and winning on conflicts),
    then any ``secondary`` rows whose ticker is not already present.

    Used to inject live IBKR positions (``primary``) while **preserving** the
    user's manual rows (``secondary``) — never locking or discarding them.
    """
    def _norm(df):
        if df is None or len(df) == 0:
            return normalise_holdings_df(pd.DataFrame(columns=list(HOLDINGS_COLUMNS)))
        return normalise_holdings_df(df)

    primary, secondary = _norm(primary), _norm(secondary)
    seen: set[str] = set()
    rows: list[dict] = []
    for frame in (primary, secondary):
        for _, row in frame.iterrows():
            ticker = str(row["Ticker"]).strip().upper()
            if not ticker or ticker in seen:
                continue
            seen.add(ticker)
            rows.append({c: row[c] for c in HOLDINGS_COLUMNS})
    return normalise_holdings_df(pd.DataFrame(rows, columns=list(HOLDINGS_COLUMNS)))


# --------------------------------------------------------------------------- #
# PAC performance ledger (Time-Weighted & Money-Weighted returns)
# --------------------------------------------------------------------------- #
@dataclass
class PacMetrics:
    """Reconstructed savings-plan performance from the tranche history."""

    twr_annual: float
    mwr_annual: float
    twr_cumulative: float
    total_contributed: float
    current_value: float
    total_gain: float
    period_years: float
    n_events: int


def _xnpv(rate: float, cashflows: list[tuple]) -> float:
    """Net present value of dated cashflows at an annual ``rate``."""
    t0 = cashflows[0][0]
    return sum(amt / (1.0 + rate) ** ((d - t0).days / 365.0) for d, amt in cashflows)


def xirr(cashflows: list[tuple], lo: float = -0.999999, hi: float = 10.0,
         tol: float = 1e-7, max_iter: int = 200) -> float:
    """Money-weighted IRR for dated cashflows via robust bisection.

    ``cashflows`` is a list of ``(datetime, amount)`` with at least one negative
    (investment) and one positive (final value). Returns ``nan`` when no root
    brackets the interval.
    """
    if len(cashflows) < 2:
        return float("nan")
    try:
        f_lo, f_hi = _xnpv(lo, cashflows), _xnpv(hi, cashflows)
    except (ZeroDivisionError, OverflowError):
        return float("nan")
    if not (f_lo == f_lo and f_hi == f_hi) or f_lo * f_hi > 0:
        return float("nan")
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        f_mid = _xnpv(mid, cashflows)
        if abs(f_mid) < tol:
            return mid
        if f_lo * f_mid < 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return (lo + hi) / 2.0


def compute_pac_metrics(
    history: pd.DataFrame,
    current_value: float,
    as_of: str | None = None,
) -> PacMetrics | None:
    """Compute TWR and MWR (XIRR) from the recorded tranche-deployment history.

    Each event stores the portfolio value just before deployment and the active
    capital pool deployed; the external cashflow is their difference. Returns
    ``None`` when there is insufficient data.
    """
    if history is None or len(history) == 0 or current_value is None or current_value <= 0:
        return None
    df = history.copy()
    df["dt"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["dt"]).sort_values("dt")
    if df.empty:
        return None

    as_of_dt = (pd.to_datetime(as_of, utc=True) if as_of
                else pd.Timestamp.now(tz="UTC")).to_pydatetime()
    events = []
    for _, row in df.iterrows():
        value = float(row["portfolio_value_at_deployment"])
        deployed = float(row["active_capital_deployed"])
        events.append({"dt": row["dt"].to_pydatetime(),
                       "value": value, "cashflow": max(deployed - value, 0.0)})

    # Time-Weighted Return: chained sub-period growth factors.
    factors = []
    for i, ev in enumerate(events):
        start_val = ev["value"] + ev["cashflow"]
        end_val = events[i + 1]["value"] if i + 1 < len(events) else current_value
        if start_val > 0:
            factors.append(end_val / start_val)
    twr_cum = float(np.prod(factors) - 1.0) if factors else float("nan")
    period_years = max((as_of_dt - events[0]["dt"]).days / 365.0, 1e-9)
    if period_years >= 0.08 and twr_cum == twr_cum and (1.0 + twr_cum) > 0:
        twr_annual = (1.0 + twr_cum) ** (1.0 / period_years) - 1.0
    else:
        twr_annual = twr_cum

    # Money-Weighted Return (XIRR): investments out, final value in.
    cashflows = [(events[0]["dt"], -(events[0]["value"] + events[0]["cashflow"]))]
    cashflows += [(ev["dt"], -ev["cashflow"]) for ev in events[1:]]
    cashflows.append((as_of_dt, float(current_value)))
    mwr = xirr(cashflows)

    contributed = events[0]["value"] + sum(ev["cashflow"] for ev in events)
    return PacMetrics(
        twr_annual=twr_annual, mwr_annual=mwr, twr_cumulative=twr_cum,
        total_contributed=contributed, current_value=float(current_value),
        total_gain=float(current_value) - contributed,
        period_years=period_years, n_events=len(events))


# --------------------------------------------------------------------------- #
# Executive investment report (printable dark-mode HTML)
# --------------------------------------------------------------------------- #
def build_executive_report_html(
    *,
    title: str,
    base_currency: str,
    generated_at: str,
    pnl: dict,
    ef_chart_html: str = "",
    donut_chart_html: str = "",
    basket_df: pd.DataFrame | None = None,
    extra_note: str = "",
) -> str:
    """Assemble a self-contained, printable dark-mode HTML report.

    Pure string assembly (no Streamlit / network) so it is fully unit-testable;
    chart HTML fragments are produced by the caller via ``fig.to_html(...)``.
    """
    sym = {"USD": "$", "EUR": "€"}.get((base_currency or "USD").upper(), "$")

    def money(value) -> str:
        try:
            return f"{sym}{float(value):,.0f}"
        except (TypeError, ValueError):
            return "—"

    total_value = pnl.get("total_value", 0.0)
    total_pnl = pnl.get("total_pnl", 0.0)
    pnl_pct = pnl.get("pnl_pct", 0.0)
    tone = "pos" if (total_pnl or 0) >= 0 else "neg"

    if basket_df is not None and not basket_df.empty:
        basket_html = basket_df.to_html(index=False, border=0, classes="tbl")
    else:
        basket_html = "<p class='muted'>No trades to stage — portfolio within tolerance.</p>"

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  :root {{ --bg:#0e1117; --card:#1a1d24; --border:#2a2e39; --text:#e6e9ef;
           --muted:#8b95a5; --green:#2ecc71; --red:#e74c3c; }}
  * {{ box-sizing:border-box; }}
  body {{ background:var(--bg); color:var(--text); font-family:-apple-system,Segoe UI,Roboto,sans-serif;
          margin:0; padding:32px; }}
  h1 {{ font-size:1.6rem; margin:0 0 4px; }}
  h2 {{ font-size:1.1rem; margin:28px 0 12px; border-bottom:1px solid var(--border); padding-bottom:6px; }}
  .muted {{ color:var(--muted); font-size:.85rem; }}
  .cards {{ display:flex; gap:16px; flex-wrap:wrap; margin:18px 0; }}
  .card {{ background:var(--card); border:1px solid var(--border); border-radius:14px;
           padding:16px 20px; min-width:190px; }}
  .card .label {{ color:var(--muted); font-size:.72rem; letter-spacing:.08em; text-transform:uppercase; }}
  .card .value {{ font-size:1.5rem; font-weight:700; margin-top:6px; }}
  .card.pos .value {{ color:var(--green); }} .card.neg .value {{ color:var(--red); }}
  .card.pos {{ border-left:4px solid var(--green); }} .card.neg {{ border-left:4px solid var(--red); }}
  table.tbl {{ width:100%; border-collapse:collapse; font-size:.9rem; }}
  table.tbl th, table.tbl td {{ border:1px solid var(--border); padding:8px 10px; text-align:left; }}
  table.tbl th {{ background:#20242d; color:var(--muted); text-transform:uppercase; font-size:.72rem; }}
  .chart {{ background:var(--card); border:1px solid var(--border); border-radius:14px; padding:8px; margin:8px 0; }}
  @media print {{ body {{ -webkit-print-color-adjust:exact; print-color-adjust:exact; }} }}
</style></head>
<body>
  <h1>{title}</h1>
  <div class="muted">Generated {generated_at} · Base currency {base_currency} · {extra_note}</div>
  <h2>Portfolio P&amp;L</h2>
  <div class="cards">
    <div class="card"><div class="label">Portfolio Value</div><div class="value">{money(total_value)}</div></div>
    <div class="card {tone}"><div class="label">Unrealised P/L</div><div class="value">{money(total_pnl)} ({pnl_pct:+.1f}%)</div></div>
    <div class="card"><div class="label">Invested Cost Basis</div><div class="value">{money(pnl.get('total_cost', 0.0))}</div></div>
  </div>
  <h2>Efficient Frontier</h2>
  <div class="chart">{ef_chart_html}</div>
  <h2>Custodian Allocation</h2>
  <div class="chart">{donut_chart_html}</div>
  <h2>IBKR Basket Trader — Actionable Orders</h2>
  {basket_html}
  <p class="muted" style="margin-top:28px">Educational tool. Not investment advice.</p>
</body></html>"""
