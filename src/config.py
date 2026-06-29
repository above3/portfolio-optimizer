"""Shared defaults and constants for the portfolio optimizer."""

from __future__ import annotations

# Number of trading days used to annualise daily statistics.
TRADING_DAYS_PER_YEAR: int = 252

# Default lookback window (in years) for the historical price download.
DEFAULT_LOOKBACK_YEARS: int = 5
LOOKBACK_CHOICES: tuple[int, ...] = (1, 2, 3, 5, 10)

# Default annual risk-free rate (as a decimal, i.e. 0.02 == 2%).
DEFAULT_RISK_FREE_RATE: float = 0.02

# Yahoo Finance ticker for the US 3-Month Treasury Bill yield (quoted in %).
RISK_FREE_TICKER: str = "^IRX"

# Default market benchmark used for comparison and Black-Litterman priors.
DEFAULT_BENCHMARK: str = "SPY"

# Maximum allocation per asset (as a decimal). 1.0 == no concentration cap.
DEFAULT_MAX_WEIGHT: float = 1.0

# Core "global anchor" All-World ETF auto-injected into the universe.
ANCHOR_TICKER: str = "VWCE.DE"
DEFAULT_ANCHOR_PCT: int = 20          # default slider value (%)
ANCHOR_MAX_PCT: int = 50             # slider upper bound (%)

# Example watchlist candidates screened for diversification.
DEFAULT_WATCHLIST: str = "GLD, NVDA, AGGH.MI"

# Default simulated transaction cost charged per executed trade (in dollars).
DEFAULT_TRANSACTION_COST: float = 0.0

# ---- Base display currency & FX normalisation ---------------------------- #
BASE_CURRENCIES: tuple[str, ...] = ("USD", "EUR")
DEFAULT_BASE_CURRENCY: str = "USD"
CURRENCY_SYMBOLS: dict[str, str] = {"USD": "$", "EUR": "€"}

# ---- Asset typing & Italian fiscal friction ------------------------------ #
ASSET_TYPES: tuple[str, ...] = ("Stock/ETF", "Bond/Fixed Income")
DEFAULT_ASSET_TYPE: str = "Stock/ETF"
# Italian capital-gains tax: 26% on equities/ETFs, 12.5% on white-list bonds.
TAX_RATE_EQUITY: float = 0.26
TAX_RATE_BOND: float = 0.125

# ---- Rebalancing tolerance band ------------------------------------------ #
DEFAULT_TOLERANCE_PCT: int = 3      # |w_cur - w_tgt| below this => HOLD
TOLERANCE_MIN_PCT: int = 1
TOLERANCE_MAX_PCT: int = 10

# ---- Sector concentration guardrail -------------------------------------- #
DEFAULT_SECTOR_MAX_PCT: int = 40
SECTOR_MIN_PCT: int = 10
SECTOR_MAX_PCT: int = 100
UNKNOWN_SECTOR: str = "Unknown"

# ---- Tactical liquidity deployment (lump sum vs DCA tranches) ------------ #
DEPLOYMENT_STRATEGIES: tuple[str, ...] = (
    "Lump Sum (Deploy All Now)",
    "Tranches / DCA (Gradual Deployment)",
)
DEFAULT_TRANCHES: int = 4
TRANCHE_MIN: int = 2
TRANCHE_MAX: int = 24

# ---- Macro-regime backtest scenarios ------------------------------------- #
MACRO_REGIMES: tuple[str, ...] = (
    "None / Live Lookback",
    "2022 Stagflation Regime",
    "2008 Great Financial Crisis (GFC)",
    "2000 Dot-Com Bubble Burst",
)

# ---- GBM Monte-Carlo wealth projection ----------------------------------- #
DEFAULT_HORIZON_YEARS: int = 15
HORIZON_MIN_YEARS: int = 1
HORIZON_MAX_YEARS: int = 40
GBM_N_PATHS: int = 10_000
GBM_STEPS_PER_YEAR: int = 12        # monthly compounding

# Expected-return models exposed in the UI.
RETURNS_MODELS: dict[str, str] = {
    "Historical mean returns": "historical",
    "Black-Litterman (market-implied)": "black_litterman",
}

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
# "Purchase Price" is optional and only used to compute unrealised P/L.
DEMO_PORTFOLIO: list[dict[str, object]] = [
    {"Ticker": "AAPL", "Shares": 10, "Purchase Price": 175.0, "Asset Type": "Stock/ETF"},
    {"Ticker": "MSFT", "Shares": 8, "Purchase Price": 330.0, "Asset Type": "Stock/ETF"},
    {"Ticker": "JNJ", "Shares": 12, "Purchase Price": 155.0, "Asset Type": "Stock/ETF"},
    {"Ticker": "JPM", "Shares": 15, "Purchase Price": 145.0, "Asset Type": "Stock/ETF"},
    {"Ticker": "AGGH.MI", "Shares": 30, "Purchase Price": 4.7, "Asset Type": "Bond/Fixed Income"},
]
