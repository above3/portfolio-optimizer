"""Shared defaults and constants for the portfolio optimizer."""

from __future__ import annotations

# Number of trading days used to annualise daily statistics.
TRADING_DAYS_PER_YEAR: int = 252

# Default lookback window (in years) for the historical price download.
DEFAULT_LOOKBACK_YEARS: int = 5
LOOKBACK_CHOICES: tuple[int, ...] = (1, 2, 3, 5, 10)

# Default annual risk-free rate (as a decimal). 2.5% is a EUR short-rate proxy
# (ECB deposit-facility / €STR area) — the consistent reference for a EUR
# investor. The live ^IRX fetch remains available but quotes a *USD* rate, so
# it is off by default to keep a EUR portfolio internally consistent.
DEFAULT_RISK_FREE_RATE: float = 0.025

# Yahoo Finance ticker for the US 3-Month Treasury Bill yield (quoted in %).
RISK_FREE_TICKER: str = "^IRX"

# Default market benchmark used for comparison and Black-Litterman priors.
# VWCE (FTSE All-World UCITS, Xetra) is what an Italian/EU buy-and-hold
# investor actually owns and compares against — not SPY.
DEFAULT_BENCHMARK: str = "VWCE.DE"

# Maximum allocation per asset (as a decimal). 1.0 == no concentration cap.
DEFAULT_MAX_WEIGHT: float = 1.0

# ---- Data architecture (provider framework) ------------------------------ #
ARCH_YAHOO: str = "Yahoo Finance (Decoupled/Offline-Safe)"
ARCH_IBKR: str = "Interactive Brokers Live API (TWS/Gateway Required)"
DATA_ARCHITECTURES: tuple[str, ...] = (ARCH_YAHOO, ARCH_IBKR)
# Native broker sockets on localhost.
IBKR_PORT_CHOICES: dict[str, int] = {"TWS (7496)": 7496, "IB Gateway (4001)": 4001}

# Core "global anchor" All-World ETF auto-injected into the universe.
ANCHOR_TICKER: str = "VWCE.DE"
DEFAULT_ANCHOR_PCT: int = 20          # default slider value (%)
ANCHOR_MAX_PCT: int = 50             # slider upper bound (%)

# Example watchlist candidates screened for diversification (a euro aggregate
# bond ETF, a European quality name, a healthcare compounder).
DEFAULT_WATCHLIST: str = "AGGH.MI, MC.PA, NVO"

# Default simulated transaction cost charged per executed trade (in dollars).
DEFAULT_TRANSACTION_COST: float = 0.0

# ---- Base display currency & FX normalisation ---------------------------- #
# EUR is the default: the target user is an Italian investor whose brokers
# (Trade Republic, IBKR Europe) settle in euros.
BASE_CURRENCIES: tuple[str, ...] = ("EUR", "USD")
DEFAULT_BASE_CURRENCY: str = "EUR"
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
# Dynamic adaptive band: Threshold_i = base_factor * sigma_i (annualised vol).
DEFAULT_BAND_FACTOR: float = 0.2

# ---- Bucket-based target allocation (the user's investment plan) ---------- #
# The rebalancer's PRIMARY target is the user's own plan, expressed as buckets,
# not whatever the Markowitz optimiser happens to produce this week. Percent
# targets are normalised at run time; within a bucket assets split its target
# equally (a deliberate, documented simplification).
BUCKETS: tuple[str, ...] = (
    "Core", "Thematic", "Satellite Quality", "Speculative", "Gold/Defensive",
)
DEFAULT_BUCKET: str = "Core"
DEFAULT_BUCKET_TARGETS: dict[str, float] = {
    "Core": 40.0,
    "Thematic": 12.0,
    "Satellite Quality": 20.0,
    "Speculative": 8.0,       # hard ceiling by design — Nike lives here
    "Gold/Defensive": 20.0,
}
# Materiality threshold: below this max drift the verdict is "no action needed".
DEFAULT_DRIFT_THRESHOLD_PCT: float = 3.0

# ---- Reverse Monte-Carlo savings goal ------------------------------------ #
DEFAULT_GOAL_WEALTH: float = 1_000_000.0
DEFAULT_GOAL_HORIZON_YEARS: int = 20
GOAL_SUCCESS_THRESHOLD: float = 0.85   # below this -> suggest extra savings

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

# Covariance estimators exposed in the UI. Ledoit-Wolf shrinkage is first (the
# default): with real-world short/mixed histories it is far more stable than
# the raw sample covariance, which is exactly what a practical tool needs.
COVARIANCE_METHODS: dict[str, str] = {
    "Ledoit-Wolf shrinkage": "ledoit_wolf",
    "Sample covariance": "sample_cov",
}

# Starter portfolio mirroring the real target user: an Italian investor with
# core/thematic UCITS ETFs, quality single stocks, a gold ETC and speculative
# names split across Trade Republic and IBKR Europe. "Purchase Price" is
# optional and only used to compute unrealised P/L and the tax estimate.
DEMO_PORTFOLIO: list[dict[str, object]] = [
    {"Ticker": "VWCE.DE", "Shares": 45, "Purchase Price": 105.0,
     "Asset Type": "Stock/ETF", "Bucket": "Core"},
    {"Ticker": "EIMI.MI", "Shares": 120, "Purchase Price": 28.0,
     "Asset Type": "Stock/ETF", "Bucket": "Core"},
    {"Ticker": "DFND.MI", "Shares": 25, "Purchase Price": 38.0,
     "Asset Type": "Stock/ETF", "Bucket": "Thematic"},
    {"Ticker": "XDWH.MI", "Shares": 18, "Purchase Price": 52.0,
     "Asset Type": "Stock/ETF", "Bucket": "Thematic"},
    {"Ticker": "UNA.AS", "Shares": 30, "Purchase Price": 47.0,
     "Asset Type": "Stock/ETF", "Bucket": "Satellite Quality"},
    {"Ticker": "ABT", "Shares": 12, "Purchase Price": 105.0,
     "Asset Type": "Stock/ETF", "Bucket": "Satellite Quality"},
    {"Ticker": "KO", "Shares": 25, "Purchase Price": 58.0,
     "Asset Type": "Stock/ETF", "Bucket": "Satellite Quality"},
    {"Ticker": "SGLD.MI", "Shares": 8, "Purchase Price": 190.0,
     "Asset Type": "Stock/ETF", "Bucket": "Gold/Defensive"},
    {"Ticker": "NKE", "Shares": 40, "Purchase Price": 95.0,
     "Asset Type": "Stock/ETF", "Bucket": "Speculative"},
    {"Ticker": "SOFI", "Shares": 60, "Purchase Price": 8.0,
     "Asset Type": "Stock/ETF", "Bucket": "Speculative"},
]
