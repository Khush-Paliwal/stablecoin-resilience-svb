"""Project-wide paths, event dates and model parameters.

Every analysis module reads its defaults from here so that a single edit
(e.g. moving the crisis window) propagates consistently through the pipeline.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# The package uses a src/ layout, so the repository root is two levels above
# this file. Set STABLECOIN_RISK_ROOT to run against a data folder elsewhere.
ROOT = Path(os.environ.get("STABLECOIN_RISK_ROOT", Path(__file__).resolve().parents[2]))

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESERVES_DIR = DATA_DIR / "reserves"
REGRESSION_DIR = DATA_DIR / "regression"
EXTERNAL_DIR = DATA_DIR / "external"  # licensed data (Bloomberg, CoinAPI) - git-ignored

REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"

# Exchange folder names under data/raw and data/processed.
EXCHANGES = ("binance", "coinbase", "crypto_com", "kraken")

# --------------------------------------------------------------------------- #
# SVB / USDC de-peg event study (March 2023)
# --------------------------------------------------------------------------- #
SVB_WINDOW = ("2023-03-01", "2023-03-21")

REGIMES: dict[str, tuple[str, str]] = {
    "pre_crisis": ("2023-03-01", "2023-03-09"),
    "peak_crisis": ("2023-03-10", "2023-03-13"),
    "recovery": ("2023-03-14", "2023-03-21"),
}

# Intraday event markers (UTC) used on every SVB-window chart.
EVENTS: dict[str, str] = {
    "SVB failure": "2023-03-10 12:00",
    "USDC de-peg": "2023-03-11 08:00",
    "Re-peg": "2023-03-13 12:00",
}
CRISIS_SHADE = ("2023-03-10 12:00", "2023-03-13 12:00")

STABLECOIN_PAIRS = ("USDCUSD", "USDTUSD")

# Coinbase merged its USD and USDC order books in 2022: BTC-USDC prints are the BTC-USD
# book quoted 1:1, so the series duplicates BTC/USD and is excluded from per-market metrics.
MIRRORED_MARKETS = ("coinbase_BTCUSDC",)
BTC_PAIRS = ("BTCUSD", "BTCUSDC", "BTCUSDT")

# --------------------------------------------------------------------------- #
# Data cleaning (1-minute bars)
# --------------------------------------------------------------------------- #
BAR_MS = 60_000
SHORT_GAP_MAX = 5  # <= 5 missing bars  -> last observation carried forward
MEDIUM_GAP_MAX = 30  # 6-30 missing bars -> simple exponential smoothing forecast
ETS_LOOKBACK = 1_440  # bars of pre-gap history used to fit the ETS model (1 day)

# A market is treated as "dead" (stale placeholder quotes) if it never trades.
MIN_ACTIVE_SHARE = 0.01

# --------------------------------------------------------------------------- #
# Microstructure metrics
# --------------------------------------------------------------------------- #
IMPACT_WINDOW = "60min"  # Amihud, Kyle-Obizhaeva, bar-level Kyle lambda
SPREAD_WINDOW = 1_440  # Roll, Corwin-Schultz, Abdi-Ranaldo (bars)
EFFICIENCY_WINDOW = 60  # Kaufman efficiency ratio (bars)
VPIN_BUCKETS_PER_DAY = 50
VPIN_MIN_BUCKET_USD = 1_000.0
KYLE_OFI_WINDOW = 1_440  # rolling OLS window for order-flow-imbalance lambda

# Market Quality Index weights (spread, price impact, efficiency stress, VPIN).
MQI_WEIGHTS = {"spread": 0.30, "impact": 0.30, "efficiency": 0.20, "vpin": 0.20}

# --------------------------------------------------------------------------- #
# Volatility spillovers (TVP-VAR, Antonakakis & Gabauer 2017)
# --------------------------------------------------------------------------- #
SPILLOVER_HORIZON = 10
TVP_LAMBDA = 0.99  # coefficient forgetting factor
TVP_KAPPA = 0.96  # covariance (EWMA) decay factor
TVP_LAGS = 2

# --------------------------------------------------------------------------- #
# Peg-deviation panel regressions
# --------------------------------------------------------------------------- #
REG_START = "2022-09-30"
CALM_WINDOW = ("2023-07-01", "2025-12-31")
SVB_STRESS_WINDOW = ("2023-03-09", "2023-03-20")
HIGH_VIX_QUANTILE = 0.90
HAC_LAGS = 7
WINSOR_QUANTILES = (0.001, 0.999)

# --------------------------------------------------------------------------- #
# Reserve stress tests (Basel III-style)
# --------------------------------------------------------------------------- #
RUN_SHOCK = 0.30  # Scenario A: share of supply redeemed
FIRE_SALE_HAIRCUT = 0.02  # Scenario A: slippage on assets sold to meet redemptions
BTC_SHOCK = 0.50  # Scenario B
GOLD_SHOCK = 0.15  # Scenario B
RATE_SHOCK = 0.03  # Scenario B: +300 bp parallel shift
CORP_BOND_DEFAULT = 0.10  # Scenario C
SECURED_LOAN_HAIRCUT = 0.25  # Scenario C
OTHER_INV_HAIRCUT = 0.50  # Scenario C
CIRCLE_TBILL_DURATION_DAYS = 45
