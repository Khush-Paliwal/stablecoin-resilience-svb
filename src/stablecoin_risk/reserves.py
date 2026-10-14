"""Issuer balance-sheet facts from public attestations (Figures 4 and 5).

Quarter-end totals, tokens outstanding and equity ratios come from the
attestation reports of Tether (tether.to/transparency) and Circle
(circle.com/transparency), compiled in ``data/reserves``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from stablecoin_risk import config

_WORKBOOK = "reserves_analysis.xlsx"

# Tether attestation line items -> five reporting buckets used in Figure 5.
TETHER_BUCKETS: dict[str, list[str]] = {
    "Cash deposits": ["Cash&BanksDepo"],
    "US Treasuries": ["Tbills"],
    "Repo": ["O/N Repo", "Reverse Repo"],
    "Other cash equivalents": ["Money Mkts", "CDs/CPs", "Non US Tbills"],
    "Corporate bonds & longer-term": ["SecuredLoans", "Corporate Bonds", "Metals and CB", "BTC",
                                      "Other Inv (Digital tokens)"],
}
CIRCLE_BUCKETS = {"Cash Deposits": "Cash deposits", "US Treasuries": "US Treasuries", "Repo": "Repo"}


def quarterly_equity(path: str | Path | None = None) -> pd.DataFrame:
    """Quarter-end total assets, tokens issued and equity % for both issuers."""
    path = Path(path or config.RESERVES_DIR / _WORKBOOK)
    df = pd.read_excel(path, sheet_name="ReservesUSDT").dropna(subset=["Date"])
    df = df.rename(columns={
        "TETHER Total Assets": "tether_assets", "USDT Issued": "usdt_issued", "Tether Equity %": "tether_equity_pct",
        "CIRCLE Total Assets": "circle_assets", "USDC Issued": "usdc_issued", "Circle Equity %": "circle_equity_pct",
    })
    df["Date"] = pd.to_datetime(df["Date"])
    cols = ["tether_assets", "usdt_issued", "tether_equity_pct", "circle_assets", "usdc_issued", "circle_equity_pct"]
    return df.set_index("Date")[cols].astype(float)


def tether_composition(path: str | Path | None = None, start: str = "2022-09-30") -> pd.DataFrame:
    """Quarter-end Tether reserve shares (rows sum to 1) in the Figure-5 buckets."""
    path = Path(path or config.RESERVES_DIR / _WORKBOOK)
    raw = pd.read_excel(path, sheet_name="ReservesCompositionDetailed").dropna(subset=["Date"])
    raw["Date"] = pd.to_datetime(raw["Date"])
    raw = raw.set_index("Date").loc[start:]
    values = pd.DataFrame({
        bucket: sum(pd.to_numeric(raw[c], errors="coerce").fillna(0) for c in cols)
        for bucket, cols in TETHER_BUCKETS.items()
    })
    return values.div(values.sum(axis=1), axis=0)


def circle_composition(path: str | Path | None = None, start: str = "2022-09-30") -> pd.DataFrame:
    """Quarter-end Circle reserve shares (cash deposits / Treasuries / repo)."""
    path = Path(path or config.RESERVES_DIR / "circle_attestations.xlsx")
    raw = pd.read_excel(path, sheet_name="Reserves by category")
    raw.columns = ["asset", "value_bn", "Date"]
    raw = raw[raw["asset"].isin(CIRCLE_BUCKETS)].copy()
    raw["value_bn"] = pd.to_numeric(raw["value_bn"], errors="coerce")
    wide = raw.pivot_table(index="Date", columns="asset", values="value_bn", aggfunc="sum").fillna(0)
    wide = wide.rename(columns=CIRCLE_BUCKETS)[list(CIRCLE_BUCKETS.values())]
    # Attestation dates do not always fall on the last calendar day: use the last report per quarter.
    wide = wide.groupby(wide.index.to_period("Q")).tail(1).loc[start:]
    wide.index = wide.index.to_period("Q").to_timestamp(how="end").normalize()
    return wide.div(wide.sum(axis=1), axis=0)
