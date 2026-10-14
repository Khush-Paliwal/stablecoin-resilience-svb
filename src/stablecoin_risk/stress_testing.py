"""Basel III-style stress tests on reconstructed daily issuer balance sheets.

Equity cushion E_t = marked-to-market reserves - tokens outstanding.

Scenario A - Liquidity run
    A ``RUN_SHOCK`` share of supply is redeemed. Cash + overnight repo pay first;
    any shortfall is met by selling T-bills/bonds at a ``FIRE_SALE_HAIRCUT``
    discount, and that loss hits equity. (Basel LCR assumes 5-10% run-off on
    stable retail deposits; USDC lost ~$3.3bn of reserve access, roughly 8% of
    supply, in one weekend - 30% is a deliberately severe tail.)

Scenario B - Market flash crash
    BTC -50%, gold -15%, and a +300 bp parallel rate shock applied to T-bill
    holdings through duration: loss = value x (duration_days / 365) x 3%.

Scenario C - Credit contagion ("doomsday")
    Scenario B plus 10% default on corporate bonds, 25% haircut on secured loans
    and 50% on other investments (2008-crisis loss rates; crypto-collateralised
    loans behaved this way at Celsius and FTX).

Circle holds only cash, overnight repo and T-bills, so its Scenario C equals B.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stablecoin_risk import config


@dataclass(frozen=True)
class StressParameters:
    run_shock: float = config.RUN_SHOCK
    fire_sale_haircut: float = config.FIRE_SALE_HAIRCUT
    btc_shock: float = config.BTC_SHOCK
    gold_shock: float = config.GOLD_SHOCK
    rate_shock: float = config.RATE_SHOCK
    corp_bond_default: float = config.CORP_BOND_DEFAULT
    secured_loan_haircut: float = config.SECURED_LOAN_HAIRCUT
    other_inv_haircut: float = config.OTHER_INV_HAIRCUT
    circle_tbill_duration_days: float = config.CIRCLE_TBILL_DURATION_DAYS


SCENARIOS = ["Baseline", "A: Liquidity run", "B: Flash crash", "C: Credit contagion"]


@dataclass(frozen=True)
class IssuerColumns:
    """Column names for one issuer's reconstructed balance sheet."""

    supply: str
    cash: str
    overnight_repo: str
    tbills: str
    tbill_duration: str | None = None
    btc: str | None = None
    gold: str | None = None
    corporate_bonds: str | None = None
    secured_loans: str | None = None
    other_investments: str | None = None
    reserves: str = "New Estimate Reserves"


ISSUERS = {
    "tether": IssuerColumns(
        supply="USDT Market Cap",
        cash="Cash + Banks Depo",
        overnight_repo="O/n Repo MtM",
        tbills="Tbills Notional",
        tbill_duration="Tbills Duration",
        btc="BTC Value",
        gold="Gold Value",
        corporate_bonds="Corporate Bonds",
        secured_loans="Secured Loans",
        other_investments="Other inv",
    ),
    "circle": IssuerColumns(
        supply="USDC Market Cap",
        cash="Cash & Deposits",
        overnight_repo="O/N Repo",
        tbills="Tbills",
    ),
}


def _col(df: pd.DataFrame, name: str | None) -> pd.Series:
    return df[name].fillna(0.0) if name else pd.Series(0.0, index=df.index)


def run_stress_tests(df: pd.DataFrame, issuer: str, params: StressParameters | None = None) -> pd.DataFrame:
    """Daily equity cushion (USD) under the baseline and Scenarios A-C."""
    params = params or StressParameters()
    c = ISSUERS[issuer.lower()]
    baseline = df[c.reserves] - df[c.supply]

    redemptions = params.run_shock * df[c.supply]
    liquid = _col(df, c.cash) + _col(df, c.overnight_repo)
    fire_sale_loss = (redemptions - liquid).clip(lower=0) * params.fire_sale_haircut

    duration = df[c.tbill_duration] if c.tbill_duration else params.circle_tbill_duration_days
    market_loss = (
        params.btc_shock * _col(df, c.btc)
        + params.gold_shock * _col(df, c.gold)
        + _col(df, c.tbills) * (duration / 365) * params.rate_shock
    )
    credit_loss = (
        params.corp_bond_default * _col(df, c.corporate_bonds)
        + params.secured_loan_haircut * _col(df, c.secured_loans)
        + params.other_inv_haircut * _col(df, c.other_investments)
    )
    return pd.DataFrame(
        {
            "Baseline": baseline,
            "A: Liquidity run": baseline - fire_sale_loss,
            "B: Flash crash": baseline - market_loss,
            "C: Credit contagion": baseline - market_loss - credit_loss,
        },
        index=df.index,
    )


def summarize(
    cushions: pd.DataFrame, supply: pd.Series, window: tuple[str, str] = config.SVB_STRESS_WINDOW
) -> pd.DataFrame:
    """Per-scenario solvency summary.

    * ``days_insolvent_pct`` - share of all days with a negative cushion (headline metric;
      robust to the day-to-day noise of the interpolated reconstruction)
    * ``median_cushion_pct_supply`` - typical loss-absorbing capacity
    * ``min_cushion_bn`` / date - worst day in the sample
    * ``svb_window_*`` - worst cushion inside the SVB stress window and its verdict
    """
    rows = {}
    in_window = cushions.loc[window[0] : window[1]]
    for s in cushions:
        col = cushions[s]
        w = in_window[s]
        rows[s] = {
            "days_insolvent_pct": float((col < 0).mean() * 100),
            "median_cushion_pct_supply": float((col / supply).median() * 100),
            "min_cushion_bn": col.min() / 1e9,
            "min_cushion_date": col.idxmin().date(),
            "svb_window_min_bn": w.min() / 1e9,
            "svb_window_days_insolvent": int((w < 0).sum()),
            "svb_window_verdict": "FAIL" if (w < 0).any() else "pass",
        }
    return pd.DataFrame(rows).T


def equity_ratio(df: pd.DataFrame, issuer: str) -> pd.Series:
    """Equity buffer K = (reserves - supply) / reserves."""
    c = ISSUERS[issuer.lower()]
    return ((df[c.reserves] - df[c.supply]) / df[c.reserves]).rename("equity_ratio")


def liquidity_coverage(df: pd.DataFrame, issuer: str, run_shock: float = config.RUN_SHOCK) -> pd.Series:
    """(cash + overnight repo) / redemptions under the run scenario (>= 1 means fully covered)."""
    c = ISSUERS[issuer.lower()]
    liquid = _col(df, c.cash) + _col(df, c.overnight_repo)
    return (liquid / (run_shock * df[c.supply])).replace([np.inf], np.nan).rename("liquidity_coverage")
