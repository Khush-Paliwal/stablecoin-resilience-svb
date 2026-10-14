"""Peg-deviation panel regressions on issuer balance-sheet fundamentals (Tables III-IV).

    y_{e,t} = a + rho y_{e,t-1} + B' f(L_t, K_t) + g VIX_t + d1 HL_{e,t-1} + d2 lnVol_{e,t-1}
              + exchange FE + e_{e,t}

* y      : log peg deviation ln(P/1) of the stablecoin on exchange e, day t
           (winsorised at the 0.1% / 99.9% quantiles)
* L      : liquidity ratio  = (cash + deposits) / reserves
* K      : equity buffer    = (reserves - tokens outstanding) / reserves
* f(.)   : USDC uses a composite safety score z(L) + z(K) (L and K are nearly
           collinear in Circle's narrow-bank balance sheet); USDT enters z(L), z(K)
           separately.

Stress specifications interact the balance-sheet terms and (centred) VIX with a
regime dummy: top-decile VIX (main) or the SVB window (appendix).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

from stablecoin_risk import config
from stablecoin_risk.data.loaders import load_regression_inputs


@dataclass(frozen=True)
class CoinSpec:
    coin: str
    separate_lk: bool  # False -> composite safety score z(L) + z(K)


COIN_SPECS = {"USDC": CoinSpec("USDC", separate_lk=False), "USDT": CoinSpec("USDT", separate_lk=True)}


def winsorize(s: pd.Series, quantiles: tuple[float, float] = config.WINSOR_QUANTILES) -> pd.Series:
    lo, hi = s.quantile(quantiles[0]), s.quantile(quantiles[1])
    return s.clip(lower=lo, upper=hi)


def zscore(s: pd.Series) -> pd.Series:
    """In-sample z-score (population SD), NaN if the series is constant."""
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if np.isfinite(sd) and sd > 0 else s * np.nan


def build_panel(
    daily: pd.DataFrame,
    balance_sheet: pd.DataFrame,
    start: str = config.REG_START,
    svb_window: tuple[str, str] = config.SVB_STRESS_WINDOW,
    vix_quantile: float = config.HIGH_VIX_QUANTILE,
) -> pd.DataFrame:
    """Merge the exchange-day panel with daily balance-sheet ratios and build regressors."""
    df = daily.merge(balance_sheet[["date", "L", "K", "VIX"]], on="date", how="left")
    df["exchange"] = df["exchange"].astype(str).str.lower().str.strip()
    df = df[df["date"] >= pd.Timestamp(start)].sort_values(["exchange", "date"])
    df = df.replace([np.inf, -np.inf], np.nan)

    num = lambda c: pd.to_numeric(df[c], errors="coerce")  # noqa: E731
    df["y"] = winsorize(num("log_peg"))
    df["hl"] = num("hl_spread")
    df["lvol"] = np.log1p(num("volume"))
    df["L"], df["K"], df["vix"] = num("L"), num("K"), num("VIX")

    by_ex = df.groupby("exchange")
    df["y_lag1"] = by_ex["y"].shift(1)
    df["hl_lag1"] = by_ex["hl"].shift(1)
    df["lvol_lag1"] = by_ex["lvol"].shift(1)

    df["stress_svb"] = df["date"].between(pd.Timestamp(svb_window[0]), pd.Timestamp(svb_window[1])).astype(int)
    df["stress_vix"] = (df["vix"] >= df["vix"].quantile(vix_quantile)).astype(int)
    return df.reset_index(drop=True)


def design_matrix(
    panel: pd.DataFrame, separate_lk: bool, stress_col: str | None = None
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Return (y, X, sample) for the main (``stress_col=None``) or stress-interaction spec."""
    required = ["y", "y_lag1", "exchange", "date", "L", "K", "vix", "hl_lag1", "lvol_lag1"]
    if stress_col:
        required.append(stress_col)
    d = panel.dropna(subset=required).copy()
    d["zL"], d["zK"] = zscore(d["L"]), zscore(d["K"])
    bs_terms = ["zL", "zK"] if separate_lk else ["safety"]
    if not separate_lk:
        d["safety"] = d["zL"] + d["zK"]

    if stress_col is None:
        rhs = ["y_lag1", *bs_terms, "vix", "hl_lag1", "lvol_lag1"]
    else:
        d["stress"] = d[stress_col].astype(int)
        d["vix_c"] = d["vix"] - d["vix"].mean()
        for term in [*bs_terms, "vix_c"]:
            d[f"{term}_x_stress"] = d[term] * d["stress"]
        rhs = ["y_lag1", *bs_terms, "vix_c", "stress", *[f"{t}_x_stress" for t in bs_terms], "vix_c_x_stress",
               "hl_lag1", "lvol_lag1"]

    fe = pd.get_dummies(d["exchange"], prefix="fe", drop_first=True).astype(float)
    X = sm.add_constant(pd.concat([d[rhs].astype(float), fe], axis=1), has_constant="add")
    return d["y"].astype(float), X, d


def fit(y: pd.Series, X: pd.DataFrame, sample: pd.DataFrame, cov: str = "hac", hac_lags: int = config.HAC_LAGS):
    """OLS with a choice of covariance estimator.

    hac            : Newey-West on the stacked panel (as reported in the paper)
    hac-panel      : Newey-West within each exchange
    driscoll-kraay : cross-sectional and serial correlation robust (time-grouped HAC)
    """
    model = sm.OLS(y, X)
    if cov == "hac":
        return model.fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    if cov == "hac-panel":
        groups = pd.factorize(sample["exchange"])[0]
        return model.fit(cov_type="hac-panel", cov_kwds={"groups": groups, "maxlags": hac_lags})
    if cov == "driscoll-kraay":
        time = pd.factorize(sample["date"], sort=True)[0]
        return model.fit(cov_type="hac-groupsum", cov_kwds={"time": time, "maxlags": hac_lags})
    raise ValueError(f"unknown cov '{cov}'")


def run_paper_specs(coin: str, cov: str = "hac", panel: pd.DataFrame | None = None) -> dict:
    """Estimate the four paper specifications for one coin.

    Returns ``{"Full": res, "Calm": res, "High-VIX x stress": res, "SVB x stress": res}``.
    """
    spec = COIN_SPECS[coin.upper()]
    if panel is None:
        panel = build_panel(*load_regression_inputs(spec.coin))
    calm = panel[panel["date"].between(pd.Timestamp(config.CALM_WINDOW[0]), pd.Timestamp(config.CALM_WINDOW[1]))]
    runs = {
        "Full": (panel, None),
        "Calm": (calm, None),
        "High-VIX x stress": (panel, "stress_vix"),
        "SVB x stress": (panel, "stress_svb"),
    }
    out = {}
    for name, (data, stress) in runs.items():
        y, X, sample = design_matrix(data, spec.separate_lk, stress)
        out[name] = fit(y, X, sample, cov=cov)
    return out


def results_table(results: dict, sig: int = 3) -> pd.DataFrame:
    """Publication-style table: coefficient with stars over (standard error).

    Peg deviations are O(1e-4), so numbers are shown to ``sig`` significant figures.
    """
    stars = lambda p: "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""  # noqa: E731
    terms = []
    for res in results.values():
        terms += [t for t in res.params.index if t not in terms and not t.startswith("fe_")]
    rows = {}
    for t in terms:
        rows[t] = {n: (f"{r.params[t]:.{sig}g}{stars(r.pvalues[t])}" if t in r.params else "") for n, r in results.items()}
        rows[f"  ({t})"] = {n: (f"({r.bse[t]:.{sig}g})" if t in r.params else "") for n, r in results.items()}
    rows["Exchange FE"] = {n: "Yes" for n in results}
    rows["Observations"] = {n: f"{int(r.nobs):,}" for n, r in results.items()}
    rows["R-squared"] = {n: f"{r.rsquared:.3f}" for n, r in results.items()}
    return pd.DataFrame(rows).T
