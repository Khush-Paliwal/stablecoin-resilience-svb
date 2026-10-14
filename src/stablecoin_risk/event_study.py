"""Intraday volatility event study (exploratory; GENIUS Act milestones, 2025).

Question: did stablecoin volatility jump when the GENIUS Act passed the Senate
(2025-06-17), the House (2025-07-17) and was signed (2025-07-18)?

Method
------
1. Keep only *active* minutes (price present and a trade printed) so stale
   quotes do not masquerade as zero volatility.
2. Sample log prices every ``sampling`` (default 5 min) and build hourly
   realised variance RV_h = sum r^2; require ``min_returns`` per hour.
3. Remove hour-of-week seasonality estimated away from the events.
4. For each event and window W, regress deseasonalised log RV on a post dummy
   and a linear trend with Newey-West errors (a "jump" regression).
5. Placebo inference: re-run the regression at random dates with the same
   hour-of-week and report the share of |beta| at least as large.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

GENIUS_ACT_EVENTS = {
    "Senate passed": pd.Timestamp("2025-06-17 21:11", tz="UTC"),
    "House passed": pd.Timestamp("2025-07-17 19:53", tz="UTC"),
    "Signed into law": pd.Timestamp("2025-07-18 00:00", tz="UTC"),
}
EPS = 1e-12


def active_prices(df: pd.DataFrame) -> pd.Series:
    """Close on minutes with a trade, NaN otherwise, on a strict 1-minute grid."""
    grid = df.asfreq("1min")
    traded = grid["num_trades"].fillna(0) > 0 if grid["num_trades"].notna().any() else grid["volume"].fillna(0) > 0
    return grid["close"].where(traded)


def realized_variance(price: pd.Series, sampling: str = "5min", bin_size: str = "1h",
                      min_returns: int = 8) -> pd.DataFrame:
    """Hourly realised variance from sparsely sampled log returns."""
    r = np.log(price.resample(sampling).last()).diff()
    out = pd.DataFrame({"rv": (r**2).resample(bin_size).sum(min_count=1),
                        "n_ret": r.notna().resample(bin_size).sum()})
    out["log_rv"] = np.log(out["rv"] + EPS).where(out["n_ret"] >= min_returns)
    return out


def hour_of_week(idx: pd.DatetimeIndex) -> np.ndarray:
    return np.asarray(idx.weekday * 24 + idx.hour)


def event_mask(idx: pd.DatetimeIndex, events, days: int = 7) -> np.ndarray:
    mask = np.zeros(len(idx), dtype=bool)
    for t0 in events:
        mask |= (idx >= t0 - pd.Timedelta(days=days)) & (idx <= t0 + pd.Timedelta(days=days))
    return mask


def deseasonalize(y: pd.Series, exclude: np.ndarray) -> pd.Series:
    """Subtract the hour-of-week mean estimated on non-event hours."""
    how = pd.Series(hour_of_week(y.index), index=y.index)
    means = y[~exclude].groupby(how[~exclude]).mean()
    return y - how.map(means)


def jump_regression(y: pd.Series, t0: pd.Timestamp, window_h: int, buffer_min: int = 10, maxlags: int = 6):
    """OLS of y on [1, post, hours-from-event] in +/- ``window_h`` hours; None if too few obs."""
    d = y.loc[t0 - pd.Timedelta(hours=window_h): t0 + pd.Timedelta(hours=window_h)].dropna()
    d = d[(d.index < t0 - pd.Timedelta(minutes=buffer_min)) | (d.index >= t0 + pd.Timedelta(minutes=buffer_min))]
    post = (d.index >= t0).astype(float)
    if len(d) < 10 or post.sum() < 5 or (1 - post).sum() < 5:
        return None
    X = sm.add_constant(pd.DataFrame({"post": post, "trend": (d.index - t0) / pd.Timedelta(hours=1)}, index=d.index))
    return sm.OLS(d, X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})


def placebo_pvalue(y: pd.Series, t0: pd.Timestamp, window_h: int, beta: float, exclude: np.ndarray,
                   n: int = 200, seed: int = 7) -> tuple[float, int]:
    """Share of placebo |beta_post| >= |beta| at random same-hour-of-week dates."""
    rng = np.random.default_rng(seed)
    cand = y.index[(hour_of_week(y.index) == hour_of_week(pd.DatetimeIndex([t0]))[0]) & ~exclude]
    betas = []
    for t in rng.choice(cand, size=min(n, len(cand)), replace=False) if len(cand) else []:
        res = jump_regression(y, pd.Timestamp(t), window_h)
        if res is not None:
            betas.append(res.params["post"])
    if not betas:
        return np.nan, 0
    return float(np.mean(np.abs(betas) >= abs(beta))), len(betas)


def run_event_study(log_rv: pd.Series, events: dict[str, pd.Timestamp] = GENIUS_ACT_EVENTS,
                    windows_h=(6, 12, 24, 48), n_placebo: int = 200) -> pd.DataFrame:
    """Jump estimates, HAC p-values and placebo p-values for every event x window."""
    exclude = event_mask(log_rv.index, events.values())
    y = deseasonalize(log_rv, exclude)
    rows = []
    for name, t0 in events.items():
        for w in windows_h:
            res = jump_regression(y, t0, w)
            row = {"event": name, "window_h": w, "n": 0}
            if res is not None:
                beta = res.params["post"]
                p_plc, n_plc = placebo_pvalue(y, t0, w, beta, exclude, n=n_placebo)
                row.update(n=int(res.nobs), beta_post=beta, rv_change_pct=(np.exp(beta) - 1) * 100,
                           hac_p=res.pvalues["post"], placebo_p=p_plc, n_placebo=n_plc)
            rows.append(row)
    return pd.DataFrame(rows)
