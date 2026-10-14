"""Market-microstructure estimators for 1-minute OHLCV bars and L1 order books.

All bar-level functions take the canonical OHLCV frame from
:mod:`stablecoin_risk.data.loaders` (UTC DatetimeIndex; open, high, low, close,
volume) and return a Series aligned to that index. Rolling windows are
backward-looking, so the value at minute *t* uses information up to *t* only.

References
----------
Amihud (2002) J. Financial Markets 5(1)            - illiquidity ratio
Roll (1984) J. Finance 39(4)                       - serial-covariance spread
Corwin & Schultz (2012) J. Finance 67(2)           - high-low spread
Abdi & Ranaldo (2017) Rev. Financial Studies 30(12) - close-high-low spread
Kyle & Obizhaeva (2016) Econometrica 84(4)         - invariance illiquidity
Easley, Lopez de Prado & O'Hara (2012) RFS 25(5)   - VPIN
Cont, Kukanov & Stoikov (2014) J. Fin. Econometrics 12(1) - order-flow imbalance
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd

from stablecoin_risk import config

_K = 3 - 2 * np.sqrt(2)  # Corwin-Schultz constant


def _returns(close: pd.Series) -> pd.Series:
    return close.pct_change()


def _tick_sign(df: pd.DataFrame) -> np.ndarray:
    """+1 / -1 / 0 trade-direction proxy: close above / below / at the bar midpoint."""
    mid = (df["high"] + df["low"]) / 2
    return np.sign(df["close"] - mid).to_numpy()


# --------------------------------------------------------------------------- #
# Price impact
# --------------------------------------------------------------------------- #
def amihud(df: pd.DataFrame, window: str = config.IMPACT_WINDOW) -> pd.Series:
    """Amihud (2002) illiquidity: rolling mean of |r_t| / dollar volume_t.

    Zero-volume bars are excluded (the ratio is undefined there).
    """
    dollar_volume = (df["close"] * df["volume"]).where(df["volume"] > 0)
    return (_returns(df["close"]).abs() / dollar_volume).rolling(window).mean().rename("amihud")


def kyle_lambda_bars(df: pd.DataFrame, window: str = config.IMPACT_WINDOW) -> pd.Series:
    """Bar-level Kyle's lambda: rolling Cov(r, signed volume) / Var(signed volume).

    Trade direction is inferred from the close's position in the bar range, so this
    is a coarse proxy; prefer :func:`kyle_lambda_ofi` when order-book data exist.
    Negative estimates (economically meaningless) are set to NaN.
    """
    signed_volume = pd.Series(df["volume"].to_numpy() * _tick_sign(df), index=df.index)
    ret = _returns(df["close"])
    lam = ret.rolling(window).cov(signed_volume) / signed_volume.rolling(window).var()
    return lam.where(lam >= 0).rename("kyle_lambda")


def kyle_obizhaeva(df: pd.DataFrame, window: str = config.IMPACT_WINDOW) -> pd.Series:
    """Kyle-Obizhaeva (2016) invariance illiquidity: (sigma^2 / dollar volume)^(1/3)."""
    ret = _returns(df["close"])
    var = (ret**2).rolling(window).mean()
    dollar_volume = (df["close"] * df["volume"]).rolling(window).sum()
    return ((var / dollar_volume.where(dollar_volume > 0)) ** (1 / 3)).rename("kyle_obizhaeva")


# --------------------------------------------------------------------------- #
# Spread estimators
# --------------------------------------------------------------------------- #
def roll_spread(df: pd.DataFrame, window: int = config.SPREAD_WINDOW) -> pd.Series:
    """Roll (1984) spread as a fraction of price: 2 sqrt(-Cov(dP_t, dP_{t-1})) / P_t.

    Positive autocovariance windows (no bid-ask bounce detected) map to zero.
    """
    dp = df["close"].diff()
    cov = dp.rolling(window).cov(dp.shift(1))
    return (2 * np.sqrt((-cov).clip(lower=0)) / df["close"]).rename("roll_spread")


def corwin_schultz(df: pd.DataFrame, window: int = config.SPREAD_WINDOW) -> pd.Series:
    """Corwin & Schultz (2012) high-low spread, rolling mean over ``window`` bars.

    S = 2 (e^alpha - 1) / (1 + e^alpha), with the two-bar beta/gamma computed from
    bars (t-1, t) so the estimate is causal. Negative alphas are set to zero, as in
    the original paper.

    Note: the competition code omitted the ``1 + e^alpha`` denominator, which
    roughly doubles the spread level; z-scored indices (MQI) are almost unaffected.
    """
    log_hl = np.log(df["high"] / df["low"])
    beta = log_hl**2 + log_hl.shift(1) ** 2
    high2 = np.maximum(df["high"], df["high"].shift(1))
    low2 = np.minimum(df["low"], df["low"].shift(1))
    gamma = np.log(high2 / low2) ** 2
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / _K - np.sqrt(gamma / _K)
    alpha = alpha.clip(lower=0)
    spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    return spread.rolling(window, min_periods=window // 2).mean().rename("cs_spread")


def abdi_ranaldo(df: pd.DataFrame, window: int = config.SPREAD_WINDOW) -> pd.Series:
    """Abdi & Ranaldo (2017) close-high-low spread.

    s^2 = 4 E[(c_t - eta_t)(c_t - eta_{t+1})] with eta the log mid-range. The term is
    lagged one bar so that it only uses data up to *t*, averaged over ``window``
    bars, truncated at zero and only then square-rooted (the paper's averaging
    order; the competition code took per-bar square roots, which biases upward).
    """
    log_c = np.log(df["close"])
    eta = (np.log(df["high"]) + np.log(df["low"])) / 2
    term = (4 * (log_c - eta) * (log_c - eta.shift(-1))).shift(1)
    s2 = term.rolling(window, min_periods=window // 2).mean()
    return np.sqrt(s2.clip(lower=0)).rename("ar_spread")


def effective_spread_hl(df: pd.DataFrame) -> pd.Series:
    """Bar range relative to midpoint, (H - L) / ((H + L) / 2): an upper bound on the quoted spread."""
    mid = (df["high"] + df["low"]) / 2
    return ((df["high"] - df["low"]) / mid).rename("hl_spread")


# --------------------------------------------------------------------------- #
# Efficiency, order-flow toxicity, resilience
# --------------------------------------------------------------------------- #
def efficiency_ratio(df: pd.DataFrame, window: int = config.EFFICIENCY_WINDOW) -> pd.Series:
    """Kaufman efficiency ratio: |P_t - P_{t-n}| / sum |dP| over the last ``window`` bars."""
    close = df["close"]
    path = close.diff().abs().rolling(window).sum()
    return ((close - close.shift(window)).abs() / path.where(path > 0)).rename("efficiency_ratio")


def vpin(
    df: pd.DataFrame,
    buckets_per_day: int = config.VPIN_BUCKETS_PER_DAY,
    min_bucket: float = config.VPIN_MIN_BUCKET_USD,
) -> pd.Series:
    """Volume-synchronised probability of informed trading (Easley et al., 2012).

    Bars are grouped into equal dollar-volume buckets sized at average daily dollar
    volume / ``buckets_per_day``; VPIN is |signed $vol| / $vol per bucket, mapped
    back to each bar. Bucket size uses the full sample (a mild look-ahead that only
    rescales bucket boundaries; documented in docs/methodology.md).
    """
    dollar_volume = df["close"] * df["volume"]
    n_days = max((df.index[-1] - df.index[0]).days, 1)
    bucket_size = max(dollar_volume.sum() / n_days / buckets_per_day, min_bucket)
    bucket = (dollar_volume.cumsum() // bucket_size).astype(int)
    signed = dollar_volume * _tick_sign(df)
    grouped = pd.DataFrame({"signed": signed, "total": dollar_volume, "bucket": bucket}).groupby("bucket")
    per_bucket = grouped["signed"].sum().abs() / grouped["total"].sum().replace(0, np.nan)
    return bucket.map(per_bucket).rename("vpin")


def liquidity_resilience(
    metric: pd.Series,
    baseline: tuple[str, str] = config.REGIMES["pre_crisis"],
    shock: tuple[str, str] = ("2023-03-10", "2023-03-12"),
    recovery: tuple[str, str] = ("2023-03-13", "2023-03-21"),
) -> pd.Series:
    """Share of the shock still outstanding: (m_t - baseline) / (peak - baseline) during recovery.

    1 = still at the shock peak, 0 = fully back to the pre-crisis median.
    """
    base = metric.loc[baseline[0] : baseline[1]].median()
    peak = metric.loc[shock[0] : shock[1]].max()
    return ((metric.loc[recovery[0] : recovery[1]] - base) / (peak - base)).rename("resilience")


# --------------------------------------------------------------------------- #
# Panel helpers
# --------------------------------------------------------------------------- #
def compute_all_metrics(df: pd.DataFrame, window: tuple[str, str] | None = config.SVB_WINDOW) -> pd.DataFrame:
    """Every bar-level metric for one market, as columns of one DataFrame.

    Zero-volume bars are dropped first: several estimators divide by volume and
    the stablecoin books trade in fewer than half of all minutes.
    """
    d = df[df["volume"] > 0]
    out = pd.concat(
        [
            d["close"],
            d["volume"],
            _returns(d["close"]).rename("ret"),
            amihud(d),
            kyle_lambda_bars(d),
            kyle_obizhaeva(d),
            roll_spread(d),
            corwin_schultz(d),
            abdi_ranaldo(d),
            effective_spread_hl(d),
            efficiency_ratio(d),
            vpin(d),
        ],
        axis=1,
    )
    if window is not None:
        out = out.loc[window[0] : f"{window[1]} 23:59:59"]
    return out


def regime_statistics(
    metrics: dict[str, pd.DataFrame],
    columns: Iterable[str] = ("amihud", "kyle_obizhaeva", "roll_spread", "cs_spread", "ar_spread", "vpin"),
    regimes: dict[str, tuple[str, str]] = config.REGIMES,
) -> pd.DataFrame:
    """Median and 95th percentile of each metric per (market, regime)."""
    rows = []
    for key, df in metrics.items():
        for regime, (start, end) in regimes.items():
            sub = df.loc[start : f"{end} 23:59:59"]
            row = {"market": key, "regime": regime, "n_bars": len(sub)}
            for col in columns:
                s = sub[col].dropna()
                row[f"{col}_median"] = s.median() if len(s) else np.nan
                row[f"{col}_p95"] = s.quantile(0.95) if len(s) else np.nan
            row["ret_std"] = sub["ret"].std()
            rows.append(row)
    return pd.DataFrame(rows).set_index(["market", "regime"])


def crisis_multipliers(regime_stats: pd.DataFrame, stat: str = "median") -> pd.DataFrame:
    """Peak-crisis / pre-crisis ratio of each metric's ``stat`` (median or p95)."""
    cols = [c for c in regime_stats.columns if c.endswith(f"_{stat}")]
    pre = regime_stats.xs("pre_crisis", level="regime")[cols]
    peak = regime_stats.xs("peak_crisis", level="regime")[cols]
    ratio = peak / pre.replace(0, np.nan)
    ratio.columns = [c.replace(f"_{stat}", "_x") for c in cols]
    return ratio


# --------------------------------------------------------------------------- #
# Cross-venue price dislocation
# --------------------------------------------------------------------------- #
def arbitrage_gap_bps(price_a: pd.Series, price_b: pd.Series) -> pd.Series:
    """|P_a - P_b| / P_b in basis points on the common timestamps."""
    a, b = price_a.align(price_b, join="inner")
    return ((a - b).abs() / b * 1e4).dropna()


def original_close(df: pd.DataFrame) -> pd.Series:
    """Close prices on traded, non-imputed bars only (for cross-venue comparisons)."""
    mask = df["volume"] > 0
    if "imputed" in df.columns:
        mask &= ~df["imputed"].astype(bool)
    return df.loc[mask, "close"]


def cross_quote_basis(usd_close: pd.Series, quote_close: pd.Series) -> pd.Series:
    """Relative basis (P_USD - P_Q) / P_Q of the same asset quoted in USD and in Q."""
    a, b = usd_close.align(quote_close, join="inner")
    return ((a - b) / b).dropna()


# --------------------------------------------------------------------------- #
# Market Quality Index
# --------------------------------------------------------------------------- #
def baseline_score(s: pd.Series, baseline: tuple[str, str] = config.REGIMES["pre_crisis"]) -> pd.Series:
    """Map a 'bad when high' metric to [0, 1] via its z-score vs the pre-crisis baseline.

    z in [-2, 6] is mapped linearly onto [0, 1]: 0.25 = baseline mean, 1 = six
    baseline standard deviations worse.
    """
    pre = s.loc[baseline[0] : f"{baseline[1]} 23:59:59"].dropna()
    if len(pre) < 5 or not np.isfinite(pre.std()) or pre.std() == 0:
        return pd.Series(np.nan, index=s.index)
    z = (s - pre.mean()) / pre.std()
    return ((z + 2) / 8).clip(0, 1)


def efficiency_stress(s: pd.Series, baseline: tuple[str, str] = config.REGIMES["pre_crisis"]) -> pd.Series:
    """|ER_t - baseline mean| / (4 baseline sd), clipped to [0, 1].

    Both trending panic (ER -> 1) and chaotic noise (ER -> 0) count as stress.
    """
    pre = s.loc[baseline[0] : f"{baseline[1]} 23:59:59"].dropna()
    if len(pre) < 5 or pre.std() == 0:
        return pd.Series(np.nan, index=s.index)
    return ((s - pre.mean()).abs() / pre.std() / 4).clip(0, 1)


def market_quality_index(metrics: pd.DataFrame, weights: dict[str, float] = config.MQI_WEIGHTS) -> pd.Series:
    """Composite Market Quality Index on a 0-100 scale (100 = pristine).

    MQI = 100 * sum_k w_k (1 - score_k) over k in {spread (Corwin-Schultz),
    impact (Amihud), efficiency stress, VPIN}. If VPIN is unavailable the remaining
    weights are renormalised to sum to one.
    """
    components = {
        "spread": baseline_score(metrics["cs_spread"]),
        "impact": baseline_score(metrics["amihud"]),
        "efficiency": efficiency_stress(metrics["efficiency_ratio"]),
    }
    if metrics.get("vpin") is not None and metrics["vpin"].notna().sum() > 10:
        components["vpin"] = baseline_score(metrics["vpin"])
    used = {k: weights[k] for k in components}
    total = sum(used.values())
    scores = pd.concat(components, axis=1).dropna()
    mqi = sum((used[k] / total) * (1 - scores[k]) for k in components) * 100
    return mqi.rename("mqi")


def compute_mqi_panel(
    metrics: dict[str, pd.DataFrame], select: Callable[[str], bool] = lambda key: True
) -> dict[str, pd.Series]:
    """MQI for every market in ``metrics`` whose key passes ``select``."""
    return {k: market_quality_index(m) for k, m in metrics.items() if select(k)}


# --------------------------------------------------------------------------- #
# Order-book Kyle's lambda
# --------------------------------------------------------------------------- #
def order_flow_imbalance(ob: pd.DataFrame) -> pd.Series:
    """Cont-Kukanov-Stoikov (2014) best-level order-flow imbalance between snapshots.

    e_t = 1{Pb_t >= Pb_t-1} qb_t - 1{Pb_t <= Pb_t-1} qb_t-1
        - 1{Pa_t <= Pa_t-1} qa_t + 1{Pa_t >= Pa_t-1} qa_t-1
    """
    pb, qb = ob["bid_price_1"], ob["bid_size_1"]
    pa, qa = ob["ask_price_1"], ob["ask_size_1"]
    pb0, qb0, pa0, qa0 = pb.shift(), qb.shift(), pa.shift(), qa.shift()
    # Cast indicators to float: unary minus on a boolean Series is a logical NOT in pandas.
    ind = lambda cond: cond.astype(float)  # noqa: E731
    bid = ind(pb >= pb0) * qb - ind(pb <= pb0) * qb0
    ask = ind(pa >= pa0) * qa0 - ind(pa <= pa0) * qa
    return (bid + ask).where(pb0.notna()).rename("ofi")


def kyle_lambda_ofi(ob: pd.DataFrame, bar: str = "1min", window: int = config.KYLE_OFI_WINDOW) -> pd.Series:
    """Rolling OLS of mid-price changes on OFI: dP_t = a + lambda * OFI_t + e_t.

    Snapshots are aggregated to ``bar`` buckets (sum of dP and OFI); bars without
    any snapshot are dropped rather than zero-filled.
    """
    from statsmodels.regression.rolling import RollingOLS

    frame = pd.DataFrame({"dP": ob["mid_price"].diff(), "OFI": order_flow_imbalance(ob)}).dropna()
    counts = frame["dP"].resample(bar).count()
    bars = frame.resample(bar).sum()[counts > 0]
    if len(bars) <= window:
        raise ValueError(f"Need more than {window} bars for the rolling regression, got {len(bars)}")
    X = np.column_stack([np.ones(len(bars)), bars["OFI"].to_numpy()])
    fit = RollingOLS(bars["dP"].to_numpy(), X, window=window).fit(params_only=True)
    return pd.Series(fit.params[:, 1], index=bars.index, name="kyle_lambda_ofi")
