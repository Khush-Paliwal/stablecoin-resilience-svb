"""Volatility connectedness between stablecoins, crypto and traditional markets.

Pipeline
--------
1. Parkinson (1980) range-based daily volatility for each asset.
2. Descriptive statistics (ADF, Jarque-Bera, Ljung-Box) and a stablecoin
   volatility regression with Newey-West errors (Table I).
3. Diebold-Yilmaz (2012) static spillover table from a VAR(p) GFEVD.
4. TVP-VAR dynamic connectedness (Antonakakis & Gabauer 2017; Koop & Korobilis
   2014 forgetting-factor Kalman filter), averaged over time (Table II).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stablecoin_risk import config

# Asset label -> (high column, low column) in the merged daily frame.
PARKINSON_ASSETS: dict[str, tuple[str, str]] = {
    "USDT": ("USDT_USD_High", "USDT_USD_Low"),
    "USDC": ("USDC_USD_High", "USDC_USD_Low"),
    "BTC": ("BTC_USD_High", "BTC_USD_Low"),
    "ETH": ("ETH_USD_High", "ETH_USD_Low"),
    "DXY": ("High_DXY", "Low_DXY"),
    "XAU": ("High_XAU", "Low_XAU"),
    "S&P500": ("High_SP", "Low_SP"),
    "KBW": ("High_KBW", "Low_KBW"),
}
SPILLOVER_ASSETS = list(PARKINSON_ASSETS)


# --------------------------------------------------------------------------- #
# Volatility panel and descriptive statistics
# --------------------------------------------------------------------------- #
def parkinson(high: pd.Series, low: pd.Series) -> pd.Series:
    """Parkinson (1980) daily volatility: |ln(H/L)| / sqrt(4 ln 2)."""
    return np.abs(np.log(high / low)) / np.sqrt(4 * np.log(2))


def build_volatility_panel(crypto: pd.DataFrame, traditional: pd.DataFrame) -> pd.DataFrame:
    """Merge daily crypto and traditional OHLC on trading days and compute volatilities.

    Crypto trades 24/7; weekends are dropped so every row is a day on which
    traditional markets were open. VIX enters as a level (mid of high/low).
    """
    merged = crypto.join(traditional, how="outer")
    merged = merged[merged.index.dayofweek < 5]
    vol = pd.DataFrame({name: parkinson(merged[h], merged[lo]) for name, (h, lo) in PARKINSON_ASSETS.items()})
    vol["VIX"] = (merged["High_VIX"] + merged["Low_VIX"]) / 2
    return vol.dropna()


def _stars(p: float) -> str:
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def descriptive_statistics(vol: pd.DataFrame) -> pd.DataFrame:
    """N, mean, SD, ADF, Jarque-Bera and Ljung-Box Q(10) per column (stars = significance)."""
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.stats.stattools import jarque_bera
    from statsmodels.tsa.stattools import adfuller

    rows = {}
    for col in vol:
        v = vol[col].dropna()
        adf, adf_p, *_ = adfuller(v, autolag="AIC")
        jb, jb_p, *_ = jarque_bera(v)
        lb = acorr_ljungbox(v, lags=[10], return_df=True).iloc[0]
        rows[col] = {
            "N": len(v),
            "Mean": round(v.mean(), 4),
            "SD": round(v.std(), 4),
            "ADF": f"{adf:.2f}{_stars(adf_p)}",
            "JB": f"{jb:.0f}{_stars(jb_p)}",
            "Q(10)": f"{lb['lb_stat']:.0f}{_stars(lb['lb_pvalue'])}",
        }
    return pd.DataFrame(rows).T


def stablecoin_volatility_regression(vol: pd.DataFrame, target: str, hac_lags: int = 10):
    """OLS of a stablecoin's volatility on market factors, both stablecoin lags and year FE.

    V_t = b0 + b'X_t + c1 V^USDT_{t-1} + c2 V^USDC_{t-1} + year dummies + e_t,
    with Newey-West HAC standard errors.
    """
    import statsmodels.api as sm

    df = vol.copy()
    df["USDT_lag1"] = df["USDT"].shift(1)
    df["USDC_lag1"] = df["USDC"].shift(1)
    years = pd.get_dummies(df.index.year, prefix="Y", drop_first=True).astype(float)
    years.index = df.index
    factors = ["BTC", "ETH", "DXY", "S&P500", "USDT_lag1", "USDC_lag1", "KBW", "XAU", "VIX"]
    X = sm.add_constant(pd.concat([df[factors], years], axis=1)).dropna()
    y = df.loc[X.index, target]
    return sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})


# --------------------------------------------------------------------------- #
# Generalised forecast-error variance decomposition
# --------------------------------------------------------------------------- #
def gfevd(ma_coefs: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Row-normalised generalised FEVD (Pesaran & Shin 1998), in percent.

    Parameters
    ----------
    ma_coefs : (H, N, N) moving-average matrices A_0..A_{H-1}.
    sigma    : (N, N) residual covariance.

    theta_ij = sigma_jj^-1 sum_h (e_i' A_h Sigma e_j)^2 / sum_h e_i' A_h Sigma A_h' e_i
    """
    a_sigma = np.einsum("hij,jk->hik", ma_coefs, sigma)
    num = (a_sigma**2).sum(axis=0) / np.diag(sigma)[None, :]
    den = np.einsum("hij,hij->i", a_sigma, ma_coefs)
    raw = num / den[:, None]
    return raw / raw.sum(axis=1, keepdims=True) * 100


def ma_coefficients(var_coefs: np.ndarray, horizon: int) -> np.ndarray:
    """MA(inf) matrices A_0..A_{H-1} from VAR lag matrices B_1..B_p (shape (p, N, N)).

    A_0 = I, A_h = sum_{j=1}^{min(h,p)} A_{h-j} B_j  (Lutkepohl 2005, eq. 2.1.22).
    """
    p, n, _ = var_coefs.shape
    out = np.zeros((horizon, n, n))
    out[0] = np.eye(n)
    for h in range(1, horizon):
        for j in range(1, min(h, p) + 1):
            out[h] += out[h - j] @ var_coefs[j - 1]
    return out


@dataclass
class SpilloverResult:
    table: pd.DataFrame  # N x N GFEVD plus FROM / TO / NET margins
    tsi: float  # total spillover index (%)
    lags: int


def spillover_table(theta: np.ndarray, labels: list[str]) -> SpilloverResult:
    """Diebold-Yilmaz connectedness table from a row-normalised GFEVD in percent.

    Rows = receivers, columns = transmitters. FROM_i = sum_{j!=i} theta_ij,
    TO_j = sum_{i!=j} theta_ij, NET = TO - FROM, TSI = mean(FROM).
    """
    diag = np.diag(theta)
    frm = theta.sum(axis=1) - diag
    to = theta.sum(axis=0) - diag
    table = pd.DataFrame(theta, index=labels, columns=labels)
    table["FROM others"] = frm
    table.loc["TO others"] = list(to) + [np.nan]
    table.loc["NET"] = list(to - frm) + [np.nan]
    return SpilloverResult(table=table, tsi=float(frm.mean()), lags=0)


def static_spillover(
    data: pd.DataFrame, horizon: int = config.SPILLOVER_HORIZON, maxlags: int = 10, lags: int | None = None
) -> SpilloverResult:
    """Diebold-Yilmaz (2012) static spillovers from a constant-parameter VAR (BIC lag order)."""
    from statsmodels.tsa.api import VAR

    model = VAR(data)
    p = lags if lags is not None else max(1, model.select_order(maxlags=maxlags).bic)
    res = model.fit(p)
    theta = gfevd(res.ma_rep(horizon - 1), np.asarray(res.sigma_u))
    out = spillover_table(theta, list(data.columns))
    out.lags = p
    return out


# --------------------------------------------------------------------------- #
# TVP-VAR with forgetting factors (Koop & Korobilis 2014)
# --------------------------------------------------------------------------- #
@dataclass
class TVPVARResult:
    coefs: np.ndarray  # (T, N, 1 + N*p) intercept + lag coefficients per equation
    sigmas: np.ndarray  # (T, N, N) EWMA residual covariances
    index: pd.Index
    labels: list[str]
    lags: int
    start: int  # first filtered observation used for inference


def tvp_var(
    data: pd.DataFrame,
    lags: int = config.TVP_LAGS,
    lam: float = config.TVP_LAMBDA,
    kappa: float = config.TVP_KAPPA,
    init_share: float = 0.10,
    prior_scale: float = 10.0,
) -> TVPVARResult:
    """Kalman-filter a TVP-VAR(p) with coefficient and covariance forgetting factors.

    beta_t = beta_{t-1} + u_t,  P_{t|t-1} = P_{t-1} / lambda
    Sigma_t = kappa Sigma_{t-1} + (1 - kappa) e_t e_t'
    The filter is initialised with OLS on the first ``init_share`` of the sample.
    """
    y = data.to_numpy(dtype=float)
    T, N = y.shape
    k = 1 + N * lags

    def regressors(t: int) -> np.ndarray:
        return np.concatenate([[1.0], y[t - lags : t][::-1].ravel()])

    init = max(int(init_share * T), N * lags + 5)
    X0 = np.array([regressors(t) for t in range(lags, init)])
    B0 = np.linalg.lstsq(X0, y[lags:init], rcond=None)[0]  # (k, N)
    resid = y[lags:init] - X0 @ B0
    sigma = resid.T @ resid / (init - lags - k)

    beta = B0.T.ravel()  # equation-major: [eq1 coefs, eq2 coefs, ...]
    P = np.eye(N * k) * prior_scale
    coefs = np.full((T, N, k), np.nan)
    sigmas = np.full((T, N, N), np.nan)
    eye_n = np.eye(N)
    for t in range(lags, T):
        Z = np.kron(eye_n, regressors(t))  # (N, N*k)
        P_pred = P / lam
        err = y[t] - Z @ beta
        sigma = kappa * sigma + (1 - kappa) * np.outer(err, err)
        sigma = (sigma + sigma.T) / 2
        S = Z @ P_pred @ Z.T + sigma
        K = np.linalg.solve(S, Z @ P_pred).T  # P Z' S^-1  (S, P symmetric)
        beta = beta + K @ err
        P = P_pred - K @ Z @ P_pred
        coefs[t] = beta.reshape(N, k)
        sigmas[t] = sigma
    return TVPVARResult(coefs, sigmas, data.index, list(data.columns), lags, init)


def tvp_connectedness(
    result: TVPVARResult, horizon: int = config.SPILLOVER_HORIZON, burn_in: int = 20
) -> tuple[SpilloverResult, pd.Series, np.ndarray]:
    """Time-varying GFEVD from a filtered TVP-VAR.

    Returns the time-averaged connectedness table, the dynamic total spillover
    index TSI_t, and the full (T', N, N) GFEVD array.
    """
    N, p = len(result.labels), result.lags
    start = result.start + burn_in
    thetas, dates = [], []
    for t in range(start, len(result.index)):
        lag_mats = result.coefs[t, :, 1:].reshape(N, p, N).transpose(1, 0, 2)  # (p, N, N)
        theta = gfevd(ma_coefficients(lag_mats, horizon), result.sigmas[t])
        if np.isfinite(theta).all():
            thetas.append(theta)
            dates.append(result.index[t])
    thetas = np.asarray(thetas)
    avg = spillover_table(thetas.mean(axis=0), result.labels)
    avg.lags = p
    off_diag = thetas.sum(axis=2) - np.einsum("tii->ti", thetas)
    tsi_t = pd.Series(off_diag.mean(axis=1), index=pd.Index(dates, name=result.index.name), name="TSI")
    return avg, tsi_t, thetas
