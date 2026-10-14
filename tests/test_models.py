import numpy as np
import pandas as pd
import pytest

from stablecoin_risk import regressions
from stablecoin_risk import spillover as sp
from stablecoin_risk import stress_testing as st


# --------------------------------------------------------------------------- spillover
@pytest.fixture
def var_data(rng):
    n, k = 600, 3
    A = np.array([[0.5, 0.2, 0.0], [0.0, 0.4, 0.1], [0.1, 0.0, 0.3]])
    y = np.zeros((n, k))
    for t in range(1, n):
        y[t] = A @ y[t - 1] + rng.normal(size=k)
    return pd.DataFrame(y, columns=["a", "b", "c"], index=pd.bdate_range("2020-01-01", periods=n))


def test_parkinson_formula():
    v = sp.parkinson(pd.Series([110.0]), pd.Series([100.0]))
    assert np.isclose(v.iloc[0], np.log(1.1) / np.sqrt(4 * np.log(2)))


def test_ma_coefficients_match_statsmodels(var_data):
    from statsmodels.tsa.api import VAR

    res = VAR(var_data).fit(2)
    ours = sp.ma_coefficients(res.coefs, 8)
    assert np.allclose(ours, res.ma_rep(7))


def test_gfevd_rows_sum_to_100_and_tsi_bounds(var_data):
    out = sp.static_spillover(var_data, lags=1)
    theta = out.table.iloc[:3, :3].to_numpy()
    assert np.allclose(theta.sum(axis=1), 100)
    assert 0 < out.tsi < 100
    assert np.isclose(out.table.loc["NET"].iloc[:3].sum(), 0, atol=1e-8)


def test_tvp_var_tracks_constant_var(var_data):
    res = sp.tvp_var(var_data, lags=1, lam=0.999, kappa=0.99)
    avg, tsi_t, thetas = sp.tvp_connectedness(res, horizon=5)
    static = sp.static_spillover(var_data, lags=1, horizon=5)
    assert thetas.shape[1:] == (3, 3)
    assert abs(avg.tsi - static.tsi) < 10  # same DGP -> similar connectedness
    assert tsi_t.between(0, 100).all()


# --------------------------------------------------------------------------- regressions
def test_winsorize_and_zscore():
    s = pd.Series(np.arange(1_000, dtype=float))
    w = regressions.winsorize(s, (0.01, 0.99))
    assert w.min() == s.quantile(0.01) and w.max() == s.quantile(0.99)
    z = regressions.zscore(pd.Series([1.0, 2.0, 3.0]))
    assert np.isclose(z.mean(), 0) and np.isclose(z.std(ddof=0), 1)
    assert regressions.zscore(pd.Series([1.0, 1.0])).isna().all()


def test_design_matrix_specs(rng):
    dates = pd.date_range("2023-01-01", periods=200, freq="D")
    daily = pd.concat([pd.DataFrame({"date": dates, "exchange": ex, "log_peg": rng.normal(0, 1e-4, 200),
                                     "hl_spread": rng.uniform(0, 1e-3, 200), "volume": rng.uniform(1, 9, 200)})
                       for ex in ("kraken", "coinbase")])
    bs = pd.DataFrame({"date": dates, "L": rng.uniform(0.1, 0.3, 200), "K": rng.uniform(0, 0.05, 200),
                       "VIX": rng.uniform(12, 35, 200)})
    panel = regressions.build_panel(daily, bs, start="2023-01-01", svb_window=("2023-03-09", "2023-03-20"))
    y, X, _ = regressions.design_matrix(panel, separate_lk=False)
    assert {"const", "y_lag1", "safety", "vix", "hl_lag1", "lvol_lag1", "fe_kraken"} <= set(X.columns)
    y, X, sample = regressions.design_matrix(panel, separate_lk=True, stress_col="stress_svb")
    assert {"zL_x_stress", "zK_x_stress", "vix_c_x_stress"} <= set(X.columns)
    for cov in ("hac", "hac-panel", "driscoll-kraay"):
        assert np.isfinite(regressions.fit(y, X, sample, cov=cov).bse).all()


# --------------------------------------------------------------------------- stress tests
def test_stress_scenarios_hand_computed():
    df = pd.DataFrame({
        "New Estimate Reserves": [105.0], "USDT Market Cap": [100.0], "Cash + Banks Depo": [10.0],
        "O/n Repo MtM": [5.0], "Tbills Notional": [60.0], "Tbills Duration": [73.0], "BTC Value": [4.0],
        "Gold Value": [2.0], "Corporate Bonds": [10.0], "Secured Loans": [8.0], "Other inv": [2.0],
    }, index=pd.to_datetime(["2023-03-10"]))
    c = st.run_stress_tests(df, "tether").iloc[0]
    assert np.isclose(c["Baseline"], 5.0)
    assert np.isclose(c["A: Liquidity run"], 5.0 - (30 - 15) * 0.02)
    market = 0.5 * 4 + 0.15 * 2 + 60 * (73 / 365) * 0.03
    assert np.isclose(c["B: Flash crash"], 5.0 - market)
    assert np.isclose(c["C: Credit contagion"], 5.0 - market - (1.0 + 2.0 + 1.0))


def test_circle_scenario_c_equals_b():
    df = pd.DataFrame({"New Estimate Reserves": [101.0], "USDC Market Cap": [100.0], "Cash & Deposits": [20.0],
                       "O/N Repo": [0.0], "Tbills": [81.0]}, index=pd.to_datetime(["2023-03-10"]))
    c = st.run_stress_tests(df, "circle").iloc[0]
    assert c["B: Flash crash"] == c["C: Credit contagion"]
    assert np.isclose(c["A: Liquidity run"], 1.0 - 10 * 0.02)
