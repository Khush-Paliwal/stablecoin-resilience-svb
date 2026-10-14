import numpy as np
import pandas as pd

from stablecoin_risk import microstructure as ms
from tests.conftest import make_bars


def test_roll_spread_recovers_simulated_bounce():
    """Pure bid-ask bounce around a constant mid: Roll estimate ~ quoted spread."""
    rng = np.random.default_rng(1)
    n, s = 5_000, 0.002
    idx = pd.date_range("2023-03-01", periods=n, freq="1min", tz="UTC")
    close = 100 * (1 + s / 2 * rng.choice([-1, 1], n))
    df = pd.DataFrame({"close": close}, index=idx)
    est = ms.roll_spread(df, window=2_000).dropna()
    assert abs(est.median() - s) / s < 0.10


def test_corwin_schultz_zero_without_spread_and_bounded():
    flat = make_bars(spread=0.0)
    cs = ms.corwin_schultz(flat, window=500).dropna()
    assert (cs >= 0).all() and (cs < 2).all()
    wide = make_bars(spread=0.004, seed=3)
    assert ms.corwin_schultz(wide, window=500).median() > cs.median()


def test_abdi_ranaldo_is_causal():
    df = make_bars(spread=0.002)
    full = ms.abdi_ranaldo(df, window=200)
    cut = ms.abdi_ranaldo(df.iloc[:1_500], window=200)
    assert np.allclose(full.iloc[:1_500].dropna(), cut.dropna())


def test_amihud_ignores_zero_volume(bars):
    bars.iloc[::5, bars.columns.get_loc("volume")] = 0
    a = ms.amihud(bars).dropna()
    assert np.isfinite(a).all() and (a >= 0).all()


def test_vpin_in_unit_interval(bars):
    v = ms.vpin(bars)
    assert v.between(0, 1).all()


def test_efficiency_ratio_trend_vs_noise():
    idx = pd.date_range("2023-03-01", periods=300, freq="1min", tz="UTC")
    trend = pd.DataFrame({"close": np.linspace(100, 110, 300)}, index=idx)
    assert np.allclose(ms.efficiency_ratio(trend, 60).dropna(), 1.0)


def test_arbitrage_gap_bps():
    idx = pd.date_range("2023-03-01", periods=3, freq="1min", tz="UTC")
    a = pd.Series([101.0, 100.0, 99.0], index=idx)
    b = pd.Series([100.0, 100.0, 100.0], index=idx[:3])
    assert np.allclose(ms.arbitrage_gap_bps(a, b), [100, 0, 100])


def test_market_quality_index_bounds_and_direction():
    calm = make_bars(n=20_000, seed=2)
    m = ms.compute_all_metrics(calm, window=None)
    mqi = ms.market_quality_index(m)
    assert mqi.between(0, 100).all()
    stressed = m.copy()
    stressed.loc["2023-03-12":, ["cs_spread", "amihud"]] *= 20
    assert ms.market_quality_index(stressed).loc["2023-03-12":].median() < mqi.loc["2023-03-12":].median()


def test_order_flow_imbalance_hand_computed():
    ob = pd.DataFrame({"bid_price_1": [10.0, 10.0, 10.1, 9.9], "bid_size_1": [5.0, 7.0, 2.0, 4.0],
                       "ask_price_1": [10.2, 10.2, 10.2, 10.0], "ask_size_1": [3.0, 3.0, 1.0, 6.0]})
    ofi = ms.order_flow_imbalance(ob)
    # t1: bid same price (+7-5), ask same price (-3+3)  -> +2
    # t2: bid up (+2), ask same (-1+3)                   -> +4
    # t3: bid down (-2), ask down (-6)                   -> -8
    assert np.isnan(ofi.iloc[0])
    assert list(ofi.iloc[1:]) == [2.0, 4.0, -8.0]


def test_kyle_lambda_ofi_recovers_true_lambda(rng):
    n, lam = 4_000, 0.05
    idx = pd.date_range("2023-03-01", periods=n, freq="1min", tz="UTC")
    bid = 100 + np.cumsum(rng.normal(0, 0.01, n))
    ob = pd.DataFrame({"bid_price_1": bid, "ask_price_1": bid + 0.01,
                       "bid_size_1": rng.uniform(1, 5, n), "ask_size_1": rng.uniform(1, 5, n)}, index=idx)
    ofi = ms.order_flow_imbalance(ob).fillna(0).to_numpy()
    mid = 100 + np.cumsum(lam * ofi + rng.normal(0, 0.01, n))
    ob["mid_price"] = mid
    est = ms.kyle_lambda_ofi(ob, window=1_000).dropna()
    assert abs(est.median() - lam) < 0.005
