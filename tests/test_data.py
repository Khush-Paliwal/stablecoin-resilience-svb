import numpy as np
import pandas as pd
import pytest

from stablecoin_risk.data import cleaning, loaders
from stablecoin_risk.data.collectors import merge_by_priority, validate_daily
from stablecoin_risk.data.quality import apply_corrections, audit, flag_peg_outliers


# --------------------------------------------------------------------------- loaders
def test_standardize_binance_schema():
    raw = pd.DataFrame({"symbol": ["BTCUSD"] * 2, "open_time_ms": [1677628860000, 1677628800000],
                        "open": [2, 1], "high": [2, 1], "low": [2, 1], "close": [2, 1], "volume": [1, 1],
                        "num_trades": [3, 4]})
    out = loaders.standardize_ohlcv(raw)
    assert list(out.columns) == loaders.OHLCV_COLUMNS
    assert out.index.is_monotonic_increasing and str(out.index.tz) == "UTC"
    assert out.index[0] == pd.Timestamp("2023-03-01 00:00", tz="UTC")


def test_standardize_coinbase_seconds_and_coinapi_schema():
    cb = pd.DataFrame({"start": [1677628800, 1677628800], "low": [1, 1], "high": [2, 2], "open": [1, 1],
                       "close": [2, 2], "volume": [5, 5]})
    out = loaders.standardize_ohlcv(cb)
    assert len(out) == 1  # duplicate timestamp dropped
    assert out.index[0] == pd.Timestamp("2023-03-01", tz="UTC")
    assert out["num_trades"].isna().all()

    coinapi = pd.DataFrame({"time_period_start": ["2023-03-01T00:00:00.0000000Z"], "price_open": [1],
                            "price_high": [1], "price_low": [1], "price_close": [1], "volume_traded": [1],
                            "trades_count": [9]})
    assert loaders.standardize_ohlcv(coinapi)["num_trades"].iloc[0] == 9


@pytest.mark.parametrize("name,exchange,pair", [
    ("binanceus_USDCUSD_1m_20230301_20230321", "binance", "USDCUSD"),
    ("crypto_com_BTCUSDT_1m_20230301_20230321_cleaned", "crypto_com", "BTCUSDT"),
    ("coinbase_BTCUSD_1m_x", "coinbase", "BTCUSD"),
])
def test_file_name_parsing(name, exchange, pair):
    assert loaders.parse_exchange(name + ".csv") == exchange
    assert loaders.parse_pair(name + ".csv") == pair


def test_inactive_market_detection(bars):
    assert loaders.is_active_market(bars)
    dead = bars.assign(volume=0.0, close=20_100.0)
    assert not loaders.is_active_market(dead)


# --------------------------------------------------------------------------- cleaning
def _with_gap(bars, start, n):
    return bars.drop(bars.index[start : start + n])


def test_find_gaps(bars):
    gaps = cleaning.find_gaps(_with_gap(bars, 100, 7).index, pd.Timedelta("1min"))
    assert len(gaps) == 1 and gaps["n_missing"].iloc[0] == 7
    assert gaps["prev_ts"].iloc[0] == bars.index[99]


@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
def test_find_gaps_independent_of_datetime_resolution(bars, unit):
    """pandas >= 3 keeps the source resolution; gap detection must not assume nanoseconds."""
    idx = _with_gap(bars, 100, 7).index.as_unit(unit)
    gaps = cleaning.find_gaps(idx, pd.Timedelta("1min"))
    assert len(gaps) == 1 and gaps["n_missing"].iloc[0] == 7


def test_locf_ets_and_long_gaps(bars):
    holed = _with_gap(_with_gap(_with_gap(bars, 2_000, 40), 1_000, 12), 500, 3)
    filled, report = cleaning.fill_gaps(holed, name="x")
    assert report.n_locf == 3 and report.n_ets == 12 and len(report.long_gaps) == 1
    assert len(filled) == len(bars) - 40
    locf = filled[filled["impute_method"] == "LOCF"]
    assert np.allclose(locf["close"], bars["close"].iloc[499])
    assert (filled.loc[filled["imputed"], "volume"] == 0).all()
    imputed = filled[filled["imputed"]]
    assert (imputed["high"] >= imputed["low"]).all()


def test_imputation_is_causal(bars):
    """Changing data *after* a gap must not change the imputed values."""
    holed = _with_gap(bars, 1_000, 12)
    shocked = holed.copy()
    shocked.loc[shocked.index > bars.index[1_011], ["open", "high", "low", "close"]] *= 2
    a, _ = cleaning.fill_gaps(holed)
    b, _ = cleaning.fill_gaps(shocked)
    assert np.allclose(a.loc[a["imputed"], "close"], b.loc[b["imputed"], "close"])


# --------------------------------------------------------------------------- collectors / quality
def test_merge_by_priority_keeps_first_source():
    idx = pd.date_range("2023-01-01", periods=3, freq="D", tz="UTC")
    a = pd.DataFrame({"close": [1.0, 1.0, np.nan], "source": "a"}, index=idx).iloc[:2]
    b = pd.DataFrame({"close": [9.0, 9.0, 9.0], "source": "b"}, index=idx)
    out = merge_by_priority(a, b)
    assert list(out["source"]) == ["a", "a", "b"]


def test_validate_daily_flags_peg_and_seams():
    idx = pd.date_range("2023-01-01", periods=4, freq="D", tz="UTC")
    df = pd.DataFrame({"open": [1, 1, 1.2, 1], "high": [1, 5, 1.2, 1], "low": [1, 1, 1.2, 1],
                       "close": [1, 1, 1.2, 1], "source": ["a", "a", "b", "b"]}, index=idx)
    checks = set(validate_daily(df, "USDT/USD")["check"])
    assert {"peg_band", "source_seam"} <= checks


def test_quality_audit_with_offline_sources():
    df = pd.DataFrame({"USDT_USD_High": [1.0, 5.0, 3.0], "USDT_USD_Low": [1.0, 1.0, 1.0],
                       "USDT_USD_Close": [1.0, 1.0, 0.999]},
                      index=pd.to_datetime(["2020-02-18", "2020-02-19", "2020-02-20"]))
    flags = flag_peg_outliers(df, "USDT/USD")
    assert list(flags.index.strftime("%Y-%m-%d")) == ["2020-02-19", "2020-02-20"]
    clean = {"high": 1.002, "low": 0.998}
    sources = {"trades": lambda p, d: None,
               "kline": lambda p, d: {"high": 5.0, "low": 1.0},
               "kraken": lambda p, d: clean if d == "2020-02-19" else None,
               "coinbase": lambda p, d: None}
    res = audit(flags, sources)
    assert res.loc[0, "best_source"] == "kraken" and pd.isna(res.loc[1, "best_source"])
    fixed, log = apply_corrections(df, res)
    assert fixed.loc["2020-02-19", "USDT_USD_High"] == 1.002
    assert fixed.loc["2020-02-20", "USDT_USD_High"] == 0.999  # close-anchor strategy B
    assert list(log["strategy"].str[0]) == ["A", "B"]
