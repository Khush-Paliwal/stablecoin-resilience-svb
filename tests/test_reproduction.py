"""Regression tests pinning headline numbers reported in the IAQF 2026 paper.

They run on the data shipped in ``data/``; the spillover test additionally needs
the licensed Bloomberg file and is skipped when it is absent.
"""

import numpy as np
import pytest

from stablecoin_risk import config, regressions
from stablecoin_risk import spillover as sp
from stablecoin_risk import stress_testing as st
from stablecoin_risk.data import loaders

needs_regression_data = pytest.mark.skipif(
    not (config.REGRESSION_DIR / "usdc_regression_inputs.xlsx").exists(), reason="regression inputs missing")
needs_bloomberg = pytest.mark.skipif(
    not (config.EXTERNAL_DIR / "traditional_market_data.xlsx").exists(), reason="Bloomberg data not available")


@needs_regression_data
def test_usdc_full_sample_regression_matches_paper():
    res = regressions.run_paper_specs("USDC")["Full"]
    assert int(res.nobs) == 2_460
    assert np.isclose(res.rsquared, 0.301, atol=5e-4)


@needs_regression_data
def test_usdt_uses_separate_balance_sheet_terms():
    res = regressions.run_paper_specs("USDT")["Full"]
    assert {"zL", "zK"} <= set(res.params.index) and int(res.nobs) == 3_709


@needs_bloomberg
def test_tvp_var_total_spillover_matches_paper():
    vol = sp.build_volatility_panel(loaders.load_daily_crypto(), loaders.load_traditional_markets())
    avg, _, _ = sp.tvp_connectedness(sp.tvp_var(vol[sp.SPILLOVER_ASSETS], lags=2))
    assert len(vol) == 1_603
    assert np.isclose(avg.tsi, 41.43, atol=0.01)
    assert avg.table.loc["NET", "USDT"] > 0 and avg.table.loc["NET", "USDC"] > 0  # net transmitters


def test_circle_cushion_thinner_than_tether():
    out = {}
    for issuer in ("tether", "circle"):
        bs = loaders.load_reserves(issuer)
        out[issuer] = st.summarize(st.run_stress_tests(bs, issuer), bs[st.ISSUERS[issuer].supply])
    assert (out["circle"].loc["A: Liquidity run", "days_insolvent_pct"]
            > 3 * out["tether"].loc["A: Liquidity run", "days_insolvent_pct"])
