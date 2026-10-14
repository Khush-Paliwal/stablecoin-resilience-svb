"""End-to-end analysis steps. Each step writes tables to ``reports/tables`` and
figures to ``reports/figures`` and returns a dict of headline numbers that is
collected into ``reports/RESULTS.md``.

Steps that need licensed data (Bloomberg, CoinAPI) are skipped with a clear
message when the files are absent, so the public pipeline always runs.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from stablecoin_risk import basis, config, plotting, regressions, reserves  # noqa: E402
from stablecoin_risk import microstructure as ms
from stablecoin_risk import spillover as sp  # noqa: E402
from stablecoin_risk import stress_testing as st  # noqa: E402
from stablecoin_risk.data import cleaning, loaders  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*frequency.*")


class MissingData(RuntimeError):
    """Raised when a step's (licensed) inputs are not available locally."""


def _save_table(df: pd.DataFrame, name: str, floatfmt: str = ".4g") -> None:
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.TABLES_DIR / f"{name}.csv")
    (config.TABLES_DIR / f"{name}.md").write_text(df.to_markdown(floatfmt=floatfmt), encoding="utf-8")


def _save_fig(fig: plt.Figure, name: str) -> None:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIGURES_DIR / f"{name}.png")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 0. Cleaning
# --------------------------------------------------------------------------- #
def step_clean() -> dict:
    reports = cleaning.clean_directory(config.RAW_DIR, config.PROCESSED_DIR, verbose=False)
    summary = cleaning.summarize_reports(reports)
    _save_table(summary, "data_gap_fill_summary")
    raw = loaders.load_svb_panel(source="raw", drop_inactive=False)
    activity = pd.DataFrame({
        k: {"bars": len(d), "zero_volume_share": float((d["volume"] == 0).mean()),
            "distinct_closes": d["close"].nunique(), "active": loaders.is_active_market(d)}
        for k, d in raw.items()
    }).T
    _save_table(activity, "data_market_activity")
    return {"series_cleaned": len(reports), "inactive_markets": sorted(activity.index[~activity["active"].astype(bool)])}


# --------------------------------------------------------------------------- #
# 1. SVB-window liquidity, fragmentation and market quality
# --------------------------------------------------------------------------- #
def _implied_usd(panel: dict, eurusd: pd.Series, key: str) -> pd.Series | None:
    """Stablecoin/USD implied from a stablecoin/EUR book and EUR/USD."""
    if key not in panel:
        return None
    s = ms.original_close(panel[key])
    return (s * eurusd.reindex(s.index)).dropna()


def arbitrage_gap_panel(panel: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cross-venue gaps (bps) for BTC pairs and for stablecoin/USD rates."""
    close = {k: ms.original_close(v) for k, v in panel.items()}
    btc, stable = {}, {}
    venues = sorted({k.rsplit("_", 1)[0] for k in panel})
    for pair in config.BTC_PAIRS:
        others = [v for v in venues if v != "binance" and f"{v}_{pair}" in close]
        for v in others:
            if f"binance_{pair}" in close:
                label = f"{pair} binance vs {v}" + (" (unified USD book)" if f"{v}_{pair}" in config.MIRRORED_MARKETS else "")
                btc[label] = ms.arbitrage_gap_bps(close[f"binance_{pair}"], close[f"{v}_{pair}"])

    eurusd = loaders.load_eurusd()
    implied = {"coinbase_USDCUSD": _implied_usd(panel, eurusd, "coinbase_USDCEUR"),
               "coinbase_USDTUSD": _implied_usd(panel, eurusd, "coinbase_USDTEUR")}
    for pair in config.STABLECOIN_PAIRS:
        base = close.get(f"binance_{pair}")
        if base is None:
            continue
        if f"kraken_{pair}" in close:
            stable[f"{pair} binance vs kraken"] = ms.arbitrage_gap_bps(base, close[f"kraken_{pair}"])
        if implied.get(f"coinbase_{pair}") is not None:
            stable[f"{pair} binance vs coinbase (EUR-implied)"] = ms.arbitrage_gap_bps(base, implied[f"coinbase_{pair}"])
    return pd.DataFrame(btc), pd.DataFrame(stable)


def step_liquidity() -> dict:
    panel = loaders.load_svb_panel(source="processed")
    if not panel:
        raise MissingData("no processed data - run the 'clean' step first")
    targets = [k for k in panel if k.rsplit("_", 1)[1] in config.BTC_PAIRS + config.STABLECOIN_PAIRS
               and k not in config.MIRRORED_MARKETS]
    metrics = {k: ms.compute_all_metrics(panel[k]) for k in targets}

    regime = ms.regime_statistics(metrics)
    _save_table(regime, "liquidity_regime_statistics")
    mult = ms.crisis_multipliers(regime)
    _save_table(mult, "liquidity_crisis_multipliers")

    # Market Quality Index (Binance BTC books by quote currency) and Stablecoin Peg Quality Index
    mqi = ms.compute_mqi_panel(metrics, lambda k: k.startswith("binance_BTC"))
    spqi = ms.compute_mqi_panel(metrics, lambda k: k.rsplit("_", 1)[1] in config.STABLECOIN_PAIRS)
    mqi_regime = pd.DataFrame({
        k: {r: s.loc[a : f"{b} 23:59:59"].median() for r, (a, b) in config.REGIMES.items()}
        for k, s in {**mqi, **spqi}.items()
    }).T
    _save_table(mqi_regime, "mqi_by_regime", ".1f")
    _save_fig(plotting.plot_svb_panels(
        {"BTC Market Quality Index by quote currency - Binance": {k.rsplit("_", 1)[1]: v for k, v in mqi.items()},
         "Stablecoin Peg Quality Index (SPQI)": spqi}, "Composite score (0-100)", resample="60min", ylim=(0, 100), legend_loc="lower left"),
        "fig2_market_quality_index")

    # Cross-venue arbitrage gaps
    btc_gaps, stable_gaps = arbitrage_gap_panel(panel)
    gaps = pd.concat([btc_gaps, stable_gaps], axis=1)
    gap_stats = pd.DataFrame({
        c: {**{f"{r}_median": gaps[c].loc[a : f"{b} 23:59:59"].median() for r, (a, b) in config.REGIMES.items()},
            "max": gaps[c].max(), "max_time": gaps[c].idxmax()}
        for c in gaps
    }).T
    _save_table(gap_stats, "arbitrage_gap_summary_bps", ".1f")
    _save_fig(plotting.plot_svb_panels({"Cross-exchange arbitrage gaps: BTC pairs": dict(btc_gaps.items()),
                                        "Cross-exchange arbitrage gaps: stablecoin / USD": dict(stable_gaps.items())},
                                       "Basis points", resample="5min", legend_loc="upper right"), "fig1_arbitrage_gaps")

    # Peg deviation and flight-to-safety ratio (Binance.US direct books)
    depeg = {k: (ms.original_close(panel[k]) - 1).abs() * 1e4 for k in panel if k.rsplit("_", 1)[1] in config.STABLECOIN_PAIRS}
    _save_fig(plotting.plot_svb_panels({"Peg deviation |P - 1| (60-min mean)": depeg}, "Basis points",
                                       resample="60min"), "fig_peg_deviation")
    hourly = {k: v.resample("60min").mean() for k, v in depeg.items()}
    flight = (hourly["binance_USDCUSD"] / hourly["binance_USDTUSD"].clip(lower=1)).dropna()

    out = {
        "markets_analysed": len(metrics),
        "max_usdc_depeg_bps": float(depeg["binance_USDCUSD"].max()),
        "max_usdt_depeg_bps": float(depeg["binance_USDTUSD"].max()),
        "usdc_usdt_depeg_ratio_peak_median": float(flight.loc["2023-03-10":"2023-03-13"].median()),
    }
    for k, _s in {**mqi, **spqi}.items():
        out[f"mqi_{k}_pre"] = float(mqi_regime.loc[k, "pre_crisis"])
        out[f"mqi_{k}_peak"] = float(mqi_regime.loc[k, "peak_crisis"])
    for c in gaps:
        out[f"gap_max_bps[{c}]"] = float(gaps[c].max())
    return out


# --------------------------------------------------------------------------- #
# 2. Cross-quote basis and regime switching
# --------------------------------------------------------------------------- #
def step_basis() -> dict:
    panel = loaders.load_svb_panel(source="processed")
    b = basis.basis_panel(panel)
    _save_table(basis.basis_summary(b, config.REGIMES), "basis_by_regime_bps", ".2f")
    fits = {c: basis.fit_markov_switching(b[c]) for c in b if b[c].notna().sum() > 500}
    failed = sorted(c for c, f in fits.items() if f is None)
    fits = {c: f for c, f in fits.items() if f is not None}
    table = basis.regime_table(fits)
    _save_table(table, "basis_markov_regimes")
    _save_fig(plotting.plot_svb_panels({"BTC cross-quote basis (P_USD - P_Q) / P_Q": {c: b[c] * 1e4 for c in b}},
                                       "Basis points", resample="15min"), "fig_btc_cross_quote_basis")
    out = {f"stress_mean_bps[{k}]": float(v) for k, v in table["stress_mean_bps"].items()}
    out.update({f"first_stress_time[{k}]": str(v) for k, v in table["first_stress_time"].items()})
    if failed:
        out["markov_fit_not_converged"] = failed
    return out


# --------------------------------------------------------------------------- #
# 3. Order-book Kyle's lambda (CoinAPI, licensed)
# --------------------------------------------------------------------------- #
def step_kyle(pairs: tuple[str, ...] = ("COINBASE_BTC_USD", "COINBASE_BTC_USDT", "BINANCE_BTC_USDT")) -> dict:
    root = config.EXTERNAL_DIR / "orderbook"
    if not root.exists():
        raise MissingData(f"{root} not found (CoinAPI order books are licensed)")
    series, coverage = {}, {}
    for pair in pairs:
        try:
            ob = loaders.load_orderbook(pair, root)
            series[pair] = ms.kyle_lambda_ofi(ob)
            coverage[pair] = ob.index.normalize().nunique()
        except (FileNotFoundError, ValueError) as exc:
            coverage[pair] = f"skipped: {exc}"
    if not series:
        raise MissingData("no order-book pair has enough data for a 1,440-bar rolling window")
    _save_fig(plotting.plot_svb_panels({"Rolling Kyle's lambda from order-flow imbalance (1,440-min OLS)": series},
                                       "lambda (price change per unit OFI)", resample="30min"), "fig3_kyle_lambda_ofi")
    stats = pd.DataFrame({k: {r: s.loc[a : f"{b} 23:59:59"].median() for r, (a, b) in config.REGIMES.items()}
                          for k, s in series.items()}).T
    stats["days_of_data"] = pd.Series(coverage)
    _save_table(stats, "kyle_lambda_ofi_by_regime")
    return {"orderbook_pairs": coverage}


# --------------------------------------------------------------------------- #
# 4. Volatility spillovers (needs Bloomberg traditional-market data)
# --------------------------------------------------------------------------- #
def step_spillover() -> dict:
    try:
        trad = loaders.load_traditional_markets()
    except FileNotFoundError as exc:
        raise MissingData(str(exc)) from exc
    vol = sp.build_volatility_panel(loaders.load_daily_crypto(), trad)
    _save_table(sp.descriptive_statistics(vol), "spillover_table1_descriptive")
    reg = {c: sp.stablecoin_volatility_regression(vol, c) for c in ("USDT", "USDC")}
    _save_table(regressions.results_table(reg), "spillover_table1_volatility_regression")

    data = vol[sp.SPILLOVER_ASSETS]
    static = sp.static_spillover(data)
    _save_table(static.table, "spillover_static_var", ".2f")
    out = {"n_days": len(vol), "static_var_lags": static.lags, "static_tsi": static.tsi}
    for p in (1, 2):
        avg, tsi_t, _ = sp.tvp_connectedness(sp.tvp_var(data, lags=p))
        _save_table(avg.table, f"spillover_table2_tvp_var_p{p}", ".2f")
        out[f"tvp_tsi_p{p}"] = avg.tsi
        if p == config.TVP_LAGS:
            tsi_t.to_frame().to_csv(config.TABLES_DIR / "spillover_dynamic_tsi.csv")
            _save_fig(plotting.plot_connectedness(tsi_t), "fig_dynamic_connectedness")
            _save_fig(plotting.plot_net_spillovers(avg.table), "fig_net_spillovers")
            out["net_usdt"] = float(avg.table.loc["NET", "USDT"])
            out["net_usdc"] = float(avg.table.loc["NET", "USDC"])
    return out


# --------------------------------------------------------------------------- #
# 5. Peg-deviation regressions
# --------------------------------------------------------------------------- #
def step_regressions() -> dict:
    out = {}
    for coin in ("USDC", "USDT"):
        panel = regressions.build_panel(*loaders.load_regression_inputs(coin))
        main = regressions.run_paper_specs(coin, cov="hac", panel=panel)
        _save_table(regressions.results_table(main), f"regression_{coin.lower()}_paper")
        robust = regressions.run_paper_specs(coin, cov="driscoll-kraay", panel=panel)
        _save_table(regressions.results_table(robust), f"regression_{coin.lower()}_driscoll_kraay")
        for name, res in main.items():
            out[f"{coin} {name} R2"] = float(res.rsquared)
            out[f"{coin} {name} N"] = int(res.nobs)
    return out


# --------------------------------------------------------------------------- #
# 6. Reserves and stress tests
# --------------------------------------------------------------------------- #
def step_stress() -> dict:
    out = {}
    q = reserves.quarterly_equity()
    _save_fig(plotting.plot_reserve_evolution(q), "fig4_reserve_equity_evolution")
    _save_fig(plotting.plot_reserve_composition(reserves.tether_composition(), reserves.circle_composition()),
              "fig5_reserve_composition")
    for issuer, fig_name in (("tether", "fig6_tether_stress_test"), ("circle", "fig7_circle_stress_test")):
        bs = loaders.load_reserves(issuer)
        cushions = st.run_stress_tests(bs, issuer)
        summary = st.summarize(cushions, bs[st.ISSUERS[issuer].supply])
        _save_table(summary, f"stress_test_{issuer}")
        _save_fig(plotting.plot_stress_test(cushions, f"{issuer.title()}: equity cushion under stress"), fig_name)
        for s in summary.index:
            out[f"{issuer} {s} days insolvent %"] = float(summary.loc[s, "days_insolvent_pct"])
        out[f"{issuer} median equity % supply"] = float(summary.loc["Baseline", "median_cushion_pct_supply"])
        out[f"{issuer} liquidity coverage on 2023-03-10"] = float(st.liquidity_coverage(bs, issuer).loc["2023-03-10"])
    return out


STEPS: dict[str, Callable[[], dict]] = {
    "clean": step_clean,
    "liquidity": step_liquidity,
    "basis": step_basis,
    "kyle": step_kyle,
    "spillover": step_spillover,
    "regressions": step_regressions,
    "stress": step_stress,
}


def _fmt(v) -> str:
    if isinstance(v, float | np.floating):
        return f"{v:,.4g}"
    return str(v)


def write_results(results: dict[str, dict | str], path: Path | None = None) -> Path:
    """Render headline numbers of every step into a Markdown report."""
    path = path or config.REPORTS_DIR / "RESULTS.md"
    lines = ["# Pipeline results", "", "_Auto-generated by `stablecoin-risk run all`; do not edit by hand._", ""]
    for step, res in results.items():
        lines += [f"## {step}", ""]
        if isinstance(res, str):
            lines += [f"> {res}", ""]
            continue
        lines += ["| metric | value |", "|---|---|"] + [f"| {k} | {_fmt(v)} |" for k, v in res.items()] + [""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run(steps: list[str] | None = None, verbose: bool = True) -> dict[str, dict | str]:
    """Run the requested steps (default: all) and write ``reports/RESULTS.md``."""
    results: dict[str, dict | str] = {}
    for name in steps or list(STEPS):
        if verbose:
            print(f"[{name}] running ...", flush=True)
        try:
            results[name] = STEPS[name]()
        except MissingData as exc:
            results[name] = f"skipped - {exc}"
        if verbose:
            print(f"[{name}] {'done' if isinstance(results[name], dict) else results[name]}", flush=True)
    write_results(results)
    return results
