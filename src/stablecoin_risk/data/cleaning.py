"""Causal gap-filling for 1-minute OHLCV bars.

Replaces the four per-exchange scripts of the competition repo with one
implementation that works on the canonical schema produced by
:mod:`stablecoin_risk.data.loaders`.

Imputation rule (no look-ahead: only pre-gap data is ever used)

=============  ==============================================================
gap (bars)     action
=============  ==============================================================
1 - 5          LOCF - repeat the last observed close as a flat O=H=L=C bar
6 - 30         ETS  - simple exponential smoothing fitted on the previous
               ``ETS_LOOKBACK`` closes; forecast used as a flat bar
> 30           left missing and written to an audit log
=============  ==============================================================

Imputed bars carry zero volume / trades and are flagged with ``imputed``,
``gap_size`` and ``impute_method`` so every downstream statistic can be
re-run on original bars only.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from stablecoin_risk import config
from stablecoin_risk.data.loaders import parse_exchange, read_ohlcv_csv


@dataclass
class GapFillReport:
    """Summary of one cleaned series."""

    name: str
    n_original: int
    n_locf: int = 0
    n_ets: int = 0
    long_gaps: list[dict] = field(default_factory=list)

    @property
    def n_imputed(self) -> int:
        return self.n_locf + self.n_ets


def ets_level_forecast(history: np.ndarray) -> float:
    """One flat SES forecast from ``history`` (falls back to the last value)."""
    history = history[np.isfinite(history)]
    if len(history) < 4 or np.ptp(history) == 0:
        return float(history[-1])
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = SimpleExpSmoothing(history, initialization_method="estimated").fit(optimized=True)
    return float(fit.forecast(1)[0])


def find_gaps(index: pd.DatetimeIndex, bar: pd.Timedelta) -> pd.DataFrame:
    """Locate runs of missing bars in a sorted DatetimeIndex.

    Returns one row per gap with the last observed timestamp before the gap
    (``prev_ts``) and the number of missing bars (``n_missing``).
    """
    if len(index) < 2:
        return pd.DataFrame(columns=["prev_ts", "n_missing"])
    steps = np.round(np.diff(index.asi8) / bar.value).astype(np.int64) - 1
    pos = np.flatnonzero(steps > 0)
    return pd.DataFrame({"prev_ts": index[pos], "n_missing": steps[pos]})


def fill_gaps(
    df: pd.DataFrame,
    name: str = "",
    short_gap_max: int = config.SHORT_GAP_MAX,
    medium_gap_max: int = config.MEDIUM_GAP_MAX,
    ets_lookback: int = config.ETS_LOOKBACK,
    bar: str = "1min",
) -> tuple[pd.DataFrame, GapFillReport]:
    """Impute short/medium gaps in a canonical OHLCV frame (see module docstring)."""
    bar_td = pd.Timedelta(bar)
    df = df.sort_index()
    out = df.copy()
    out["imputed"] = False
    out["gap_size"] = 0
    out["impute_method"] = "none"
    report = GapFillReport(name=name, n_original=len(df))

    closes = df["close"].to_numpy(dtype=float)
    positions = {ts: i for i, ts in enumerate(df.index)}
    synthetic = []
    for prev_ts, n in find_gaps(df.index, bar_td).itertuples(index=False):
        if n > medium_gap_max:
            report.long_gaps.append(
                {
                    "series": name,
                    "gap_start_utc": prev_ts + bar_td,
                    "gap_end_utc": prev_ts + n * bar_td,
                    "n_missing_bars": int(n),
                }
            )
            continue
        i = positions[prev_ts]
        if n <= short_gap_max:
            level, method = closes[i], "LOCF"
            report.n_locf += n
        else:
            level = ets_level_forecast(closes[max(0, i + 1 - ets_lookback) : i + 1])
            method = "ETS"
            report.n_ets += n
        idx = pd.date_range(prev_ts + bar_td, periods=n, freq=bar_td, name=df.index.name)
        synthetic.append(
            pd.DataFrame(
                {
                    "open": level,
                    "high": level,
                    "low": level,
                    "close": level,
                    "volume": 0.0,
                    "num_trades": 0.0 if df["num_trades"].notna().any() else np.nan,
                    "imputed": True,
                    "gap_size": n,
                    "impute_method": method,
                },
                index=idx,
            )
        )

    if synthetic:
        out = pd.concat([out, *synthetic]).sort_index()
    return out, report


def clean_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    window: tuple[str, str] | None = config.SVB_WINDOW,
    verbose: bool = True,
) -> list[GapFillReport]:
    """Clean every ``*.csv`` under ``input_dir/<exchange>/`` into ``output_dir/<exchange>/``.

    Writes ``<stem>_cleaned.csv`` per file and a ``long_gaps_dropped.csv`` audit log.
    """
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    reports: list[GapFillReport] = []
    for path in sorted(input_dir.glob("*/*.csv")):
        exchange = parse_exchange(path)
        if exchange is None:
            continue
        df = read_ohlcv_csv(path, window)
        if df.empty:
            continue
        cleaned, report = fill_gaps(df, name=path.stem)
        dest = output_dir / exchange
        dest.mkdir(parents=True, exist_ok=True)
        cleaned.to_csv(dest / f"{path.stem}_cleaned.csv", index_label="timestamp")
        reports.append(report)
        if verbose:
            print(
                f"  {path.stem:<45s} rows={report.n_original:>6,d}  LOCF={report.n_locf:>5,d}  "
                f"ETS={report.n_ets:>5,d}  long gaps={len(report.long_gaps)}"
            )

    long_gaps = [g for r in reports for g in r.long_gaps]
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(long_gaps, columns=["series", "gap_start_utc", "gap_end_utc", "n_missing_bars"]).to_csv(
        output_dir / "long_gaps_dropped.csv", index=False
    )
    return reports


def summarize_reports(reports: list[GapFillReport]) -> pd.DataFrame:
    """Tabulate gap-fill reports (one row per series)."""
    rows = [
        {
            "series": r.name,
            "original_bars": r.n_original,
            "locf_bars": r.n_locf,
            "ets_bars": r.n_ets,
            "imputed_share": r.n_imputed / max(r.n_original + r.n_imputed, 1),
            "long_gaps": len(r.long_gaps),
            "long_gap_bars": sum(g["n_missing_bars"] for g in r.long_gaps),
        }
        for r in reports
    ]
    return pd.DataFrame(rows).set_index("series")
