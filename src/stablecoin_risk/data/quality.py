"""Stablecoin OHLCV quality audit.

A stablecoin pegged at $1 cannot trade at $5 on a liquid book; daily High/Low
values far outside the peg band are almost always single stale orders hitting a
thin order book (or an exchange data error). For each flagged day the audit:

1. recomputes the true High/Low from Binance.US raw trade files,
2. re-reads the Binance.US daily kline,
3. cross-checks Kraken and (USDT only) Coinbase,

and takes the first source whose High and Low both sit inside the band.

Corrections
-----------
A  cross-exchange replacement - High/Low replaced by the chosen source.
B  close anchor - no clean source exists: High = Close, Low = min(Low, Close)
   (keeps the close, removes the uninformative intraday spike).
"""

from __future__ import annotations

import io
import zipfile
from collections.abc import Callable

import numpy as np
import pandas as pd
import requests

from stablecoin_risk.data.collectors import (
    BINANCE_US_BULK,
    _verified_zip,
    fetch_coinbase,
    fetch_kraken,
    http_session,
)

PEG_BAND = (0.86, 1.14)
SOURCE_PRIORITY = ("trades", "kline", "kraken", "coinbase")


def flag_peg_outliers(df: pd.DataFrame, pair: str, band: tuple[float, float] = PEG_BAND) -> pd.DataFrame:
    """Rows of a daily ``<PAIR>_High/_Low`` frame with a High or Low outside the peg band."""
    prefix = pair.replace("/", "_")
    hi, lo = df[f"{prefix}_High"], df[f"{prefix}_Low"]
    mask = (hi > band[1]) | (lo < band[0])
    return pd.DataFrame({"pair": pair, "reported_high": hi[mask], "reported_low": lo[mask]})


def binance_trade_range(symbol: str, day: str, session: requests.Session) -> dict | None:
    """High/Low/N from the Binance.US raw trade file of one day (None if unavailable)."""
    blob = _verified_zip(f"{BINANCE_US_BULK}/daily/trades/{symbol}/{symbol}-trades-{day}.zip", session)
    if blob is None:
        return None
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        trades = pd.read_csv(io.BytesIO(zf.read(zf.namelist()[0])), header=None, usecols=[1])
    prices = pd.to_numeric(trades[1], errors="coerce").dropna()
    return {"high": prices.max(), "low": prices.min(), "n": len(prices)} if len(prices) else None


def binance_kline(symbol: str, day: str, session: requests.Session) -> dict | None:
    blob = _verified_zip(f"{BINANCE_US_BULK}/daily/klines/{symbol}/1d/{symbol}-1d-{day}.zip", session)
    if blob is None:
        return None
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        k = zf.read(zf.namelist()[0]).decode().splitlines()[0].split(",")
    return {"high": float(k[2]), "low": float(k[3])}


def _daily_bar(df: pd.DataFrame, day: str) -> dict | None:
    ts = pd.Timestamp(day, tz="UTC")
    return {"high": df.at[ts, "high"], "low": df.at[ts, "low"]} if ts in df.index else None


def default_sources(session: requests.Session | None = None) -> dict[str, Callable[[str, str], dict | None]]:
    """Live fetchers ``source -> f(pair, day)``. Tests inject offline stubs instead."""
    session = session or http_session()
    sym = {"USDT/USD": "USDTUSD", "USDC/USD": "USDCUSD"}
    return {
        "trades": lambda pair, day: binance_trade_range(sym[pair], day, session),
        "kline": lambda pair, day: binance_kline(sym[pair], day, session),
        "kraken": lambda pair, day: _daily_bar(fetch_kraken(sym[pair], day, session=session), day),
        "coinbase": lambda pair, day: (
            _daily_bar(fetch_coinbase("USDT-USD", day, day, session=session), day) if pair == "USDT/USD" else None
        ),
    }


def audit(
    flags: pd.DataFrame,
    sources: dict[str, Callable[[str, str], dict | None]] | None = None,
    band: tuple[float, float] = PEG_BAND,
) -> pd.DataFrame:
    """Cross-check each flagged day; returns per-source High/Low and the chosen replacement."""
    sources = sources or default_sources()
    rows = []
    for day, flag in flags.iterrows():
        day_str = pd.Timestamp(day).strftime("%Y-%m-%d")
        row = {"date": day_str, **flag.to_dict(), "best_source": np.nan, "best_high": np.nan, "best_low": np.nan}
        for name in SOURCE_PRIORITY:
            try:
                res = sources[name](flag["pair"], day_str)
            except Exception as exc:  # network errors should not abort the audit
                res, row[f"{name}_error"] = None, str(exc)
            row[f"{name}_high"] = res["high"] if res else np.nan
            row[f"{name}_low"] = res["low"] if res else np.nan
            if res and pd.isna(row["best_source"]) and band[0] <= res["low"] and res["high"] <= band[1]:
                row.update(best_source=name, best_high=res["high"], best_low=res["low"])
        rows.append(row)
    return pd.DataFrame(rows)


def apply_corrections(df: pd.DataFrame, audit_results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply strategy A / B to a wide daily frame; returns (corrected frame, change log)."""
    out, log = df.copy(), []
    for _, r in audit_results.iterrows():
        prefix, ts = r["pair"].replace("/", "_"), pd.Timestamp(r["date"])
        hi_col, lo_col, close_col = f"{prefix}_High", f"{prefix}_Low", f"{prefix}_Close"
        old = (out.at[ts, hi_col], out.at[ts, lo_col])
        if pd.notna(r["best_source"]):
            new, strategy = (r["best_high"], r["best_low"]), f"A ({r['best_source']})"
        else:
            close = out.at[ts, close_col]
            new, strategy = (close, min(old[1], close)), "B (close anchor)"
        out.at[ts, hi_col], out.at[ts, lo_col] = new
        log.append({"date": r["date"], "pair": r["pair"], "strategy": strategy,
                    "old_high": old[0], "old_low": old[1], "new_high": new[0], "new_low": new[1]})
    return out, pd.DataFrame(log)
