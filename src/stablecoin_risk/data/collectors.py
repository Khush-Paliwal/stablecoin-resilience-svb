"""Public-API collectors for daily and intraday OHLCV.

Source priority for the daily panel used in the spillover analysis
(highest first; a lower source only fills dates the higher ones miss):

=====  ============================  =========================================
rank   source                        notes
=====  ============================  =========================================
1      Binance.US bulk klines        monthly/daily zip files, SHA-256 verified
2      Binance.US REST /api/v3       fills dates missing from the bulk archive
3      Coinbase Exchange candles     BTC, ETH, USDT (no USDC/USD book)
4      Kraken public OHLC            USDC only (720-bar history limit)
=====  ============================  =========================================

All functions return the canonical OHLCV frame (UTC DatetimeIndex) plus a
``source`` column so every row stays traceable after merging.
"""

from __future__ import annotations

import hashlib
import io
import time
import zipfile
from collections.abc import Iterable

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BINANCE_US_REST = "https://api.binance.us/api/v3/klines"
BINANCE_US_BULK = "https://data.binance.us/public_data/spot"
COINBASE_REST = "https://api.exchange.coinbase.com/products/{product}/candles"
KRAKEN_REST = "https://api.kraken.com/0/public/OHLC"

BINANCE_SYMBOLS = {"BTC/USD": "BTCUSD", "ETH/USD": "ETHUSD", "USDC/USD": "USDCUSD", "USDT/USD": "USDTUSD"}
COINBASE_PRODUCTS = {"BTC/USD": "BTC-USD", "ETH/USD": "ETH-USD", "USDT/USD": "USDT-USD"}
KRAKEN_PAIRS = {"USDC/USD": "USDCUSD", "USDT/USD": "USDTUSD"}

_COLS = ["open", "high", "low", "close", "volume"]
_INTERVAL_MS = {"1m": 60_000, "1h": 3_600_000, "1d": 86_400_000}


def http_session(retries: int = 4, backoff: float = 0.5) -> requests.Session:
    """Session with exponential back-off on 429/5xx, shared by all collectors."""
    s = requests.Session()
    retry = Retry(total=retries, backoff_factor=backoff, status_forcelist=(429, 500, 502, 503, 504))
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers["User-Agent"] = "stablecoin-risk-research/1.0"
    return s


def _frame(rows: list[list], time_unit: str, source: str) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=[*_COLS, "source"], index=pd.DatetimeIndex([], tz="UTC", name="timestamp"))
    df = pd.DataFrame(rows, columns=["t", *_COLS])
    df.index = pd.DatetimeIndex(pd.to_datetime(df.pop("t").astype("int64"), unit=time_unit, utc=True), name="timestamp")
    df = df.astype(float)
    df["source"] = source
    return df[~df.index.duplicated()].sort_index()


# --------------------------------------------------------------------------- #
# Binance.US
# --------------------------------------------------------------------------- #
def fetch_binance_us_rest(
    symbol: str, start: str, end: str, interval: str = "1d", session: requests.Session | None = None
) -> pd.DataFrame:
    """Paginate Binance.US ``/api/v3/klines`` (1,000 bars per call) over [start, end]."""
    session = session or http_session()
    step = _INTERVAL_MS[interval]
    cursor = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ms = int((pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)).timestamp() * 1000) - 1
    rows: list[list] = []
    while cursor <= end_ms:
        r = session.get(BINANCE_US_REST, params={"symbol": symbol, "interval": interval, "startTime": cursor,
                                                 "endTime": end_ms, "limit": 1000}, timeout=30)
        r.raise_for_status()
        batch = r.json()
        if not batch or isinstance(batch, dict):
            break
        rows += [[k[0], k[1], k[2], k[3], k[4], k[5]] for k in batch]
        cursor = batch[-1][0] + step
        if len(batch) < 1000:
            break
        time.sleep(0.2)
    return _frame(rows, "ms", "binance.us-rest")


def _verified_zip(url: str, session: requests.Session) -> bytes | None:
    r = session.get(url, timeout=60)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    check = session.get(url + ".CHECKSUM", timeout=15)
    if check.status_code == 200:
        expected = check.text.split()[0].lower()
        if hashlib.sha256(r.content).hexdigest() != expected:
            raise OSError(f"checksum mismatch for {url}")
    return r.content


def _klines_from_zip(blob: bytes) -> list[list]:
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        raw = zf.read(next(n for n in zf.namelist() if n.endswith(".csv"))).decode()
    rows = []
    for line in raw.splitlines():
        parts = line.split(",")
        if parts and parts[0].isdigit():
            rows.append([int(parts[0]), *parts[1:6]])
    return rows


def fetch_binance_us_bulk(
    symbol: str, start: str, end: str, interval: str = "1d", session: requests.Session | None = None
) -> pd.DataFrame:
    """Download Binance.US monthly kline archives (daily archives as fallback)."""
    session = session or http_session()
    rows: list[list] = []
    for month in pd.period_range(start, end, freq="M"):
        blob = _verified_zip(f"{BINANCE_US_BULK}/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{month}.zip", session)
        if blob is not None:
            rows += _klines_from_zip(blob)
            continue
        for day in pd.date_range(month.start_time, month.end_time, freq="D"):
            blob = _verified_zip(
                f"{BINANCE_US_BULK}/daily/klines/{symbol}/{interval}/{symbol}-{interval}-{day:%Y-%m-%d}.zip", session
            )
            if blob is not None:
                rows += _klines_from_zip(blob)
    df = _frame(rows, "ms", "binance.us-bulk")
    return df.loc[start : f"{end} 23:59:59"]


# --------------------------------------------------------------------------- #
# Coinbase / Kraken
# --------------------------------------------------------------------------- #
def fetch_coinbase(
    product: str, start: str, end: str, granularity: int = 86_400, session: requests.Session | None = None
) -> pd.DataFrame:
    """Coinbase Exchange candles (max 300 per request). Row format: [time, low, high, open, close, volume]."""
    session = session or http_session()
    t0 = pd.Timestamp(start, tz="UTC")
    t_end = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    chunk = pd.Timedelta(seconds=granularity * 300)
    rows: list[list] = []
    while t0 < t_end:
        t1 = min(t0 + chunk, t_end)
        r = session.get(COINBASE_REST.format(product=product),
                        params={"granularity": granularity, "start": t0.isoformat(), "end": t1.isoformat()}, timeout=30)
        r.raise_for_status()
        rows += [[k[0], k[3], k[2], k[1], k[4], k[5]] for k in r.json()]
        t0 = t1
        time.sleep(0.25)
    df = _frame(rows, "s", "coinbase")
    return df.loc[: t_end - pd.Timedelta(seconds=1)]


def fetch_kraken(pair: str, start: str, interval_min: int = 1440, session: requests.Session | None = None) -> pd.DataFrame:
    """Kraken public OHLC. Note: Kraken only serves the most recent 720 bars per interval."""
    session = session or http_session()
    since = int(pd.Timestamp(start, tz="UTC").timestamp())
    r = session.get(KRAKEN_REST, params={"pair": pair, "interval": interval_min, "since": since}, timeout=30)
    r.raise_for_status()
    payload = r.json()
    if payload.get("error"):
        raise RuntimeError(f"Kraken API error: {payload['error']}")
    key = next(k for k in payload["result"] if k != "last")
    rows = [[k[0], k[1], k[2], k[3], k[4], k[6]] for k in payload["result"][key]]
    return _frame(rows, "s", "kraken")


# --------------------------------------------------------------------------- #
# Daily panel with source cascade + validation
# --------------------------------------------------------------------------- #
def merge_by_priority(*frames: pd.DataFrame) -> pd.DataFrame:
    """Combine frames, keeping the first (highest-priority) row for each timestamp."""
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame(columns=[*_COLS, "source"])
    combined = pd.concat(frames)
    return combined[~combined.index.duplicated(keep="first")].sort_index()


def collect_daily_pair(pair: str, start: str, end: str, session: requests.Session | None = None) -> pd.DataFrame:
    """Daily OHLCV for one pair via the Binance.US -> Coinbase / Kraken cascade."""
    session = session or http_session()
    expected = pd.date_range(start, end, freq="D", tz="UTC")
    symbol = BINANCE_SYMBOLS[pair]
    df = fetch_binance_us_bulk(symbol, start, end, session=session)
    if expected.difference(df.index).size:
        df = merge_by_priority(df, fetch_binance_us_rest(symbol, start, end, session=session))
    missing = expected.difference(df.index)
    if missing.size and pair in COINBASE_PRODUCTS:
        df = merge_by_priority(df, fetch_coinbase(COINBASE_PRODUCTS[pair], f"{missing.min():%Y-%m-%d}",
                                                  f"{missing.max():%Y-%m-%d}", session=session))
    elif missing.size and pair in KRAKEN_PAIRS:
        df = merge_by_priority(df, fetch_kraken(KRAKEN_PAIRS[pair], f"{missing.min():%Y-%m-%d}", session=session))
    return df.loc[df.index.isin(expected)]


def collect_daily_panel(pairs: Iterable[str], start: str, end: str) -> pd.DataFrame:
    """Wide daily panel with columns ``BTC_USD_Open ... USDT_USD_Volume`` (as used by spillover)."""
    session = http_session()
    wide = []
    for pair in pairs:
        df = collect_daily_pair(pair, start, end, session=session)
        prefix = pair.replace("/", "_")
        wide.append(df[_COLS].rename(columns={c: f"{prefix}_{c.capitalize()}" for c in _COLS}))
    out = pd.concat(wide, axis=1)
    out.index = out.index.tz_localize(None).rename("Date")
    return out


def validate_daily(
    df: pd.DataFrame, pair: str, peg_band: tuple[float, float] = (0.86, 1.14), seam_threshold: float = 0.05
) -> pd.DataFrame:
    """Data-quality checks on one collected pair; returns one row per issue found.

    Checks: duplicate dates, calendar gaps, stablecoin prices outside ``peg_band``,
    and price jumps > ``seam_threshold`` where the data source switches.
    """
    issues = []
    if df.index.duplicated().any():
        issues.append({"check": "duplicates", "detail": f"{df.index.duplicated().sum()} duplicate dates"})
    full = pd.date_range(df.index.min(), df.index.max(), freq="D", tz=df.index.tz)
    if (missing := full.difference(df.index)).size:
        issues.append({"check": "coverage", "detail": f"{missing.size} missing dates, first {missing.min():%Y-%m-%d}"})
    if pair in KRAKEN_PAIRS:
        bad = df[(df["high"] > peg_band[1]) | (df["low"] < peg_band[0])]
        issues += [{"check": "peg_band", "detail": f"{t:%Y-%m-%d} H={r.high:.4f} L={r.low:.4f} ({r.source})"}
                   for t, r in bad.iterrows()]
    if "source" in df:
        switch = df["source"].ne(df["source"].shift()) & df["source"].shift().notna()
        jump = (df["open"] / df["close"].shift() - 1).abs()
        issues += [{"check": "source_seam", "detail": f"{t:%Y-%m-%d} jump {jump[t]:.1%}"}
                   for t in df.index[switch & (jump > seam_threshold)]]
    return pd.DataFrame(issues, columns=["check", "detail"]).assign(pair=pair)
