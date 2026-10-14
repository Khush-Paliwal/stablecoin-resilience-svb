"""Loaders that turn heterogeneous exchange / issuer files into tidy DataFrames.

Exchange 1-minute files arrive in four different schemas (Binance klines,
two Coinbase exports, Crypto.com candles and CoinAPI/Kraken OHLCV). Every
loader here returns the same canonical frame:

    index   : ``timestamp`` - tz-aware UTC DatetimeIndex, sorted, unique
    columns : open, high, low, close, volume, num_trades (float)

so downstream analytics never need to know where the data came from.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from stablecoin_risk import config

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume", "num_trades"]

_RENAME = {
    # CoinAPI / Kraken export
    "price_open": "open",
    "price_high": "high",
    "price_low": "low",
    "price_close": "close",
    "volume_traded": "volume",
    "trades_count": "num_trades",
    # Kraken native export
    "trades": "num_trades",
}

_PAIR_RE = re.compile(r"_([A-Z]{6,10})_1m")
_EXCHANGE_PREFIXES = {
    "binanceus": "binance",
    "binance": "binance",
    "coinbase": "coinbase",
    "crypto_com": "crypto_com",
    "kraken": "kraken",
}


# --------------------------------------------------------------------------- #
# Timestamp / schema normalisation
# --------------------------------------------------------------------------- #
def _epoch_to_datetime(values: pd.Series) -> pd.Series:
    """Convert unix epochs to UTC datetimes, inferring seconds vs milliseconds."""
    nums = pd.to_numeric(values, errors="coerce")
    unit = "ms" if nums.dropna().abs().median() > 1e11 else "s"
    return pd.to_datetime(nums, unit=unit, utc=True, errors="coerce")


def parse_timestamp(df: pd.DataFrame) -> pd.Series:
    """Return a UTC timestamp series from whichever time column the file carries.

    Numeric epoch columns are preferred over string columns because they are
    unambiguous and an order of magnitude faster to parse.
    """
    for col in ("open_time_ms", "start", "timestamp", "ts"):
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            return _epoch_to_datetime(df[col])
    for col in ("time_period_start", "timestamp_utc", "open_time_utc", "datetime", "timestamp", "ts"):
        if col in df.columns:
            return pd.to_datetime(df[col], utc=True, errors="coerce")
    raise ValueError(f"No recognised timestamp column in {list(df.columns)}")


def standardize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Map any supported exchange schema onto the canonical OHLCV frame."""
    out = df.rename(columns={k: v for k, v in _RENAME.items() if k in df.columns}).copy()
    out.index = pd.DatetimeIndex(parse_timestamp(df).to_numpy(), name="timestamp")
    out = out[out.index.notna()]

    missing = {"open", "high", "low", "close", "volume"} - set(out.columns)
    if missing:
        raise ValueError(f"Missing OHLCV columns: {sorted(missing)}")
    if "num_trades" not in out.columns:
        out["num_trades"] = np.nan

    flags = out["imputed"].astype(str).str.lower().eq("true") if "imputed" in out.columns else None
    out = out[OHLCV_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if flags is not None:  # processed files keep the gap-fill audit flag
        out["imputed"] = flags.to_numpy()
    out = out[~out.index.duplicated(keep="first")].sort_index()
    return out.dropna(subset=["open", "high", "low", "close"])


def clip_window(df: pd.DataFrame, window: tuple[str, str] | None) -> pd.DataFrame:
    """Keep rows inside an inclusive [start-day, end-day] UTC window."""
    if window is None:
        return df
    start = pd.Timestamp(window[0], tz="UTC")
    end = pd.Timestamp(window[1], tz="UTC") + pd.Timedelta(days=1)
    return df.loc[(df.index >= start) & (df.index < end)]


def read_ohlcv_csv(path: str | Path, window: tuple[str, str] | None = config.SVB_WINDOW) -> pd.DataFrame:
    """Read one exchange CSV into the canonical schema, clipped to ``window``."""
    return clip_window(standardize_ohlcv(pd.read_csv(path)), window)


# --------------------------------------------------------------------------- #
# File discovery
# --------------------------------------------------------------------------- #
def parse_pair(path: str | Path) -> str | None:
    """Extract the trading pair from a file name, e.g. ``binance_BTCUSDC_1m_...`` -> ``BTCUSDC``."""
    match = _PAIR_RE.search(Path(path).stem)
    return match.group(1) if match else None


def parse_exchange(path: str | Path) -> str | None:
    """Extract the canonical exchange name from a file name prefix."""
    stem = Path(path).stem.lower()
    for prefix in sorted(_EXCHANGE_PREFIXES, key=len, reverse=True):
        if stem.startswith(prefix + "_"):
            return _EXCHANGE_PREFIXES[prefix]
    return None


def is_active_market(df: pd.DataFrame, min_active_share: float = config.MIN_ACTIVE_SHARE) -> bool:
    """False for "dead" feeds: no trades and a frozen price (stale placeholder quotes).

    Example: Crypto.com BTC/USDC in March 2023 reports a constant 20,100 close with
    zero volume for all 30,240 minutes; treating it as a live price would create a
    spurious 42% cross-exchange "arbitrage gap".
    """
    if df.empty:
        return False
    traded = (df["volume"].fillna(0) > 0).mean()
    return bool(traded >= min_active_share and df["close"].nunique() > 1)


def load_exchange(
    exchange: str,
    root: str | Path | None = None,
    window: tuple[str, str] | None = config.SVB_WINDOW,
    pairs: Iterable[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Load every 1-minute file for one exchange, keyed by pair (``BTCUSDC`` ...)."""
    folder = Path(root or config.PROCESSED_DIR) / exchange
    wanted = set(pairs) if pairs else None
    out: dict[str, pd.DataFrame] = {}
    for path in sorted(folder.glob("*.csv")):
        pair = parse_pair(path)
        if pair is None or (wanted and pair not in wanted):
            continue
        df = read_ohlcv_csv(path, window)
        if not df.empty:
            out[pair] = df
    return out


def load_svb_panel(
    source: str = "processed",
    exchanges: Iterable[str] = config.EXCHANGES,
    window: tuple[str, str] | None = config.SVB_WINDOW,
    drop_inactive: bool = True,
    verbose: bool = False,
) -> dict[str, pd.DataFrame]:
    """Load the March-2023 cross-exchange panel as ``{"binance_BTCUSDC": df, ...}``.

    Parameters
    ----------
    source : "processed" (gap-filled, default) or "raw".
    drop_inactive : drop dead feeds (see :func:`is_active_market`).
    """
    root = config.PROCESSED_DIR if source == "processed" else config.RAW_DIR
    panel: dict[str, pd.DataFrame] = {}
    for exchange in exchanges:
        for pair, df in load_exchange(exchange, root, window).items():
            key = f"{exchange}_{pair}"
            if drop_inactive and not is_active_market(df):
                if verbose:
                    print(f"  dropping inactive market {key}")
                continue
            panel[key] = df
    return panel


def load_eurusd(path: str | Path | None = None) -> pd.Series:
    """EUR/USD 1-minute close, forward-filled over weekends (FX is shut Sat/Sun)."""
    path = Path(path or config.RAW_DIR / "fx" / "eurusd_clean_march_2023.csv")
    fx = standardize_ohlcv(pd.read_csv(path))["close"]
    grid = pd.date_range(fx.index.min(), fx.index.max(), freq="1min", tz="UTC", name="timestamp")
    return fx.reindex(grid).ffill().rename("EURUSD")


# --------------------------------------------------------------------------- #
# Issuer balance sheets, regression inputs and daily market data
# --------------------------------------------------------------------------- #
def _to_number(x):
    if isinstance(x, str):
        x = x.replace("$", "").replace(",", "").strip()
        if x in {"", "-"}:
            return 0.0
        try:
            return float(x)
        except ValueError:
            return np.nan
    return x


def load_reserves(issuer: str, path: str | Path | None = None) -> pd.DataFrame:
    """Daily mark-to-market reserve reconstruction for ``"tether"`` or ``"circle"``.

    Source CSVs interpolate quarterly attestations using duration-based MTM
    (see ``docs/methodology.md``). Duplicate column names (the Tether file has two
    ``Corporate Bonds`` columns: notional and duration) are disambiguated.
    """
    issuer = issuer.lower()
    if issuer not in {"tether", "circle"}:
        raise ValueError("issuer must be 'tether' or 'circle'")
    path = Path(path or config.RESERVES_DIR / f"{issuer}assets.csv")
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    if issuer == "tether":
        df = df.rename(columns={"Corporate Bonds.1": "Corporate Bonds Duration"})
    for col in df.columns.drop("Date"):
        df[col] = df[col].map(_to_number).astype(float)
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%y")
    return df.set_index("Date").sort_index()


def load_regression_inputs(coin: str, path: str | Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(daily_exchange_panel, issuer_balance_sheet)`` for USDC or USDT."""
    coin = coin.upper()
    sheets = {"USDC": ("USDCDaily", "Circle"), "USDT": ("USDTDaily", "Tether")}
    if coin not in sheets:
        raise ValueError("coin must be 'USDC' or 'USDT'")
    path = Path(path or config.REGRESSION_DIR / f"{coin.lower()}_regression_inputs.xlsx")
    daily_sheet, bs_sheet = sheets[coin]
    daily = pd.read_excel(path, sheet_name=daily_sheet)
    bs = pd.read_excel(path, sheet_name=bs_sheet).rename(columns={"Date": "date"})
    for frame in (daily, bs):
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.tz_localize(None).dt.normalize()
    return daily, bs


def load_daily_crypto(path: str | Path | None = None) -> pd.DataFrame:
    """Daily BTC/ETH/USDC/USDT OHLCV (Binance.US -> Coinbase -> Kraken priority)."""
    path = Path(path or config.RAW_DIR / "daily" / "crypto_ohlcv_daily.xlsx")
    df = pd.read_excel(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.set_index("Date").sort_index()


def load_traditional_markets(path: str | Path | None = None) -> pd.DataFrame:
    """Daily DXY / XAU / VIX / S&P 500 / KBW OHLC.

    The original series are Bloomberg-licensed and therefore *not* distributed.
    Place your own export at ``data/external/traditional_market_data.xlsx`` with
    columns ``Dates, High _DXY, Low _DXY, ...`` (see ``data/README.md``).
    """
    path = Path(path or config.EXTERNAL_DIR / "traditional_market_data.xlsx")
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Traditional-market data is Bloomberg-licensed; "
            "see data/README.md for how to supply it."
        )
    df = pd.read_excel(path).rename(columns={"Dates": "Date"})
    df.columns = [c.replace(" _", "_").strip() for c in df.columns]
    df["Date"] = pd.to_datetime(df["Date"])
    return df.set_index("Date").sort_index()


def load_orderbook(pair: str, root: str | Path | None = None, days: Iterable[str] | None = None) -> pd.DataFrame:
    """Concatenate CoinAPI level-1 order-book snapshots for ``pair`` (e.g. ``BINANCE_BTC_USDT``).

    Expected layout: ``<root>/<YYYY-MM-DD>/<pair>_ob_<YYYY-MM-DD>.csv``. CoinAPI data
    is licensed, so it lives under ``data/external/orderbook`` and is git-ignored.
    """
    root = Path(root or config.EXTERNAL_DIR / "orderbook")
    if days is None:
        days = sorted(p.name for p in root.iterdir() if p.is_dir())
    cols = ["time_exchange", "bid_price_1", "bid_size_1", "ask_price_1", "ask_size_1"]
    frames = []
    for day in days:
        path = root / day / f"{pair}_ob_{day}.csv"
        if path.exists():
            frames.append(pd.read_csv(path, usecols=lambda c: c.strip() in cols))
    if not frames:
        raise FileNotFoundError(f"No order-book files for {pair} under {root}")
    ob = pd.concat(frames, ignore_index=True)
    ob.columns = ob.columns.str.strip()
    ob["time_exchange"] = pd.to_datetime(ob["time_exchange"], utc=True, format="ISO8601")
    ob = ob.sort_values("time_exchange").set_index("time_exchange")
    ob["mid_price"] = (ob["bid_price_1"] + ob["ask_price_1"]) / 2
    return ob
