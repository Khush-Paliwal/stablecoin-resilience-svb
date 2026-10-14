import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(7)


def make_bars(n: int = 3_000, start: str = "2023-03-01", seed: int = 0, spread: float = 0.0) -> pd.DataFrame:
    """Synthetic 1-minute OHLCV: random-walk mid, optional bid-ask bounce on the close."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq="1min", tz="UTC", name="timestamp")
    mid = 100 * np.exp(np.cumsum(rng.normal(0, 2e-4, n)))
    close = mid * (1 + spread / 2 * rng.choice([-1, 1], n))
    high = np.maximum(mid, close) * (1 + np.abs(rng.normal(0, 1e-4, n)))
    low = np.minimum(mid, close) * (1 - np.abs(rng.normal(0, 1e-4, n)))
    return pd.DataFrame({"open": mid, "high": high, "low": low, "close": close,
                         "volume": rng.uniform(0.5, 2.0, n), "num_trades": rng.integers(1, 50, n).astype(float)},
                        index=idx)


@pytest.fixture
def bars():
    return make_bars()
