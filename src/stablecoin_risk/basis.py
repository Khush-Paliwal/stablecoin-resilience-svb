"""Cross-quote BTC basis and Markov-switching regime detection.

If USDC (USDT) trades at par, BTC/USD and BTC/USDC (BTC/USDT) on the same venue
should coincide. The relative basis

    b_t = (P_t^{BTC/USD} - P_t^{BTC/Q}) / P_t^{BTC/Q}

therefore measures the market-implied discount on stablecoin Q. A two-state
Markov-switching mean model separates "calm" and "stress" regimes and dates the
switch without imposing event windows.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from stablecoin_risk.microstructure import cross_quote_basis, original_close


def venue_basis(panel: dict[str, pd.DataFrame], exchange: str, quote: str, asset: str = "BTC") -> pd.Series | None:
    """Basis of ``asset``/USD vs ``asset``/``quote`` on one exchange (None if a leg is missing)."""
    usd, other = panel.get(f"{exchange}_{asset}USD"), panel.get(f"{exchange}_{asset}{quote}")
    if usd is None or other is None:
        return None
    return cross_quote_basis(original_close(usd), original_close(other)).rename(f"{exchange}_{asset}_USD_{quote}")


def basis_panel(panel: dict[str, pd.DataFrame], quotes=("USDC", "USDT"), asset: str = "BTC") -> pd.DataFrame:
    """All available venue/quote basis series as columns (decimal, not bps)."""
    exchanges = sorted({k.split("_BTC")[0] for k in panel if "_BTC" in k})
    series = [venue_basis(panel, ex, q, asset) for ex in exchanges for q in quotes]
    # Drop identically-zero bases (mirrored books, e.g. Coinbase BTC-USDC == BTC-USD).
    return pd.concat([s for s in series if s is not None and s.abs().max() > 0], axis=1)


def basis_summary(basis: pd.DataFrame, regimes: dict[str, tuple[str, str]]) -> pd.DataFrame:
    """Mean / std / min / max of each basis series by regime, in basis points."""
    rows = []
    for col in basis:
        for regime, (start, end) in regimes.items():
            s = basis[col].loc[start : f"{end} 23:59:59"].dropna() * 1e4
            rows.append({"series": col, "regime": regime, "n": len(s), "mean_bps": s.mean(),
                         "std_bps": s.std(), "min_bps": s.min(), "max_bps": s.max()})
    return pd.DataFrame(rows).set_index(["series", "regime"])


def fit_markov_switching(
    y: pd.Series, k_regimes: int = 2, switching_variance: bool = True, resample: str | None = "5min"
):
    """Fit a Markov-switching mean model; returns the statsmodels results object.

    The series is optionally resampled (last value) to keep the EM/BFGS fit fast
    and to damp microstructure noise. Regime 1 is relabelled, if needed, to be the
    high-|mean| ("stress") state. Returns ``None`` if the optimiser fails to converge.
    """
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

    y = y.replace([np.inf, -np.inf], np.nan).dropna()
    if resample:
        y = y.resample(resample).last().dropna()
    # The basis is O(1e-4); rescale to bps so the optimiser is well conditioned.
    model = MarkovRegression(y * 1e4, k_regimes=k_regimes, trend="c", switching_variance=switching_variance)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return model.fit(disp=False)
        except (np.linalg.LinAlgError, ValueError):
            return None


def regime_table(results: dict[str, object]) -> pd.DataFrame:
    """Regime means (bps), persistence and expected durations for several fits."""
    rows = []
    for name, res in results.items():
        params = res.params
        means = np.array([params["const[0]"], params["const[1]"]])
        stress = int(np.argmax(np.abs(means)))
        calm = 1 - stress
        p = res.regime_transition[:, :, 0] if res.regime_transition.ndim == 3 else res.regime_transition
        probs = res.smoothed_marginal_probabilities[stress]
        rows.append(
            {
                "series": name,
                "calm_mean_bps": means[calm],
                "stress_mean_bps": means[stress],
                "p_stay_calm": p[calm, calm],
                "p_stay_stress": p[stress, stress],
                "expected_stress_duration_bars": res.expected_durations[stress],
                "first_stress_time": probs.index[probs > 0.5][0] if (probs > 0.5).any() else pd.NaT,
                "share_in_stress": float((probs > 0.5).mean()),
            }
        )
    return pd.DataFrame(rows).set_index("series")
