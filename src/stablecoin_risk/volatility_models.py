"""GARCH(1,1) and DCC(1,1) for the BTC/USD vs BTC/USDT cross-currency study (exploratory).

GARCH uses the optional ``arch`` package (``pip install -e .[exploratory]``);
the bivariate DCC of Engle (2002) is implemented here with a closed-form 2x2
likelihood so it runs on ~30k one-minute observations in seconds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def fit_garch(returns: pd.Series, scale: float = 1e4, dist: str = "t"):
    """GARCH(1,1) on ``scale``-ed returns; returns the arch result.

    * One-minute returns are O(1e-4); scaling to basis points keeps the optimiser well
      conditioned (with percent scaling it stalls at the starting values).
    * Student-t innovations by default: minute returns have tail index nu ~ 5.
    * When persistence sits at the alpha + beta = 1 boundary (IGARCH) SLSQP reports
      code 8; check ``result.convergence_flag`` rather than relying on the warning.
    """
    import warnings

    from arch import arch_model

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return arch_model(returns.dropna() * scale, vol="GARCH", p=1, q=1, dist=dist, rescale=False).fit(disp="off")


def _dcc_path(z: np.ndarray, a: float, b: float) -> tuple[float, np.ndarray]:
    """Negative DCC log-likelihood (up to a constant) and the correlation path for 2 series."""
    qbar = np.cov(z.T)
    q11, q22, q12 = qbar[0, 0], qbar[1, 1], qbar[0, 1]
    c = 1 - a - b
    rho = np.empty(len(z))
    nll = 0.0
    for t in range(len(z)):
        if t:
            z1, z2 = z[t - 1]
            q11 = c * qbar[0, 0] + a * z1 * z1 + b * q11
            q22 = c * qbar[1, 1] + a * z2 * z2 + b * q22
            q12 = c * qbar[0, 1] + a * z1 * z2 + b * q12
        r = q12 / np.sqrt(q11 * q22)
        rho[t] = r
        x, y = z[t]
        det = 1 - r * r
        nll += 0.5 * (np.log(det) + (x * x - 2 * r * x * y + y * y) / det - (x * x + y * y))
    return nll, rho


def fit_dcc(std_resid: pd.DataFrame) -> tuple[dict, pd.Series]:
    """Estimate DCC(1,1) on two standardised-residual series; returns (params, rho_t)."""
    z = std_resid.dropna().to_numpy()

    def objective(p):
        a, b = p
        if a <= 0 or b <= 0 or a + b >= 0.999:
            return 1e10
        return _dcc_path(z, a, b)[0]

    opt = minimize(objective, x0=[0.05, 0.90], method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-3})
    a, b = opt.x
    _, rho = _dcc_path(z, a, b)
    return {"alpha": float(a), "beta": float(b), "persistence": float(a + b), "converged": bool(opt.success)}, pd.Series(
        rho, index=std_resid.dropna().index, name="dcc_corr")
