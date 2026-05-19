"""Ornstein-Uhlenbeck process estimation via AR(1) regression.

Discretised OU process at dt=1 trading day:

    s_t = a + b * s_{t-1} + eps

where:
    b   = exp(-theta)          AR coefficient
    a   = mu * (1 - b)         intercept
    eps ~ N(0, sigma^2)        residuals

Parameters are recovered as:
    theta     = -ln(b)         mean-reversion speed (per trading day)
    mu        = a / (1 - b)    long-run mean
    sigma     = std(eps)       diffusion magnitude
    half_life = ln(2) / theta  trading days for spread to halve
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

log = logging.getLogger(__name__)

_MIN_OBS = 60  # Minimum observations for stable AR(1) estimation


def estimate_ou(spread: pd.Series) -> dict[str, float]:
    """Fit AR(1) to a spread series and extract OU parameters.

    Parameters
    ----------
    spread : pd.Series
        Cointegration residual (spread) time series.

    Returns
    -------
    dict with keys:
        theta     : float  mean-reversion speed (per trading day); NaN on failure
        mu        : float  long-run mean of the spread
        sigma     : float  std of AR(1) residuals
        half_life : float  trading days; np.inf when b >= 1 (no mean reversion)

    Notes
    -----
    Returns all-NaN dict if len(spread.dropna()) < 60 — AR(1) estimates are
    unstable on short series and the half-life would be unreliable.

    b >= 1 → random walk or explosive process; half_life is set to np.inf and
    theta to 0.0 so downstream filters (half_life in [5, 60]) reject the pair.

    b <= 0 → oscillatory behaviour; not consistent with OU; all NaN returned.
    """
    vals = spread.dropna().values
    if len(vals) < _MIN_OBS:
        log.debug(
            "estimate_ou: %d obs below minimum %d, returning NaN",
            len(vals), _MIN_OBS,
        )
        return dict(theta=np.nan, mu=np.nan, sigma=np.nan, half_life=np.nan)

    y = vals[1:]
    X = add_constant(vals[:-1])
    try:
        model = OLS(y, X).fit()
    except Exception as exc:
        log.debug("estimate_ou: OLS failed (%s), returning NaN", exc)
        return dict(theta=np.nan, mu=np.nan, sigma=np.nan, half_life=np.nan)

    a = float(model.params[0])   # intercept
    b = float(model.params[1])   # AR coefficient

    sigma = float(np.std(model.resid, ddof=1))

    if b >= 1.0:
        return dict(theta=0.0, mu=np.nan, sigma=sigma, half_life=np.inf)

    if b <= 0.0:
        # Oscillatory — not an OU process
        return dict(theta=np.nan, mu=np.nan, sigma=sigma, half_life=np.nan)

    theta = -np.log(b)                # exact discretisation
    mu = a / (1.0 - b)
    half_life = np.log(2.0) / theta

    return dict(theta=theta, mu=mu, sigma=sigma, half_life=half_life)
