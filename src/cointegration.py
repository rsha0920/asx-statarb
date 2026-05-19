"""Cointegration screening for pairs trading.

Implements the Engle-Granger two-step procedure, R/S-based Hurst exponent
estimation, Benjamini-Hochberg FDR correction, and a full sector-level
pair screening function.

Typical usage:
    prices = get_close_panel(df)          # wide panel from src.data
    candidates = screen_pairs(
        prices, ASX_UNIVERSE,
        window_start="2018-01-01",
        window_end="2020-12-31",
    )
"""

from __future__ import annotations

import itertools
import logging
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.stattools import adfuller

from .ou_process import estimate_ou

log = logging.getLogger(__name__)


def engle_granger(y: pd.Series, x: pd.Series) -> dict[str, Any]:
    """Engle-Granger two-step cointegration test.

    Parameters
    ----------
    y, x : pd.Series
        Log-price series. Need not be aligned — the function takes the inner
        intersection of non-NaN observations.

    Returns
    -------
    dict with keys:
        beta      : float      cointegrating coefficient (slope)
        alpha     : float      intercept
        residuals : pd.Series  spread = y - beta*x - alpha
        adf_stat  : float      ADF test statistic on residuals
        adf_pvalue: float      MacKinnon p-value

    Notes
    -----
    ADF on residuals uses regression='c' (constant included). Although the
    first-stage OLS de-means the residuals, using regression='n' (no constant)
    is anti-conservative in finite samples: OLS pre-selection biases the
    residuals toward apparent stationarity, and the no-constant ADF critical
    values don't account for that. regression='c' is more conservative and
    matches standard EG implementations (R po.test, Enders 2014).

    Returns NaN fields if fewer than 30 aligned observations are available.
    """
    combined = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    if len(combined) < 30:
        return dict(
            beta=np.nan, alpha=np.nan,
            residuals=pd.Series(dtype=float),
            adf_stat=np.nan, adf_pvalue=np.nan,
        )

    y_vals = combined["y"].values
    x_vals = combined["x"].values

    x_const = add_constant(x_vals)
    model = OLS(y_vals, x_const).fit()
    beta = float(model.params[1])
    alpha = float(model.params[0])
    residuals = pd.Series(
        y_vals - beta * x_vals - alpha,
        index=combined.index,
    )

    adf_result = adfuller(residuals.values, regression="c", autolag="AIC")
    return dict(
        beta=beta,
        alpha=alpha,
        residuals=residuals,
        adf_stat=float(adf_result[0]),
        adf_pvalue=float(adf_result[1]),
    )


def hurst_exponent(series: pd.Series, min_lag: int = 2, max_lag: int = 100) -> float:
    """Estimate the Hurst exponent via DFA (Detrended Fluctuation Analysis).

    H < 0.5  → anti-persistent / sub-diffusive
    H ≈ 0.5  → uncorrelated / random walk
    H > 0.5  → persistent / super-diffusive

    DFA integrates the mean-subtracted series into Y, then for each window
    size s fits and removes a linear trend inside each segment and measures
    the RMS fluctuation F(s). H is the slope of log(F(s)) vs log(s).

    DFA is preferred over R/S for financial series: it removes local trends,
    has lower small-sample bias, and handles non-stationarities better.

    When screen_pairs calls this on the FIRST DIFFERENCES of the spread, Y
    is approximately the spread level itself (cumsum of differences ≈ spread).
    For a mean-reverting (OU) spread, Y is sub-diffusive → F(s) ∝ s^H with
    H < 0.5. For a random walk spread, Y grows as sqrt(s) → H ≈ 0.5.

    Returns np.nan when fewer than 5 usable scales can be computed.
    """
    vals = np.asarray(series.dropna(), dtype=float)
    n = len(vals)

    # Integrate mean-subtracted series
    Y = np.cumsum(vals - vals.mean())

    # Linear detrending inside each DFA window is degenerate for windows
    # smaller than 4 points (2-point fit is exact → F ≈ 0 → log(F) = -∞).
    effective_min_lag = max(min_lag, 4)

    # Cap max_lag so we have at least 4 non-overlapping windows at the largest scale
    max_lag = min(max_lag, n // 4)
    if max_lag < effective_min_lag + 1:
        return np.nan

    # Log-spaced scales: even coverage across orders of magnitude, avoids
    # small-lag overweighting that inflates H in linear-spaced R/S.
    n_scales = min(25, max_lag - effective_min_lag + 1)
    scales = np.unique(
        np.round(
            np.logspace(np.log10(effective_min_lag), np.log10(max_lag), n_scales)
        ).astype(int)
    )

    log_scales: list[float] = []
    log_flucts: list[float] = []

    for s in scales:
        if s < effective_min_lag or s > n // 2:
            continue
        n_windows = n // s
        if n_windows < 2:
            continue

        t = np.arange(s, dtype=float)
        flucts_sq: list[float] = []
        for j in range(n_windows):
            seg = Y[j * s : (j + 1) * s]
            poly = np.polyfit(t, seg, 1)
            residuals = seg - np.polyval(poly, t)
            flucts_sq.append(float(np.mean(residuals ** 2)))

        F = np.sqrt(np.mean(flucts_sq))
        if F > 0:
            log_scales.append(np.log(s))
            log_flucts.append(np.log(F))

    if len(log_scales) < 5:
        return np.nan

    coef = np.polyfit(log_scales, log_flucts, 1)
    return float(coef[0])


def benjamini_hochberg(pvalues: np.ndarray, fdr: float = 0.10) -> np.ndarray:
    """Benjamini-Hochberg FDR correction (Benjamini & Hochberg 1995).

    Parameters
    ----------
    pvalues : array-like of float
    fdr : float
        Target false discovery rate (default 0.10).

    Returns
    -------
    np.ndarray of bool
        True where the null hypothesis is rejected.

    Notes
    -----
    For each hypothesis ranked i (1-indexed) by ascending p-value, the BH
    threshold is (i / m) * fdr. All hypotheses up to and including the last
    one where p_{(i)} <= threshold are rejected.
    """
    pvalues = np.asarray(pvalues, dtype=float)
    n = len(pvalues)
    if n == 0:
        return np.array([], dtype=bool)

    sort_order = np.argsort(pvalues)
    sorted_pvals = pvalues[sort_order]
    thresholds = (np.arange(1, n + 1) / n) * fdr

    passed = sorted_pvals <= thresholds
    reject_up_to = int(np.where(passed)[0][-1]) if passed.any() else -1

    result = np.zeros(n, dtype=bool)
    if reject_up_to >= 0:
        result[sort_order[: reject_up_to + 1]] = True
    return result


def screen_pairs(
    prices: pd.DataFrame,
    sectors: dict[str, list[str]],
    window_start,
    window_end,
    fdr: float = 0.10,
    min_history_days: int = 504,
) -> pd.DataFrame:
    """Screen all within-sector pairs for cointegration over a given window.

    Parameters
    ----------
    prices : pd.DataFrame
        Wide panel, rows=date (DatetimeIndex), columns=ticker.
        Use adj_close values; the function takes log internally.
    sectors : dict[str, list[str]]
        Sector groupings (e.g. ASX_UNIVERSE from src.data).
    window_start, window_end : date-like
        Inclusive window bounds.
    fdr : float
        BH FDR target, applied independently within each sector.
    min_history_days : int
        Minimum shared non-NaN observations required for a pair to be tested.
        Default 504 ≈ 2 trading years. Pairs where either ticker has shorter
        history are silently skipped and excluded from the BH hypothesis count.

    Returns
    -------
    pd.DataFrame sorted by adf_pvalue with columns:
        ticker_a, ticker_b, sector, beta, alpha, adf_stat, adf_pvalue,
        bh_pass, half_life_days, hurst, n_obs, single_test_sector

    Notes
    -----
    BH correction is applied within each sector independently. Sectors are
    treated as separate hypothesis families because the economic relationships
    differ across sectors.

    single_test_sector=True marks pairs from sectors where only one pair
    survived the min-history guard. BH correction is degenerate on a single
    test, so the raw ADF p-value < 0.05 threshold is used instead.
    """
    window = prices.loc[window_start:window_end]
    log_prices = np.log(window.replace(0.0, np.nan))

    _COLS = [
        "ticker_a", "ticker_b", "sector", "beta", "alpha",
        "adf_stat", "adf_pvalue", "bh_pass", "half_life_days",
        "ou_sigma", "hurst", "n_obs", "single_test_sector",
    ]
    records: list[dict[str, Any]] = []

    for sector, tickers in sectors.items():
        available = [t for t in tickers if t in log_prices.columns]
        if len(available) < 2:
            continue

        sector_records: list[dict[str, Any]] = []

        for ticker_a, ticker_b in itertools.combinations(available, 2):
            series_a = log_prices[ticker_a]
            series_b = log_prices[ticker_b]

            # Min-history guard: count observations where BOTH tickers are valid
            shared_idx = series_a.dropna().index.intersection(series_b.dropna().index)
            if len(shared_idx) < min_history_days:
                log.debug(
                    "Skipping %s/%s: %d shared obs (need %d)",
                    ticker_a, ticker_b, len(shared_idx), min_history_days,
                )
                continue

            try:
                eg = engle_granger(series_a.loc[shared_idx], series_b.loc[shared_idx])
            except Exception as exc:
                log.warning("engle_granger failed for %s/%s: %s", ticker_a, ticker_b, exc)
                continue

            if np.isnan(eg["adf_pvalue"]):
                continue

            ou = estimate_ou(eg["residuals"])
            # Hurst on spread CHANGES, not levels. AR(1) levels are positively
            # autocorrelated (H > 0.5 on levels even for strong mean reversion).
            # Differencing flips the sign: mean-reverting spreads produce
            # negatively autocorrelated changes → H < 0.5 on differences.
            h = hurst_exponent(eg["residuals"].diff().dropna())

            sector_records.append({
                "ticker_a": ticker_a,
                "ticker_b": ticker_b,
                "sector": sector,
                "beta": eg["beta"],
                "alpha": eg["alpha"],
                "adf_stat": eg["adf_stat"],
                "adf_pvalue": eg["adf_pvalue"],
                "half_life_days": ou.get("half_life", np.nan),
                "ou_sigma": ou.get("sigma", np.nan),
                "hurst": h,
                "n_obs": len(shared_idx),
            })

        if not sector_records:
            continue

        single_test = len(sector_records) == 1

        if single_test:
            for rec in sector_records:
                rec["bh_pass"] = bool(rec["adf_pvalue"] < 0.05)
                rec["single_test_sector"] = True
        else:
            pvals = np.array([r["adf_pvalue"] for r in sector_records])
            bh_mask = benjamini_hochberg(pvals, fdr=fdr)
            for rec, passed in zip(sector_records, bh_mask):
                rec["bh_pass"] = bool(passed)
                rec["single_test_sector"] = False

        records.extend(sector_records)

    if not records:
        return pd.DataFrame(columns=_COLS)

    return (
        pd.DataFrame(records)[_COLS]
        .sort_values("adf_pvalue")
        .reset_index(drop=True)
    )
