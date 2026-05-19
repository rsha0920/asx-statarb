"""Phase 5 analytical functions: alpha decay, capacity, and multiple-testing diagnostics."""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .cointegration import screen_pairs
from .filters import apply_all_filters

logger = logging.getLogger(__name__)


def sharpe_by_year(
    daily_pnl: pd.Series,
    annual_factor: float = 252,
    min_obs: int = 20,
) -> pd.DataFrame:
    """Compute annualised Sharpe ratio and related metrics for each calendar year.

    Parameters
    ----------
    daily_pnl : pd.Series
        Daily P&L series with a DatetimeIndex.
    annual_factor : float
        Number of trading days used for annualisation.
    min_obs : int
        Minimum observations required in a year to compute meaningful stats.

    Returns
    -------
    pd.DataFrame
        Index: year (int), columns: sharpe, ann_return, ann_vol, n_days.
        Rows with insufficient observations or zero variance have NaN for the
        first three columns.
    """
    records: list[dict[str, Any]] = []
    for year, group in daily_pnl.groupby(daily_pnl.index.year):
        n = len(group)
        std = float(group.std(ddof=1))
        if n < min_obs or std == 0:
            records.append(
                {"year": int(year), "sharpe": np.nan, "ann_return": np.nan, "ann_vol": np.nan, "n_days": n}
            )
        else:
            ann_return = float(group.mean()) * annual_factor
            ann_vol = std * math.sqrt(annual_factor)
            sharpe = ann_return / ann_vol
            records.append(
                {"year": int(year), "sharpe": sharpe, "ann_return": ann_return, "ann_vol": ann_vol, "n_days": n}
            )
    df = pd.DataFrame(records).set_index("year")
    return df


def sharpe_decay_regression(sharpe_df: pd.DataFrame) -> dict[str, Any]:
    """OLS regression of annual Sharpe on calendar year to detect alpha decay.

    Parameters
    ----------
    sharpe_df : pd.DataFrame
        Output of :func:`sharpe_by_year`. Index must be named ``year``.

    Returns
    -------
    dict
        Keys: slope, t_stat, p_value, r_squared, n, intercept.
        All float fields are np.nan when fewer than 3 valid observations exist.
    """
    valid = sharpe_df["sharpe"].dropna()
    n = int(len(valid))
    if n < 3:
        return {
            "slope": np.nan,
            "t_stat": np.nan,
            "p_value": np.nan,
            "r_squared": np.nan,
            "n": n,
            "intercept": np.nan,
        }

    years = valid.index.astype(float).values
    sharpes = valid.values.astype(float)
    result = stats.linregress(years, sharpes)
    slope = float(result.slope)
    se = float(result.stderr)
    t_stat = slope / se if se != 0 else np.nan
    return {
        "slope": slope,
        "t_stat": t_stat,
        "p_value": float(result.pvalue),
        "r_squared": float(result.rvalue ** 2),
        "n": n,
        "intercept": float(result.intercept),
    }


def capacity_analysis(
    pairs: pd.DataFrame,
    prices: pd.DataFrame,
    adv_window: int = 252,
    adv_fraction: float = 0.01,
) -> pd.DataFrame:
    """Estimate position capacity for each pair based on average daily value.

    Parameters
    ----------
    pairs : pd.DataFrame
        Pair candidates, must contain columns ``ticker_a`` and ``ticker_b``.
    prices : pd.DataFrame
        Long-form DataFrame indexed by ``(date, ticker)`` with columns
        ``adj_close`` and ``volume``.
    adv_window : int
        Number of most recent rows (per ticker) used to estimate ADV.
    adv_fraction : float
        Maximum position as a fraction of ADV.

    Returns
    -------
    pd.DataFrame
        Same row order as ``pairs``, columns:
        ticker_a, ticker_b, adv_a, adv_b, max_pos_a, max_pos_b, pair_capacity.
    """
    adv_map: dict[str, float] = {}
    for ticker, grp in prices.groupby(level="ticker"):
        tail = grp.tail(adv_window)
        adv_map[str(ticker)] = float((tail["adj_close"] * tail["volume"]).mean())

    rows: list[dict[str, Any]] = []
    for _, row in pairs.iterrows():
        ta, tb = str(row["ticker_a"]), str(row["ticker_b"])
        adv_a = adv_map.get(ta, np.nan)
        adv_b = adv_map.get(tb, np.nan)
        max_pos_a = adv_a * adv_fraction
        max_pos_b = adv_b * adv_fraction
        pair_cap = min(max_pos_a, max_pos_b)
        rows.append(
            {
                "ticker_a": ta,
                "ticker_b": tb,
                "adv_a": adv_a,
                "adv_b": adv_b,
                "max_pos_a": max_pos_a,
                "max_pos_b": max_pos_b,
                "pair_capacity": pair_cap,
            }
        )
    return pd.DataFrame(rows)


def multiple_testing_diagnostic(
    prices: pd.DataFrame,
    sectors: dict[str, list[str]],
    n_shuffles: int = 100,
    seed: int = 42,
    window_start: str = "2018-01-01",
    window_end: str = "2020-12-31",
) -> dict[str, Any]:
    """Compare surviving pairs on real data against column-permuted (null) data.

    Parameters
    ----------
    prices : pd.DataFrame
        Wide-form panel (rows=date, columns=ticker) as expected by
        :func:`~src.cointegration.screen_pairs`.
    sectors : dict[str, list[str]]
        Sector groupings passed through to ``screen_pairs``.
    n_shuffles : int
        Number of independent permutation replicates.
    seed : int
        Seed for :func:`numpy.random.default_rng`.
    window_start, window_end : str
        Date bounds forwarded to ``screen_pairs``.

    Returns
    -------
    dict
        n_real, n_shuffled_mean, n_shuffled_median, n_shuffled_p95,
        n_shuffled_counts (list of ints, length == n_shuffles).
    """
    real_pairs = apply_all_filters(
        screen_pairs(prices, sectors, window_start=window_start, window_end=window_end)
    )
    n_real = int(len(real_pairs))
    logger.info("Real data survivors: %d", n_real)

    rng = np.random.default_rng(seed)
    shuffled_counts: list[int] = []
    index = prices.index
    for i in range(n_shuffles):
        shuffled = pd.DataFrame(
            {col: prices[col].values[rng.permutation(len(prices))] for col in prices.columns},
            index=index,
        )
        try:
            n_surv = int(len(apply_all_filters(
                screen_pairs(shuffled, sectors, window_start=window_start, window_end=window_end)
            )))
        except Exception:
            logger.exception("Shuffle %d failed; recording 0 survivors", i)
            n_surv = 0
        shuffled_counts.append(n_surv)
        logger.debug("Shuffle %d/%d: %d survivors", i + 1, n_shuffles, n_surv)

    counts_arr = np.array(shuffled_counts, dtype=float)
    return {
        "n_real": n_real,
        "n_shuffled_mean": float(counts_arr.mean()),
        "n_shuffled_median": float(np.median(counts_arr)),
        "n_shuffled_p95": float(np.percentile(counts_arr, 95)),
        "n_shuffled_counts": shuffled_counts,
    }
