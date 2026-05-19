"""Filter functions for cointegration screening output.

All functions accept the DataFrame produced by screen_pairs() and return
a filtered copy. Filters are pure: they do not mutate the input.

Typical pipeline (matches CLAUDE.md Phase 2 spec):
    surviving = apply_all_filters(screen_pairs(...))
"""

from __future__ import annotations

import pandas as pd


def filter_half_life(df: pd.DataFrame, lo: float = 5, hi: float = 60) -> pd.DataFrame:
    """Keep pairs with mean-reversion half-life in [lo, hi] trading days.

    Pairs with half_life=NaN (estimation failed) or half_life=inf (no mean
    reversion) are dropped.
    """
    return df[df["half_life_days"].between(lo, hi)].copy()


def filter_hurst(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Keep pairs with Hurst exponent strictly below threshold.

    H < 0.5 confirms anti-persistent (mean-reverting) behaviour beyond the
    ADF test. Pairs with hurst=NaN are dropped.
    """
    return df[df["hurst"] < threshold].copy()


def filter_significance(df: pd.DataFrame) -> pd.DataFrame:
    """Keep pairs that pass the significance criterion.

    Multi-test sectors: bh_pass=True (BH-FDR corrected at 10%).
    Single-test sectors: raw ADF p-value < 0.05 (bh_pass already reflects
    this threshold, set during screen_pairs).

    The distinction is preserved explicitly here so the filter is readable
    without needing to understand how bh_pass was populated.
    """
    multi_pass = df["bh_pass"] & ~df["single_test_sector"]
    single_pass = df["single_test_sector"] & (df["adf_pvalue"] < 0.05)
    return df[multi_pass | single_pass].copy()


def apply_all_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Apply significance → half-life → Hurst filters in sequence.

    Order matters: significance first (cheapest to compute; already in df),
    then half-life, then Hurst. Each step narrows the candidate set.
    """
    return filter_hurst(filter_half_life(filter_significance(df)))
