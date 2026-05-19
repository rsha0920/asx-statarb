"""Trading signal generation for cointegration-based pairs trading.

Provides z-score computation and a state-machine signal generator that
implements the entry/exit/stop rules from CLAUDE.md.

Typical usage:
    spread = log_a - beta * log_b - alpha          # from engle_granger
    zscore = compute_zscore(spread, window=60)
    signals = generate_signals(zscore, half_life=ou['half_life'])
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def compute_zscore(spread: pd.Series, window: int = 60) -> pd.Series:
    """Rolling z-score of the spread.

    Parameters
    ----------
    spread : pd.Series
        Cointegration residual (spread) time series.
    window : int
        Rolling window length in trading days (default 60).

    Returns
    -------
    pd.Series
        (spread - rolling_mean) / rolling_std.  NaN for the first
        window - 1 bars; min_periods=window ensures no partial-window
        estimates leak into the signal.
    """
    mean = spread.rolling(window, min_periods=window).mean()
    std = spread.rolling(window, min_periods=window).std(ddof=1)
    return (spread - mean) / std


def generate_signals(
    zscore: pd.Series,
    half_life: float,
    entry: float = 2.0,
    exit: float = 0.5,
    hard_stop: float = 4.0,
) -> pd.DataFrame:
    """State-machine signal generator for a single spread.

    Parameters
    ----------
    zscore : pd.Series
        Rolling z-score series. NaN values are skipped (position held).
    half_life : float
        Mean-reversion half-life in trading days (from OU estimation).
        Time stop fires after ceil(2 * half_life) bars in a position.
    entry : float
        Enter long when z <= -entry, short when z >= +entry (default 2.0).
    exit : float
        Exit when |z| <= exit (default 0.5).
    hard_stop : float
        Emergency exit when |z| >= hard_stop while in a position (default 4.0).

    Returns
    -------
    pd.DataFrame with columns:
        position     : int    -1 (short), 0 (flat), +1 (long)
        entry_signal : bool   True on the bar a position was opened
        exit_signal  : bool   True on the bar a position was closed
        exit_reason  : object None | "mean_revert" | "hard_stop" | "time_stop"

    Notes
    -----
    Implemented as an explicit sequential loop to prevent look-ahead bias.

    Per-bar decision order (when flat):
        z <= -entry  → enter long (+1)
        z >= +entry  → enter short (-1)

    Per-bar decision order (when in a position), checked in priority order:
        1. |z| >= hard_stop            → exit "hard_stop"   (most urgent)
        2. bars_held >= ceil(2 * HL)   → exit "time_stop"
        3. |z| <= exit                 → exit "mean_revert"

    Constraints:
        - Cannot enter and exit the same bar (exits are only checked
          from the bar *after* entry).
        - Direct flip (long → short without intermediate flat) is not
          allowed. After an exit the position is flat; a new entry
          fires on the first subsequent bar where the threshold is met.
    """
    max_hold = int(np.ceil(2.0 * half_life))
    n = len(zscore)
    z_vals = zscore.values

    position = np.zeros(n, dtype=np.intp)
    entry_signal = np.zeros(n, dtype=bool)
    exit_signal = np.zeros(n, dtype=bool)
    exit_reason: list[str | None] = [None] * n

    current_pos = 0
    entry_bar = -1

    for i in range(n):
        z = z_vals[i]

        if np.isnan(z):
            position[i] = current_pos
            continue

        if current_pos == 0:
            if z <= -entry:
                current_pos = 1
                entry_bar = i
                entry_signal[i] = True
            elif z >= entry:
                current_pos = -1
                entry_bar = i
                entry_signal[i] = True
        else:
            bars_held = i - entry_bar
            reason: str | None = None

            if abs(z) >= hard_stop:
                reason = "hard_stop"
            elif bars_held >= max_hold:
                reason = "time_stop"
            elif abs(z) <= exit:
                reason = "mean_revert"

            if reason is not None:
                exit_signal[i] = True
                exit_reason[i] = reason
                current_pos = 0
                entry_bar = -1

        position[i] = current_pos

    return pd.DataFrame(
        {
            "position": position,
            "entry_signal": entry_signal,
            "exit_signal": exit_signal,
            "exit_reason": exit_reason,
        },
        index=zscore.index,
    )
