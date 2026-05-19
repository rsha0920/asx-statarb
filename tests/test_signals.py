"""Tests for signal generation state machine correctness.

All tests use synthetic z-score series with deterministic behaviour so
the expected positions and exit reasons can be verified exactly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.signals import generate_signals


def _zseries(values: list[float]) -> pd.Series:
    idx = pd.date_range("2020-01-01", periods=len(values), freq="B")
    return pd.Series(values, index=idx, dtype=float)


# ---------------------------------------------------------------------------
# Test 1: Standard mean-revert cycle
# ---------------------------------------------------------------------------

def test_mean_revert_exit():
    """Enter long on z=-3, hold through z=-1, exit at z=0.4 (mean_revert).

    Sequence: -3, -2.5, -1.0, 0.6, 0.4
    - Bar 0 (z=-3):   enter long; |z|=3 > exit=0.5, no exit same bar
    - Bar 1 (z=-2.5): hold; |z|=2.5 > 0.5
    - Bar 2 (z=-1.0): hold; |z|=1.0 > 0.5
    - Bar 3 (z=0.6):  hold; |z|=0.6 > 0.5, does not reach exit threshold
    - Bar 4 (z=0.4):  exit; |z|=0.4 <= 0.5 → mean_revert
    """
    z = _zseries([-3.0, -2.5, -1.0, 0.6, 0.4])
    sig = generate_signals(z, half_life=10.0)
    idx = z.index

    assert sig.loc[idx[0], "position"] == 1, "Should enter long at z=-3"
    assert sig.loc[idx[0], "entry_signal"], "entry_signal must be set at entry bar"
    assert not sig.loc[idx[0], "exit_signal"], "Cannot exit on entry bar"
    assert sig.loc[idx[2], "position"] == 1, "Should remain long through z=-1"
    assert not sig.loc[idx[3], "exit_signal"], "z=0.6 is above exit threshold 0.5"
    assert sig.loc[idx[4], "exit_signal"], "Should exit at z=0.4"
    assert sig.loc[idx[4], "exit_reason"] == "mean_revert"
    assert sig.loc[idx[4], "position"] == 0


# ---------------------------------------------------------------------------
# Test 2: Hard stop
# ---------------------------------------------------------------------------

def test_hard_stop_exit():
    """Enter long on z=-2.5; hard stop fires when z drops to -5.

    Sequence: -2.5, -5.0, -4.5
    - Bar 0 (z=-2.5): enter long
    - Bar 1 (z=-5.0): |z|=5.0 >= hard_stop=4.0 → hard_stop exit
    - Bar 2 (z=-4.5): flat (previous exit fired); z <= -2.0 so new entry fires
    """
    z = _zseries([-2.5, -5.0, -4.5])
    sig = generate_signals(z, half_life=10.0)
    idx = z.index

    assert sig.loc[idx[0], "position"] == 1, "Should enter long at z=-2.5"
    assert sig.loc[idx[1], "exit_signal"], "Hard stop must fire at z=-5"
    assert sig.loc[idx[1], "exit_reason"] == "hard_stop"
    assert sig.loc[idx[1], "position"] == 0, "Position must be flat on hard-stop bar"


# ---------------------------------------------------------------------------
# Test 3: Time stop
# ---------------------------------------------------------------------------

def test_time_stop_exit():
    """Position stuck at z=-2.5 exits via time stop after ceil(2 * half_life) bars.

    With half_life=5.0, max_hold = ceil(10.0) = 10 bars.
    Entry on bar 0 (bars_held=0); time stop fires on bar 10 (bars_held=10 >= 10).
    """
    half_life = 5.0
    max_hold = int(np.ceil(2.0 * half_life))  # 10

    # Series length: max_hold + 2 so we can verify the bar after the stop too
    z = _zseries([-2.5] * (max_hold + 2))
    sig = generate_signals(z, half_life=half_life)
    idx = z.index

    assert sig.loc[idx[0], "position"] == 1, "Should enter long on bar 0"
    assert sig.loc[idx[max_hold - 1], "position"] == 1, (
        f"Should still be in position on bar {max_hold - 1}"
    )
    assert sig.loc[idx[max_hold], "exit_signal"], (
        f"Time stop must fire on bar {max_hold} (bars_held={max_hold} >= {max_hold})"
    )
    assert sig.loc[idx[max_hold], "exit_reason"] == "time_stop"
    assert sig.loc[idx[max_hold], "position"] == 0
    # z=-2.5 on bar max_hold+1 triggers a re-entry (flat → can enter next bar),
    # so we only assert the stop bar itself is flat.
