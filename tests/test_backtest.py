"""Sanity tests for the backtest engine.

Uses synthetic price data so the tests are fast and deterministic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest import _walk_forward_windows, compute_metrics, run_backtest
from src.costs import BASE_BPS, STRESS_BPS, trade_cost


# ---------------------------------------------------------------------------
# Test 1: Walk-forward window generation
# ---------------------------------------------------------------------------

def test_walk_forward_windows_non_overlapping():
    """Trading windows must be contiguous and non-overlapping."""
    start = pd.Timestamp("2010-01-01")
    end   = pd.Timestamp("2015-12-31")
    windows = list(_walk_forward_windows(start, end, train_months=24, trade_months=6))

    assert len(windows) > 0, "Should produce at least one window"

    for i, (_, _, ts, te) in enumerate(windows):
        assert ts <= te, f"Window {i}: trade_start > trade_end"
        if i > 0:
            prev_te = windows[i - 1][3]
            assert ts > prev_te, f"Windows {i-1} and {i} overlap"

    # First trading window starts at start + 24 months
    expected_first = start + pd.DateOffset(months=24)
    assert windows[0][2] == pd.Timestamp(expected_first)


# ---------------------------------------------------------------------------
# Test 2: Costs reduce PnL (stress > base drag)
# ---------------------------------------------------------------------------

def test_stress_costs_lower_pnl():
    """Stress cost scenario must produce lower or equal equity than base."""
    from src.costs import trade_cost
    # Stress (25 bps) > base (10 bps) on any positive notional
    assert trade_cost(100_000, STRESS_BPS) > trade_cost(100_000, BASE_BPS)


# ---------------------------------------------------------------------------
# Test 3: compute_metrics recovers expected Sharpe on a simple PnL series
# ---------------------------------------------------------------------------

def test_compute_metrics_flat_pnl():
    """All-zero PnL should produce NaN Sharpe (avoid div-by-zero)."""
    pnl = pd.Series(np.zeros(252))
    m = compute_metrics(pnl)
    assert np.isnan(m["sharpe"]), "Zero-vol series should give NaN Sharpe"
    assert m["total_pnl"] == 0.0


def test_compute_metrics_constant_positive():
    """Constant positive PnL of $1/day: std=0 → NaN Sharpe, but total PnL correct."""
    pnl = pd.Series(np.ones(252))
    m = compute_metrics(pnl)
    assert np.isnan(m["sharpe"]), "Constant series has zero std → NaN Sharpe"
    assert m["total_pnl"] == pytest.approx(252.0)
    assert m["max_drawdown"] == pytest.approx(0.0)


def test_compute_metrics_sharpe_sign():
    """Positive-mean PnL gives positive Sharpe; negative-mean gives negative."""
    rng = np.random.default_rng(99)
    pos_pnl = pd.Series(rng.normal(loc=100, scale=50, size=1000))
    neg_pnl = pd.Series(rng.normal(loc=-100, scale=50, size=1000))
    assert compute_metrics(pos_pnl)["sharpe"] > 0
    assert compute_metrics(neg_pnl)["sharpe"] < 0


# ---------------------------------------------------------------------------
# Test 4: run_backtest on minimal synthetic universe (smoke test)
# ---------------------------------------------------------------------------

def _make_synthetic_prices(n_days: int = 1800, seed: int = 0) -> pd.DataFrame:
    """Two cointegrated pairs in two sectors, plus two standalone tickers."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-01", periods=n_days)

    x1 = np.cumsum(rng.normal(0, 1, n_days))
    x2 = np.cumsum(rng.normal(0, 1, n_days))

    b = np.exp(-0.05)  # theta=0.05, HL≈14d — well within [5, 60] filter
    spread1 = np.zeros(n_days)
    spread2 = np.zeros(n_days)
    for t in range(1, n_days):
        spread1[t] = b * spread1[t - 1] + rng.normal(0, 0.3)
        spread2[t] = b * spread2[t - 1] + rng.normal(0, 0.3)

    # Tick prices: exp(log-prices) to get positive prices
    base = 50.0
    df = pd.DataFrame(
        {
            "A1.AX": base * np.exp(x1 + spread1),
            "A2.AX": base * np.exp(x1),
            "B1.AX": base * np.exp(x2 + spread2),
            "B2.AX": base * np.exp(x2),
        },
        index=dates,
    )
    return df


def test_run_backtest_smoke():
    """Backtest runs without error and returns expected keys and shapes."""
    prices = _make_synthetic_prices(n_days=1800)
    sectors = {
        "SectorA": ["A1.AX", "A2.AX"],
        "SectorB": ["B1.AX", "B2.AX"],
    }

    result = run_backtest(
        prices,
        sectors,
        start="2012-01-01",
        end="2014-12-31",
        train_months=24,
        trade_months=6,
        total_capital=100_000,
        max_pairs=4,
        min_history_days=200,  # relaxed for synthetic data
    )

    assert "daily_pnl_base" in result
    assert "daily_pnl_stress" in result
    assert "equity_base" in result
    assert "trade_log" in result
    assert "window_log" in result

    # Equity series should be aligned to trading dates
    assert len(result["equity_base"]) > 0

    # Stress equity must be <= base equity (more costs → lower cumulative PnL)
    assert (result["equity_stress"] <= result["equity_base"] + 1e-6).all(), (
        "Stress scenario should not outperform base scenario"
    )
