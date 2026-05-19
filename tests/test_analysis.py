"""Tests for src/analysis.py — Phase 5 analytical functions."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis import (
    capacity_analysis,
    multiple_testing_diagnostic,
    sharpe_by_year,
    sharpe_decay_regression,
)


# ---------------------------------------------------------------------------
# sharpe_by_year
# ---------------------------------------------------------------------------

def _constant_pnl_series(daily_pnl: float, years: list[int]) -> pd.Series:
    dates = pd.date_range(f"{years[0]}-01-01", f"{years[-1]}-12-31", freq="B")
    return pd.Series(daily_pnl, index=dates)


def test_sharpe_by_year_returns_one_row_per_year():
    s = _constant_pnl_series(100.0, [2020, 2021, 2022])
    result = sharpe_by_year(s)
    assert set(result.index) == {2020, 2021, 2022}


def test_sharpe_by_year_constant_pnl_is_nan():
    # Constant daily PnL → zero variance → Sharpe should be NaN
    s = _constant_pnl_series(100.0, [2020])
    result = sharpe_by_year(s)
    assert np.isnan(result.loc[2020, "sharpe"])


def test_sharpe_by_year_zero_pnl_is_nan():
    s = _constant_pnl_series(0.0, [2020])
    result = sharpe_by_year(s)
    assert np.isnan(result.loc[2020, "sharpe"])


def test_sharpe_by_year_correct_value():
    rng = np.random.default_rng(42)
    daily = rng.normal(loc=10, scale=100, size=252)
    dates = pd.date_range("2020-01-01", periods=252, freq="B")
    s = pd.Series(daily, index=dates)
    result = sharpe_by_year(s)
    expected_sharpe = (daily.mean() * 252) / (daily.std(ddof=1) * np.sqrt(252))
    assert pytest.approx(result.loc[2020, "sharpe"], rel=0.02) == expected_sharpe


def test_sharpe_by_year_returns_required_columns():
    s = _constant_pnl_series(100.0, [2020, 2021])
    result = sharpe_by_year(s)
    for col in ["sharpe", "ann_return", "ann_vol", "n_days"]:
        assert col in result.columns, f"Missing column: {col}"


def test_sharpe_by_year_short_year_returns_nan():
    # A year with fewer than 20 observations → NaN sharpe
    dates = pd.date_range("2020-12-20", "2020-12-31", freq="B")
    s = pd.Series(np.random.randn(len(dates)), index=dates)
    result = sharpe_by_year(s)
    assert np.isnan(result.loc[2020, "sharpe"])


# ---------------------------------------------------------------------------
# sharpe_decay_regression
# ---------------------------------------------------------------------------

def _make_sharpe_df(years, sharpes) -> pd.DataFrame:
    df = pd.DataFrame({"sharpe": sharpes}, index=pd.Index(years, name="year"))
    df["ann_return"] = np.nan
    df["ann_vol"] = np.nan
    df["n_days"] = 252
    return df


def test_sharpe_decay_regression_negative_slope():
    years = list(range(2014, 2024))
    sharpes = [2.0 - 0.2 * i for i in range(len(years))]
    df = _make_sharpe_df(years, sharpes)
    result = sharpe_decay_regression(df)
    assert result["slope"] < 0


def test_sharpe_decay_regression_returns_required_keys():
    years = list(range(2014, 2024))
    sharpes = list(np.random.randn(len(years)))
    df = _make_sharpe_df(years, sharpes)
    result = sharpe_decay_regression(df)
    for key in ["slope", "t_stat", "p_value", "r_squared", "n", "intercept"]:
        assert key in result, f"Missing key: {key}"


def test_sharpe_decay_regression_p_value_in_range():
    years = list(range(2014, 2024))
    sharpes = list(np.random.default_rng(0).normal(size=len(years)))
    df = _make_sharpe_df(years, sharpes)
    result = sharpe_decay_regression(df)
    assert 0.0 <= result["p_value"] <= 1.0


def test_sharpe_decay_regression_fewer_than_3_returns_nan():
    df = _make_sharpe_df([2020, 2021], [1.0, np.nan])
    result = sharpe_decay_regression(df)
    assert np.isnan(result["slope"])
    assert np.isnan(result["t_stat"])


def test_sharpe_decay_regression_r_squared_in_range():
    years = list(range(2014, 2024))
    sharpes = [1.0 - 0.1 * i for i in range(len(years))]
    df = _make_sharpe_df(years, sharpes)
    result = sharpe_decay_regression(df)
    assert 0.0 <= result["r_squared"] <= 1.0


# ---------------------------------------------------------------------------
# capacity_analysis
# ---------------------------------------------------------------------------

def _make_capacity_pairs() -> pd.DataFrame:
    return pd.DataFrame({
        "ticker_a": ["CBA.AX", "WBC.AX"],
        "ticker_b": ["WBC.AX", "ANZ.AX"],
        "sector": ["Financials_Banks", "Financials_Banks"],
        "beta": [1.1, 0.95],
        "alpha": [0.02, -0.01],
        "half_life_days": [15.0, 20.0],
        "ou_sigma": [0.01, 0.012],
    })


def _make_long_prices(tickers, n_days=252) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    rows = []
    dates = pd.date_range("2022-01-01", periods=n_days, freq="B")
    for tkr in tickers:
        price = 100.0 + rng.normal(0, 1, n_days).cumsum()
        volume = rng.integers(500_000, 5_000_000, n_days).astype(float)
        for d, p, v in zip(dates, price, volume):
            rows.append({"date": d, "ticker": tkr, "adj_close": max(p, 1.0), "volume": v})
    df = pd.DataFrame(rows).set_index(["date", "ticker"])
    return df


def test_capacity_analysis_returns_dataframe():
    pairs = _make_capacity_pairs()
    prices = _make_long_prices(["CBA.AX", "WBC.AX", "ANZ.AX"])
    result = capacity_analysis(pairs, prices)
    assert isinstance(result, pd.DataFrame)


def test_capacity_analysis_required_columns():
    pairs = _make_capacity_pairs()
    prices = _make_long_prices(["CBA.AX", "WBC.AX", "ANZ.AX"])
    result = capacity_analysis(pairs, prices)
    for col in ["ticker_a", "ticker_b", "adv_a", "adv_b", "max_pos_a", "max_pos_b", "pair_capacity"]:
        assert col in result.columns, f"Missing column: {col}"


def test_capacity_analysis_adv_positive():
    pairs = _make_capacity_pairs()
    prices = _make_long_prices(["CBA.AX", "WBC.AX", "ANZ.AX"])
    result = capacity_analysis(pairs, prices)
    assert (result["adv_a"] > 0).all()
    assert (result["adv_b"] > 0).all()


def test_capacity_analysis_max_pos_is_1pct_adv():
    pairs = _make_capacity_pairs()
    prices = _make_long_prices(["CBA.AX", "WBC.AX", "ANZ.AX"])
    result = capacity_analysis(pairs, prices)
    pd.testing.assert_series_equal(
        result["max_pos_a"],
        result["adv_a"] * 0.01,
        check_names=False,
        rtol=1e-6,
    )


def test_capacity_analysis_pair_capacity_is_min_of_legs():
    pairs = _make_capacity_pairs()
    prices = _make_long_prices(["CBA.AX", "WBC.AX", "ANZ.AX"])
    result = capacity_analysis(pairs, prices)
    expected = result[["max_pos_a", "max_pos_b"]].min(axis=1)
    pd.testing.assert_series_equal(result["pair_capacity"], expected, check_names=False)


def test_capacity_analysis_row_count_matches_pairs():
    pairs = _make_capacity_pairs()
    prices = _make_long_prices(["CBA.AX", "WBC.AX", "ANZ.AX"])
    result = capacity_analysis(pairs, prices)
    assert len(result) == len(pairs)


# ---------------------------------------------------------------------------
# multiple_testing_diagnostic
# ---------------------------------------------------------------------------

def _make_synthetic_sectors_and_prices():
    """Tiny universe: 2 sectors, 2 tickers each, 600 days."""
    rng = np.random.default_rng(7)
    dates = pd.date_range("2018-01-01", periods=600, freq="B")
    tickers = ["A.AX", "B.AX", "C.AX", "D.AX"]
    prices = pd.DataFrame(
        {t: np.exp(rng.normal(0, 0.01, 600).cumsum()) * 100 for t in tickers},
        index=dates,
    )
    sectors = {"SectorX": ["A.AX", "B.AX"], "SectorY": ["C.AX", "D.AX"]}
    return prices, sectors


def test_multiple_testing_diagnostic_returns_required_keys():
    prices, sectors = _make_synthetic_sectors_and_prices()
    result = multiple_testing_diagnostic(
        prices, sectors, n_shuffles=3, seed=42,
        window_start="2018-01-01", window_end="2020-06-30",
    )
    for key in ["n_real", "n_shuffled_mean", "n_shuffled_median",
                "n_shuffled_p95", "n_shuffled_counts"]:
        assert key in result, f"Missing key: {key}"


def test_multiple_testing_diagnostic_counts_length():
    prices, sectors = _make_synthetic_sectors_and_prices()
    result = multiple_testing_diagnostic(
        prices, sectors, n_shuffles=5, seed=0,
        window_start="2018-01-01", window_end="2020-06-30",
    )
    assert len(result["n_shuffled_counts"]) == 5


def test_multiple_testing_diagnostic_nonnegative():
    prices, sectors = _make_synthetic_sectors_and_prices()
    result = multiple_testing_diagnostic(
        prices, sectors, n_shuffles=3, seed=1,
        window_start="2018-01-01", window_end="2020-06-30",
    )
    assert result["n_real"] >= 0
    assert result["n_shuffled_mean"] >= 0


def test_multiple_testing_diagnostic_reproducible():
    prices, sectors = _make_synthetic_sectors_and_prices()
    r1 = multiple_testing_diagnostic(
        prices, sectors, n_shuffles=3, seed=99,
        window_start="2018-01-01", window_end="2020-06-30",
    )
    r2 = multiple_testing_diagnostic(
        prices, sectors, n_shuffles=3, seed=99,
        window_start="2018-01-01", window_end="2020-06-30",
    )
    assert r1["n_shuffled_counts"] == r2["n_shuffled_counts"]
