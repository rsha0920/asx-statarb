"""Synthetic tests for cointegration primitives and OU estimation.

All tests use fixed RNG seeds for reproducibility.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.cointegration import benjamini_hochberg, engle_granger
from src.ou_process import estimate_ou


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_walk(n: int, rng: np.random.Generator) -> pd.Series:
    return pd.Series(np.cumsum(rng.normal(0, 1, n)))


def _cointegrated_pair(
    n: int,
    beta: float,
    theta: float,
    sigma_eps: float,
    rng: np.random.Generator,
) -> tuple[pd.Series, pd.Series]:
    """Return (x, y) where y = beta*x + spread and spread is OU(theta)."""
    x = np.cumsum(rng.normal(0, 1, n))
    b = np.exp(-theta)
    spread = np.zeros(n)
    for t in range(1, n):
        spread[t] = b * spread[t - 1] + rng.normal(0, sigma_eps)
    y = beta * x + spread
    return pd.Series(x), pd.Series(y)


# ---------------------------------------------------------------------------
# Test 1: Independent random walks should not appear cointegrated
# ---------------------------------------------------------------------------

def test_independent_random_walks_no_cointegration():
    """Two independent random walks must not produce a low ADF p-value."""
    rng = np.random.default_rng(0)
    n = 600
    x = _random_walk(n, rng)
    y = _random_walk(n, rng)
    result = engle_granger(y, x)
    assert not np.isnan(result["adf_pvalue"]), "engle_granger returned NaN p-value"
    assert result["adf_pvalue"] > 0.10, (
        f"Expected p > 0.10 for independent random walks, got {result['adf_pvalue']:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 2: Known OU process — half-life within 20% of true value
# ---------------------------------------------------------------------------

def test_ou_half_life_estimation_accuracy():
    """estimate_ou should recover half-life within 20% for a known OU process."""
    rng = np.random.default_rng(1)
    theta = 0.05                          # per trading day
    expected_hl = np.log(2.0) / theta     # ≈ 13.86 days

    n = 1000
    b = np.exp(-theta)
    vals = np.zeros(n)
    for t in range(1, n):
        vals[t] = b * vals[t - 1] + rng.normal(0, 0.5)

    result = estimate_ou(pd.Series(vals))

    assert not np.isnan(result["half_life"]), "estimate_ou returned NaN half_life"
    assert not np.isinf(result["half_life"]), "estimate_ou returned inf half_life (b >= 1)"

    rel_error = abs(result["half_life"] - expected_hl) / expected_hl
    assert rel_error < 0.20, (
        f"half_life {result['half_life']:.2f}d not within 20% of true "
        f"{expected_hl:.2f}d (error: {rel_error:.1%})"
    )


# ---------------------------------------------------------------------------
# Test 3: BH correction detects cointegrated pairs, controls false positives
# ---------------------------------------------------------------------------

def test_benjamini_hochberg_detects_cointegrated_pairs():
    """BH at 10% FDR should flag cointegrated pairs and reject most nulls.

    Setup: 8 pairs total — 2 cointegrated (pairs 0 and 1), 6 independent
    random walks. BH should flag both cointegrated pairs and produce at most
    1 false positive out of 6 null pairs (consistent with 10% FDR).
    """
    rng = np.random.default_rng(2)
    n = 800
    theta = 0.08          # half-life ≈ 8.7 days — clear mean reversion
    sigma_eps = 0.5

    pvalues: list[float] = []
    labels: list[str] = []

    # Two cointegrated pairs — expected small p-values
    for _ in range(2):
        x, y = _cointegrated_pair(n, beta=1.5, theta=theta, sigma_eps=sigma_eps, rng=rng)
        res = engle_granger(y, x)
        pvalues.append(res["adf_pvalue"])
        labels.append("cointegrated")

    # Six independent random-walk pairs — expected large p-values
    for _ in range(6):
        x = _random_walk(n, rng)
        y = _random_walk(n, rng)
        res = engle_granger(y, x)
        pvalues.append(res["adf_pvalue"])
        labels.append("null")

    mask = benjamini_hochberg(np.array(pvalues), fdr=0.10)

    cointegrated_idx = [i for i, lbl in enumerate(labels) if lbl == "cointegrated"]
    null_idx = [i for i, lbl in enumerate(labels) if lbl == "null"]

    # Both cointegrated pairs must be detected
    for i in cointegrated_idx:
        assert mask[i], (
            f"BH failed to detect cointegrated pair {i} "
            f"(p={pvalues[i]:.4f})"
        )

    # False positives must not exceed 1 (10% FDR over 6 nulls → expect ≤ 0.6)
    false_positives = sum(mask[i] for i in null_idx)
    assert false_positives <= 1, (
        f"BH produced {false_positives} false positives out of "
        f"{len(null_idx)} null pairs"
    )
