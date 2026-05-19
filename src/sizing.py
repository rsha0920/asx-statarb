"""Portfolio-level position sizing for pairs trading.

Sizing is applied at pair-formation time (start of each walk-forward
trading window), not per-trade. Allocations are re-computed at each
window roll.

Typical usage:
    units_a, units_b = pair_dollar_legs(price_a, price_b, beta, 100_000)
    allocations = inverse_vol_weights(
        {pair_id: ou_sigma for ...}, total_capital=1_000_000
    )
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


def pair_dollar_legs(
    price_a: float,
    price_b: float,
    beta: float,
    dollar_per_pair: float,
) -> tuple[float, float]:
    """Compute absolute share counts for one spread unit, scaled to a dollar amount.

    Parameters
    ----------
    price_a, price_b : float
        Current prices of the two legs.
    beta : float
        Cointegrating coefficient from Engle-Granger (y = beta*x + alpha).
    dollar_per_pair : float
        Total notional to allocate to this pair (long leg + short leg combined).

    Returns
    -------
    (units_a, units_b) : tuple[float, float]
        Absolute share counts. Always positive — callers apply sign based on
        the position direction (-1 short spread, +1 long spread).

    Notes
    -----
    The cointegrating vector (1, -beta) sets the ratio: one spread unit is
    1 share of A against beta shares of B. We scale the whole unit so that

        price_a * units_a + price_b * units_b = dollar_per_pair

    which gives:

        units_a = dollar_per_pair / (price_a + beta * price_b)
        units_b = beta * units_a

    Dollar split: leg_a / total = price_a / (price_a + beta * price_b).
    For a well-cointegrated pair beta ≈ price_a / price_b, so the split is
    close to 50/50 but is determined by the data, not forced equal.

    This differs from a naive 50/50 equal-dollar split in that it respects
    the hedge ratio. Forcing 50/50 would create a residual beta mismatch
    that is not in the cointegrating vector and would generate spurious PnL
    from the non-spread component.
    """
    units_a = dollar_per_pair / (price_a + beta * price_b)
    units_b = beta * units_a
    return float(units_a), float(units_b)


def inverse_vol_weights(
    spread_vols: dict[str, float],
    total_capital: float,
    max_pairs: int = 10,
) -> dict[str, float]:
    """Inverse-volatility capital allocation across a portfolio of pairs.

    Parameters
    ----------
    spread_vols : dict[str, float]
        Mapping of pair identifier → spread volatility (use sigma from
        estimate_ou; this is the AR(1) residual std in log-price units).
    total_capital : float
        Total capital to allocate across all selected pairs.
    max_pairs : int
        Maximum number of pairs to hold simultaneously (default 10).

    Returns
    -------
    dict[str, float]
        Pair identifier → dollar allocation. Allocations sum to total_capital.
        Only the selected pairs appear; others are omitted.

    Notes
    -----
    **Selection**: the max_pairs lowest-volatility pairs are selected.
    Rationale: lower spread volatility means faster, cleaner mean reversion
    relative to the spread level and better signal-to-noise after transaction
    costs. High-vol pairs have wider oscillations that absorb a larger fraction
    of the fixed cost budget per trade.

    **Weighting**: allocations are proportional to 1/sigma. A pair with
    sigma=0.01 receives twice the notional of one with sigma=0.02. This
    equalises the expected spread-level dollar volatility across pairs,
    so no single pair dominates the portfolio's risk.

    **Timing**: this is portfolio-level sizing applied once per walk-forward
    window at pair-formation time, not dynamic per-trade sizing. It is
    re-computed at each window roll when new EG/OU fits become available.
    """
    if not spread_vols:
        return {}

    sorted_pairs = sorted(spread_vols.items(), key=lambda kv: kv[1])[:max_pairs]
    pairs = [k for k, _ in sorted_pairs]
    vols = np.array([v for _, v in sorted_pairs], dtype=float)

    if np.any(vols <= 0):
        raise ValueError(
            "All spread volatilities must be strictly positive; "
            f"got: {dict(zip(pairs, vols.tolist()))}"
        )

    inv_vols = 1.0 / vols
    weights = inv_vols / inv_vols.sum()

    return {pair: float(total_capital * w) for pair, w in zip(pairs, weights)}
