"""Transaction cost model for pairs trading.

CLAUDE.md specifies: 10 bps base, 25 bps stress — per side per leg.

"Per side per leg" means each trade on each stock incurs bps of its notional.
For a spread with two legs (A and B), entering costs bps on each leg; exiting
costs bps on each leg again. Total round-trip cost: 2 * bps * total_notional.
"""

from __future__ import annotations

BASE_BPS: float = 10.0
STRESS_BPS: float = 25.0


def trade_cost(dollar_notional: float, bps: float) -> float:
    """One-way transaction cost to enter OR exit both legs of a spread.

    Parameters
    ----------
    dollar_notional : float
        Combined dollar value of both legs: price_a * units_a + price_b * units_b.
    bps : float
        Cost in basis points per side per leg.

    Returns
    -------
    float
        Dollar cost for one entry or one exit.

    Notes
    -----
    Round-trip cost (entry + exit) = 2 * trade_cost(notional, bps).
    At 10 bps on a $100k pair: one-way = $100, round trip = $200.
    """
    return dollar_notional * bps / 10_000
