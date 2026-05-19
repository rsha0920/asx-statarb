"""Walk-forward backtest engine for cointegration-based pairs trading.

Implements the locked methodology from CLAUDE.md:
    - 24-month training / 6-month trading windows, rolled forward 6 months
    - Re-screens pairs from scratch each window (no look-ahead)
    - Inverse-volatility sizing, capped at max_pairs
    - Two cost scenarios: base (10 bps) and stress (25 bps) per side per leg
    - All performance numbers are out-of-sample by construction

Typical usage:
    results = run_backtest(prices, ASX_UNIVERSE)
    metrics = compute_metrics(results['daily_pnl_base'])
"""

from __future__ import annotations

import logging
from typing import Generator

import numpy as np
import pandas as pd

from .cointegration import screen_pairs
from .costs import BASE_BPS, STRESS_BPS, trade_cost
from .filters import apply_all_filters
from .signals import compute_zscore, generate_signals
from .sizing import inverse_vol_weights, pair_dollar_legs

log = logging.getLogger(__name__)

_WARMUP_BUFFER = 5  # extra trading days beyond zscore_window for z-score warmup


# ---------------------------------------------------------------------------
# Walk-forward window generation
# ---------------------------------------------------------------------------

def _walk_forward_windows(
    start: pd.Timestamp,
    end: pd.Timestamp,
    train_months: int = 24,
    trade_months: int = 6,
) -> Generator[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp], None, None]:
    """Yield non-overlapping (train_start, train_end, trade_start, trade_end) windows.

    Each trading window advances by trade_months. The training window is always
    the train_months immediately preceding the trading window.
    """
    trade_start = start + pd.DateOffset(months=train_months)
    while trade_start <= end:
        trade_end = min(
            trade_start + pd.DateOffset(months=trade_months) - pd.Timedelta(days=1),
            end,
        )
        train_end = trade_start - pd.Timedelta(days=1)
        train_start = trade_start - pd.DateOffset(months=train_months)
        yield pd.Timestamp(train_start), pd.Timestamp(train_end), pd.Timestamp(trade_start), pd.Timestamp(trade_end)
        trade_start = trade_start + pd.DateOffset(months=trade_months)


# ---------------------------------------------------------------------------
# Per-pair PnL computation
# ---------------------------------------------------------------------------

def _pair_pnl(
    prices: pd.DataFrame,
    log_prices: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    beta: float,
    alpha: float,
    half_life: float,
    units_a: float,
    units_b: float,
    warmup_start: pd.Timestamp,
    trade_start: pd.Timestamp,
    trade_end: pd.Timestamp,
    zscore_window: int,
    entry: float,
    exit_z: float,
    hard_stop: float,
    trade_dates: pd.DatetimeIndex,
) -> tuple[pd.Series, pd.Series, list[dict]]:
    """Compute gross daily PnL (in dollars) for one pair in one window.

    Returns (pnl_gross, notional_per_bar, trades) where:
        pnl_gross   : daily dollar PnL excluding costs
        notional    : combined dollar notional (units_a*p_a + units_b*p_b) per bar
        trades      : list of trade dicts (entry/exit info without cost split)
    """
    # Build spread with warmup
    log_a = log_prices.loc[str(warmup_start.date()):trade_end, ticker_a].dropna()
    log_b = log_prices.loc[str(warmup_start.date()):trade_end, ticker_b].dropna()
    shared = log_a.index.intersection(log_b.index)

    if len(shared) < zscore_window + 2:
        zero = pd.Series(0.0, index=trade_dates)
        return zero, zero, []

    spread_full = log_a.loc[shared] - beta * log_b.loc[shared] - alpha
    zscore_full = compute_zscore(spread_full, window=zscore_window)
    zscore_trade = zscore_full.loc[trade_start:trade_end]

    if zscore_trade.notna().sum() < 2:
        zero = pd.Series(0.0, index=trade_dates)
        return zero, zero, []

    signals = generate_signals(
        zscore_trade,
        half_life=half_life,
        entry=entry,
        exit=exit_z,
        hard_stop=hard_stop,
    )

    # Actual prices for dollar PnL
    price_a = prices.loc[trade_start:trade_end, ticker_a].reindex(trade_dates)
    price_b = prices.loc[trade_start:trade_end, ticker_b].reindex(trade_dates)

    pos = signals["position"].reindex(trade_dates, fill_value=0)
    delta_a = price_a.diff().fillna(0.0)
    delta_b = price_b.diff().fillna(0.0)

    # Long spread: long A, short B
    pnl_gross = pos.shift(1).fillna(0.0) * (units_a * delta_a - units_b * delta_b)

    # Notional per bar (for cost calculation)
    notional = (units_a * price_a.ffill() + units_b * price_b.ffill()).fillna(0.0)

    # Trade log (cost-free; costs applied by caller per scenario)
    entry_bars = signals["entry_signal"].reindex(trade_dates, fill_value=False)
    exit_bars  = signals["exit_signal"].reindex(trade_dates, fill_value=False)

    # Force-close any open position at end of window
    if pos.iloc[-1] != 0:
        exit_bars.iloc[-1] = True

    exit_dates_set = set(exit_bars[exit_bars].index)
    trades: list[dict] = []
    for entry_date in entry_bars[entry_bars].index:
        entry_pos = signals.loc[entry_date, "position"] if entry_date in signals.index else 0
        exits_after = sorted(d for d in exit_dates_set if d > entry_date)
        if not exits_after:
            continue
        exit_date   = exits_after[0]
        exit_reason = (
            signals.loc[exit_date, "exit_reason"]
            if exit_date in signals.index
            else "window_end"
        )
        holding = len(trade_dates[(trade_dates >= entry_date) & (trade_dates <= exit_date)])
        trades.append(
            dict(
                ticker_a=ticker_a,
                ticker_b=ticker_b,
                trade_start=trade_start,
                entry_date=entry_date,
                exit_date=exit_date,
                direction="long" if entry_pos == 1 else "short",
                exit_reason=exit_reason,
                holding_bars=holding,
                entry_notional=float(notional.loc[entry_date]) if entry_date in notional.index else np.nan,
            )
        )

    return pnl_gross.fillna(0.0), notional, trades


# ---------------------------------------------------------------------------
# Main backtest
# ---------------------------------------------------------------------------

def run_backtest(
    prices: pd.DataFrame,
    sectors: dict[str, list[str]],
    start: str = "2012-01-01",
    end: str | None = None,
    train_months: int = 24,
    trade_months: int = 6,
    total_capital: float = 1_000_000.0,
    max_pairs: int = 10,
    zscore_window: int = 60,
    entry: float = 2.0,
    exit_z: float = 0.5,
    hard_stop: float = 4.0,
    fdr: float = 0.10,
    min_history_days: int = 504,
) -> dict:
    """Run walk-forward backtest with base and stress cost scenarios.

    Parameters
    ----------
    prices : pd.DataFrame
        Wide panel of adj_close prices (rows=date, columns=ticker).
    sectors : dict[str, list[str]]
        Universe groupings (e.g. ASX_UNIVERSE from src.data).
    start : str
        First date of the first trading window (must be at least
        train_months after the start of prices).
    end : str or None
        Last date to include. Defaults to the last date in prices.
    train_months, trade_months : int
        Walk-forward window lengths. Default 24/6 months.
    total_capital : float
        Total portfolio capital (AUD). Divided across max_pairs pairs.
    max_pairs : int
        Maximum concurrent pairs (inverse-vol allocation cap).
    zscore_window : int
        Rolling z-score window in trading days (default 60).
    entry, exit_z, hard_stop : float
        Signal thresholds (default: 2.0 / 0.5 / 4.0).
    fdr : float
        Benjamini-Hochberg FDR target for pair screening.
    min_history_days : int
        Minimum shared history required per pair (default 504).

    Returns
    -------
    dict with keys:
        daily_pnl_base   : pd.Series  daily dollar PnL at 10 bps
        daily_pnl_stress : pd.Series  daily dollar PnL at 25 bps
        equity_base      : pd.Series  cumulative dollar PnL at 10 bps
        equity_stress    : pd.Series  cumulative dollar PnL at 25 bps
        trade_log        : pd.DataFrame
        window_log       : pd.DataFrame
    """
    end_ts = pd.Timestamp(end) if end else prices.index.max()
    start_ts = pd.Timestamp(start)

    log_prices = np.log(prices.replace(0.0, np.nan))

    all_dates = prices.loc[start_ts:end_ts].index
    daily_pnl_base   = pd.Series(0.0, index=all_dates)
    daily_pnl_stress = pd.Series(0.0, index=all_dates)
    trade_log: list[dict]  = []
    window_log: list[dict] = []

    windows = list(_walk_forward_windows(start_ts, end_ts, train_months, trade_months))
    log.info("Running backtest: %d windows from %s to %s", len(windows), start_ts.date(), end_ts.date())

    for win_idx, (train_start, train_end, trade_start, trade_end) in enumerate(windows):
        log.info(
            "Window %d/%d  train %s–%s  trade %s–%s",
            win_idx + 1, len(windows),
            train_start.date(), train_end.date(),
            trade_start.date(), trade_end.date(),
        )

        # ── Screen and filter pairs ───────────────────────────────────────
        try:
            all_pairs = screen_pairs(
                prices, sectors,
                window_start=train_start, window_end=train_end,
                fdr=fdr, min_history_days=min_history_days,
            )
        except Exception as exc:
            log.warning("screen_pairs failed for window %d: %s", win_idx + 1, exc)
            continue

        pairs = apply_all_filters(all_pairs)

        if len(pairs) == 0:
            log.info("  No pairs survived filters.")
            window_log.append(dict(
                train_start=train_start, trade_start=trade_start,
                n_pairs_screened=len(all_pairs), n_pairs_filtered=0,
                window_pnl_base=0.0, window_pnl_stress=0.0,
            ))
            continue

        # ── Size pairs ────────────────────────────────────────────────────
        sigma_map = {
            f"{r.ticker_a}/{r.ticker_b}": r.ou_sigma
            for _, r in pairs.iterrows()
            if not np.isnan(r.ou_sigma) and r.ou_sigma > 0
        }
        if not sigma_map:
            continue

        allocations = inverse_vol_weights(sigma_map, total_capital, max_pairs=max_pairs)

        # ── Warmup start for z-score ──────────────────────────────────────
        pre_trade = prices.loc[:trade_start - pd.Timedelta(days=1)].index
        warmup_start = (
            pre_trade[-(zscore_window + _WARMUP_BUFFER)]
            if len(pre_trade) > zscore_window + _WARMUP_BUFFER
            else pre_trade[0]
        )

        trade_dates = prices.loc[trade_start:trade_end].index
        window_pnl_base   = pd.Series(0.0, index=trade_dates)
        window_pnl_stress = pd.Series(0.0, index=trade_dates)

        # ── Per-pair PnL ──────────────────────────────────────────────────
        for _, row in pairs.iterrows():
            pair_id = f"{row.ticker_a}/{row.ticker_b}"
            if pair_id not in allocations:
                continue

            dollar_alloc = allocations[pair_id]
            p_a = prices.loc[trade_start:trade_end, row.ticker_a].dropna()
            p_b = prices.loc[trade_start:trade_end, row.ticker_b].dropna()

            if p_a.empty or p_b.empty:
                continue

            units_a, units_b = pair_dollar_legs(
                float(p_a.iloc[0]),
                float(p_b.iloc[0]),
                float(row.beta),
                dollar_alloc,
            )

            pnl_gross, notional, pair_trades = _pair_pnl(
                prices=prices,
                log_prices=log_prices,
                ticker_a=row.ticker_a,
                ticker_b=row.ticker_b,
                beta=float(row.beta),
                alpha=float(row.alpha),
                half_life=float(row.half_life_days),
                units_a=units_a,
                units_b=units_b,
                warmup_start=warmup_start,
                trade_start=trade_start,
                trade_end=trade_end,
                zscore_window=zscore_window,
                entry=entry,
                exit_z=exit_z,
                hard_stop=hard_stop,
                trade_dates=trade_dates,
            )

            # Apply costs: entry cost + exit cost per trade bar
            signals_dates = pnl_gross.index
            for cost_bps, pnl_acc in [(BASE_BPS, window_pnl_base), (STRESS_BPS, window_pnl_stress)]:
                costs = pd.Series(0.0, index=signals_dates)
                for t in pair_trades:
                    ed, xd = t["entry_date"], t["exit_date"]
                    n_entry = float(notional.loc[ed]) if ed in notional.index else 0.0
                    n_exit  = float(notional.loc[xd]) if xd in notional.index else 0.0
                    if ed in costs.index:
                        costs.loc[ed] += trade_cost(n_entry, cost_bps)
                    if xd in costs.index:
                        costs.loc[xd] += trade_cost(n_exit, cost_bps)
                pnl_acc += (pnl_gross - costs)

            # Annotate trade log with pair metadata
            for t in pair_trades:
                trade_log.append({**t, "sector": row.sector, "pair_id": pair_id,
                                   "beta": row.beta, "half_life": row.half_life_days,
                                   "dollar_alloc": dollar_alloc})

        # ── Accumulate into portfolio PnL ─────────────────────────────────
        daily_pnl_base.loc[trade_dates]   += window_pnl_base
        daily_pnl_stress.loc[trade_dates] += window_pnl_stress

        window_log.append(dict(
            train_start=train_start,
            trade_start=trade_start,
            trade_end=trade_end,
            n_pairs_screened=len(all_pairs),
            n_pairs_filtered=len(pairs),
            window_pnl_base=float(window_pnl_base.sum()),
            window_pnl_stress=float(window_pnl_stress.sum()),
        ))

    equity_base   = daily_pnl_base.cumsum()
    equity_stress = daily_pnl_stress.cumsum()

    return dict(
        daily_pnl_base=daily_pnl_base,
        daily_pnl_stress=daily_pnl_stress,
        equity_base=equity_base,
        equity_stress=equity_stress,
        trade_log=pd.DataFrame(trade_log) if trade_log else pd.DataFrame(),
        window_log=pd.DataFrame(window_log),
    )


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def compute_metrics(daily_pnl: pd.Series, annual_factor: float = 252) -> dict:
    """Compute standard performance metrics from a daily dollar PnL series.

    Parameters
    ----------
    daily_pnl : pd.Series
        Daily dollar profit/loss (not return — absolute dollars).
    annual_factor : float
        Trading days per year (default 252).

    Returns
    -------
    dict with keys:
        total_pnl        : float  total dollar PnL
        ann_return       : float  annualised mean daily PnL * 252
        ann_vol          : float  annualised std of daily PnL
        sharpe           : float  ann_return / ann_vol (risk-free = 0)
        sortino          : float  ann_return / ann_downside_vol
        max_drawdown     : float  peak-to-trough dollar drawdown (negative)
        max_drawdown_pct : float  max drawdown as fraction of peak equity
        avg_daily_pnl    : float
        trading_days     : int    number of days with non-zero PnL
    """
    pnl = daily_pnl.dropna()
    total_pnl = float(pnl.sum())

    if len(pnl) == 0:
        return dict(
            total_pnl=total_pnl, ann_return=0.0, ann_vol=0.0,
            sharpe=np.nan, sortino=np.nan, max_drawdown=0.0,
            max_drawdown_pct=np.nan, avg_daily_pnl=0.0, trading_days=0,
        )

    ann_return = float(pnl.mean() * annual_factor)
    ann_vol    = float(pnl.std(ddof=1) * np.sqrt(annual_factor))
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan

    downside = pnl[pnl < 0]
    ann_downside_vol = float(downside.std(ddof=1) * np.sqrt(annual_factor)) if len(downside) > 1 else np.nan
    sortino = ann_return / ann_downside_vol if ann_downside_vol and ann_downside_vol > 0 else np.nan

    equity = pnl.cumsum()
    running_max = equity.cummax()
    drawdown = equity - running_max
    max_drawdown = float(drawdown.min())
    peak_equity = float(running_max.max())
    max_drawdown_pct = max_drawdown / peak_equity if peak_equity > 0 else np.nan

    trading_days = int((pnl != 0).sum())

    return dict(
        total_pnl=total_pnl,
        ann_return=ann_return,
        ann_vol=ann_vol,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=max_drawdown,
        max_drawdown_pct=max_drawdown_pct,
        avg_daily_pnl=float(pnl[pnl != 0].mean()) if trading_days > 0 else 0.0,
        trading_days=trading_days,
    )
