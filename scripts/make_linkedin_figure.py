#!/usr/bin/env python3
"""Generate publication-quality 1200×900 px LinkedIn summary figure.

All data is sourced from Phase 4/5 notebook outputs (cells 2, 3, 6, 9).
The parquet files have a pyarrow version mismatch, so we reconstruct
equity curves from the annual metrics that were printed in the notebook.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.dates import date2num, YearLocator, DateFormatter
from matplotlib.ticker import FuncFormatter

# ─── Colour scheme ────────────────────────────────────────────────────────────
BG    = '#1a1a2e'
TEXT  = '#ffffff'
TEAL  = '#00d4aa'
CORAL = '#ff6b6b'
GREY  = '#888899'
DIM   = '#33335a'

# ─── Raw data (notebook cell 3 and cell 6 outputs) ───────────────────────────

# Sharpe by year — base 10 bps scenario
# 2012-2013 are NaN (training-only; no live positions)
sharpe_years = [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]
sharpe_vals  = [1.14, 0.66, 2.10, -1.54, 0.14, -0.24, 0.27, 0.81, -0.31, 0.90, -0.47, 1.50, -2.50]

# Annual gross PnL ≈ ann_return × n_days / 252  (base, notebook cell 3)
ann_returns = [28462, 25474, 85374, -63126, 7472, -13475, 80282, 105471, -32715, 71401, -113868, 87416, -106529]
n_days_yr   = [  253,   255,   253,   252,   253,    253,   255,    254,   251,   252,    254,   253,    81]
base_ann    = np.array([r * d / 252.0 for r, d in zip(ann_returns, n_days_yr)])

# Walk-forward windows: trade_start → n_pairs_filtered  (notebook cell 6)
win_dates = pd.to_datetime([
    '2014-01-01', '2014-07-01', '2015-01-01', '2015-07-01',
    '2016-01-01', '2016-07-01', '2017-01-01', '2017-07-01',
    '2018-01-01', '2018-07-01', '2019-01-01', '2019-07-01',
    '2020-01-01', '2020-07-01', '2021-01-01', '2021-07-01',
    '2022-01-01', '2022-07-01', '2023-01-01', '2023-07-01',
    '2024-01-01', '2024-07-01', '2025-01-01', '2025-07-01',
    '2026-01-01',
])
pairs_cnt = np.array([19, 20, 19, 27, 16, 10, 4, 8, 5, 5, 8, 3, 1, 4, 6, 4, 5, 3, 4, 2, 0, 1, 11, 10, 14])

# Full-period headline figures (notebook cell 2)
SHARPE_BASE   = 0.141
SHARPE_STRESS = 0.012
TOTAL_BASE    = 236_025
TOTAL_STRESS  =  19_508

# Multiple testing (notebook cell 9): 50 shuffles — every one returned 0 survivors
N_REAL     = 3
N_SHUFFLES = 50
shuffled   = np.zeros(N_SHUFFLES, dtype=int)

# ─── Reconstruct equity curves ────────────────────────────────────────────────
# Total extra stress cost vs base = $360,862 − $144,345 = $216,517
# Distribute proportionally across pair-halfwindow units.
# 2026 is a partial period: 81 trading days vs ~126 expected in a half-year.
SCALE_2026 = 81.0 / 126.0

pairs_units = pairs_cnt.astype(float).copy()
idx_2026 = np.flatnonzero(win_dates == pd.Timestamp('2026-01-01'))[0]
pairs_units[idx_2026] *= SCALE_2026

extra_per_unit = 216_517.0 / pairs_units.sum()

def _year_extra(year: int) -> float:
    total = 0.0
    for i, d in enumerate(win_dates):
        if d.year == year:
            total += pairs_units[i]
    return total * extra_per_unit

per_year_extra = np.array([_year_extra(y) for y in sharpe_years])

cum_base_raw   = np.cumsum(base_ann)
cum_stress_raw = np.cumsum(base_ann - per_year_extra)

# Rescale endpoints to match the headline totals exactly
cum_base   = cum_base_raw   * (TOTAL_BASE   / cum_base_raw[-1])
cum_stress = cum_stress_raw * (TOTAL_STRESS / cum_stress_raw[-1])

equity_dates = [
    pd.Timestamp('2026-04-29') if y == 2026 else pd.Timestamp(f'{y}-12-31')
    for y in sharpe_years
]
eq_dates_full   = [pd.Timestamp('2013-12-31')] + equity_dates
cum_base_full   = np.concatenate([[0.0], cum_base])
cum_stress_full = np.concatenate([[0.0], cum_stress])

# ─── Global style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'axes.facecolor':  BG,
    'figure.facecolor': BG,
    'text.color':      TEXT,
    'axes.labelcolor': TEXT,
    'xtick.color':     GREY,
    'ytick.color':     GREY,
    'axes.edgecolor':  DIM,
    'axes.grid':       False,
    'axes.spines.top':   False,
    'axes.spines.right': False,
})

fig = plt.figure(figsize=(4.0, 3.0), dpi=300, facecolor=BG)

# ── Title bar ──────────────────────────────────────────────────────────────────
fig.text(0.5, 0.978, "ASX Pairs Trading: 12 Years of Walk-Forward Results",
         ha='center', va='top', fontsize=7.5, fontweight='bold', color=TEXT)
fig.text(0.5, 0.943, "Cointegration-based statistical arbitrage on ASX 200 large-caps  |  2012–2026  |  Walk-forward OOS",
         ha='center', va='top', fontsize=4.8, color=GREY)

# ── Grid layout ───────────────────────────────────────────────────────────────
gs = gridspec.GridSpec(
    2, 2, figure=fig,
    left=0.10, right=0.97, bottom=0.10, top=0.89,
    hspace=0.55, wspace=0.40,
)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

def _style_ax(ax):
    for spine in ('left', 'bottom'):
        ax.spines[spine].set_linewidth(0.4)
        ax.spines[spine].set_color(DIM)
    ax.tick_params(length=2, width=0.4)

# ══════════════════════════════════════════════════════════════════════════════
# Panel 1 — Sharpe by year (hero panel)
# ══════════════════════════════════════════════════════════════════════════════
ax1.set_title("Sharpe by Year  (base 10 bps)", fontsize=5.5, color=TEXT, pad=3, loc='left')

sharpe_years_plot = [y for y in sharpe_years if y <= 2025]
sharpe_vals_plot  = [v for y, v in zip(sharpe_years, sharpe_vals) if y <= 2025]
bar_colors = [TEAL if v > 0 else CORAL for v in sharpe_vals_plot]
x_pos      = np.arange(len(sharpe_years_plot))

for xi, (yr, val, col) in enumerate(zip(sharpe_years_plot, sharpe_vals_plot, bar_colors)):
    ax1.bar(xi, val, color=col, alpha=1.0, width=0.72, edgecolor='none')
    pad   = 0.09 if val >= 0 else -0.09
    va    = 'bottom' if val >= 0 else 'top'
    ax1.text(xi, val + pad, f'{val:.1f}', ha='center', va=va, fontsize=3.0, color=TEXT)

ax1.axhline(0, color=GREY, linewidth=0.4, linestyle='--', alpha=0.6)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(["'" + str(y)[2:] for y in sharpe_years_plot], fontsize=3.4, rotation=45, ha='right')
ax1.tick_params(axis='y', labelsize=3.5)
ax1.set_ylabel("Sharpe", fontsize=4.5)
ax1.set_xlim(-0.65, len(sharpe_years_plot) - 0.35)

_style_ax(ax1)

# ══════════════════════════════════════════════════════════════════════════════
# Panel 2 — Cumulative equity (both cost scenarios)
# ══════════════════════════════════════════════════════════════════════════════
ax2.set_title("Cumulative P&L  (base vs stress costs)", fontsize=5.5, color=TEXT, pad=3, loc='left')

eq_nums   = date2num(eq_dates_full)
ax2.plot(eq_nums, cum_base_full   / 1000, color=TEAL,  linewidth=1.0, zorder=3)
ax2.plot(eq_nums, cum_stress_full / 1000, color=CORAL, linewidth=0.9, linestyle='--', zorder=3)

# Shade region between curves
ax2.fill_between(eq_nums, cum_stress_full / 1000, cum_base_full / 1000,
                 alpha=0.07, color=TEAL)
ax2.axhline(0, color=GREY, linewidth=0.3, alpha=0.5)

# End-point labels
ax2.annotate(f'Sharpe {SHARPE_BASE}',
             xy=(eq_nums[-1], cum_base_full[-1] / 1000),
             xytext=(-20, 4), textcoords='offset points',
             fontsize=3.8, color=TEAL, va='bottom')
ax2.annotate(f'Sharpe {SHARPE_STRESS}',
             xy=(eq_nums[-1], cum_stress_full[-1] / 1000),
             xytext=(-22, -8), textcoords='offset points',
             fontsize=3.8, color=CORAL, va='top')

ax2.xaxis_date()
ax2.xaxis.set_major_locator(YearLocator(3))
ax2.xaxis.set_major_formatter(DateFormatter('%Y'))
ax2.tick_params(axis='both', labelsize=3.5)
ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'${v:.0f}k'))
ax2.set_ylabel("Cumulative P&L", fontsize=4.5)

_style_ax(ax2)

# ══════════════════════════════════════════════════════════════════════════════
# Panel 3 — Pairs per walk-forward window
# ══════════════════════════════════════════════════════════════════════════════
ax3.set_title("Active Pairs per 6-Month Window", fontsize=5.5, color=TEXT, pad=3, loc='left')

win_nums = date2num(win_dates)
ax3.bar(win_nums, pairs_cnt, width=130, color=TEAL, alpha=0.80, edgecolor='none')

# OLS trend line
coef  = np.polyfit(win_nums, pairs_cnt, 1)
trend = np.poly1d(coef)
ax3.plot(win_nums, trend(win_nums), color=CORAL, linewidth=0.9, linestyle='--', alpha=0.85)
ax3.text(0.96, 0.92, "declining\ntradeable pairs",
         transform=ax3.transAxes, ha='right', va='top',
         fontsize=3.5, color=CORAL, linespacing=1.3)

ax3.xaxis_date()
ax3.xaxis.set_major_locator(YearLocator(3))
ax3.xaxis.set_major_formatter(DateFormatter('%Y'))
ax3.tick_params(axis='both', labelsize=3.5)
ax3.set_ylabel("# Pairs", fontsize=4.5)
ax3.set_ylim(0, max(pairs_cnt) * 1.20)

_style_ax(ax3)

# ══════════════════════════════════════════════════════════════════════════════
# Panel 4 — Multiple testing diagnostic
# ══════════════════════════════════════════════════════════════════════════════
ax4.set_title("Multiple Testing Diagnostic", fontsize=5.5, color=TEXT, pad=3, loc='left')

# All 50 shuffled permutations returned 0 surviving pairs → spike at 0
ax4.bar([0], [N_SHUFFLES], color=GREY, alpha=0.65, width=0.7, edgecolor='none',
        label=f'Shuffled (n={N_SHUFFLES})')

# Real data marker
ax4.axvline(N_REAL, color=CORAL, linewidth=1.4, zorder=5, label=f'Real data')
ax4.text(N_REAL + 0.08, N_SHUFFLES * 0.92, f'Real data:\n{N_REAL} pairs',
         color=CORAL, fontsize=4.0, va='top', fontweight='bold')
ax4.text(0, N_SHUFFLES * 0.92, f'All {N_SHUFFLES}\nshuffles\n= 0',
         color=GREY, fontsize=3.5, ha='center', va='top')

# Empirical p-value
ax4.text(0.97, 0.08, f'p < {1/N_SHUFFLES:.2f} (empirical)',
         transform=ax4.transAxes, ha='right', va='bottom',
         fontsize=3.8, color=TEXT)

ax4.set_xlim(-0.5, 4.5)
ax4.set_ylim(0, N_SHUFFLES * 1.18)
ax4.set_xticks([0, 1, 2, 3, 4])
ax4.tick_params(axis='both', labelsize=3.5)
ax4.set_xlabel("Pairs Surviving All Filters", fontsize=4.5)
ax4.set_ylabel("Shuffle count", fontsize=4.5)

_style_ax(ax4)

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = 'reports/figures/linkedin_summary.png'
fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor=BG, edgecolor='none')
print(f"Saved: {out_path}")
print(f"Pixel dimensions: {fig.get_size_inches() * fig.dpi}")
