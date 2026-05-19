"""Script to create notebooks/05_analysis.ipynb programmatically."""
import nbformat
from pathlib import Path

nb = nbformat.v4.new_notebook()

cells = []

# Cell 1: Markdown title
cells.append(nbformat.v4.new_markdown_cell(
    "# Phase 5: Alpha Decay Analysis and Multiple Testing Diagnostic\n\n"
    "Alpha decay study, capacity analysis, and placebo test for the ASX pairs trading strategy.\n"
    "Results feed directly into reports/writeup.md."
))

# Cell 2: Imports
cells.append(nbformat.v4.new_code_cell(
    """import logging
import os
import sys
import warnings
from pathlib import Path

# Ensure project root is on the path regardless of where the kernel was launched
_project_root = str(Path("..").resolve())
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Change working directory to project root so relative paths work
os.chdir(_project_root)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis import (
    capacity_analysis,
    multiple_testing_diagnostic,
    sharpe_by_year,
    sharpe_decay_regression,
)
from src.data import ASX_UNIVERSE, fetch_history, get_close_panel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
warnings.filterwarnings("ignore", category=FutureWarning)

FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "sans-serif",
    "font.size": 10,
})"""
))

# Cell 3: Load backtest results
cells.append(nbformat.v4.new_code_cell(
    """daily_pnl  = pd.read_parquet("data/results/backtest_daily_pnl.parquet")
trade_log  = pd.read_parquet("data/results/backtest_trade_log.parquet")
window_log = pd.read_parquet("data/results/backtest_window_log.parquet")

print(f"Daily PnL: {len(daily_pnl)} rows, {daily_pnl.index[0].date()} to {daily_pnl.index[-1].date()}")
print(f"Trades: {len(trade_log)}")
print(f"Windows: {len(window_log)}")
print(f"\\nFull-period Sharpe (base):   {daily_pnl['base'].mean()*252 / (daily_pnl['base'].std()*np.sqrt(252)):.3f}")
print(f"Full-period Sharpe (stress): {daily_pnl['stress'].mean()*252 / (daily_pnl['stress'].std()*np.sqrt(252)):.3f}")
print(f"Total PnL base:   ${daily_pnl['base'].sum():>10,.0f}")
print(f"Total PnL stress: ${daily_pnl['stress'].sum():>10,.0f}")"""
))

# Cell 4: Sharpe by year and decay regression
cells.append(nbformat.v4.new_code_cell(
    """sby_base   = sharpe_by_year(daily_pnl["base"])
sby_stress = sharpe_by_year(daily_pnl["stress"])

decay = sharpe_decay_regression(sby_base)

print("=== Sharpe by Year (Base) ===")
print(sby_base[["sharpe", "ann_return", "ann_vol", "n_days"]].to_string())
print()
print("=== Alpha Decay OLS Regression ===")
print(f"  slope     = {decay['slope']:.4f} Sharpe units/year")
print(f"  t-stat    = {decay['t_stat']:.2f}")
print(f"  p-value   = {decay['p_value']:.3f}")
print(f"  R²        = {decay['r_squared']:.3f}")
print(f"  n years   = {decay['n']}")
print()
if decay['p_value'] < 0.1:
    print("Decay is statistically significant at 10%.")
else:
    print(f"Decay is NOT statistically significant (p={decay['p_value']:.3f}). "
          "Directional but noisy.")"""
))

# Cell 5: Save sharpe_by_year.png
cells.append(nbformat.v4.new_code_cell(
    """full_sharpe_base = daily_pnl['base'].mean()*252 / (daily_pnl['base'].std()*np.sqrt(252))

fig, ax = plt.subplots(figsize=(10, 4))

valid_years = sby_base.index[sby_base["sharpe"].notna()]
valid_sharpe = sby_base.loc[valid_years, "sharpe"]
colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in valid_sharpe]

ax.bar(valid_years, valid_sharpe, color=colors, width=0.7, edgecolor="none")
ax.axhline(0, color="black", linewidth=0.8, linestyle="-")
ax.axhline(full_sharpe_base, color="grey", linewidth=1, linestyle="--",
           label=f"Full-period Sharpe ({full_sharpe_base:.2f})")

x_fit = np.array(valid_years, dtype=float)
y_fit = decay["intercept"] + decay["slope"] * x_fit
ax.plot(valid_years, y_fit, color="navy", linewidth=1.5, linestyle="-",
        label=f"OLS trend (slope={decay['slope']:.2f}/yr, p={decay['p_value']:.2f})")

ax.set_xlabel("Year")
ax.set_ylabel("Annual Sharpe Ratio")
ax.set_title("Annual Sharpe by Year — ASX Pairs Trading (Base Costs, 10bps)")
ax.legend(fontsize=8)
ax.set_xticks(valid_years)
ax.tick_params(axis="x", rotation=45)

fig.tight_layout()
fig.savefig(FIG_DIR / "sharpe_by_year.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved: {FIG_DIR / 'sharpe_by_year.png'}")"""
))

# Cell 6: Save drawdown.png
cells.append(nbformat.v4.new_code_cell(
    """equity_base   = daily_pnl["base"].cumsum()
equity_stress = daily_pnl["stress"].cumsum()

dd_base   = equity_base   - equity_base.cummax()
dd_stress = equity_stress - equity_stress.cummax()

fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

ax = axes[0]
ax.plot(equity_base.index, equity_base / 1000, color="#2980b9", linewidth=1.2, label="Base (10bps)")
ax.plot(equity_stress.index, equity_stress / 1000, color="#e67e22", linewidth=1.2,
        linestyle="--", label="Stress (25bps)")
ax.axhline(0, color="black", linewidth=0.6)
ax.set_ylabel("Cumulative PnL ($k)")
ax.set_title("Equity Curve and Drawdown — ASX Pairs Trading Walk-Forward Backtest")
ax.legend(fontsize=9)

ax = axes[1]
ax.fill_between(dd_base.index, dd_base / 1000, 0, color="#2980b9", alpha=0.4, label="Base drawdown")
ax.fill_between(dd_stress.index, dd_stress / 1000, 0, color="#e67e22", alpha=0.3,
                label="Stress drawdown")
ax.axhline(0, color="black", linewidth=0.6)
ax.set_ylabel("Drawdown ($k)")
ax.set_xlabel("Date")
ax.legend(fontsize=9)

fig.tight_layout()
fig.savefig(FIG_DIR / "drawdown.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved: {FIG_DIR / 'drawdown.png'}")"""
))

# Cell 7: Cost sensitivity
cells.append(nbformat.v4.new_code_cell(
    """base_net_total   = float(daily_pnl["base"].sum())
stress_net_total = float(daily_pnl["stress"].sum())
gross_total      = (2.5 * base_net_total - stress_net_total) / 1.5
base_cost_total  = gross_total - base_net_total
stress_cost_total= gross_total - stress_net_total

print("=== Cost Sensitivity ===")
print(f"  Estimated gross PnL: ${gross_total:>10,.0f}")
print(f"  Net PnL (base  10bps): ${base_net_total:>10,.0f}  (costs: ${base_cost_total:,.0f}, {base_cost_total/gross_total*100:.0f}% of gross)")
print(f"  Net PnL (stress 25bps): ${stress_net_total:>9,.0f}  (costs: ${stress_cost_total:,.0f}, {stress_cost_total/gross_total*100:.0f}% of gross)")
print()
print("Exit reason breakdown:")
print(trade_log["exit_reason"].value_counts().to_string())
print()
print("Pairs per window:")
print(window_log[["trade_start", "n_pairs_filtered"]].to_string(index=False))"""
))

# Cell 8: Capacity analysis
cells.append(nbformat.v4.new_code_cell(
    """prices_long = fetch_history(
    list({t for s in ASX_UNIVERSE.values() for t in s}),
    cache_dir="data/cache",
)

pair_counts = (
    trade_log.groupby(["ticker_a", "ticker_b", "sector"]).size()
    .reset_index(name="n_trades")
    .sort_values("n_trades", ascending=False)
    .head(15)
)
pair_counts["beta"] = trade_log.groupby(["ticker_a", "ticker_b"])["beta"].mean().reindex(
    pd.MultiIndex.from_frame(pair_counts[["ticker_a", "ticker_b"]])
).values
pair_counts["alpha"] = 0.0
pair_counts["half_life_days"] = 20.0
pair_counts["ou_sigma"] = 0.01

cap = capacity_analysis(pair_counts, prices_long)

print("=== Capacity Analysis (top pairs by trade count) ===")
cap_display = cap.copy()
cap_display["adv_a_m"]   = (cap_display["adv_a"] / 1e6).round(1)
cap_display["adv_b_m"]   = (cap_display["adv_b"] / 1e6).round(1)
cap_display["capacity_k"] = (cap_display["pair_capacity"] / 1e3).round(0)
print(cap_display[["ticker_a", "ticker_b", "adv_a_m", "adv_b_m", "capacity_k"]].to_string(index=False))
print()
print(f"Median pair capacity at 1% ADV: ${cap['pair_capacity'].median():,.0f}")
print(f"Tightest constraint:            ${cap['pair_capacity'].min():,.0f}")"""
))

# Cell 9: Save capacity_analysis.png
cells.append(nbformat.v4.new_code_cell(
    """cap_sorted = cap.sort_values("pair_capacity")
labels = [f"{r.ticker_a.replace('.AX','')}/{r.ticker_b.replace('.AX','')}" for _, r in cap_sorted.iterrows()]

fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(labels, cap_sorted["pair_capacity"] / 1000, color="#3498db", edgecolor="none")
ax.axvline(100, color="#e74c3c", linewidth=1, linestyle="--", label="$100k position reference")
ax.set_xlabel("Maximum Position Size at 1% ADV ($k)")
ax.set_title("Capacity Analysis: Binding Constraint per Pair (1% of Smaller Leg's ADV)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG_DIR / "capacity_analysis.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved: {FIG_DIR / 'capacity_analysis.png'}")"""
))

# Cell 10: Multiple testing diagnostic
cells.append(nbformat.v4.new_code_cell(
    """prices_wide = get_close_panel(prices_long)

diag = multiple_testing_diagnostic(
    prices_wide,
    ASX_UNIVERSE,
    n_shuffles=50,
    seed=42,
    window_start="2018-01-01",
    window_end="2020-12-31",
)

print("=== Multiple Testing Diagnostic ===")
print(f"  Real data (2018-2020): {diag['n_real']} pairs survived all filters")
print(f"  Shuffled data (n=50):")
print(f"    Mean pairs surviving:     {diag['n_shuffled_mean']:.1f}")
print(f"    Median pairs surviving:   {diag['n_shuffled_median']:.1f}")
print(f"    95th percentile:          {diag['n_shuffled_p95']:.1f}")
print()
ratio = diag['n_real'] / max(diag['n_shuffled_mean'], 0.01)
print(f"  Signal-to-noise ratio: {ratio:.1f}x (real / shuffled mean)")
if ratio < 2:
    print("  WARNING: Real data finds fewer than 2x as many pairs as shuffled data.")
elif ratio < 5:
    print("  Moderate signal: real data finds meaningfully more pairs than random data.")
else:
    print("  Strong signal: real data produces substantially more pairs than shuffled data.")"""
))

# Cell 11: Save multiple_testing_diagnostic.png
cells.append(nbformat.v4.new_code_cell(
    """counts = diag["n_shuffled_counts"]

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(counts, bins=max(10, len(set(counts))), color="#95a5a6", edgecolor="white")
ax.axvline(diag["n_real"], color="#e74c3c", linewidth=2, label=f"Real data ({diag['n_real']} pairs)")
ax.axvline(diag["n_shuffled_mean"], color="#2c3e50", linewidth=1.5, linestyle="--",
           label=f"Shuffled mean ({diag['n_shuffled_mean']:.1f})")
ax.set_xlabel("Pairs surviving all filters")
ax.set_ylabel("Frequency (out of 50 shuffles)")
ax.set_title("Multiple Testing Diagnostic: Real vs Shuffled Data (2018-2020 Window)")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(FIG_DIR / "multiple_testing_diagnostic.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved: {FIG_DIR / 'multiple_testing_diagnostic.png'}")"""
))

nb.cells = cells

# Set kernel info
nb.metadata["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3"
}
nb.metadata["language_info"] = {
    "name": "python",
    "version": "3.11.0"
}

out_path = Path("/Users/rajveershah/Desktop/ASX_statistical_arbritage/notebooks/05_analysis.ipynb")
with open(out_path, "w") as f:
    nbformat.write(nb, f)

print(f"Notebook written to {out_path}")
print(f"Cell count: {len(nb.cells)}")
