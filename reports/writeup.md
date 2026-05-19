# ASX Pairs Trading: What 12 Years of Data Actually Shows

*Rajveer Shah | May 2026*

---

## What I Built

I built a cointegration-based pairs trading system on 45 ASX large-caps and backtested it across 12 years of daily data, 2014 to 2026. Walk-forward out-of-sample, realistic transaction costs, no cherry-picked windows.

The short version: the strategy doesn't make money.

The longer version is more interesting.

I chose the ASX because the market has structural features that should, in theory, favour pairs trading. The big four banks move as a near-bloc, driven by the same macro inputs: housing credit, RBA rates, net interest margins. BHP, RIO, and FMG are tethered to the same iron ore price. REITs cluster tightly on rate sensitivity. If cointegration exists anywhere in equities, it should show up here.

The methodology is textbook. Engle-Granger cointegration with Benjamini-Hochberg correction for multiple testing. Ornstein-Uhlenbeck spread modelling. Z-score signals at +/-2 entry, +/-0.5 exit. Walk-forward design: 24-month training, 6-month trading, rolled monthly. No pair is ever traded in the window it was selected on. Full details are in the [README](../README.md).

---

## What the Numbers Say

Overall Sharpe: **0.141** at 10bps per side (base), **0.012** at 25bps (stress). On a $1M notional, that's $236k net under base costs and $19k under stress. Gross PnL was roughly $380k. Base costs consumed 38% of that. Stress costs consumed 95%.

The year-by-year picture tells a better story than the aggregate:

| Year | Sharpe |
|------|--------|
| 2014 | 1.14 |
| 2015 | 0.66 |
| 2016 | 2.10 |
| 2017 | -1.54 |
| 2018 | 0.14 |
| 2019 | -0.24 |
| 2020 | 0.27 |
| 2021 | 0.81 |
| 2022 | -0.31 |
| 2023 | 0.90 |
| 2024 | -0.47 |
| 2025 | 1.50 |

2014-2016 was genuinely good. A 2.10 Sharpe in 2016. Then 2017 collapsed to -1.54, coinciding with APRA's interest-only lending cap in March 2017 and diverging commodity prices that broke the spread relationships the model had learned. The spreads simply stopped reverting.

Of 447 trades, 55% exited via mean reversion (the strategy working as designed), 31% via time stop (the spread never converged), and 3% via hard stop (something structural changed fast). That time-stop rate matters. Nearly a third of all positions paid entry and exit costs for nothing.

One window in particular: H1 2024 produced zero pairs surviving all filters. Not a bug. The ASX large-cap universe had become so tightly priced that no within-sector spread showed clean enough reversion to pass the screen.

---

## Why It Fails

The alpha decay regression gives a slope of -0.11 Sharpe units per year (t = -1.25, p = 0.24, R² = 0.12). Not statistically significant with only 12 annual observations, but the direction is clear.

The most likely culprit is passive money. ETF flows into sector baskets (financials ETFs buying all four banks simultaneously on rebalance) compress the relative price dispersion that pairs trading needs. Electronic market making has tightened spreads at the daily close. The arbitrage opportunity hasn't disappeared, it has shrunk below the cost of capturing it at daily frequency.

BHP/RIO is the case I keep coming back to. Everyone assumes this is the canonical Australian pairs trade. Two iron ore miners, massive market caps, obvious co-movement. But across every window I tested, BHP/RIO almost never survived the Hurst filter. The spread is more persistent than it looks. BHP's petroleum and copper exposure, plus its London listing dynamics, create enough structural divergence that the pair wanders rather than reverts. The "obvious" pair isn't one. The filter is doing its job.

The cost finding is what really sticks. Moving from 10bps to 25bps per side flips the outcome from $236k to $19k over 12 years. The trading strategy and the execution strategy are the same thing. You cannot evaluate one without the other. At 10bps you need direct market access and algorithmic execution. At 25bps you're losing money. There's no comfortable middle ground.

---

## The Placebo Test

I ran 50 permutation shuffles of the 2018-2020 data and re-ran the full cointegration screen on each. Zero shuffles produced any surviving pairs. Real data: 3 survivors. Empirical p-value under 0.02.

The screen finds real statistical relationships, not noise. That's the counterintuitive part. The cointegration is genuine. The problem is that genuine cointegration and profitable trading are separated by transaction costs, execution slippage, and time-to-convergence. The ASX large-cap market is efficient enough that the relationships exist but the edges they create are too thin for daily-frequency capture without institutional infrastructure.

---

## What I Know Is Wrong With This

Survivorship bias. I used current ASX constituents back-filled to 2010. Companies that delisted or fell out of the index aren't represented. This flatters the results.

No market impact modelling. The cost model is a flat per-side rate. For CBA and BHP, fine. For the smaller names on the edge of my universe, real market impact would be worse.

Daily resolution only. Spreads computed from daily closes can look more stationary than they are intraday. The signal fires on the close; execution happens at the next open. That gap erodes the edge.

Single-test sectors. Telecom and Gold each had one or two pairs for most of the backtest. BH correction on a single test is degenerate. Those results deserve more scepticism than the financials results.

---

## What I Actually Learned

Walk-forward backtesting is a disappointment machine by design. Every training window looks better than the trading window that follows it, because the model is fit to the training data. The useful output isn't the headline Sharpe. It's the gap between training and trading performance. That gap is where fragility lives. I didn't fully appreciate this until I watched it happen across 20+ rolling windows.

The second thing is about the relationship between costs and strategy design. The 31% time-stop rate means 138 trades paid full entry and exit costs for zero mean-reversion payoff. Tighter stops wouldn't fix this; the positions legitimately hadn't converged. They were paying the cost of the strategy's design constraint that each trade lives at most twice its half-life. That's not slippage or bad luck. It's the strategy taxing itself.

And the thing I didn't expect: finding real signal doesn't mean finding real profit. The placebo test confirmed the cointegration screen works. The backtest confirmed it doesn't matter. The ASX large-cap universe has become efficient enough that the statistical relationships I can detect are real but the edges they produce sit below the cost floor. You need institutional execution to capture them, and that bar has been rising every year since 2014.

Not that pairs trading is dead. But the infrastructure it requires has outgrown what a daily-frequency backtest on free data can prove.
