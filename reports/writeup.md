# ASX Statistical Arbitrage: A Pairs Trading Post-Mortem

**Rajveer Shah | May 2026**

---

## 1. What I Built and Why

This project applies cointegration-based pairs trading to ASX 200 large-cap equities, with a strict walk-forward out-of-sample backtest spanning 2014 to 2026. The ASX was the right testing ground for a few structural reasons. The big four banks (CBA, WBC, ANZ, NAB) move as a near-bloc driven by the same macro inputs: housing credit growth, RBA rate decisions, and net interest margin dynamics. Materials is similar: BHP, RIO, and FMG are tethered to the same iron ore price, though with different cost structures and balance sheet timing. REITs cluster tightly on rate sensitivity. These structural relationships give cointegration screens something real to find. The differentiator in this project is not the methodology, which is standard, but the honesty about what it actually produces. No cherry-picked windows. No in-sample Sharpe presented as tradeable.

---

## 2. Methodology

Pairs are screened within GICS sectors to limit the search space and reduce spurious correlations across unrelated industries. The cointegration test uses the Engle-Granger two-step procedure: regress one asset on another, then run an ADF test on the residuals. Multiple testing across the pair universe is handled with Benjamini-Hochberg FDR correction at 10%, applied per sector per window.

Pairs that pass cointegration also need a mean-reversion half-life between 5 and 60 trading days (via AR(1) on the spread), and a Hurst exponent below 0.5. The half-life filter excludes noise (too fast) and capital-inefficient slow reversion (too slow). The Hurst filter confirms anti-persistent behaviour beyond what ADF alone captures.

Spreads are modelled as Ornstein-Uhlenbeck processes. Signals use a rolling 60-day z-score: enter at +/-2, exit at +/-0.5, hard stop at +/-4, time stop at twice the estimated half-life. Sizing is equal-dollar per leg with inverse-volatility scaling across pairs, capped at 10 concurrent positions.

The backtest is walk-forward: 24-month training windows, 6-month trading windows, rolled monthly. The cointegration screen runs fresh in each training window. Trading window signals are strictly out-of-sample. Full methodology details are in the README.

---

## 3. Results

The headline is honest: the strategy does not reliably make money after realistic transaction costs.

Out-of-sample Sharpe over the full 2014-2026 period is **0.141** at 10 basis points per side per leg (base scenario), and **0.012** at 25 basis points (stress scenario). Net PnL on a $1M notional portfolio is $236,025 under base costs and $19,508 under stress costs. Estimated gross PnL is roughly $380,000. Base costs consume about 38% of gross. Stress costs consume 95%.

The year-by-year picture is volatile:

| Year | Annual Sharpe (base) |
|------|----------------------|
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

The 2014-2016 run looks genuinely good: a 2.10 Sharpe in 2016 with positive years on either side. Then 2017 collapses to -1.54. That drawdown is not noise. It coincides with a period of unusual sector dispersion in Australian equities, driven partly by APRA's macroprudential tightening on investor lending in 2017 and commodity price divergence in materials. The spread relationships the training windows built simply broke down.

Of 447 total trades across the backtest, 55% exited via mean reversion, 31% via time stop, and 3% via hard stop. The time-stop rate is telling: nearly one-third of positions never converged within the allowed holding period.

One window stands out: the H1 2024 training window produced zero pairs surviving all filters. That is not a bug. It reflects how tight the large-cap ASX had become by that period, with almost no within-sector spread showing statistically clean reversion.

---

## 4. Why It Fails (and When It Worked)

The alpha decay regression runs OLS on annual Sharpe versus year. The slope is -0.11 Sharpe units per year. The t-statistic is -1.25, the p-value is 0.24, and R-squared is 0.12. The decline is real in direction but not statistically significant at conventional levels, which is about what you should expect from 12 annual observations with high variance.

The most plausible explanation for the directional decline is the structural shift in how ASX large-caps are traded. Passive ETF flows into sector ETFs (particularly financials and materials) have tightened the within-sector spread distribution since approximately 2017. When a large sector ETF buys all four banks simultaneously on an index rebalance, it compresses the relative price dispersion that pairs trading needs. Electronic market making and improved price discovery at the daily close have contributed. None of this is provable from the data here, but the pattern is consistent.

BHP/RIO is the instructive case. The "obvious" pair in Australian materials almost never survives the Hurst filter. The two stocks look cointegrated in simple ADF tests, but the spread is more persistent than it appears. The filter is doing its job.

Cost sensitivity is the central finding. Moving from 10bps to 25bps costs removes $217,000 from the 12-year net outcome. That gap, $236k versus $19k, means the entire strategy stands or falls on execution quality. 10bps is achievable only with direct market access and algorithmic execution. Any retail or semi-institutional cost structure puts this firmly in the unprofitable bucket.

The multiple testing diagnostic adds an interesting wrinkle. For the 2018-2020 training window, the cointegration screen found 3 pairs surviving all filters in real data. Across 50 permutation shuffles of the same data, zero shuffles produced any survivors. The screen is finding genuine statistical structure, not noise. But genuine statistical structure and profitable trading are different questions, and the backtest answers the second one clearly.

---

## 5. What Would Need to Be True

For a strategy like this to be viable, several conditions need to hold simultaneously.

Transaction costs below 10 basis points require direct market access, exchange connectivity, and algorithmic execution. That is not realistic for an individual or small fund without a prime broker relationship. Even 15bps per side pushes the economics into marginal territory given the gross PnL observed here.

The frequency question matters. Daily data is convenient and reproducible, but cointegration relationships that are visible at daily resolution and survive the Hurst filter may be better captured intraday, where the spread can be managed more precisely and exit timing is more flexible. Intraday data introduces its own costs: infrastructure, data licensing, and a much more complex execution layer.

The universe also needs updating. Several ASX 200 names have been absorbed, delisted, or fundamentally restructured over the backtest period. A live version of this strategy would need continuous universe maintenance and would face index inclusion/exclusion events that create artificial spread dislocations.

None of these are insurmountable. They are, however, institutional-grade problems. This is not a retail-executable strategy at the cost levels the backtest requires.

---

## 6. Limitations I Know About

**Survivorship bias.** The universe was constructed using current ASX 200 constituents back-filled. Names that were in the index during 2014-2018 but later delisted or fell out of the index are not fully represented. The pairs that looked good historically may have looked good partly because we selected companies that survived.

**No market impact modelling.** The cost model applies a flat basis-point cost per side, but does not model market impact from position sizing relative to average daily volume. For the larger-cap names (CBA, BHP, WBC) this is probably fine. For smaller names on the fringe of the 50-stock universe, it is a real omission. Position sizes at the pair level were not calibrated to ADV limits.

**Daily data resolution.** Spreads computed from daily closing prices can look more stationary than they are intraday. The z-score signal triggers on the daily close, which means execution happens the next open. The spread at next open may have already partially reverted, or may have gapped further. This creates a systematic gap between the theoretical signal and the achievable entry.

**Single-test sectors.** The Telecom and Gold sub-sectors each had only one or two tradeable pairs for most of the backtest period. Statistical tests in very small populations have low power. Any cointegration finding in these sectors should be treated with more scepticism than the financials results.

**yfinance corporate action handling.** Splits, spin-offs, and special dividends are adjusted in yfinance, but the adjustment quality is not verified against ASX announcements. Some anomalous return spikes in the QC data were flagged and investigated, but a production system would use a professionally maintained price series.

---

## 7. What I Learned

The most concrete thing I took from this project is what walk-forward backtesting actually reveals. It is, by design, a disappointment machine. Every training window looks better than the corresponding trading window, because the model is fitted to the training data. The actual output of a walk-forward exercise is not the trading window Sharpe in isolation; it is the gap between training window and trading window performance. That gap is where the strategy's fragility lives. Running walk-forward once taught me to read that gap, not the headline number.

The second thing is about cost sensitivity. The difference between 10bps and 25bps per side produced a $217,000 spread in net PnL over 12 years on a $1M portfolio. That gap becomes sharper when you see where the $217k goes. Of 447 total trades, 31% exited via time stop (138 trades), paying entry and exit costs for no mean-reversion payoff. The time-stop rate compounds the cost problem. These are not losses that tighter stops could fix; they reflect the strategy's design constraint that each position lives at most twice its half-life. That is not rounding error. The trading strategy and the execution strategy are the same thing. You cannot evaluate one without specifying the other.

The third observation is the most counterintuitive. The multiple testing diagnostic showed that the cointegration screen finds real signal: zero out of 50 shuffles of the 2018-2020 data produced any surviving pairs, compared to 3 in the real data. The statistical relationships exist. The problem is not that the screen is detecting noise. The problem is that detecting a relationship and being able to profitably trade it are different things, separated by transaction costs, execution slippage, and the time it takes a spread to converge. The ASX large-cap market has become efficient enough that the relationships are real but the edges they create are thin. You need institutional execution quality to capture them, and that threshold has been rising over the full backtest period.

That is the honest conclusion. Not that pairs trading is dead, but that it requires better infrastructure than this project assumes, and that the infrastructure gap has widened since 2014.
