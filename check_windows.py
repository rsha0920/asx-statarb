from src.data import fetch_history, get_close_panel, ASX_UNIVERSE, all_tickers
from src.cointegration import screen_pairs

prices = get_close_panel(fetch_history(all_tickers(), cache_dir="data/cache"))

for start, end in [("2012-01-01", "2014-12-31"),
                   ("2015-01-01", "2017-12-31"),
                   ("2018-01-01", "2020-12-31")]:
    df = screen_pairs(prices, ASX_UNIVERSE, start, end)
    pairs_of_interest = df[df.apply(
        lambda r: {r.ticker_a, r.ticker_b} in [
            {"CBA.AX", "WBC.AX"}, {"ANZ.AX", "NAB.AX"},
            {"BHP.AX", "RIO.AX"}, {"GMG.AX", "SGP.AX"}
        ], axis=1)]
    print(f"\n{start} to {end}:")
    print(pairs_of_interest[["ticker_a", "ticker_b", "adf_pvalue",
                              "half_life_days", "hurst", "bh_pass"]])
