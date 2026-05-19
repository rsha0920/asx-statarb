"""Data acquisition for ASX equity universe.

Uses yfinance for daily OHLCV. Acknowledged limitations:
- yfinance handles splits but corporate actions like demergers are imperfect
- Universe is current ASX large-caps. Survivorship bias is present and is
  flagged in the writeup rather than corrected for.
- Daily close only. No intraday.

Run as a script to download the universe and emit a quality report:

    python -m src.data
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)


# Static universe organised by GICS sector. Tickers use the yfinance .AX suffix.
# Names chosen for liquidity, continuity through the 2010-present period, and
# sector clustering that makes within-sector cointegration plausible.
ASX_UNIVERSE: dict[str, list[str]] = {
    # Big four banks plus the larger second-tier and Macquarie.
    "Financials_Banks": [
        "CBA.AX",  # Commonwealth Bank
        "WBC.AX",  # Westpac
        "ANZ.AX",  # ANZ
        "NAB.AX",  # National Australia Bank
        "MQG.AX",  # Macquarie
        "BEN.AX",  # Bendigo & Adelaide
        "BOQ.AX",  # Bank of Queensland
    ],
    "Financials_Insurance": [
        "IAG.AX",  # Insurance Australia
        "SUN.AX",  # Suncorp
        "QBE.AX",  # QBE Insurance
    ],
    # Iron ore and diversified miners. Tightest commodity cluster on ASX.
    "Materials_IronOre": [
        "BHP.AX",
        "RIO.AX",
        "FMG.AX",
        "MIN.AX",  # Mineral Resources
    ],
    # Gold. NCM/NEM transition (Newcrest acquired by Newmont 2023) makes those
    # tickers awkward, so we restrict to continuous names.
    "Materials_Gold": [
        "NST.AX",  # Northern Star
        "EVN.AX",  # Evolution Mining
    ],
    "Materials_Diversified": [
        "S32.AX",  # South32 (post-2015 BHP spinoff)
        "JHX.AX",  # James Hardie
        "AMC.AX",  # Amcor
        "ORI.AX",  # Orica
    ],
    # REITs. High clustering, similar rate sensitivity.
    "REITs": [
        "GMG.AX",  # Goodman Group
        "SCG.AX",  # Scentre Group (only post-2014; flag in QC)
        "SGP.AX",  # Stockland
        "DXS.AX",  # Dexus
        "MGR.AX",  # Mirvac
        "VCX.AX",  # Vicinity Centres
        "CHC.AX",  # Charter Hall
    ],
    # Energy. WPL renamed to WDS in 2022 but yfinance handles the continuity.
    "Energy": [
        "WDS.AX",  # Woodside (formerly WPL)
        "STO.AX",  # Santos
        "WHC.AX",  # Whitehaven Coal
        "NHC.AX",  # New Hope Coal
        "YAL.AX",  # Yancoal
    ],
    "Telecom": [
        "TLS.AX",  # Telstra
        "TPG.AX",  # TPG Telecom
    ],
    "Healthcare": [
        "CSL.AX",  # CSL
        "COH.AX",  # Cochlear
        "RMD.AX",  # ResMed
        "SHL.AX",  # Sonic Healthcare
        "RHC.AX",  # Ramsay Health Care
        "FPH.AX",  # Fisher & Paykel Healthcare
    ],
    # COL only since 2018 demerger; EDV only since 2021 demerger. Flagged in QC.
    "Consumer_Staples_Retail": [
        "WOW.AX",  # Woolworths
        "COL.AX",  # Coles
        "WES.AX",  # Wesfarmers
        "EDV.AX",  # Endeavour
        "TWE.AX",  # Treasury Wine
    ],
}


def all_tickers() -> list[str]:
    """Flat list of every ticker in the universe."""
    return [t for tickers in ASX_UNIVERSE.values() for t in tickers]


def ticker_to_sector() -> dict[str, str]:
    """Reverse lookup: ticker -> sector key."""
    return {t: sector for sector, tickers in ASX_UNIVERSE.items() for t in tickers}


def fetch_history(
    tickers: Iterable[str],
    start: str = "2010-01-01",
    end: str | None = None,
    cache_dir: Path | str | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Download daily OHLCV for the given tickers from yfinance.

    Returns a long-form DataFrame indexed by (date, ticker) with columns:
    open, high, low, close, adj_close, volume.

    If `cache_dir` is given, results are cached to a single parquet file
    keyed by (start, end). Pass `force_refresh=True` to bypass the cache.
    """
    tickers = list(tickers)
    cache_path: Path | None = None

    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"prices_{start}_{end or 'latest'}.parquet"
        if cache_path.exists() and not force_refresh:
            log.info("Loading cached prices from %s (mtime: %s)", cache_path,
                     pd.Timestamp(cache_path.stat().st_mtime, unit='s').date())
            return pd.read_parquet(cache_path)

    log.info("Downloading %d tickers from yfinance (start=%s, end=%s)",
             len(tickers), start, end)

    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    # yfinance returns a wide multi-index DataFrame when downloading multiple
    # tickers. Reshape into long form.
    rows: list[pd.DataFrame] = []
    available = set(raw.columns.get_level_values(0))
    for tkr in tickers:
        if tkr not in available:
            log.warning("No data returned for %s", tkr)
            continue
        sub = raw[tkr].copy()
        sub.columns = [str(c).lower().replace(" ", "_") for c in sub.columns]
        if sub["close"].dropna().empty:
            log.warning("Empty close series for %s, skipping", tkr)
            continue
        sub["ticker"] = tkr
        sub = sub.reset_index().rename(columns={"Date": "date"})
        rows.append(sub)

    if not rows:
        raise RuntimeError("No data fetched for any ticker")

    df = pd.concat(rows, ignore_index=True)
    df = df.set_index(["date", "ticker"]).sort_index()

    if cache_path is not None:
        df.to_parquet(cache_path)
        log.info("Cached to %s", cache_path)

    return df


def quality_check(df: pd.DataFrame) -> pd.DataFrame:
    """Per-ticker QC summary.

    Returns a DataFrame indexed by ticker with columns:
    n_rows, first_date, last_date, n_missing_close, max_abs_daily_return.

    The max_abs_daily_return column flags potential data errors.
    Anything above ~30% in a day on a large-cap is suspicious and worth
    investigating before using the ticker in a backtest.
    """
    out: list[dict] = []
    for tkr, sub in df.groupby(level="ticker"):
        dates = sub.index.get_level_values("date")
        rets = sub["adj_close"].pct_change()
        out.append({
            "ticker": tkr,
            "n_rows": len(sub),
            "first_date": dates.min().date() if len(dates) else None,
            "last_date": dates.max().date() if len(dates) else None,
            "n_missing_close": int(sub["close"].isna().sum()),
            "max_abs_daily_return": float(rets.abs().max()) if len(rets) else None,
        })
    qc = pd.DataFrame(out).set_index("ticker")
    return qc.sort_index()


def get_close_panel(df: pd.DataFrame, price_col: str = "adj_close") -> pd.DataFrame:
    """Pivot long-form prices into a wide panel: rows=date, columns=ticker.

    Use adj_close by default. Pass price_col='close' if you want raw close.
    Cointegration tests should run on log-prices: take np.log of the result.
    """
    return df[price_col].unstack(level="ticker")


def main() -> None:
    """Download the full universe and print a QC report."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    tickers = all_tickers()
    log.info("Universe size: %d tickers across %d sectors",
             len(tickers), len(ASX_UNIVERSE))

    df = fetch_history(tickers, cache_dir="data/cache")
    qc = quality_check(df)

    log.info("%s", "=" * 80)
    log.info("QUALITY REPORT")
    log.info("%s", "=" * 80)
    log.info("\n%s", qc.to_string())
    log.info("Total rows: %d", len(df))
    log.info(
        "Tickers with data: %d / %d",
        df.index.get_level_values("ticker").nunique(),
        len(tickers),
    )
    log.info(
        "Date range: %s to %s",
        df.index.get_level_values("date").min().date(),
        df.index.get_level_values("date").max().date(),
    )


if __name__ == "__main__":
    main()
