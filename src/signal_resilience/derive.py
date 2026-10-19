"""Load the solved weights and derive every rebalance period from them.

Only the lags are solved, daily. A rebalance_period=N, phase=p portfolio is the
lag-0 daily portfolio kept on days p, p+N, p+2N, ...: each date's optimization
depends only on that date's inputs, so the solve is identical to a direct
rebalance_period=N run (checked byte-for-byte against one).
"""

import polars as pl
import sf_quant.data as sfd
from sf_quant.backtester import rebalance_dates

from .config import CONFIG
from .paths import alphas_path, longshort_dir, weights_dir


def load_calendar() -> pl.Series:
    """The calendar every backtest's grid is built on (worker.py uses the same)."""
    return sfd.load_trading_days(CONFIG.start, CONFIG.end)


def phases(rebalance_period: int) -> list[int]:
    """Every starting day: phase p trades on days p, p+N, p+2N, ..."""
    return list(range(rebalance_period))


def grid_weights(daily: pl.DataFrame, calendar: pl.Series, rebalance_period: int, phase: int) -> pl.DataFrame:
    """The weights a direct (rebalance_period, phase) run would hold.

    A direct run starts at calendar[phase] and collapse() anchors its grid
    there, so its rebalance dates are rebalance_dates(calendar[phase:], N).
    """
    grid = rebalance_dates(calendar[phase:], rebalance_period)
    return daily.filter(pl.col("date").is_in(grid.implode()))


def expected_years(signal: str, lag: int, calendar: pl.Series) -> set[int]:
    """Years a lag-L daily run should have written a file for.

    A date gets a portfolio iff the date L trading days earlier has alpha, so a
    year can legitimately be empty (a signal whose history starts in late
    December, lagged 21 days); worker.py writes no file for it.
    """
    alpha_dates = pl.scan_parquet(alphas_path(signal)).select("date").unique().collect()["date"]
    cal = pl.DataFrame({"date": calendar}).with_row_index("t")
    sources = cal.filter(pl.col("date").is_in(alpha_dates.implode()))["t"] + lag
    return set(cal.filter(pl.col("t").is_in(sources.implode()))["date"].dt.year().to_list())


def load_daily(signal: str, lag: int, calendar: pl.Series) -> tuple[pl.DataFrame | None, list[int]]:
    """(weights through analysis_end, missing years). Weights are None if incomplete."""
    directory = weights_dir(signal, lag)
    found = {int(f.stem) for f in directory.glob("*.parquet")}
    missing = sorted(expected_years(signal, lag, calendar) - found)
    if missing:
        return None, missing
    weights = pl.read_parquet(sorted(directory.glob("*.parquet")))
    return weights.filter(pl.col("date") <= CONFIG.analysis_end), []


def load_longshort(signal: str) -> pl.DataFrame:
    weights = pl.read_parquet(sorted(longshort_dir(signal).glob("*.parquet")))
    return weights.filter(pl.col("date") <= CONFIG.analysis_end)
