"""Compute every number the figures show. No plotting here.

Per signal: every constrained setting (each lag, solved daily; each rebalance
period at every starting day, derived from lag 0) plus the long/short reference.
For each portfolio the holding-period returns are computed once and reused for
the Sharpe, the Transfer Coefficient and the trajectory.

Writes to output/analysis/:
    runs.csv       one row per (signal, portfolio, lag, rebalance_period, phase)
    summary.csv    one row per setting: phase-averaged Sharpe, plus mean TC
    paths.parquet  cumulative log active return per setting on the daily axis
"""

import polars as pl
import sf_quant.performance as sfp
from joblib import Parallel, delayed

from .config import CONFIG, connect
from .derive import grid_weights, load_calendar, load_daily, load_longshort, phases
from .paths import ANALYSIS


def score(returns: pl.Series, rebalance_period: int) -> dict:
    # Returns are per holding period, not per day: annualize by periods per
    # year, or a rebal=21 Sharpe would be overstated by about sqrt(21).
    periods_per_year = 252 / rebalance_period
    mean = returns.mean() * periods_per_year
    vol = returns.std() * periods_per_year**0.5
    return {"sharpe": mean / vol if vol else None, "mean_%": mean * 100, "vol_%": vol * 100, "n_periods": returns.len()}


def transfer_coefficient(actual: pl.DataFrame, ideal: pl.DataFrame) -> pl.Series:
    """Per-date cross-sectional correlation of active weights (Clarke/de Silva/Thorley).

    Inner join: only dates both portfolios were formed on count.
    """
    joined = actual.select("date", "barrid", "active_weight").join(
        ideal.select("date", "barrid", "active_weight"), on=["date", "barrid"], suffix="_ideal"
    )
    return joined.group_by("date").agg(pl.corr("active_weight", "active_weight_ideal"))["active_weight"]


def cumulative_log_path(weights: pl.DataFrame, returns: pl.DataFrame, axis_dates: pl.Series) -> pl.Series:
    """Cumulative log active return, marked on every date of axis_dates.

    A return row is dated at its formation date t_k but earned over
    (t_k, t_{k+1}], so it is booked at t_{k+1}; between rebalances the path
    holds its last mark. A shared daily axis is what lets paths with different
    rebalance days be averaged.
    """
    formation = weights["date"].unique().sort()
    ends = pl.DataFrame({"date": formation, "end": formation.shift(-1)})
    marks = (
        returns.sort("date")
        .join(ends, on="date")
        .select(pl.col("end").alias("date"), pl.col("return").log1p().cum_sum().alias("cum_log"))
    )
    return (
        pl.DataFrame({"date": axis_dates})
        .join(marks, on="date", how="left")
        .with_columns(pl.col("cum_log").forward_fill().fill_null(0.0))["cum_log"]
    )


def analyze_signal(signal: str, calendar: pl.Series) -> dict:
    connect()  # joblib workers are fresh processes
    daily, missing = {}, {}
    for lag in CONFIG.lags:
        weights, gaps = load_daily(signal, lag, calendar)
        if gaps:
            missing[lag] = gaps
        else:
            daily[lag] = weights
    if missing:
        return {"signal": signal, "missing": missing}

    ideal = load_longshort(signal)
    axis_dates = daily[0]["date"].unique().sort()
    runs, summary, paths = [], [], []

    def record(portfolio, lag, rebal, frames, tc):
        path_columns = []
        for phase, weights in enumerate(frames):
            returns = sfp.generate_active_returns_from_weights(weights)
            runs.append(
                {"signal": signal, "portfolio": portfolio, "lag": lag, "rebalance_period": rebal, "phase": phase,
                 **score(returns["return"], rebal), "active_risk": weights["active_risk"].mean()}
            )
            path_columns.append(cumulative_log_path(weights, returns, axis_dates).alias(str(phase)))
        paths.append(
            pl.DataFrame({"date": axis_dates, "cum_log": pl.DataFrame(path_columns).mean_horizontal()}).with_columns(
                signal=pl.lit(signal), portfolio=pl.lit(portfolio), lag=pl.lit(lag),
                rebalance_period=pl.lit(rebal), n_phases=pl.lit(len(frames)),
            )
        )
        summary.append({"signal": signal, "portfolio": portfolio, "lag": lag, "rebalance_period": rebal,
                        "mean_tc": tc.mean() if tc is not None else None,
                        "tc_sd": tc.std() if tc is not None else None})

    record("long_short", 0, 1, [ideal], None)
    for lag, rebal in CONFIG.settings:
        frames = [daily[lag]] if rebal == 1 else [grid_weights(daily[0], calendar, rebal, p) for p in phases(rebal)]
        tc = pl.concat([transfer_coefficient(w, ideal) for w in frames])
        record("long_only", lag, rebal, frames, tc)

    print(f"{signal}: {len(runs)} portfolios scored", flush=True)
    return {"signal": signal, "runs": runs, "summary": summary, "paths": paths}


def run(workers: int) -> None:
    calendar = load_calendar()
    results = Parallel(n_jobs=min(workers, len(CONFIG.signals)))(
        delayed(analyze_signal)(signal, calendar) for signal in CONFIG.signals
    )
    complete = [r for r in results if "missing" not in r]
    if not complete:
        raise SystemExit("No signal has a complete set of daily runs.")

    runs = pl.DataFrame([row for r in complete for row in r["runs"]])
    keys = ["signal", "portfolio", "lag", "rebalance_period"]
    summary = (
        runs.group_by(keys)
        .agg(
            pl.col("sharpe").mean(),
            pl.col("sharpe").std().alias("sharpe_sd"),  # spread across starting days
            pl.len().alias("n_phases"),
            pl.col("mean_%").mean(),
            pl.col("vol_%").mean(),
            pl.col("active_risk").mean(),
        )
        .join(pl.DataFrame([row for r in complete for row in r["summary"]]), on=keys)
        .sort("signal", "portfolio", "rebalance_period", "lag")
    )

    ANALYSIS.mkdir(parents=True, exist_ok=True)
    runs.sort(*keys, "phase").write_csv(ANALYSIS / "runs.csv")
    summary.write_csv(ANALYSIS / "summary.csv")
    pl.concat([p for r in complete for p in r["paths"]]).write_parquet(ANALYSIS / "paths.parquet")
    print(f"Wrote runs.csv, summary.csv, paths.parquet to {ANALYSIS}")

    incomplete = [r for r in results if "missing" in r]
    if incomplete:
        # Loud but non-fatal: a signal with any missing year is left out of
        # every table and chart rather than scored on a partial sample.
        print("\nWARNING: these signals are missing years and were excluded:")
        for r in incomplete:
            for lag, years in r["missing"].items():
                print(f"    {r['signal']} lag={lag}: missing {years}")
