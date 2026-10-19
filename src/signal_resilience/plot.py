"""Render every figure from output/analysis/.

Reads no market data and imports nothing from the data lake, so it runs in
seconds anywhere and a chart change never means recomputing results.

Writes to output/figures/:
    lag_curve.png, rebalance_curve.png        Sharpe vs. lag / rebalance period
    trajectories/{lag,rebalance}/{signal}.png  growth of 1, long/short vs. constrained
"""

import polars as pl
from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator, NullFormatter, NullLocator

from . import style
from .config import CONFIG
from .paths import ANALYSIS, FIGURES

REFERENCE_COLOR = "#1b1b1b"  # dark ink: the long/short line reads as the baseline
SETTING_PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7", "#e34948"]
SETTING_COLOR = dict(zip(CONFIG.settings, SETTING_PALETTE))


def dollar_axis(ax) -> None:
    """Log y-axis labelled in dollars, dense enough to read values off.

    Ticks at 1-2-5 per decade across several decades; evenly spaced ticks when
    the lines span less than one decade, so both sides of $1 get labels.
    "\\$" because a bare "$" starts math text.
    """
    low, high = ax.get_ylim()
    if high / low >= 10:
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0), numticks=30))
    else:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=9, steps=[1, 2, 2.5, 5, 10]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"\\${v:g}"))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_formatter(NullFormatter())


def sharpe_curve(summary: pl.DataFrame, x: str, xlabel: str, title: str, name: str) -> None:
    """One line per signal, Sharpe vs. a setting, evenly spaced categorical x."""
    fig, ax = style.frame(style.title(title))
    for signal in CONFIG.signals:
        line = summary.filter(pl.col("signal") == signal).sort(x)
        if line.is_empty():
            continue
        ax.plot(line[x].cast(pl.Utf8), line["sharpe"], marker="o", markersize=9, markeredgewidth=2,
                markeredgecolor="white", linewidth=2.5, color=style.SIGNAL_COLOR[signal], label=signal)
    style.labels(ax, xlabel=xlabel, ylabel="Sharpe")
    style.legend(ax)
    style.save(fig, FIGURES / f"{name}.png")


def trajectories(signal: str, axis: str, summary: pl.DataFrame, paths: pl.DataFrame) -> None:
    """Growth of 1 in active return: long/short reference vs. one axis's settings."""
    settings = [(lag, 1) for lag in CONFIG.lags] if axis == "lag" else [(0, n) for n in CONFIG.rebalance_periods]
    name = (lambda lag, rebal: f"lag={lag}") if axis == "lag" else (lambda lag, rebal: f"rebal={rebal}")
    tc = {
        (r["lag"], r["rebalance_period"]): r["mean_tc"]
        for r in summary.filter((pl.col("signal") == signal) & (pl.col("portfolio") == "long_only")).iter_rows(named=True)
    }
    mine = paths.filter(pl.col("signal") == signal)

    fig, ax = style.frame(style.title(f"{signal}: active return trajectories, varying {'lag' if axis == 'lag' else 'rebalance period'}"))
    reference = mine.filter(pl.col("portfolio") == "long_short").sort("date")
    ax.plot(reference["date"], reference["cum_log"].exp(), color=REFERENCE_COLOR, linewidth=2.2,
            label="long/short (reference)")
    averaged = []
    for lag, rebal in settings:
        line = mine.filter(
            (pl.col("portfolio") == "long_only") & (pl.col("lag") == lag) & (pl.col("rebalance_period") == rebal)
        ).sort("date")
        ax.plot(line["date"], line["cum_log"].exp(), color=SETTING_COLOR[(lag, rebal)], linewidth=2.2,
                label=f"{name(lag, rebal)} (TC={tc[(lag, rebal)]:.2f})")
        if line["n_phases"][0] > 1:
            averaged.append(f"{name(lag, rebal)} averages {line['n_phases'][0]}")

    # Log scale: the long/short reference compounds to 100x+ gross of costs
    # while constrained lines stay near 1x, so a linear axis flattens all but it.
    ax.set_yscale("log")
    ax.axhline(1.0, color=style.MUTED_INK, linewidth=1)
    dollar_axis(ax)
    style.labels(ax, ylabel="Growth of \\$1 in active return (log scale)")
    style.legend(ax)
    if averaged:
        style.footnote(fig, f"Starting days: {', '.join(averaged)}; every other line is a single path.")
    style.save(fig, FIGURES / "trajectories" / axis / f"{signal}.png")


def run() -> None:
    style.apply()
    summary = pl.read_csv(ANALYSIS / "summary.csv")
    paths = pl.read_parquet(ANALYSIS / "paths.parquet")
    long_only = summary.filter(pl.col("portfolio") == "long_only")

    sharpe_curve(long_only.filter(pl.col("rebalance_period") == 1), "lag", "Lag (days)",
                 "Lag curve (rebalance period 1), Sharpe", "lag_curve")
    sharpe_curve(long_only.filter(pl.col("lag") == 0), "rebalance_period", "Rebalance period (days)",
                 "Rebalance curve (lag 0), Sharpe", "rebalance_curve")

    for signal in summary["signal"].unique(maintain_order=True):
        trajectories(signal, "lag", summary, paths)
        trajectories(signal, "rebalance", summary, paths)
    print(f"Wrote figures to {FIGURES}")
