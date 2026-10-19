"""Build one alpha panel per signal: the backtester's input.

The alphas come from the production alpha library (source = "library") or are
built from raw Barra data in personal.py (source = "personal"). Either way each
signal gets predicted_beta and benchmark_weight joined on. Output columns:
date, barrid, alpha, predicted_beta, benchmark_weight.
"""

import polars as pl
import sf_quant.data as sfd

from .config import CONFIG
from .paths import TEMP, alphas_path


def load_betas() -> pl.DataFrame:
    return sfd.load_assets(
        start=CONFIG.start,
        end=CONFIG.end,
        columns=["date", "barrid", "predicted_beta"],
        in_universe=True,
    ).drop_nulls("predicted_beta")


def write_panel(signal: str, alphas: pl.LazyFrame, betas: pl.DataFrame, benchmark: pl.DataFrame) -> None:
    """Attach beta and benchmark weight to one signal's (date, barrid, alpha) and write it."""
    out = (
        alphas.select("date", "barrid", "alpha")
        # Inner join attaches predicted_beta and enforces universe
        # membership. The alpha library is already near-universe.
        .join(betas.lazy(), on=["date", "barrid"], how="inner")
        # Left join: null means "not in the benchmark", which collapse()
        # fills to 0 so LongOnly floors those names at h >= 0.
        .join(benchmark.lazy(), on=["date", "barrid"], how="left")
        .sort("date", "barrid")
        .collect()
    )
    path = alphas_path(signal)
    out.write_parquet(path)
    print(
        f"{signal:18s} {len(out):>10,} rows  {out['date'].n_unique():>5} dates  "
        f"{out['barrid'].n_unique():>5} barrids  -> {path.name}",
        flush=True,
    )


def run() -> None:
    TEMP.mkdir(parents=True, exist_ok=True)

    # Shared across signals, so load once.
    betas = load_betas()
    benchmark = sfd.load_benchmark(CONFIG.start, CONFIG.end).rename({"weight": "benchmark_weight"})
    print(f"universe: {len(betas):,} rows, {betas['date'].n_unique():,} dates")

    if CONFIG.source == "personal":
        from . import personal

        for signal, alphas in personal.build().items():
            write_panel(signal, alphas.lazy(), betas, benchmark)
        return

    alphas = pl.scan_parquet(CONFIG.alphas).filter(pl.col("date").is_between(CONFIG.start, CONFIG.end))
    for signal in CONFIG.signals:
        write_panel(signal, alphas.filter(pl.col("signal_name") == signal), betas, benchmark)
