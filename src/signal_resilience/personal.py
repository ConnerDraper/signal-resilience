"""Personal signals, built from raw Barra data (config.personal.toml).

With lr = log(1 + return/100) and slr = the same on Barra's specific_return,
each summed per stock over trading-day windows that must be complete:

    resid_reversal_1w   −Σ slr, last 5 days
    resid_reversal_1m   −Σ slr, last 22 days
    industry_reversal   −(Σ lr, last 22 days, minus its equal-weight industry mean that day)
    high_52w            log of the total-return index over its 252-day max (0 at a new high)
    industry_momentum   cap-weighted industry mean of Σ lr over t−252..t−22
    resid_momentum_vs   Σ slr over t−252..t−22 ÷ specific_risk   (Barra vol-scaled momentum)
    momentum_vs         Σ lr over t−252..t−22 ÷ total_risk       (vol-scaled momentum)

Vol scaling divides once, by the vol of the same kind of return. Industry is a
stock's largest Barra industry exposure.

Score and alpha follow production (alpha_signals_handoff.md), except the clip:
a daily cross-sectional z-score over the universe, clipped at ±3, then
alpha = 0.05 · z · specific_risk / 100.
"""

import datetime as dt

import polars as pl
import sf_quant.data as sfd
from sf_quant.data._factors import sector_factors

from .config import CONFIG

SIGNALS = [
    "resid_reversal_1w",
    "resid_reversal_1m",
    "industry_reversal",
    "high_52w",
    "industry_momentum",
    "resid_momentum_vs",
    "momentum_vs",
]
MOMENTUM_WINDOW, MOMENTUM_SKIP = 230, 22  # t−252..t−22
HIGH_WINDOW = 252
WARMUP = dt.timedelta(days=400)  # calendar days of history before start: fills the 252-day windows
IC = 0.05
CLIP = 3.0

ASSET_COLUMNS = ["date", "barrid", "return", "specific_return", "specific_risk", "total_risk", "market_cap", "in_universe"]


def log_return(column: str) -> pl.Expr:
    return pl.col(column).truediv(100).log1p()


def signals(raw: pl.DataFrame) -> pl.DataFrame:
    """Add one column per signal. raw: ASSET_COLUMNS plus industry (null = none)."""
    by_stock = raw.sort("barrid", "date").with_columns(
        reversal_22=log_return("return").rolling_sum(22).over("barrid"),
        momentum=log_return("return").rolling_sum(MOMENTUM_WINDOW).shift(MOMENTUM_SKIP).over("barrid"),
        resid_momentum=log_return("specific_return").rolling_sum(MOMENTUM_WINDOW).shift(MOMENTUM_SKIP).over("barrid"),
        resid_reversal_1w=-log_return("specific_return").rolling_sum(5).over("barrid"),
        resid_reversal_1m=-log_return("specific_return").rolling_sum(22).over("barrid"),
        log_index=log_return("return").cum_sum().over("barrid"),
    )

    # Industry averages are over universe stocks only: the ones that can be held.
    industry = ["date", "industry"]
    members = pl.col("in_universe") & pl.col("industry").is_not_null()
    cap = pl.when(members & pl.col("momentum").is_not_null()).then(pl.col("market_cap"))
    return by_stock.with_columns(
        high_52w=pl.col("log_index") - pl.col("log_index").rolling_max(HIGH_WINDOW).over("barrid"),
        resid_momentum_vs=pl.col("resid_momentum") / pl.col("specific_risk"),
        momentum_vs=pl.col("momentum") / pl.col("total_risk"),
        industry_reversal=-(
            pl.col("reversal_22") - pl.col("reversal_22").filter(members).mean().over(industry)
        ),
        industry_momentum=(pl.col("momentum") * cap).sum().over(industry) / cap.sum().over(industry),
    ).with_columns(
        # A stock with no industry has no industry signal.
        pl.when(pl.col("industry").is_not_null()).then(pl.col(s)).alias(s)
        for s in ("industry_reversal", "industry_momentum")
    )


def alphas(panel: pl.DataFrame, signal: str) -> pl.DataFrame:
    """(date, barrid, alpha) over universe stocks: z-score, clip, scale by specific risk."""
    value = pl.col(signal)
    return (
        panel.filter(pl.col("in_universe"), value.is_finite(), pl.col("specific_risk").is_not_null())
        .with_columns(((value - value.mean().over("date")) / value.std().over("date")).clip(-CLIP, CLIP).alias("z"))
        .select("date", "barrid", (IC * pl.col("z") * pl.col("specific_risk") / 100).alias("alpha"))
        .drop_nulls("alpha")
    )


def load_industries(start: dt.date, end: dt.date) -> pl.DataFrame:
    """date, barrid, industry: the index of each universe stock's largest industry exposure."""
    years = []
    for year in range(start.year, end.year + 1):
        exposures = sfd.load_exposures(
            max(start, dt.date(year, 1, 1)), min(end, dt.date(year, 12, 31)), True, ["date", "barrid", *sector_factors]
        )
        loadings = pl.concat_list(pl.col(sector_factors).fill_null(0))
        years.append(
            exposures.select(
                "date", "barrid",
                pl.when(loadings.list.max() > 0).then(loadings.list.arg_max()).alias("industry"),
            )
        )
        print(f"industries {year}: {exposures.height:,} rows", flush=True)
    return pl.concat(years)


def build() -> dict[str, pl.DataFrame]:
    """Every personal signal's (date, barrid, alpha) over [start, end]."""
    unknown = set(CONFIG.signals) - set(SIGNALS)
    if unknown:
        raise KeyError(f"no personal signal named {sorted(unknown)}; known: {SIGNALS}")

    raw = sfd.load_assets(CONFIG.start - WARMUP, CONFIG.end, ASSET_COLUMNS, in_universe=False)
    print(f"assets: {raw.height:,} rows from {raw['date'].min()}", flush=True)
    # Industry enters only through same-day cross-sections, so no warmup needed.
    raw = raw.join(load_industries(CONFIG.start, CONFIG.end), on=["date", "barrid"], how="left")

    panel = signals(raw).filter(pl.col("date").is_between(CONFIG.start, CONFIG.end))
    return {signal: alphas(panel, signal) for signal in CONFIG.signals}
