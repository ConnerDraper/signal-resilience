"""Personal signal definitions on a small synthetic panel. No data lake."""

import datetime as dt

import numpy as np
import polars as pl
import pytest

from signal_resilience import personal

DAYS = 300


def panel(returns: dict[str, np.ndarray], industry: dict[str, int | None]) -> pl.DataFrame:
    dates = [dt.date(2020, 1, 1) + dt.timedelta(days=i) for i in range(DAYS)]
    return pl.concat(
        pl.DataFrame(
            {
                "date": dates,
                "barrid": barrid,
                "return": r,
                "specific_return": r / 2,
                "specific_risk": 30.0,
                "total_risk": 40.0,
                "market_cap": 1e9 * (i + 1),
                "in_universe": True,
                "industry": industry[barrid],
            },
            schema_overrides={"industry": pl.UInt32},
        )
        for i, (barrid, r) in enumerate(returns.items())
    )


@pytest.fixture
def signals():
    rng = np.random.default_rng(0)
    returns = {name: rng.normal(0, 2, DAYS) for name in ("A", "B", "C", "D")}
    returns["UP"] = np.full(DAYS, 0.5)  # a new high every day
    industry = {"A": 0, "B": 0, "C": 1, "D": None, "UP": 1}
    return returns, personal.signals(panel(returns, industry)).sort("barrid", "date")


def test_high_52w_is_zero_on_a_new_high(signals):
    _, out = signals
    up = out.filter(pl.col("barrid") == "UP")["high_52w"]
    assert up[: personal.HIGH_WINDOW - 1].is_null().all()
    assert np.allclose(up[personal.HIGH_WINDOW - 1 :].to_numpy(), 0.0)
    assert (out["high_52w"].drop_nulls() <= 1e-12).all()


def test_windows_are_null_until_full(signals):
    _, out = signals
    a = out.filter(pl.col("barrid") == "A")
    assert a["resid_reversal_1w"][:4].is_null().all() and a["resid_reversal_1w"][4] is not None
    first = personal.MOMENTUM_WINDOW + personal.MOMENTUM_SKIP - 1
    assert a["momentum_vs"][:first].is_null().all() and a["momentum_vs"][first] is not None


def test_momentum_skips_the_last_month(signals):
    returns, out = signals
    a = out.filter(pl.col("barrid") == "A")
    t = DAYS - 1
    window = np.log1p(returns["A"] / 100)[t - personal.MOMENTUM_SKIP - personal.MOMENTUM_WINDOW + 1 : t - personal.MOMENTUM_SKIP + 1]
    assert a["momentum_vs"][t] == pytest.approx(window.sum() / 40.0)
    resid = np.log1p(returns["A"] / 200)[t - personal.MOMENTUM_SKIP - personal.MOMENTUM_WINDOW + 1 : t - personal.MOMENTUM_SKIP + 1]
    assert a["resid_momentum_vs"][t] == pytest.approx(resid.sum() / 30.0)


def test_reversal_is_minus_the_recent_residual_return(signals):
    returns, out = signals
    a = out.filter(pl.col("barrid") == "A")
    assert a["resid_reversal_1m"][DAYS - 1] == pytest.approx(-np.log1p(returns["A"][-22:] / 200).sum())


def test_industry_signals(signals):
    _, out = signals
    last = out.filter(pl.col("date") == out["date"].max())
    # Industry-relative reversal sums to zero within an industry; no industry, no signal.
    assert last.filter(pl.col("industry") == 0)["industry_reversal"].sum() == pytest.approx(0.0, abs=1e-12)
    assert last.filter(pl.col("barrid") == "D")["industry_reversal"].is_null().all()
    assert last.filter(pl.col("barrid") == "D")["industry_momentum"].is_null().all()
    # Industry momentum is the same for every member and cap-weighted (A: cap 1, B: cap 2).
    zero = last.filter(pl.col("industry") == 0).sort("barrid")
    assert zero["industry_momentum"][0] == zero["industry_momentum"][1]
    expected = (zero["momentum"][0] * 1 + zero["momentum"][1] * 2) / 3
    assert zero["industry_momentum"][0] == pytest.approx(expected)


def test_alphas_are_clipped_scores_times_specific_risk(signals):
    _, out = signals
    alphas = personal.alphas(out, "resid_reversal_1m")
    z = alphas["alpha"] / (personal.IC * 30.0 / 100)
    assert z.abs().max() <= personal.CLIP + 1e-12
    daily = alphas.with_columns(z=z).group_by("date").agg(pl.col("z").mean())
    assert np.allclose(daily["z"].to_numpy(), 0.0, atol=1e-12)
