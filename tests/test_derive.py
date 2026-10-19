import datetime as dt

import polars as pl
import pytest

from signal_resilience.derive import grid_weights, phases

CALENDAR = pl.Series("date", [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(30)])
DAILY = pl.DataFrame({"date": CALENDAR, "barrid": ["A"] * 30, "active_weight": range(30)})


def day_index(frame: pl.DataFrame) -> list[int]:
    return [(d - CALENDAR[0]).days for d in frame["date"]]


@pytest.mark.parametrize(
    ("rebal", "phase", "expected"),
    [(1, 0, list(range(30))), (5, 0, [0, 5, 10, 15, 20, 25]), (5, 1, [1, 6, 11, 16, 21, 26]), (21, 17, [17])],
)
def test_phase_p_keeps_days_p_p_plus_n(rebal, phase, expected):
    assert day_index(grid_weights(DAILY, CALENDAR, rebal, phase)) == expected


@pytest.mark.parametrize("rebal", [1, 5, 21])
def test_phases_partition_the_daily_run(rebal):
    days = sorted(d for p in phases(rebal) for d in day_index(grid_weights(DAILY, CALENDAR, rebal, p)))
    assert days == list(range(30))
