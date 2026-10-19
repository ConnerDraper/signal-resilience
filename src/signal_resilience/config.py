"""Typed view of config.toml, read once at import.

Every tunable lives in the TOML. Unknown or missing keys raise, so a typo stops
the run rather than silently falling back to a default.

Which TOML is read comes from RESILIENCE_CONFIG (default config.toml), so one
pipeline runs several experiments: `resilience --config config.personal.toml`.
"""

import datetime as dt
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_CONSTRAINTS = {"ZeroInvestment", "ZeroBeta"}
SOURCES = {"library", "personal"}  # production alpha library, or signals built in personal.py
CONFIG_FILE = PROJECT_ROOT / os.environ.get("RESILIENCE_CONFIG", "config.toml")


@dataclass(frozen=True)
class Slurm:
    n_cpus: int
    mem: str
    time: str
    max_concurrent_jobs: int = 1


@dataclass(frozen=True)
class Config:
    label: str  # "production" or "personal"; prefixes figure titles and job names
    source: str
    output: Path
    data_root: str
    database: str
    alphas: Path | None  # only for source = "library"
    start: dt.date
    end: dt.date
    analysis_end: dt.date
    signals: tuple[str, ...]
    target_active_risk: float
    lags: tuple[int, ...]
    rebalance_periods: tuple[int, ...]
    reference: dict[str, tuple[str, ...]]  # signal -> long/short reference constraints
    slurm: dict[str, Slurm]

    @property
    def settings(self) -> list[tuple[int, int]]:
        """Constrained (lag, rebalance_period) settings, varied one axis at a time around (0, 1)."""
        return [(lag, 1) for lag in self.lags] + [(0, n) for n in self.rebalance_periods if n != 1]

    @property
    def email(self) -> str:
        """SLURM --mail-user, from BYU_EMAIL (environment or .env)."""
        return os.environ.get("BYU_EMAIL", "")


def _take(table: dict, name: str, keys: set[str]) -> dict:
    """Return the table's values, raising on unknown or missing keys."""
    unknown = set(table) - keys
    missing = keys - set(table)
    if unknown or missing:
        raise KeyError(f"[{name}] in {CONFIG_FILE.name}: unknown {sorted(unknown)}, missing {sorted(missing)}")
    return table


def load(path: Path = CONFIG_FILE) -> Config:
    raw = tomllib.loads(path.read_text())
    _take(raw, "top level", {"experiment", "environment", "window", "sweep", "reference", "slurm"})
    experiment = _take(raw["experiment"], "experiment", {"label", "source", "output"})
    if experiment["source"] not in SOURCES:
        raise ValueError(f"experiment.source: {experiment['source']!r} not in {sorted(SOURCES)}")
    library = experiment["source"] == "library"
    env = _take(raw["environment"], "environment", {"data_root", "database"} | ({"alphas"} if library else set()))
    window = _take(raw["window"], "window", {"start", "end", "analysis_end"})
    sweep = _take(raw["sweep"], "sweep", {"signals", "target_active_risk", "lags", "rebalance_periods"})
    reference = _take(raw["reference"], "reference", {"default", "signals"})
    stages = _take(raw["slurm"], "slurm", {"data", "backtest", "longshort", "analyze", "plot"})

    slurm = {}
    for stage, table in stages.items():
        required = {"n_cpus", "mem", "time"}
        unknown = set(table) - required - {"max_concurrent_jobs"}
        if unknown or required - set(table):
            raise KeyError(f"[slurm.{stage}]: unknown {sorted(unknown)}, missing {sorted(required - set(table))}")
        slurm[stage] = Slurm(**table)

    overrides = reference["signals"]
    if set(overrides) - set(sweep["signals"]):
        raise KeyError(f"[reference.signals]: not in sweep.signals: {sorted(set(overrides) - set(sweep['signals']))}")
    resolved = {s: tuple(overrides.get(s, reference["default"])) for s in sweep["signals"]}
    for signal, names in resolved.items():
        if set(names) - REFERENCE_CONSTRAINTS:
            raise ValueError(f"reference constraints for {signal}: {sorted(set(names) - REFERENCE_CONSTRAINTS)} "
                             f"not in {sorted(REFERENCE_CONSTRAINTS)}")

    if not window["start"] <= window["analysis_end"] <= window["end"]:
        raise ValueError("window: need start <= analysis_end <= end")
    if 0 not in sweep["lags"]:
        raise ValueError("sweep.lags must include 0: every rebalance period is derived from the lag-0 run")
    if 1 not in sweep["rebalance_periods"]:
        raise ValueError("sweep.rebalance_periods must include 1, the daily baseline")

    return Config(
        label=experiment["label"],
        source=experiment["source"],
        output=PROJECT_ROOT / experiment["output"],
        data_root=env["data_root"],
        database=env["database"],
        alphas=Path(env["alphas"]) if library else None,
        start=window["start"],
        end=window["end"],
        analysis_end=window["analysis_end"],
        signals=tuple(sweep["signals"]),
        target_active_risk=float(sweep["target_active_risk"]),
        lags=tuple(sweep["lags"]),
        rebalance_periods=tuple(sweep["rebalance_periods"]),
        reference=resolved,
        slurm=slurm,
    )


def connect() -> None:
    """Point sf_quant at the data lake, here and in any subprocess."""
    import sf_quant.data as sfd

    os.environ.setdefault("ROOT", CONFIG.data_root)
    os.environ.setdefault("DATABASE", CONFIG.database)
    sfd.env(root=CONFIG.data_root, database=CONFIG.database)


CONFIG = load()
