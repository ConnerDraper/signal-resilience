"""The output tree. Everything generated lives under the experiment's output
directory (config.toml: output/; config.personal.toml: output/personal/), gitignored."""

from pathlib import Path

from .config import CONFIG

OUTPUT = CONFIG.output
TEMP = OUTPUT / "temp"  # alpha panels, the backtester's input
WEIGHTS = OUTPUT / "weights"  # daily constrained weights, one dir per (signal, lag)
LONGSHORT = OUTPUT / "longshort"  # closed-form long/short reference weights
ANALYSIS = OUTPUT / "analysis"  # every number the figures use; no plotting here
FIGURES = OUTPUT / "figures"  # rendered from ANALYSIS alone
LOGS = OUTPUT / "logs"


def alphas_path(signal: str) -> Path:
    return TEMP / f"{signal}_alphas.parquet"


def weights_dir(signal: str, lag: int) -> Path:
    return WEIGHTS / signal / f"lag{lag}"


def longshort_dir(signal: str) -> Path:
    return LONGSHORT / signal
