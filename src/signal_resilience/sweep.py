"""The backtests: one SLURM array job per (signal, lag), solved daily.

Rebalance periods and starting days are derived from the lag-0 daily weights in
derive.py, so they need no jobs of their own.
"""

from pathlib import Path

import polars as pl
from sf_quant.slurm import BacktestConfig
from sf_quant.slurm.config import SlurmConfig
from sf_quant.slurm.slurm import generate_sbatch_script

from . import slurm
from .config import CONFIG, PROJECT_ROOT
from .derive import load_calendar
from .paths import LOGS, alphas_path, weights_dir


def configs() -> list[BacktestConfig]:
    calendar = load_calendar()
    resources = CONFIG.slurm["backtest"]
    return [
        BacktestConfig(
            signal_name=signal,
            data_path=str(alphas_path(signal)),
            start=calendar[0],
            end=CONFIG.end,
            project_root=str(PROJECT_ROOT),
            byu_email=CONFIG.email,
            target_active_risk=CONFIG.target_active_risk,
            lag=lag,
            rebalance_period=1,
            output_dir=str(weights_dir(signal, lag)),
            logs_dir=str(LOGS / "backtest" / signal / f"lag{lag}"),
            slurm=SlurmConfig(
                n_cpus=resources.n_cpus,
                mem=resources.mem,
                time=resources.time,
                max_concurrent_jobs=resources.max_concurrent_jobs,
                mail_type="FAIL",
            ),
        )
        for signal in CONFIG.signals
        for lag in CONFIG.lags
    ]


def submit(dry_run: bool, after: list[str]) -> list[str]:
    """Queue every (signal, lag) array job; return their job ids."""
    jobs = configs()
    table = (
        pl.DataFrame({"lag": [c.lag for c in jobs], "array_tasks": [len(c.years) for c in jobs]})
        .group_by("lag")
        .agg(pl.len().alias("jobs"), pl.col("array_tasks").sum())
        .sort("lag")
    )
    print(f"backtests: {len(jobs)} jobs, {table['array_tasks'].sum():,} array tasks")
    print(table)

    ids = []
    for i, config in enumerate(jobs):
        for directory in (config.output_dir, config.logs_dir):
            Path(directory).mkdir(parents=True, exist_ok=True)
        script = slurm.after(generate_sbatch_script(config, config.years), after)
        if dry_run and i == 0:
            print("\nFirst backtest sbatch script:\n" + script)
        job_id = slurm.submit(script, dry_run)
        ids.append(job_id or f"<backtest_{i}>")
    return ids
