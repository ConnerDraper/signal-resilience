"""Write sbatch scripts, submit them, and chain them with --dependency=afterok."""

import os
import subprocess
import sys
import tempfile

from .config import CONFIG, CONFIG_FILE, PROJECT_ROOT, Slurm
from .paths import LOGS


def submit(script: str, dry_run: bool) -> str | None:
    """Submit an sbatch script; return its job id (None on a dry run)."""
    if dry_run:
        return None
    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as f:
        f.write(script)
        path = f.name
    try:
        out = subprocess.run(["sbatch", "--parsable", path], capture_output=True, text=True, check=True)
    finally:
        os.unlink(path)
    return out.stdout.strip().split(";")[0]


def after(script: str, job_ids: list[str]) -> str:
    """Make the script wait for job_ids to finish successfully."""
    if not job_ids:
        return script
    shebang, rest = script.split("\n", 1)
    return f"{shebang}\n#SBATCH --dependency=afterok:{':'.join(job_ids)}\n{rest}"


def stage_script(stage: str, slurm: Slurm, mail_type: str = "FAIL") -> str:
    """An sbatch script that runs `resilience <stage>` on a compute node."""
    LOGS.mkdir(parents=True, exist_ok=True)
    return f"""#!/bin/bash
#SBATCH --job-name=resilience_{CONFIG.label}_{stage}
#SBATCH --output={LOGS}/{stage}_%j.out
#SBATCH --cpus-per-task={slurm.n_cpus}
#SBATCH --mem={slurm.mem}
#SBATCH --time={slurm.time}
#SBATCH --mail-user={CONFIG.email}
#SBATCH --mail-type={mail_type}

ulimit -c 0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export ROOT={CONFIG.data_root} DATABASE={CONFIG.database} RESILIENCE_CONFIG={CONFIG_FILE}

cd {PROJECT_ROOT}
{sys.executable} -m signal_resilience.cli {stage}
"""
