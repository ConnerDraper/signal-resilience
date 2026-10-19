"""Command-line entry point.

    resilience submit [--dry-run] [--rebuild-data]   queue the whole pipeline on SLURM
    resilience submit --from longshort               keep the backtests; redo longshort -> analyze -> plot
    resilience submit --from analyze                 reuse existing weights; queue analyze -> plot
    resilience plot                                  re-render figures (seconds, no data lake)
    resilience status                                show queued/running jobs

    resilience data | longshort [--check N] | analyze  run one stage in-process
                                                     (what the queued jobs call)

    resilience --config config.personal.toml ...     any of the above, on another experiment

`submit` chains the stages with --dependency=afterok and returns in seconds:

    data (only if panels are missing) -> backtests (array, one per signal x lag)
                                       + longshort -> analyze -> plot
"""

import argparse
import getpass
import os
import subprocess


STAGES = ["backtests", "longshort", "analyze", "plot"]


def submit(dry_run: bool, rebuild_data: bool, start: str = "backtests") -> None:
    from . import slurm, sweep
    from .config import CONFIG, connect
    from .paths import alphas_path

    if not CONFIG.email:
        raise SystemExit("BYU_EMAIL is not set. Export it or add it to .env.")
    connect()  # sets ROOT/DATABASE, which the backtest workers inherit

    def queue(stage: str, after: list[str], mail_type: str = "FAIL") -> str:
        script = slurm.after(slurm.stage_script(stage, CONFIG.slurm[stage], mail_type), after)
        if dry_run:
            print(f"\n--- {stage} ---\n{script}")
        return slurm.submit(script, dry_run) or f"<{stage}>"

    stages, upstream = [], []
    if start == "backtests":
        needs_data = rebuild_data or not all(alphas_path(s).exists() for s in CONFIG.signals)
        after_data = [queue("data", [])] if needs_data else []
        backtests = sweep.submit(dry_run, after_data)
        upstream = backtests + [queue("longshort", after_data)]
        stages = (["data"] if needs_data else []) + [f"{len(backtests)} backtests + longshort"]
    if start == "longshort":
        upstream = [queue("longshort", [])]
        stages.append("longshort")
    if start in ("backtests", "longshort", "analyze"):
        upstream = [queue("analyze", upstream)]
        stages.append("analyze")
    plot = queue("plot", upstream, mail_type="END,FAIL")
    stages.append("plot")
    verb = "Would queue" if dry_run else "Queued"
    print(f"\n{verb}: {' -> '.join(stages)}" + ("" if dry_run else f" (plot job {plot}; SLURM emails when it ends)"))


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="resilience", description="Lag and rebalance-period resilience of the production alphas."
    )
    parser.add_argument("--config", help="experiment TOML, relative to the project (default: config.toml)")
    commands = parser.add_subparsers(dest="command", required=True)
    queue = commands.add_parser("submit", help="queue the whole pipeline on SLURM")
    queue.add_argument("--dry-run", action="store_true", help="print the sbatch scripts, submit nothing")
    queue.add_argument("--rebuild-data", action="store_true", help="rebuild the alpha panels even if they exist")
    queue.add_argument("--from", dest="start", choices=STAGES, default="backtests",
                       help="first stage to queue; later stages reuse existing output (default: backtests)")
    commands.add_parser("data", help="build the alpha panels")
    longshort = commands.add_parser("longshort", help="closed-form long/short reference weights")
    longshort.add_argument("--check", type=int, metavar="N", help="verify against CVXPY on N random dates per signal")
    commands.add_parser("analyze", help="compute every result from the weights")
    commands.add_parser("plot", help="render figures from output/analysis")
    commands.add_parser("status", help="show queued/running SLURM jobs")
    args = parser.parse_args()
    if args.config:
        # Before anything imports .config, which reads the file once at import.
        os.environ["RESILIENCE_CONFIG"] = args.config

    if args.command == "submit":
        submit(args.dry_run, args.rebuild_data, args.start)
    elif args.command == "plot":
        from . import plot

        plot.run()
    elif args.command == "status":
        subprocess.run(["squeue", "-u", getpass.getuser()], check=False)
    else:
        from .config import CONFIG, connect

        connect()
        if args.command == "data":
            from . import data

            data.run()
        elif args.command == "longshort":
            from . import longshort as stage

            stage.run(args.check)
        elif args.command == "analyze":
            from . import analyze

            analyze.run(CONFIG.slurm["analyze"].n_cpus)


if __name__ == "__main__":
    main()
