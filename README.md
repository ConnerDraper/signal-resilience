# signal-resilience

Are the production alphas tradeable, or only statistically present?

Two things break a signal in practice: acting on it late (a **lag**) and holding
the position instead of re-solving daily (a **rebalance period**). This sweeps
both over the six signals in the production alpha library and reports active
Sharpe for each setting. A flat lag curve means the edge survives; a curve that
collapses at lag=5 means it is only capturable with same-day execution.

Portfolios are active overlays under the fund mandate — zero active investment,
zero active beta, and no shorting beyond the benchmark (`w_a + w_b >= 0`) — with
active risk targeted at 5% annualized. Each is compared against a long/short
reference that drops only the last constraint.

## Setup

```bash
uv sync
export BYU_EMAIL=you@byu.edu     # SLURM notifications; or put it in ../../.env
```

### Fonts

Figures use Latin Modern Sans, Beamer's default Computer Modern Sans in its
modern form. Install it once per user (no root needed):

```bash
D=~/.local/share/fonts/latin-modern-sans; mkdir -p $D
for f in regular bold oblique boldoblique; do
  curl -sSL -o $D/lmsans10-$f.otf https://mirrors.ctan.org/fonts/lm/fonts/opentype/public/lm/lmsans10-$f.otf
done
```

`resilience plot` stops with a pointer here if the font is missing.

## Configuration

Every tunable lives in **`config.toml`**: data paths, the window, the signals,
lags, rebalance periods, target active risk, and SLURM resources per stage.
`src/signal_resilience/config.py` only types and reads it; unknown or missing
keys raise, so a typo stops the run instead of changing it.

## Personal signals

**`config.personal.toml`** runs the same sweep, reference and figures on seven
signals built from raw Barra data in `src/signal_resilience/personal.py`. These
are not taken from the alpha library. Results go to `output/personal/`, and
figure titles start with "Personal ·". Pass the config to every command:

```bash
uv run resilience --config config.personal.toml submit --dry-run
uv run resilience --config config.personal.toml submit
```

Notation: lr = log(1 + return/100), and slr is the same on Barra's
`specific_return`. Sums are per stock over complete trading-day windows.

| signal | definition |
|---|---|
| `resid_reversal_1w` | −Σ slr, last 5 days |
| `resid_reversal_1m` | −Σ slr, last 22 days |
| `industry_reversal` | −(Σ lr over the last 22 days − its equal-weight industry mean that day) |
| `high_52w` | log of the total-return index over its 252-day max (0 at a new high) |
| `industry_momentum` | cap-weighted industry mean of Σ lr over t−252..t−22 |
| `resid_momentum_vs` | Σ slr over t−252..t−22 ÷ `specific_risk` (Barra vol-scaled momentum) |
| `momentum_vs` | Σ lr over t−252..t−22 ÷ `total_risk` (vol-scaled momentum) |

- **Vol scaling:** each signal is divided by vol once, using the vol of the
  same kind of return.
- **Industry:** a stock's industry is its largest Barra industry exposure.
- **Score:** a daily cross-sectional z-score over the universe, clipped at ±3.
  Alpha = 0.05 · z · `specific_risk`/100, the same as production; production
  clips only `barra_reversal`, at ±2.
- **Window:** the sweep ends at `analysis_end`, because nothing later is
  analyzed.

## Run

```bash
# Queue the whole pipeline on the supercomputer (returns in seconds):
uv run resilience submit
uv run resilience submit --dry-run        # print the sbatch scripts, submit nothing
uv run resilience submit --from longshort # keep the backtests: longshort -> analyze -> plot
uv run resilience submit --from analyze   # reuse existing weights: analyze -> plot only

# Re-render every figure from existing results (seconds; no data lake):
uv run resilience plot

uv run resilience status                  # queued/running jobs
uv run pytest                             # no data lake needed
```

`submit` chains five stages with `--dependency=afterok`:

```
data ─┬─> backtests (array, one per signal × lag) ─┬─> analyze ─> plot
      └─> longshort ───────────────────────────────┘
```

`data` is queued only if the alpha panels are missing (`--rebuild-data` forces
it). SLURM emails when `plot` finishes or any stage fails; logs land in
`output/logs/`. Each stage also runs in-process — `resilience data`,
`resilience longshort [--check N]`, `resilience analyze` — which is what the
queued jobs call.

**Computing and drawing are separate stages.** `analyze` reads the weights and
the data lake and writes every number the figures use to `output/analysis/`;
`plot` reads only those files. Changing a chart never means recomputing results.

## The sweep

Backtests run 2005-01-01 to 2026-03-20, but results stop at **2025-07-03**
(`analysis_end`): from July 2025 the data lake is patchy — no alphas from
2025-07-03 to 2025-12-23, and most of Sep 2025 – Jan 2026 missing from the asset
calendar — so a portfolio would otherwise be held across a months-long hole as a
single period. Since each date's solve is independent, this filters weights; it
never re-solves.

Lag and rebalance period are varied one axis at a time around `(0, 1)`: lag 0,
5, 21 at `rebal=1`, and `rebal=1, 5, 21` at lag 0.

**Only the lags are solved.** Each date's optimization depends only on that
date's alpha, risk model, benchmark and beta — never on what was held the day
before — so a `rebal=N` portfolio on date t *is* the daily portfolio on date t.
A rebalance run is the daily run kept every N days and needs no solves of its
own (checked byte-for-byte against a directly solved `rebal=5` run). A lag pairs
a stale alpha with today's risk model, which is a different problem.

| solved | jobs | array tasks |
|---|---:|---:|
| lag 0, 5, 21 at `rebal=1`, one per signal | 18 | 396 |

About 94,000 optimizations.

### Why phases, and why all of them

Rebalancing every N days means picking which day to start on. Start on day 0 and
you trade on days 0, N, 2N; start on day 3 and you trade on 3, N+3, 2N+3 —
different trades, different Sharpe. One arbitrary starting day also gives only
~250 holding periods at N=21 against 5,200 at N=1. So every starting day is
evaluated and the Sharpes averaged — all 5 for `rebal=5`, all 21 for `rebal=21`
— which costs nothing because phases are derived, not solved. `summary.csv`
reports `sharpe_sd` across phases: how much the starting day mattered.

`derive.grid_weights` picks phase p's dates with sf_quant's own
`rebalance_dates(calendar[p:], N)`, exactly the grid a direct run started at
`calendar[p]` would solve.

## Long/short reference and Transfer Coefficient

To see how much of each portfolio's cost is specifically `LongOnly`,
`longshort` builds a reference portfolio per signal per day with only
`ZeroInvestment`/`ZeroBeta` — set per signal under `[reference]` in
`config.toml`. `beta` uses `ZeroBeta` alone: betting against beta is long
low-beta, short high-beta, and with zero net weight that portfolio has negative
beta, so zero beta would rule the bet out. With equality constraints alone the optimum is
analytic — Σ⁻¹-weighted alpha with the net-investment and beta directions
projected out, scaled so the 5% budget binds — so there is no solver (formula in
`longshort.py`). With the factor risk model Σ⁻¹ costs a K×K solve per date.
`resilience longshort --check N` verifies it against CVXPY: the closed form is
feasible to ~1e-16 and earns at least as much alpha per unit risk as the solver
on every date (the solver's ~1e-8 risk overshoot is its own tolerance).

The **Transfer Coefficient** (Clarke/de Silva/Thorley) is the per-date
cross-sectional correlation between a constrained portfolio's `active_weight`
and the reference's, averaged into `mean_tc`. Near 1, the constraint barely
changes the bet; low, it forces a meaningfully different one. Rebalance TCs
equal the `rebal=1` TC by construction: TC is measured on formation dates, and
a rebalance period changes how long a portfolio is held, not what it holds when
formed. The cost of holding shows up in the trajectories and Sharpes instead.

Returns are gross of transaction costs, which this study does not model. The
long/short reversal reference compounds to 100x+ over the window at 3.5x gross
leverage; costs would take much of that.

## Output

Everything generated lives under one gitignored tree:

```
output/
  temp/{signal}_alphas.parquet               alpha panels (data)
  weights/{signal}/lag{L}/{year}.parquet     daily constrained weights (backtests)
  longshort/{signal}/{year}.parquet          long/short reference weights (longshort)
  analysis/                                  every number the figures use (analyze)
    runs.csv          one row per (signal, portfolio, lag, rebal, phase)
    summary.csv       one row per setting: Sharpe averaged over phases, sharpe_sd, mean_tc
    paths.parquet     cumulative log active return per setting, on the daily axis
  figures/                                   rendered from analysis/ alone (plot)
    lag_curve.png / rebalance_curve.png
    trajectories/lag/{signal}.png            long/short vs. lag=0,5,21
    trajectories/rebalance/{signal}.png      long/short vs. rebal=1,5,21
  logs/
```

Weight files carry `active_weight` (the overlay `h`), `benchmark_weight`, and
`active_risk` (ex ante; sits at 0.05 because the linear objective always spends
the budget). Realized active volatility runs well above it for the reversal
signals — the risk model understates their risk.

Trajectory charts plot the growth of $1 on a log axis, so equal percentage moves
take equal height; `rebal=5`/`21` lines average every starting day, each phase
marked on the shared daily calendar (booked at the end of each holding period,
held flat between) before averaging.

`analyze` checks each daily run for every year it should have written and
leaves an incomplete signal out of every table and chart, with a loud warning
naming the missing years. "Should have" is computed from the alpha dates: a
lagged run can legitimately have no file for a year.

## Note on annualization

Sharpe annualizes by `252 / rebalance_period`, because
`generate_active_returns_from_weights` returns one row per holding period rather
than per day. Routing these through `sfp.generate_returns_summary_table`, which
hardcodes 252, would overstate the `rebal=21` Sharpe by roughly sqrt(21) — and
since rebalancing is the axis under test, that would read as "rebalancing less
improves the signal".
