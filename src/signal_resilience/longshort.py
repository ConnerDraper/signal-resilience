"""Long/short reference portfolio for every signal and trading day, in closed form.

Dropping LongOnly leaves only equality constraints (by default both of these;
config.toml can drop either per signal):

    maximize  α·h   subject to   h'Σh <= σ²,   1·h = 0,   β·h = 0

which has an analytic optimum, so no solver is involved. With A the rows of the
constraints in force ([1; β] by default):

    u = Σ⁻¹α,   W = Σ⁻¹A'
    λ = (A W)⁻¹ A u          (how much of the 1 and β directions to strip out)
    g = u − W λ              (satisfies A g = 0)
    h = σ g / √(g'α)          (g'Σg = g'α, so this makes the risk budget bind)

Σ = B F B' + diag(D) is never formed. Σ⁻¹v uses the Woodbury identity in the
form that never inverts F (the loader fills missing covariances with 0, so F
can be singular):

    Σ⁻¹v = D⁻¹v − D⁻¹B (I + F B'D⁻¹B)⁻¹ F B'D⁻¹v

The risk model is loaded once per date for the union of all signals' names and
shared across signals.
"""

import datetime as dt
import random
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import polars as pl
import sf_quant.data as sfd
from sf_quant.backtester import collapse
from sf_quant.optimizer import ZeroBeta, ZeroInvestment, active_mve_optimizer

from .config import CONFIG
from .derive import load_calendar
from .paths import alphas_path, longshort_dir

WORKERS = 8

# CLARABEL is iterative: it stops within its tolerance of optimal *and* of
# feasible (risk can overshoot 5% by ~1e-8), so the check doesn't ask the two to
# match exactly. It asks that the closed form be feasible to machine precision
# and earn at least as much alpha per unit of risk as the solver's answer --
# the optimum is unique, so that makes it the optimum.
FEASIBILITY_TOLERANCE = 1e-10
EFFICIENCY_TOLERANCE = 1e-9  # solver's alpha/risk may beat the formula's by this, relative


def sigma_inverse(v: np.ndarray, B: np.ndarray, F: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Σ⁻¹v for Σ = B F B' + diag(D), without forming Σ or inverting F."""
    d_inv_v = v / D[:, None] if v.ndim == 2 else v / D
    d_inv_b = B / D[:, None]
    core = np.eye(F.shape[0]) + F @ (B.T @ d_inv_b)
    return d_inv_v - d_inv_b @ np.linalg.solve(core, F @ (B.T @ d_inv_v))


def closed_form(
    alpha, beta, B, F, D, target: float, constraints=("ZeroInvestment", "ZeroBeta")
) -> tuple[np.ndarray, float]:
    """(h, realized active risk) under any subset of ZeroInvestment and ZeroBeta."""
    rows = {"ZeroInvestment": np.ones_like(alpha), "ZeroBeta": beta}
    u = sigma_inverse(alpha, B, F, D)
    if constraints:
        A = np.vstack([rows[name] for name in constraints])
        W = sigma_inverse(A.T, B, F, D)
        g = u - W @ np.linalg.solve(A @ W, A @ u)
    else:
        g = u
    h = g * (target / np.sqrt(g @ alpha))
    loadings = B.T @ h
    risk = float(np.sqrt(loadings @ F @ loadings + np.sum(D * h**2)))
    return h, risk


def load_panels(calendar: pl.Series) -> dict[str, dict[dt.date, pl.DataFrame]]:
    """Per signal, per date: the same lag-0 daily rows the backtests optimize."""
    panels = {}
    for signal in CONFIG.signals:
        daily = collapse(pl.scan_parquet(alphas_path(signal)), dates=calendar).collect()
        panels[signal] = {key[0]: part.sort("barrid") for key, part in daily.partition_by("date", as_dict=True).items()}
        print(f"{signal}: {len(panels[signal]):,} dates")
    return panels


def solve_date(date_: dt.date, parts: dict[str, pl.DataFrame]) -> dict[str, pl.DataFrame]:
    """One risk-model load for the date, then the closed form for every signal."""
    union = sorted(set().union(*(part["barrid"] for part in parts.values())))
    B, F, D = sfd.construct_factor_model_components(date_, union)
    position = {barrid: i for i, barrid in enumerate(union)}

    out = {}
    for signal, part in parts.items():
        idx = np.array([position[b] for b in part["barrid"]])
        h, risk = closed_form(
            part["alpha"].to_numpy(), part["predicted_beta"].to_numpy(), B[idx], F, D[idx],
            CONFIG.target_active_risk, CONFIG.reference[signal],
        )
        out[signal] = pl.DataFrame(
            {
                "date": [date_] * part.height,
                "barrid": part["barrid"],
                "active_weight": h,
                "benchmark_weight": part["benchmark_weight"],
                "active_risk": [risk] * part.height,
            }
        )
    return out


def check(panels, n_dates: int) -> None:
    """Formula vs. CVXPY (the library optimizer, LongOnly dropped) on random dates."""
    target = CONFIG.target_active_risk
    rng = random.Random(0)
    failures = 0
    for signal in CONFIG.signals:
        for date_ in rng.sample(sorted(panels[signal]), n_dates):
            part = panels[signal][date_]
            barrids = part["barrid"].to_list()
            alpha, beta = part["alpha"].to_numpy(), part["predicted_beta"].to_numpy()
            B, F, D = sfd.construct_factor_model_components(date_, barrids)
            names = CONFIG.reference[signal]
            h, risk = closed_form(alpha, beta, B, F, D, target, names)
            solved = active_mve_optimizer(
                ids=barrids,
                alphas=alpha,
                factor_exposures=B,
                factor_covariance=F,
                specific_risk=D,
                benchmark_weights=part["benchmark_weight"].to_numpy(),
                target_active_risk=target,
                betas=beta,
                constraints=[{"ZeroInvestment": ZeroInvestment, "ZeroBeta": ZeroBeta}[n]() for n in names],
            )["active_weight"].to_numpy()

            loadings = B.T @ solved
            solved_risk = float(np.sqrt(loadings @ F @ loadings + np.sum(D * solved**2)))
            residuals = {"ZeroInvestment": h.sum(), "ZeroBeta": h @ beta}
            infeasibility = max([abs(residuals[n]) for n in names] + [abs(risk - target)])
            # > 0 means the solver earned more alpha per unit of risk than the formula.
            efficiency_gap = (alpha @ solved / solved_risk) / (alpha @ h / risk) - 1
            ok = infeasibility <= FEASIBILITY_TOLERANCE and efficiency_gap <= EFFICIENCY_TOLERANCE
            failures += not ok
            print(
                f"{'ok  ' if ok else 'FAIL'} {signal:16s} {date_}  formula infeasibility={infeasibility:.1e}  "
                f"solver risk overshoot={solved_risk / target - 1:+.1e}  efficiency gap={efficiency_gap:+.1e}"
            )
    print(f"\n{'PASS' if not failures else 'FAIL'}: {failures} of {len(CONFIG.signals) * n_dates} dates failed")
    if failures:
        raise SystemExit(1)


def run(check_dates: int | None = None) -> None:
    panels = load_panels(load_calendar())
    if check_dates:
        check(panels, check_dates)
        return

    dates = sorted(set().union(*panels.values()))
    for year in sorted({d.year for d in dates}):
        year_dates = [d for d in dates if d.year == year]
        # Risk-model loads (polars) and the linear algebra (numpy/BLAS) both
        # release the GIL, so threads parallelize this cleanly.
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            results = list(
                pool.map(lambda d: solve_date(d, {s: panels[s][d] for s in CONFIG.signals if d in panels[s]}), year_dates)
            )
        for signal in CONFIG.signals:
            frames = [r[signal] for r in results if signal in r]
            if frames:
                directory = longshort_dir(signal)
                directory.mkdir(parents=True, exist_ok=True)
                pl.concat(frames).write_parquet(directory / f"{year}.parquet")
        print(f"{year}: {len(year_dates)} dates written")
