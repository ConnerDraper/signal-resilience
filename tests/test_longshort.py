"""The closed form against the library's solver. Synthetic risk model, no data lake."""

import numpy as np
import pytest
from sf_quant.optimizer import ZeroBeta, ZeroInvestment, active_mve_optimizer

from signal_resilience.longshort import closed_form

TARGET = 0.05
N, K = 60, 5
SUBSETS = [("ZeroInvestment", "ZeroBeta"), ("ZeroBeta",), ("ZeroInvestment",), ()]
CLASSES = {"ZeroInvestment": ZeroInvestment, "ZeroBeta": ZeroBeta}


@pytest.fixture(params=[0, 1, 2])
def model(request):
    rng = np.random.default_rng(request.param)
    B = rng.normal(size=(N, K))
    root = rng.normal(size=(K, K))
    F = root @ root.T / K + np.eye(K) * 0.02
    D = rng.uniform(0.02, 0.09, size=N) ** 2
    return dict(alpha=rng.normal(scale=0.02, size=N), beta=rng.uniform(0.6, 1.5, size=N), B=B, F=F, D=D)


@pytest.mark.parametrize("constraints", SUBSETS)
def test_feasible_to_machine_precision(model, constraints):
    h, risk = closed_form(**model, target=TARGET, constraints=constraints)
    if "ZeroInvestment" in constraints:
        assert abs(h.sum()) < 1e-12
    if "ZeroBeta" in constraints:
        assert abs(model["beta"] @ h) < 1e-12
    assert risk == pytest.approx(TARGET, rel=1e-12)


@pytest.mark.parametrize("constraints", SUBSETS)
def test_matches_the_solver(model, constraints):
    h, risk = closed_form(**model, target=TARGET, constraints=constraints)
    solved = active_mve_optimizer(
        ids=[str(i) for i in range(N)], alphas=model["alpha"], factor_exposures=model["B"],
        factor_covariance=model["F"], specific_risk=model["D"], benchmark_weights=np.full(N, 1 / N),
        target_active_risk=TARGET, betas=model["beta"], constraints=[CLASSES[c]() for c in constraints],
    )["active_weight"].to_numpy()
    assert np.corrcoef(h, solved)[0, 1] > 1 - 1e-8
    # The formula is the exact optimum: the solver can't earn more alpha per unit risk.
    solved_risk = np.sqrt((model["B"].T @ solved) @ model["F"] @ (model["B"].T @ solved) + model["D"] @ solved**2)
    assert model["alpha"] @ solved / solved_risk <= model["alpha"] @ h / risk * (1 + 1e-9)


def test_singular_factor_covariance(model):
    """The loader fills missing covariances with 0, so F can be singular."""
    model["F"][0, :] = model["F"][:, 0] = 0.0
    h, risk = closed_form(**model, target=TARGET)
    assert np.all(np.isfinite(h)) and risk == pytest.approx(TARGET, rel=1e-12)
