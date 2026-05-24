import pytest

from capext.problem import CapacitanceProblem
from capext.solvers import DenseBEMSolver, FRWSolver, create_solver


def test_create_bem_solver() -> None:
    solver = create_solver("bem", max_panel_size=12.0)

    assert isinstance(solver, DenseBEMSolver)
    assert solver.max_panel_size == 12.0


def test_create_frw_solver() -> None:
    solver = create_solver("frw", samples_per_observation_net=123, seed=7)

    assert isinstance(solver, FRWSolver)
    assert solver.samples_per_observation_net == 123
    assert solver.statistics().seed == 7


def test_unknown_solver_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="expected 'bem' or 'frw'"):
        create_solver("fdtd")


def test_frw_solver_framework_is_not_claiming_results() -> None:
    problem = CapacitanceProblem.from_boxes(
        [
            ("a", (0, 0, 0), (1, 1, 1)),
            ("b", (3, 0, 0), (4, 1, 1)),
        ],
        domain_min=(-1, -1, -1),
        domain_max=(5, 2, 2),
    )
    solver = FRWSolver(samples_per_observation_net=10)

    with pytest.raises(NotImplementedError, match="transition kernel"):
        solver.solve_matrix(problem)
