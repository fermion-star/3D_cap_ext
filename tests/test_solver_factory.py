import pytest
import numpy as np

from capext.problem import CapacitanceProblem
from capext.solvers import DenseBEMSolver, FRWSolver, create_solver
from capext.solvers.frw import CenteredCubeGreenSampler


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


def test_frw_solver_returns_stochastic_matrix_with_reference_node() -> None:
    problem = CapacitanceProblem.from_boxes(
        [
            ("a", (0, 0, 0), (1, 1, 1)),
            ("b", (3, 0, 0), (4, 1, 1)),
        ],
        domain_min=(-2, -2, -2),
        domain_max=(6, 3, 3),
    )
    solver = FRWSolver(
        samples_per_observation_net=200,
        seed=7,
        gaussian_padding=0.5,
        add_reference_node=True,
        max_steps_per_walk=1_000,
    )

    result = solver.solve(problem)

    assert result.net_names == ("a", "b", "enclosure")
    assert result.reference_net_index == 2
    assert result.capacitance.shape == (3, 3)
    assert result.standard_error.shape == (3, 3)
    assert result.statistics.completed_walks == 400
    assert result.statistics.escaped_walks == 0
    assert result.capacitance[0, 0] > 0
    assert result.capacitance[1, 1] > 0
    assert result.capacitance[0, 1] < 0
    assert result.capacitance[1, 0] < 0
    assert pytest.approx(result.capacitance.sum(axis=1), abs=1e-24) == [0.0, 0.0, 0.0]


def test_centered_cube_green_sampler_is_normalized_by_face_symmetry() -> None:
    sampler = CenteredCubeGreenSampler(grid_size=21, series_terms=31)

    assert np.isclose(np.sum(sampler.flat_probabilities), 1.0)
    cells_per_face = sampler.grid_size * sampler.grid_size
    face_probabilities = [
        float(np.sum(sampler.flat_probabilities[index * cells_per_face : (index + 1) * cells_per_face]))
        for index in range(6)
    ]

    assert np.allclose(face_probabilities, np.full(6, 1.0 / 6.0))
    assert sampler.cell_probabilities[sampler.grid_size // 2, sampler.grid_size // 2] > sampler.cell_probabilities[0, 0]
