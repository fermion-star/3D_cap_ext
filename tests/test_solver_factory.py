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
    assert result.statistics.total_walks == 400
    assert result.statistics.walks_per_observation_net == (200, 200)
    assert result.capacitance[0, 0] > 0
    assert result.capacitance[1, 1] > 0
    assert result.capacitance[0, 1] < 0
    assert result.capacitance[1, 0] < 0
    assert pytest.approx(result.capacitance.sum(axis=1), abs=1e-24) == [0.0, 0.0, 0.0]


def test_frw_uses_largest_transition_cube_by_default() -> None:
    solver = FRWSolver(samples_per_observation_net=10, min_samples_per_observation_net=2)

    assert solver.transition_safety == 1.0


def test_frw_outer_boundary_box_scales_with_geometry() -> None:
    problem = CapacitanceProblem.from_boxes(
        [
            ("a", (10, 20, 30), (20, 40, 50)),
            ("b", (80, 25, 35), (90, 45, 55)),
        ],
        domain_min=(0, 0, 0),
        domain_max=(100, 100, 100),
    )
    solver = FRWSolver(
        samples_per_observation_net=10,
        min_samples_per_observation_net=2,
        outer_box_scale=20.0,
    )

    outer = solver.outer_boundary_box(problem)

    assert np.allclose(outer.size, np.full(3, 20.0 * 80.0))
    assert np.allclose(outer.center, np.asarray((50.0, 32.5, 42.5)))


def test_frw_can_stop_by_estimated_error_before_max_walks() -> None:
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
        min_samples_per_observation_net=10,
        check_interval=5,
        absolute_tolerance=1.0,
        seed=7,
        gaussian_padding=0.5,
        max_steps_per_walk=1_000,
    )

    result = solver.solve(problem)

    assert result.statistics.walks_per_observation_net == (10, 10)
    assert result.statistics.total_walks == 20


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
