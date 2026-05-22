import numpy as np

from capext.problem import CapacitanceProblem
from capext.solvers import DenseBEMSolver


def test_two_separated_boxes_have_physical_sign_pattern() -> None:
    problem = CapacitanceProblem.from_boxes(
        [
            ("a", (0, 0, 0), (1, 1, 1)),
            ("b", (3, 0, 0), (4, 1, 1)),
        ],
        domain_min=(-2, -2, -2),
        domain_max=(6, 3, 3),
    )
    capacitance = DenseBEMSolver(max_panel_size=0.5).solve_matrix(problem)

    assert capacitance.shape == (2, 2)
    assert np.allclose(capacitance, capacitance.T)
    assert capacitance[0, 0] > 0
    assert capacitance[1, 1] > 0
    assert capacitance[0, 1] < 0


def test_overlapping_boxes_are_solved_as_one_net() -> None:
    problem = CapacitanceProblem.from_boxes(
        [
            ("a", (0, 0, 0), (2, 2, 2)),
            ("b", (1, 1, 1), (3, 3, 3)),
        ],
        domain_min=(-1, -1, -1),
        domain_max=(4, 4, 4),
    )
    result = DenseBEMSolver(max_panel_size=1.0).solve(problem)

    assert result.capacitance.shape == (1, 1)
    assert result.capacitance[0, 0] > 0


def test_reference_node_augmented_matrix_has_maxwell_properties() -> None:
    problem = CapacitanceProblem.from_boxes(
        [
            ("a", (0, 0, 0), (1, 1, 1)),
            ("b", (3, 0, 0), (4, 1, 1)),
        ],
        domain_min=(-2, -2, -2),
        domain_max=(6, 3, 3),
    )
    result = DenseBEMSolver(
        max_panel_size=0.5,
        add_reference_node=True,
        reference_name="enclosure",
    ).solve(problem)
    capacitance = result.capacitance

    assert result.net_names == ("a", "b", "enclosure")
    assert result.reference_net_index == 2
    assert capacitance.shape == (3, 3)
    assert np.allclose(capacitance, capacitance.T)
    assert np.allclose(capacitance.sum(axis=1), 0.0, atol=1e-24)
    assert np.all(np.diag(capacitance) > 0)
    assert np.all(capacitance[~np.eye(3, dtype=bool)] < 0)
    assert np.array_equal(result.reduced_capacitance, capacitance[:2, :2])
