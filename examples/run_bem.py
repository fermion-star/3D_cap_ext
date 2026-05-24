from __future__ import annotations

import numpy as np

from draw_geometry import count_bem_unknowns, draw, draw_discretization

from capext.problem import CapacitanceProblem
from capext.solvers import DenseBEMSolver


def main() -> None:
    max_panel_size = 5.0
    problem = CapacitanceProblem.from_boxes(
        [
            ("left", (100.0, 220.0, 220.0), (160.0, 280.0, 280.0)),
            ("right", (260.0, 220.0, 220.0), (320.0, 280.0, 280.0)),
            # This face-touching box is merged into the left net.
            ("left_extension", (160.0, 230.0, 230.0), (190.0, 270.0, 270.0)),
        ],
    )
    draw(problem, show=True)
    draw_discretization(problem, max_panel_size=max_panel_size, show=True)

    unknowns_without_outer_box = count_bem_unknowns(
        problem,
        max_panel_size=max_panel_size,
        include_physical_outer_box=False,
    )
    unknowns_with_outer_box = count_bem_unknowns(
        problem,
        max_panel_size=max_panel_size,
        include_physical_outer_box=True,
    )
    print("BEM unknowns without physical outer box:", unknowns_without_outer_box)
    print("BEM unknowns with meshed physical outer box:", unknowns_with_outer_box)
    print("Current reference-node enclosure adds no BEM unknowns.")

    solver = DenseBEMSolver(max_panel_size=max_panel_size, add_reference_node=True)
    result = solver.solve(problem)

    np.set_printoptions(precision=6, suppress=False)
    print("nets:", result.net_names)
    print("panel_count:", result.panel_count)
    print("capacitance matrix with enclosure/reference net [F]:")
    print(result.capacitance)
    print("row sums [F]:")
    print(result.capacitance.sum(axis=1))


if __name__ == "__main__":
    main()
