from __future__ import annotations

import numpy as np

from draw_run_bem_geometry import draw_gaussian_surfaces

from capext.problem import CapacitanceProblem
from capext.solvers import FRWSolver


def main() -> None:
    problem = CapacitanceProblem.from_boxes(
        [
            ("left", (100.0, 220.0, 220.0), (160.0, 280.0, 280.0)),
            ("right", (260.0, 220.0, 220.0), (320.0, 280.0, 280.0)),
            # This face-touching box is merged into the left net.
            ("left_extension", (160.0, 230.0, 230.0), (190.0, 270.0, 270.0)),
        ],
    )
    solver = FRWSolver(
        samples_per_observation_net=2_000,
        seed=1,
        gaussian_padding=10.0,
        add_reference_node=True,
        max_steps_per_walk=5_000,
    )

    draw_gaussian_surfaces(problem, solver, show=True)
    result = solver.solve(problem)

    np.set_printoptions(precision=6, suppress=False)
    print("nets:", result.net_names)
    print("capacitance matrix estimate [F]:")
    print(result.capacitance)
    print("standard error estimate [F]:")
    print(result.standard_error)
    print("row sums [F]:")
    print(result.capacitance.sum(axis=1))
    print("statistics:")
    print(result.statistics)


if __name__ == "__main__":
    main()
