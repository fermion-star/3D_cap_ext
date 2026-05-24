from __future__ import annotations

import numpy as np

from draw_geometry import draw_frw_walks, draw_gaussian_surfaces

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
        samples_per_observation_net=20_000,
        min_samples_per_observation_net=200,
        check_interval=200,
        relative_tolerance=0.05,
        absolute_tolerance=1e-12,
        seed=2,
        gaussian_padding=10.0,
        add_reference_node=True,
        max_steps_per_walk=5_000,
    )

    outer = solver.outer_boundary_box(problem)
    print("FRW outer boundary min:", outer.min_corner)
    print("FRW outer boundary max:", outer.max_corner)
    print("FRW outer boundary size:", tuple(float(value) for value in outer.size))

    draw_gaussian_surfaces(problem, solver, show=True)
    result = solver.solve(problem)
    draw_frw_walks(problem, solver, result, show=True)

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
    print("walks per observation net:", result.statistics.walks_per_observation_net)
    print("completed walks:", result.statistics.completed_walks)
    print("escaped walks:", result.statistics.escaped_walks)
    print("total walks:", result.statistics.total_walks)
    for trace in result.representative_walks:
        hit_name = "reference" if trace.hit_net_index is None else result.net_names[trace.hit_net_index]
        print(
            "representative walk:",
            result.net_names[trace.observation_net_index],
            "->",
            hit_name,
            f"points={len(trace.points)}",
            f"escaped={trace.escaped}",
        )


if __name__ == "__main__":
    main()
