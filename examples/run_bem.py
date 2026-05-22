from __future__ import annotations

import numpy as np

from capext.problem import CapacitanceProblem
from capext.solvers import DenseBEMSolver


def main() -> None:
    problem = CapacitanceProblem.from_boxes(
        [
            ("left", (100.0, 220.0, 220.0), (160.0, 280.0, 280.0)),
            ("right", (260.0, 220.0, 220.0), (320.0, 280.0, 280.0)),
            # This face-touching box is merged into the left net.
            ("left_extension", (160.0, 230.0, 230.0), (190.0, 270.0, 270.0)),
        ],
    )
    solver = DenseBEMSolver(max_panel_size=20.0)
    result = solver.solve(problem)

    np.set_printoptions(precision=6, suppress=False)
    print("nets:", result.net_names)
    print("panel_count:", result.panel_count)
    print("capacitance matrix [F]:")
    print(result.capacitance)


if __name__ == "__main__":
    main()
