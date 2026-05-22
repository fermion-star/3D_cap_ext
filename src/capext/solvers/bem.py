from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from capext.mesh import SurfacePanel, mesh_net_surfaces
from capext.problem import CapacitanceProblem
from capext.solvers.base import CapacitanceSolver


@dataclass(frozen=True)
class BEMResult:
    capacitance: np.ndarray
    net_names: tuple[str, ...]
    panel_count: int


class DenseBEMSolver(CapacitanceSolver):
    """Low-order dense collocation BEM for homogeneous free-space dielectric."""

    def __init__(
        self,
        *,
        max_panel_size: float = 20.0,
        symmetrize: bool = True,
        contact_tol: float = 1e-12,
    ) -> None:
        self.max_panel_size = max_panel_size
        self.symmetrize = symmetrize
        self.contact_tol = contact_tol

    def solve(self, problem: CapacitanceProblem) -> BEMResult:
        nets = problem.nets(tol=self.contact_tol)
        panels = mesh_net_surfaces(
            nets,
            max_panel_size=self.max_panel_size,
            tol=self.contact_tol,
        )
        if not panels:
            raise ValueError("no surface panels were generated")

        influence = self._build_influence_matrix(panels, problem.epsilon)
        panel_net_indices = np.asarray([panel.net_index for panel in panels], dtype=int)
        panel_areas = np.asarray([panel.area for panel in panels], dtype=float)

        net_count = len(nets)
        capacitance = np.zeros((net_count, net_count), dtype=float)
        for driven_net in range(net_count):
            potentials = (panel_net_indices == driven_net).astype(float)
            sigma = np.linalg.solve(influence, potentials)
            for net_index in range(net_count):
                mask = panel_net_indices == net_index
                capacitance[net_index, driven_net] = float(np.sum(sigma[mask] * panel_areas[mask]))

        if self.symmetrize:
            capacitance = 0.5 * (capacitance + capacitance.T)

        return BEMResult(
            capacitance=capacitance,
            net_names=tuple(net.name for net in nets),
            panel_count=len(panels),
        )

    def solve_matrix(self, problem: CapacitanceProblem) -> np.ndarray:
        return self.solve(problem).capacitance

    @staticmethod
    def _build_influence_matrix(panels: list[SurfacePanel], epsilon: float) -> np.ndarray:
        centers = np.asarray([panel.center for panel in panels], dtype=float)
        areas = np.asarray([panel.area for panel in panels], dtype=float)
        count = len(panels)
        matrix = np.empty((count, count), dtype=float)
        coefficient = 1.0 / (4.0 * np.pi * epsilon)

        for i in range(count):
            delta = centers[i] - centers
            distance = np.linalg.norm(delta, axis=1)
            distance[i] = np.inf
            matrix[i, :] = coefficient * areas / distance
            equivalent_radius = np.sqrt(areas[i] / np.pi)
            matrix[i, i] = equivalent_radius / (2.0 * epsilon)
        return matrix
