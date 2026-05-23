from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from capext.capacitance import augment_with_reference_node
from capext.mesh import SurfacePanel, mesh_net_surfaces
from capext.panel_integrals import (
    disk_potential_coefficient_multipole,
    equivalent_disk_radius,
    point_center_panel_coefficient,
)
from capext.problem import CapacitanceProblem
from capext.solvers.base import CapacitanceSolver


@dataclass(frozen=True)
class BEMResult:
    capacitance: np.ndarray
    net_names: tuple[str, ...]
    panel_count: int
    reference_net_index: int | None = None

    @property
    def reduced_capacitance(self) -> np.ndarray:
        if self.reference_net_index is None:
            return self.capacitance
        mask = np.ones(self.capacitance.shape[0], dtype=bool)
        mask[self.reference_net_index] = False
        return self.capacitance[np.ix_(mask, mask)]


class DenseBEMSolver(CapacitanceSolver):
    """Low-order dense collocation BEM for homogeneous free-space dielectric."""

    def __init__(
        self,
        *,
        max_panel_size: float = 20.0,
        symmetrize: bool = True,
        contact_tol: float = 1e-12,
        add_reference_node: bool = False,
        reference_name: str = "enclosure",
        point_approximation_ratio: float = 5.0,
        multipole_order: int = 24,
    ) -> None:
        self.max_panel_size = max_panel_size
        self.symmetrize = symmetrize
        self.contact_tol = contact_tol
        self.add_reference_node = add_reference_node
        self.reference_name = reference_name
        self.point_approximation_ratio = point_approximation_ratio
        self.multipole_order = multipole_order

    def solve(self, problem: CapacitanceProblem) -> BEMResult:
        nets = problem.nets(tol=self.contact_tol)
        panels = mesh_net_surfaces(
            nets,
            max_panel_size=self.max_panel_size,
            tol=self.contact_tol,
        )
        if not panels:
            raise ValueError("no surface panels were generated")

        influence = self._build_influence_matrix(
            panels,
            problem.epsilon,
            point_approximation_ratio=self.point_approximation_ratio,
            multipole_order=self.multipole_order,
        )
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

        net_names = tuple(net.name for net in nets)
        reference_net_index = None
        if self.add_reference_node:
            capacitance = augment_with_reference_node(capacitance)
            net_names = (*net_names, self.reference_name)
            reference_net_index = len(net_names) - 1

        return BEMResult(
            capacitance=capacitance,
            net_names=net_names,
            panel_count=len(panels),
            reference_net_index=reference_net_index,
        )

    def solve_matrix(self, problem: CapacitanceProblem) -> np.ndarray:
        return self.solve(problem).capacitance

    @staticmethod
    def _build_influence_matrix(
        panels: list[SurfacePanel],
        epsilon: float,
        *,
        point_approximation_ratio: float = 5.0,
        multipole_order: int = 24,
    ) -> np.ndarray:
        centers = np.asarray([panel.center for panel in panels], dtype=float)
        normals = np.asarray([panel.normal for panel in panels], dtype=float)
        areas = np.asarray([panel.area for panel in panels], dtype=float)
        radii = equivalent_disk_radius(areas)
        count = len(panels)
        matrix = np.empty((count, count), dtype=float)

        for i in range(count):
            delta = centers[i] - centers
            distance = np.linalg.norm(delta, axis=1)
            distance[i] = np.inf
            matrix[i, :] = point_center_panel_coefficient(areas, distance, epsilon)

            use_multipole = (distance > radii) & (distance <= point_approximation_ratio * radii)
            if np.any(use_multipole):
                cos_theta = np.einsum("ij,ij->i", delta[use_multipole], normals[use_multipole])
                cos_theta = cos_theta / distance[use_multipole]
                matrix[i, use_multipole] = disk_potential_coefficient_multipole(
                    distance[use_multipole],
                    cos_theta,
                    radii[use_multipole],
                    epsilon,
                    max_order=multipole_order,
                )

            matrix[i, i] = radii[i] / (2.0 * epsilon)
        return matrix
