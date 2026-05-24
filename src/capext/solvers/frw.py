from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from capext.problem import CapacitanceProblem
from capext.solvers.base import CapacitanceSolver


@dataclass(frozen=True)
class FRWStatistics:
    samples_per_observation_net: int
    seed: int | None
    relative_tolerance: float | None
    absolute_tolerance: float | None


@dataclass(frozen=True)
class FRWResult:
    capacitance: np.ndarray
    standard_error: np.ndarray
    net_names: tuple[str, ...]
    statistics: FRWStatistics
    reference_net_index: int | None = None


class FRWSolver(CapacitanceSolver):
    """Framework for a future Floating Random Walk capacitance solver."""

    def __init__(
        self,
        *,
        samples_per_observation_net: int = 10_000,
        seed: int | None = None,
        relative_tolerance: float | None = None,
        absolute_tolerance: float | None = None,
        add_reference_node: bool = False,
        reference_name: str = "enclosure",
    ) -> None:
        if samples_per_observation_net <= 0:
            raise ValueError("samples_per_observation_net must be positive")
        if relative_tolerance is not None and relative_tolerance <= 0:
            raise ValueError("relative_tolerance must be positive when provided")
        if absolute_tolerance is not None and absolute_tolerance <= 0:
            raise ValueError("absolute_tolerance must be positive when provided")

        self.samples_per_observation_net = samples_per_observation_net
        self.seed = seed
        self.relative_tolerance = relative_tolerance
        self.absolute_tolerance = absolute_tolerance
        self.add_reference_node = add_reference_node
        self.reference_name = reference_name
        self.rng = np.random.default_rng(seed)

    def statistics(self) -> FRWStatistics:
        return FRWStatistics(
            samples_per_observation_net=self.samples_per_observation_net,
            seed=self.seed,
            relative_tolerance=self.relative_tolerance,
            absolute_tolerance=self.absolute_tolerance,
        )

    def solve(self, problem: CapacitanceProblem) -> FRWResult:
        net_names = tuple(net.name for net in problem.nets())
        raise NotImplementedError(
            "FRWSolver is a framework placeholder. "
            f"Problem has {len(net_names)} merged conductor nets: {net_names}. "
            "The random-walk transition kernel and charge estimator still need implementation."
        )

    def solve_matrix(self, problem: CapacitanceProblem) -> np.ndarray:
        return self.solve(problem).capacitance
