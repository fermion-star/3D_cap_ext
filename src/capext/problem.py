from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import EPSILON_0
from .geometry import AxisAlignedBox, BoxConductor, NetConductor, merge_touching_boxes


@dataclass(frozen=True)
class CapacitanceProblem:
    domain: AxisAlignedBox
    conductors: tuple[BoxConductor, ...]
    epsilon: float = EPSILON_0

    def __post_init__(self) -> None:
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        for conductor in self.conductors:
            if not self.domain.contains_closed(conductor.box.min_array):
                raise ValueError(f"{conductor.name} min corner is outside the domain")
            if not self.domain.contains_closed(conductor.box.max_array):
                raise ValueError(f"{conductor.name} max corner is outside the domain")

    @classmethod
    def from_boxes(
        cls,
        boxes: list[tuple[str, tuple[float, float, float], tuple[float, float, float]]],
        *,
        domain_min: tuple[float, float, float] = (0.0, 0.0, 0.0),
        domain_max: tuple[float, float, float] = (500.0, 500.0, 500.0),
        epsilon: float = EPSILON_0,
    ) -> CapacitanceProblem:
        conductors = tuple(
            BoxConductor(name=name, box=AxisAlignedBox(lo, hi))
            for name, lo, hi in boxes
        )
        return cls(
            domain=AxisAlignedBox(domain_min, domain_max),
            conductors=conductors,
            epsilon=epsilon,
        )

    def nets(self, *, tol: float = 1e-12) -> list[NetConductor]:
        return merge_touching_boxes(list(self.conductors), tol=tol)
