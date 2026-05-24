from __future__ import annotations

from typing import Literal

from capext.solvers.base import CapacitanceSolver
from capext.solvers.bem import DenseBEMSolver
from capext.solvers.frw import FRWSolver

SolverName = Literal["bem", "frw"]


def create_solver(name: SolverName | str, **kwargs) -> CapacitanceSolver:
    normalized = name.lower().replace("-", "_")
    if normalized in {"bem", "dense_bem"}:
        return DenseBEMSolver(**kwargs)
    if normalized in {"frw", "floating_random_walk"}:
        return FRWSolver(**kwargs)
    raise ValueError(f"unknown solver {name!r}; expected 'bem' or 'frw'")
