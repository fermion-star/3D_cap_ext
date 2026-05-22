from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from capext.problem import CapacitanceProblem


class CapacitanceSolver(ABC):
    @abstractmethod
    def solve_matrix(self, problem: CapacitanceProblem) -> np.ndarray:
        """Return C where Q = C V for merged conductor nets."""
