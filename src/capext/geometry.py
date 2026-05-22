from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AxisAlignedBox:
    """Continuous-coordinate axis-aligned cuboid."""

    min_corner: tuple[float, float, float]
    max_corner: tuple[float, float, float]

    def __post_init__(self) -> None:
        lo = np.asarray(self.min_corner, dtype=float)
        hi = np.asarray(self.max_corner, dtype=float)
        if lo.shape != (3,) or hi.shape != (3,):
            raise ValueError("box corners must be 3D coordinates")
        if np.any(hi <= lo):
            raise ValueError(f"invalid box with min={self.min_corner}, max={self.max_corner}")

    @property
    def min_array(self) -> np.ndarray:
        return np.asarray(self.min_corner, dtype=float)

    @property
    def max_array(self) -> np.ndarray:
        return np.asarray(self.max_corner, dtype=float)

    @property
    def size(self) -> np.ndarray:
        return self.max_array - self.min_array

    @property
    def center(self) -> np.ndarray:
        return 0.5 * (self.min_array + self.max_array)

    def contains_closed(self, point: np.ndarray, tol: float = 1e-12) -> bool:
        p = np.asarray(point, dtype=float)
        return bool(np.all(p >= self.min_array - tol) and np.all(p <= self.max_array + tol))

    def contact_dimension(self, other: AxisAlignedBox, tol: float = 1e-12) -> int:
        """Return the dimension of the closed-set intersection, or -1 if disjoint."""

        lo = np.maximum(self.min_array, other.min_array)
        hi = np.minimum(self.max_array, other.max_array)
        overlap = hi - lo
        if np.any(overlap < -tol):
            return -1
        return int(np.count_nonzero(overlap > tol))

    def same_net_contact(self, other: AxisAlignedBox, tol: float = 1e-12) -> bool:
        """3D volume overlap or 2D face-area contact means the same electrical net."""

        return self.contact_dimension(other, tol=tol) >= 2


@dataclass(frozen=True)
class BoxConductor:
    name: str
    box: AxisAlignedBox


@dataclass(frozen=True)
class NetConductor:
    name: str
    boxes: tuple[BoxConductor, ...]


class _DisjointSet:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, a: int, b: int) -> None:
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a != root_b:
            self.parent[root_b] = root_a


def merge_touching_boxes(
    conductors: list[BoxConductor],
    *,
    tol: float = 1e-12,
) -> list[NetConductor]:
    """Merge boxes into nets when they have 2D or 3D contact."""

    dsu = _DisjointSet(len(conductors))
    for i, conductor_i in enumerate(conductors):
        for j in range(i + 1, len(conductors)):
            conductor_j = conductors[j]
            if conductor_i.box.same_net_contact(conductor_j.box, tol=tol):
                dsu.union(i, j)

    grouped: dict[int, list[BoxConductor]] = {}
    for i, conductor in enumerate(conductors):
        grouped.setdefault(dsu.find(i), []).append(conductor)

    nets = []
    for boxes in grouped.values():
        if len(boxes) == 1:
            name = boxes[0].name
        else:
            name = "+".join(box.name for box in boxes)
        nets.append(NetConductor(name=name, boxes=tuple(boxes)))
    return nets
