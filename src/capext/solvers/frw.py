from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from capext.capacitance import augment_with_reference_node
from capext.geometry import AxisAlignedBox, NetConductor
from capext.problem import CapacitanceProblem
from capext.solvers.base import CapacitanceSolver


@dataclass(frozen=True)
class FRWStatistics:
    samples_per_observation_net: int
    seed: int | None
    relative_tolerance: float | None
    absolute_tolerance: float | None
    gaussian_padding: float
    transition_safety: float
    transition_grid_size: int
    transition_series_terms: int
    hit_tolerance: float
    max_steps_per_walk: int
    completed_walks: int
    escaped_walks: int


@dataclass(frozen=True)
class FRWResult:
    capacitance: np.ndarray
    standard_error: np.ndarray
    net_names: tuple[str, ...]
    statistics: FRWStatistics
    reference_net_index: int | None = None

    @property
    def reduced_capacitance(self) -> np.ndarray:
        if self.reference_net_index is None:
            return self.capacitance
        mask = np.ones(self.capacitance.shape[0], dtype=bool)
        mask[self.reference_net_index] = False
        return self.capacitance[np.ix_(mask, mask)]


class FRWSolver(CapacitanceSolver):
    """Prototype Floating Random Walk solver with cubic transition domains.

    This is a first executable FRW estimator, not yet an RWCap-grade solver. It
    uses a uniformly sampled axis-aligned Gaussian box and a discretized
    centered-cube surface Green-function table for transition-cube exits.
    """

    def __init__(
        self,
        *,
        samples_per_observation_net: int = 10_000,
        seed: int | None = None,
        relative_tolerance: float | None = None,
        absolute_tolerance: float | None = None,
        add_reference_node: bool = False,
        reference_name: str = "enclosure",
        gaussian_padding: float = 5.0,
        transition_safety: float = 0.95,
        transition_grid_size: int = 31,
        transition_series_terms: int = 41,
        hit_tolerance: float = 1e-6,
        max_steps_per_walk: int = 10_000,
        symmetrize: bool = True,
    ) -> None:
        if samples_per_observation_net <= 0:
            raise ValueError("samples_per_observation_net must be positive")
        if relative_tolerance is not None and relative_tolerance <= 0:
            raise ValueError("relative_tolerance must be positive when provided")
        if absolute_tolerance is not None and absolute_tolerance <= 0:
            raise ValueError("absolute_tolerance must be positive when provided")
        if gaussian_padding <= 0:
            raise ValueError("gaussian_padding must be positive")
        if not 0 < transition_safety < 1:
            raise ValueError("transition_safety must be between 0 and 1")
        if transition_grid_size <= 0:
            raise ValueError("transition_grid_size must be positive")
        if transition_series_terms <= 0:
            raise ValueError("transition_series_terms must be positive")
        if hit_tolerance <= 0:
            raise ValueError("hit_tolerance must be positive")
        if max_steps_per_walk <= 0:
            raise ValueError("max_steps_per_walk must be positive")

        self.samples_per_observation_net = samples_per_observation_net
        self.seed = seed
        self.relative_tolerance = relative_tolerance
        self.absolute_tolerance = absolute_tolerance
        self.add_reference_node = add_reference_node
        self.reference_name = reference_name
        self.gaussian_padding = gaussian_padding
        self.transition_safety = transition_safety
        self.transition_grid_size = transition_grid_size
        self.transition_series_terms = transition_series_terms
        self.hit_tolerance = hit_tolerance
        self.max_steps_per_walk = max_steps_per_walk
        self.symmetrize = symmetrize
        self.rng = np.random.default_rng(seed)
        self._transition_sampler = CenteredCubeGreenSampler(
            grid_size=transition_grid_size,
            series_terms=transition_series_terms,
        )
        self._completed_walks = 0
        self._escaped_walks = 0

    def statistics(self) -> FRWStatistics:
        return FRWStatistics(
            samples_per_observation_net=self.samples_per_observation_net,
            seed=self.seed,
            relative_tolerance=self.relative_tolerance,
            absolute_tolerance=self.absolute_tolerance,
            gaussian_padding=self.gaussian_padding,
            transition_safety=self.transition_safety,
            transition_grid_size=self.transition_grid_size,
            transition_series_terms=self.transition_series_terms,
            hit_tolerance=self.hit_tolerance,
            max_steps_per_walk=self.max_steps_per_walk,
            completed_walks=self._completed_walks,
            escaped_walks=self._escaped_walks,
        )

    def solve(self, problem: CapacitanceProblem) -> FRWResult:
        nets = problem.nets()
        net_count = len(nets)
        net_names = tuple(net.name for net in nets)
        reduced = np.zeros((net_count, net_count), dtype=float)
        standard_error = np.zeros((net_count, net_count), dtype=float)
        self._completed_walks = 0
        self._escaped_walks = 0

        for observation_net_index, net in enumerate(nets):
            gaussian_box = self._gaussian_box(problem, net)
            gaussian_area = _box_surface_area(gaussian_box)
            flux_weight = problem.epsilon * gaussian_area / self.gaussian_padding
            samples = np.zeros((self.samples_per_observation_net, net_count), dtype=float)

            for sample_index in range(self.samples_per_observation_net):
                start = _sample_box_surface(gaussian_box, self.rng)
                hit_net = self._walk_to_boundary(start, problem, nets)
                if hit_net is not None:
                    samples[sample_index, hit_net] -= flux_weight
                samples[sample_index, observation_net_index] += flux_weight

            reduced[observation_net_index, :] = np.mean(samples, axis=0)
            if self.samples_per_observation_net > 1:
                standard_error[observation_net_index, :] = np.std(samples, axis=0, ddof=1) / np.sqrt(
                    self.samples_per_observation_net
                )

        if self.symmetrize:
            reduced = 0.5 * (reduced + reduced.T)
            standard_error = 0.5 * (standard_error + standard_error.T)

        capacitance = reduced
        reference_net_index = None
        if self.add_reference_node:
            capacitance = augment_with_reference_node(reduced)
            standard_error = _augment_standard_error_shape(standard_error)
            net_names = (*net_names, self.reference_name)
            reference_net_index = len(net_names) - 1

        return FRWResult(
            capacitance=capacitance,
            standard_error=standard_error,
            net_names=net_names,
            statistics=self.statistics(),
            reference_net_index=reference_net_index,
        )

    def solve_matrix(self, problem: CapacitanceProblem) -> np.ndarray:
        return self.solve(problem).capacitance

    def _gaussian_box(self, problem: CapacitanceProblem, net: NetConductor) -> AxisAlignedBox:
        mins = np.min(np.vstack([conductor.box.min_array for conductor in net.boxes]), axis=0)
        maxs = np.max(np.vstack([conductor.box.max_array for conductor in net.boxes]), axis=0)
        lo = mins - self.gaussian_padding
        hi = maxs + self.gaussian_padding
        domain_lo = problem.domain.min_array
        domain_hi = problem.domain.max_array
        if np.any(lo <= domain_lo) or np.any(hi >= domain_hi):
            raise ValueError(
                f"gaussian_padding={self.gaussian_padding} places the Gaussian box outside the domain"
            )
        return AxisAlignedBox(tuple(lo), tuple(hi))

    def _walk_to_boundary(
        self,
        start: np.ndarray,
        problem: CapacitanceProblem,
        nets: list[NetConductor],
    ) -> int | None:
        point = np.asarray(start, dtype=float)

        for _ in range(self.max_steps_per_walk):
            hit_net = _containing_net(point, nets, tol=self.hit_tolerance)
            if hit_net is not None:
                self._completed_walks += 1
                return hit_net

            domain_distance = _distance_to_domain_boundary_linf(point, problem.domain)
            conductor_distance, nearest_net = _nearest_conductor_linf(point, nets)
            nearest_distance = min(domain_distance, conductor_distance)

            if nearest_distance <= self.hit_tolerance:
                self._completed_walks += 1
                if conductor_distance <= domain_distance:
                    return nearest_net
                return None

            half_size = self.transition_safety * nearest_distance
            point = self._transition_sampler.sample(point, half_size, self.rng)

        self._escaped_walks += 1
        return None


def _augment_standard_error_shape(standard_error: np.ndarray) -> np.ndarray:
    augmented = np.full((standard_error.shape[0] + 1, standard_error.shape[1] + 1), np.nan)
    augmented[: standard_error.shape[0], : standard_error.shape[1]] = standard_error
    return augmented


def _box_surface_area(box: AxisAlignedBox) -> float:
    dx, dy, dz = box.size
    return float(2.0 * (dx * dy + dx * dz + dy * dz))


def _sample_box_surface(box: AxisAlignedBox, rng: np.random.Generator) -> np.ndarray:
    lo = box.min_array
    hi = box.max_array
    dx, dy, dz = box.size
    face_areas = np.asarray([dy * dz, dy * dz, dx * dz, dx * dz, dx * dy, dx * dy], dtype=float)
    face = int(rng.choice(6, p=face_areas / np.sum(face_areas)))
    u = rng.random(2)
    point = np.empty(3, dtype=float)
    if face == 0:
        point[0] = lo[0]
        point[1] = lo[1] + u[0] * dy
        point[2] = lo[2] + u[1] * dz
    elif face == 1:
        point[0] = hi[0]
        point[1] = lo[1] + u[0] * dy
        point[2] = lo[2] + u[1] * dz
    elif face == 2:
        point[0] = lo[0] + u[0] * dx
        point[1] = lo[1]
        point[2] = lo[2] + u[1] * dz
    elif face == 3:
        point[0] = lo[0] + u[0] * dx
        point[1] = hi[1]
        point[2] = lo[2] + u[1] * dz
    elif face == 4:
        point[0] = lo[0] + u[0] * dx
        point[1] = lo[1] + u[1] * dy
        point[2] = lo[2]
    else:
        point[0] = lo[0] + u[0] * dx
        point[1] = lo[1] + u[1] * dy
        point[2] = hi[2]
    return point


class CenteredCubeGreenSampler:
    """Discrete surface Green-function sampler for a centered homogeneous cube."""

    def __init__(self, *, grid_size: int = 31, series_terms: int = 41) -> None:
        self.grid_size = grid_size
        self.series_terms = series_terms
        self.cell_probabilities = _centered_cube_face_probabilities(grid_size, series_terms)
        self.flat_probabilities = np.tile(self.cell_probabilities.ravel(), 6)
        self.flat_probabilities = self.flat_probabilities / np.sum(self.flat_probabilities)

    def sample(self, center: np.ndarray, half_size: float, rng: np.random.Generator) -> np.ndarray:
        flat_index = int(rng.choice(self.flat_probabilities.size, p=self.flat_probabilities))
        cells_per_face = self.grid_size * self.grid_size
        face = flat_index // cells_per_face
        cell_index = flat_index % cells_per_face
        i = cell_index // self.grid_size
        j = cell_index % self.grid_size

        u = (i + rng.random()) / self.grid_size
        v = (j + rng.random()) / self.grid_size
        local_u = -half_size + 2.0 * half_size * u
        local_v = -half_size + 2.0 * half_size * v
        point = np.asarray(center, dtype=float).copy()

        if face == 0:
            point[0] -= half_size
            point[1] += local_u
            point[2] += local_v
        elif face == 1:
            point[0] += half_size
            point[1] += local_u
            point[2] += local_v
        elif face == 2:
            point[1] -= half_size
            point[0] += local_u
            point[2] += local_v
        elif face == 3:
            point[1] += half_size
            point[0] += local_u
            point[2] += local_v
        elif face == 4:
            point[2] -= half_size
            point[0] += local_u
            point[1] += local_v
        else:
            point[2] += half_size
            point[0] += local_u
            point[1] += local_v
        return point


def _centered_cube_face_probabilities(grid_size: int, series_terms: int) -> np.ndarray:
    coords = (np.arange(grid_size, dtype=float) + 0.5) / grid_size
    x, y = np.meshgrid(coords, coords, indexing="ij")
    density = np.zeros((grid_size, grid_size), dtype=float)

    for nx in range(1, series_terms + 1):
        sx = np.sin(np.pi * nx / 2.0)
        if abs(sx) < 1e-15:
            continue
        for ny in range(1, series_terms + 1):
            sy = np.sin(np.pi * ny / 2.0)
            if abs(sy) < 1e-15:
                continue
            nz = np.sqrt(nx * nx + ny * ny)
            scaled_nz = np.pi * nz
            density += (
                4.0
                * sx
                * sy
                * np.sinh(scaled_nz / 2.0)
                / np.sinh(scaled_nz)
                * np.sin(np.pi * nx * x)
                * np.sin(np.pi * ny * y)
            )

    density = np.maximum(density, 0.0)
    probabilities = density / (grid_size * grid_size)
    return probabilities / (6.0 * np.sum(probabilities))


def _containing_net(point: np.ndarray, nets: list[NetConductor], tol: float) -> int | None:
    for net_index, net in enumerate(nets):
        if any(conductor.box.contains_closed(point, tol=tol) for conductor in net.boxes):
            return net_index
    return None


def _distance_to_domain_boundary_linf(point: np.ndarray, domain: AxisAlignedBox) -> float:
    return float(np.min(np.minimum(point - domain.min_array, domain.max_array - point)))


def _nearest_conductor_linf(point: np.ndarray, nets: list[NetConductor]) -> tuple[float, int]:
    best_distance = np.inf
    best_net = -1
    for net_index, net in enumerate(nets):
        for conductor in net.boxes:
            distance = _linf_distance_to_box(point, conductor.box)
            if distance < best_distance:
                best_distance = distance
                best_net = net_index
    return float(best_distance), best_net


def _linf_distance_to_box(point: np.ndarray, box: AxisAlignedBox) -> float:
    lower_gap = np.maximum(box.min_array - point, 0.0)
    upper_gap = np.maximum(point - box.max_array, 0.0)
    return float(np.max(lower_gap + upper_gap))
