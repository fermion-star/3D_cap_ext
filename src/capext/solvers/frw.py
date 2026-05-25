from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from capext.capacitance import augment_with_reference_node
from capext.geometry import AxisAlignedBox, NetConductor
from capext.problem import CapacitanceProblem
from capext.solvers.base import CapacitanceSolver


@dataclass(frozen=True)
class FRWStatistics:
    """Run metadata needed to judge stochastic convergence and reproducibility."""

    max_samples_per_observation_net: int
    min_samples_per_observation_net: int
    check_interval: int
    seed: int | None
    relative_tolerance: float | None
    absolute_tolerance: float | None
    gaussian_padding: float
    outer_box_scale: float
    transition_safety: float
    transition_grid_size: int
    transition_series_terms: int
    hit_tolerance: float
    max_steps_per_walk: int
    walks_per_observation_net: tuple[int, ...]
    completed_walks: int
    escaped_walks: int

    @property
    def total_walks(self) -> int:
        return self.completed_walks + self.escaped_walks


@dataclass(frozen=True)
class FRWWalkTrace:
    """One representative walk path for visualization and debugging."""

    observation_net_index: int
    hit_net_index: int | None
    points: np.ndarray
    transition_boxes: tuple[AxisAlignedBox, ...]
    escaped: bool


@dataclass(frozen=True)
class FRWResult:
    """FRW capacitance estimate plus uncertainty and diagnostic data."""

    capacitance: np.ndarray
    standard_error: np.ndarray
    net_names: tuple[str, ...]
    statistics: FRWStatistics
    representative_walks: tuple[FRWWalkTrace, ...] = ()
    reference_net_index: int | None = None

    @property
    def reduced_capacitance(self) -> np.ndarray:
        """Return the physical-net block, dropping the analytic reference node."""

        if self.reference_net_index is None:
            return self.capacitance
        mask = np.ones(self.capacitance.shape[0], dtype=bool)
        mask[self.reference_net_index] = False
        return self.capacitance[np.ix_(mask, mask)]


class FRWSolver(CapacitanceSolver):
    """Prototype Floating Random Walk solver with cubic transition domains.

    This is a first executable FRW estimator, not yet an RWCap-grade solver. It
    uses a uniformly sampled axis-aligned Gaussian box, a closed-form first-step
    omega weight, and a discretized centered-cube surface Green-function table
    for transition-cube exits.
    """

    def __init__(
        self,
        *,
        samples_per_observation_net: int = 10_000,
        min_samples_per_observation_net: int = 30,
        check_interval: int = 100,
        seed: int | None = None,
        relative_tolerance: float | None = None,
        absolute_tolerance: float | None = None,
        add_reference_node: bool = False,
        reference_name: str = "enclosure",
        gaussian_padding: float = 5.0,
        outer_box_scale: float = 20.0,
        transition_safety: float = 1.0,
        transition_grid_size: int = 31,
        transition_series_terms: int = 41,
        hit_tolerance: float = 1e-6,
        max_steps_per_walk: int = 10_000,
        symmetrize: bool = False,  # yifan set to false until the whole thing is perfect
    ) -> None:
        if samples_per_observation_net <= 0:
            raise ValueError("samples_per_observation_net must be positive")
        if min_samples_per_observation_net <= 1:
            raise ValueError("min_samples_per_observation_net must be greater than 1")
        if min_samples_per_observation_net > samples_per_observation_net:
            raise ValueError("min_samples_per_observation_net must not exceed samples_per_observation_net")
        if check_interval <= 0:
            raise ValueError("check_interval must be positive")
        if relative_tolerance is not None and relative_tolerance <= 0:
            raise ValueError("relative_tolerance must be positive when provided")
        if absolute_tolerance is not None and absolute_tolerance <= 0:
            raise ValueError("absolute_tolerance must be positive when provided")
        if gaussian_padding <= 0:
            raise ValueError("gaussian_padding must be positive")
        if outer_box_scale <= 1:
            raise ValueError("outer_box_scale must be greater than 1")
        if not 0 < transition_safety <= 1:
            raise ValueError("transition_safety must be in (0, 1]")
        if transition_grid_size <= 0:
            raise ValueError("transition_grid_size must be positive")
        if transition_series_terms <= 0:
            raise ValueError("transition_series_terms must be positive")
        if hit_tolerance <= 0:
            raise ValueError("hit_tolerance must be positive")
        if max_steps_per_walk <= 0:
            raise ValueError("max_steps_per_walk must be positive")

        self.samples_per_observation_net = samples_per_observation_net
        self.min_samples_per_observation_net = min_samples_per_observation_net
        self.check_interval = check_interval
        self.seed = seed
        self.relative_tolerance = relative_tolerance
        self.absolute_tolerance = absolute_tolerance
        self.add_reference_node = add_reference_node
        self.reference_name = reference_name
        self.gaussian_padding = gaussian_padding
        self.outer_box_scale = outer_box_scale
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
        self._walks_per_observation_net: tuple[int, ...] = ()

    def statistics(self) -> FRWStatistics:
        """Return statistics from the most recent solve."""

        return FRWStatistics(
            max_samples_per_observation_net=self.samples_per_observation_net,
            min_samples_per_observation_net=self.min_samples_per_observation_net,
            check_interval=self.check_interval,
            seed=self.seed,
            relative_tolerance=self.relative_tolerance,
            absolute_tolerance=self.absolute_tolerance,
            gaussian_padding=self.gaussian_padding,
            outer_box_scale=self.outer_box_scale,
            transition_safety=self.transition_safety,
            transition_grid_size=self.transition_grid_size,
            transition_series_terms=self.transition_series_terms,
            hit_tolerance=self.hit_tolerance,
            max_steps_per_walk=self.max_steps_per_walk,
            walks_per_observation_net=self._walks_per_observation_net,
            completed_walks=self._completed_walks,
            escaped_walks=self._escaped_walks,
        )

    def solve(self, problem: CapacitanceProblem) -> FRWResult:
        """Estimate the capacitance matrix row by row using FRW samples.

        Each observation row starts walks on a Gaussian box around one net. The
        first cube uses the omega field weight, and the remaining walk estimates
        the potential by terminal conductor hits.
        """

        nets = problem.nets()
        net_count = len(nets)
        net_names = tuple(net.name for net in nets)
        reduced = np.zeros((net_count, net_count), dtype=float)
        standard_error = np.zeros((net_count, net_count), dtype=float)
        representative_walks: list[FRWWalkTrace] = []
        walks_per_observation_net = []
        self._completed_walks = 0
        self._escaped_walks = 0
        outer_boundary = self.outer_boundary_box(problem)

        for observation_net_index, net in enumerate(nets):
            # A Gaussian surface converts charge on this net into a flux
            # integral in the dielectric, avoiding direct starts on metal.
            gaussian_box = self.gaussian_box(problem, net)
            gaussian_area = _box_surface_area(gaussian_box)
            samples = []

            for sample_index in range(self.samples_per_observation_net):
                surface_sample = _sample_box_surface_with_normal(gaussian_box, self.rng)
                start = surface_sample.point
                record_trace = sample_index == 0

                # First FRW step: sample the transition cube and apply the
                # capacitance omega weight
                #   epsilon * A_G * (-grad_r G_phi . n_G) / proposal_density.
                first_step = self._sample_transition(
                    start,
                    outer_boundary,
                    nets,
                )
                omega = (
                    problem.epsilon
                    * gaussian_area
                    * float(np.dot(first_step.negative_green_gradient, surface_sample.normal))
                    / first_step.proposal_density
                ) # ref 00 equ 2.2.5

                # Continuation walk: after the weighted field step, ordinary
                # potential walks only need the terminal conductor identity.
                walk = self._walk_to_boundary(
                    first_step.point,
                    outer_boundary,
                    nets,
                    record_trace=record_trace,
                    initial_points=(start, first_step.point),
                    initial_transition_boxes=(_cube_box(start, first_step.half_size),),
                )
                hit_net = walk.hit_net_index
                if record_trace:
                    representative_walks.append(
                        FRWWalkTrace(
                            observation_net_index=observation_net_index,
                            hit_net_index=walk.hit_net_index,
                            points=np.asarray(walk.points, dtype=float),
                            transition_boxes=walk.transition_boxes,
                            escaped=walk.escaped,
                        )
                    )
                sample = np.zeros(net_count, dtype=float)
                if hit_net is not None:
                    # Boundary potentials are indicators under unit excitation,
                    # so a hit on h contributes omega to C[i, h].
                    sample[hit_net] += omega
                samples.append(sample)

                current_samples = np.asarray(samples)
                if self._can_stop_sampling(current_samples):
                    break

            sample_matrix = np.asarray(samples)
            walks_per_observation_net.append(sample_matrix.shape[0])
            reduced[observation_net_index, :] = np.mean(samples, axis=0)
            standard_error[observation_net_index, :] = _standard_error(sample_matrix)

        self._walks_per_observation_net = tuple(walks_per_observation_net)

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
            representative_walks=tuple(representative_walks),
            reference_net_index=reference_net_index,
        )

    def solve_matrix(self, problem: CapacitanceProblem) -> np.ndarray:
        """Compatibility wrapper returning only the capacitance matrix."""

        return self.solve(problem).capacitance

    def _can_stop_sampling(self, samples: np.ndarray) -> bool:
        """Decide whether the current row has met the configured error target."""

        sample_count = samples.shape[0]
        if sample_count < self.min_samples_per_observation_net:
            return False
        if sample_count == self.samples_per_observation_net:
            return True
        if self.relative_tolerance is None and self.absolute_tolerance is None:
            return False
        if sample_count % self.check_interval != 0:
            return False

        mean = np.mean(samples, axis=0)
        standard_error = _standard_error(samples)
        target = np.zeros_like(mean)
        if self.absolute_tolerance is not None:
            target = np.maximum(target, self.absolute_tolerance)
        if self.relative_tolerance is not None:
            target = np.maximum(target, self.relative_tolerance * np.abs(mean))
        return bool(np.all(standard_error <= target))

    def gaussian_box(self, problem: CapacitanceProblem, net: NetConductor) -> AxisAlignedBox:
        """Build the axis-aligned Gaussian box around one merged net."""

        mins = np.min(np.vstack([conductor.box.min_array for conductor in net.boxes]), axis=0)
        maxs = np.max(np.vstack([conductor.box.max_array for conductor in net.boxes]), axis=0)
        lo = mins - self.gaussian_padding
        hi = maxs + self.gaussian_padding
        outer = self.outer_boundary_box(problem)
        outer_lo = outer.min_array
        outer_hi = outer.max_array
        if np.any(lo <= outer_lo) or np.any(hi >= outer_hi):
            raise ValueError(
                f"gaussian_padding={self.gaussian_padding} places the Gaussian box outside the FRW outer boundary"
            )
        return AxisAlignedBox(tuple(lo), tuple(hi))

    def outer_boundary_box(self, problem: CapacitanceProblem) -> AxisAlignedBox:
        """Construct the absorbing reference boundary used by FRW walks."""

        boxes = [conductor.box for conductor in problem.conductors]
        mins = np.min(np.vstack([box.min_array for box in boxes]), axis=0)
        maxs = np.max(np.vstack([box.max_array for box in boxes]), axis=0)
        center = 0.5 * (mins + maxs)
        max_dimension = float(np.max(maxs - mins))
        half_size = 0.5 * self.outer_box_scale * max_dimension
        lo = center - half_size
        hi = center + half_size
        return AxisAlignedBox(tuple(lo), tuple(hi))

    def _walk_to_boundary(
        self,
        start: np.ndarray,
        outer_boundary: AxisAlignedBox,
        nets: list[NetConductor],
        *,
        record_trace: bool = False,
        initial_points: tuple[np.ndarray, ...] = (),
        initial_transition_boxes: tuple[AxisAlignedBox, ...] = (),
    ) -> _WalkOutcome:
        """Run the unweighted potential walk until metal or reference boundary.

        The optional initial points/boxes let the representative trace include
        the separately weighted first transition cube.
        """

        point = np.asarray(start, dtype=float)
        points = [np.asarray(initial_point, dtype=float).copy() for initial_point in initial_points] if record_trace else []
        transition_boxes = list(initial_transition_boxes) if record_trace else []
        if record_trace and not points:
            points.append(point.copy())

        for _ in range(self.max_steps_per_walk):
            hit_net = _containing_net(point, nets, tol=self.hit_tolerance)
            if hit_net is not None:
                self._completed_walks += 1
                return _WalkOutcome(hit_net, tuple(points), tuple(transition_boxes), escaped=False)

            domain_distance = _distance_to_domain_boundary_linf(point, outer_boundary)
            conductor_distance, nearest_net = _nearest_conductor_linf(point, nets)
            nearest_distance = min(domain_distance, conductor_distance)

            if nearest_distance <= self.hit_tolerance:
                self._completed_walks += 1
                if conductor_distance <= domain_distance:
                    return _WalkOutcome(nearest_net, tuple(points), tuple(transition_boxes), escaped=False)
                return _WalkOutcome(None, tuple(points), tuple(transition_boxes), escaped=False)

            half_size = self.transition_safety * nearest_distance
            if record_trace:
                transition_boxes.append(_cube_box(point, half_size))
            point = self._transition_sampler.sample(point, half_size, self.rng)
            if record_trace:
                points.append(point.copy())

        self._escaped_walks += 1
        return _WalkOutcome(None, tuple(points), tuple(transition_boxes), escaped=True)

    def _sample_transition(
        self,
        point: np.ndarray,
        outer_boundary: AxisAlignedBox,
        nets: list[NetConductor],
    ) -> _TransitionSample:
        """Sample one centered transition cube and return data for omega."""

        half_size = self._transition_half_size(point, outer_boundary, nets)
        return self._transition_sampler.sample_with_weight_data(point, half_size, self.rng)

    def _transition_half_size(
        self,
        point: np.ndarray,
        outer_boundary: AxisAlignedBox,
        nets: list[NetConductor],
    ) -> float:
        """Largest allowed centered cube half-size at a walk point."""

        domain_distance = _distance_to_domain_boundary_linf(point, outer_boundary)
        conductor_distance, _ = _nearest_conductor_linf(point, nets)
        nearest_distance = min(domain_distance, conductor_distance)
        if nearest_distance <= self.hit_tolerance:
            return self.hit_tolerance
        return self.transition_safety * nearest_distance


@dataclass(frozen=True)
class _WalkOutcome:
    """Internal terminal state of one potential walk."""

    hit_net_index: int | None
    points: tuple[np.ndarray, ...]
    transition_boxes: tuple[AxisAlignedBox, ...]
    escaped: bool


@dataclass(frozen=True)
class _SurfaceSample:
    """Point sampled on a box surface together with its outward normal."""

    point: np.ndarray
    normal: np.ndarray


@dataclass(frozen=True)
class _TransitionSample:
    """One cube exit sample plus the quantities needed by omega."""

    point: np.ndarray
    half_size: float
    proposal_density: float
    negative_green_gradient: np.ndarray
    face: int


def _augment_standard_error_shape(standard_error: np.ndarray) -> np.ndarray:
    """Add an all-NaN reference row/column to match the augmented matrix shape."""

    augmented = np.full((standard_error.shape[0] + 1, standard_error.shape[1] + 1), np.nan)
    augmented[: standard_error.shape[0], : standard_error.shape[1]] = standard_error
    return augmented


def _standard_error(samples: np.ndarray) -> np.ndarray:
    """Independent-sample standard error for each row entry."""

    if samples.shape[0] <= 1:
        return np.full(samples.shape[1], np.inf)
    return np.std(samples, axis=0, ddof=1) / np.sqrt(samples.shape[0])


def _cube_box(center: np.ndarray, half_size: float) -> AxisAlignedBox:
    """Create an axis-aligned cube from a center and half-size."""

    lo = np.asarray(center, dtype=float) - half_size
    hi = np.asarray(center, dtype=float) + half_size
    return AxisAlignedBox(tuple(lo), tuple(hi))


def _box_surface_area(box: AxisAlignedBox) -> float:
    """Surface area of an axis-aligned rectangular box."""

    dx, dy, dz = box.size
    return float(2.0 * (dx * dy + dx * dz + dy * dz))


def _sample_box_surface(box: AxisAlignedBox, rng: np.random.Generator) -> np.ndarray:
    """Sample a point uniformly by area on a box surface."""

    return _sample_box_surface_with_normal(box, rng).point


def _sample_box_surface_with_normal(box: AxisAlignedBox, rng: np.random.Generator) -> _SurfaceSample:
    """Sample a box-surface point and return the outward face normal."""

    lo = box.min_array
    hi = box.max_array
    dx, dy, dz = box.size
    face_areas = np.asarray([dy * dz, dy * dz, dx * dz, dx * dz, dx * dy, dx * dy], dtype=float)
    face = int(rng.choice(6, p=face_areas / np.sum(face_areas)))
    u = rng.random(2) # [0,1) ??
    point = np.empty(3, dtype=float)
    normal = np.zeros(3, dtype=float)
    if face == 0:
        point[0] = lo[0]
        point[1] = lo[1] + u[0] * dy
        point[2] = lo[2] + u[1] * dz
        normal[0] = -1.0
    elif face == 1:
        point[0] = hi[0]
        point[1] = lo[1] + u[0] * dy
        point[2] = lo[2] + u[1] * dz
        normal[0] = 1.0
    elif face == 2:
        point[0] = lo[0] + u[0] * dx
        point[1] = lo[1]
        point[2] = lo[2] + u[1] * dz
        normal[1] = -1.0
    elif face == 3:
        point[0] = lo[0] + u[0] * dx
        point[1] = hi[1]
        point[2] = lo[2] + u[1] * dz
        normal[1] = 1.0
    elif face == 4:
        point[0] = lo[0] + u[0] * dx
        point[1] = lo[1] + u[1] * dy
        point[2] = lo[2]
        normal[2] = -1.0
    else:
        point[0] = lo[0] + u[0] * dx
        point[1] = lo[1] + u[1] * dy
        point[2] = hi[2]
        normal[2] = 1.0
    return _SurfaceSample(point=point, normal=normal)


class CenteredCubeGreenSampler:
    """Discrete surface Green-function sampler for a centered homogeneous cube.

    The sampler uses a tabulated cell PDF for ordinary potential walks. For the
    first capacitance step it also evaluates the closed-form negative Green
    gradient series at the sampled surface location.
    """

    def __init__(self, *, grid_size: int = 31, series_terms: int = 41) -> None:
        self.grid_size = grid_size
        self.series_terms = series_terms
        self.cell_probabilities = _centered_cube_face_probabilities(grid_size, series_terms)
        self.flat_probabilities = np.tile(self.cell_probabilities.ravel(), 6)
        self.flat_probabilities = self.flat_probabilities / np.sum(self.flat_probabilities)
        self._series = _CenteredCubeSeries(series_terms)

    def sample(self, center: np.ndarray, half_size: float, rng: np.random.Generator) -> np.ndarray:
        """Sample only the cube exit point for an ordinary potential walk."""

        return self.sample_with_weight_data(center, half_size, rng).point

    def sample_with_weight_data(
        self,
        center: np.ndarray,
        half_size: float,
        rng: np.random.Generator,
    ) -> _TransitionSample:
        """Sample a cube exit and return proposal density plus gradient data."""

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

        side_length = 2.0 * half_size
        cell_area = (side_length / self.grid_size) ** 2
        proposal_density = float(self.cell_probabilities[i, j] / cell_area)
        if proposal_density <= 0.0:
            raise RuntimeError("sampled a zero-probability transition cell")
        negative_gradient = self.negative_green_gradient(face, u, v, side_length) # Yifan: calculate each time, even not needed (only the frist step need it)
        return _TransitionSample(
            point=point,
            half_size=half_size,
            proposal_density=proposal_density,
            negative_green_gradient=negative_gradient,
            face=face,
        )

    def pz_density_and_negative_gradient(self, u: float, v: float, side_length: float) -> tuple[float, np.ndarray]:
        """Evaluate top-face Green density and negative Green gradient."""

        return self._series.pz_density_and_negative_gradient(u, v, side_length)

    def negative_green_gradient(self, face: int, u: float, v: float, side_length: float) -> np.ndarray:
        """Map the top-face gradient formula to any cube face."""

        _, pz_negative_gradient = self.pz_density_and_negative_gradient(u, v, side_length)
        gx, gy, gz = pz_negative_gradient
        if face == 0:
            return np.asarray([-gz, gx, gy], dtype=float)
        if face == 1:
            return np.asarray([gz, gx, gy], dtype=float)
        if face == 2:
            return np.asarray([gx, -gz, gy], dtype=float)
        if face == 3:
            return np.asarray([gx, gz, gy], dtype=float)
        if face == 4:
            return np.asarray([gx, gy, -gz], dtype=float)
        if face == 5:
            return np.asarray([gx, gy, gz], dtype=float)
        raise ValueError("face must be in [0, 5]")


class _CenteredCubeSeries:
    """Precomputed coefficient arrays for the centered-cube Green series."""

    def __init__(self, series_terms: int) -> None:
        indices = np.arange(1, series_terms + 1, dtype=float)
        nx, ny = np.meshgrid(indices, indices, indexing="ij")
        nz = np.sqrt(nx * nx + ny * ny)
        sx = np.sin(np.pi * nx / 2.0)
        sy = np.sin(np.pi * ny / 2.0)
        cx = np.cos(np.pi * nx / 2.0)
        cy = np.cos(np.pi * ny / 2.0)
        half = np.pi * nz / 2.0

        self.nx = nx.ravel()
        self.ny = ny.ravel()
        self.density_coeff = (sx * sy / np.cosh(half)).ravel()
        self.negative_grad_x_coeff = (nx * cx * sy / np.cosh(half)).ravel()
        self.negative_grad_y_coeff = (ny * sx * cy / np.cosh(half)).ravel()
        self.negative_grad_z_coeff = (nz * sx * sy / np.sinh(half)).ravel()

    def pz_density_and_negative_gradient(self, u: float, v: float, side_length: float) -> tuple[float, np.ndarray]:
        """Closed-form top-face density and source-point negative gradient."""

        sin_u = np.sin(np.pi * self.nx * u)
        sin_v = np.sin(np.pi * self.ny * v)
        surface_shape = sin_u * sin_v

        density_sum = float(np.sum(self.density_coeff * surface_shape))
        grad_sum = np.asarray(
            [
                np.sum(self.negative_grad_x_coeff * surface_shape),
                np.sum(self.negative_grad_y_coeff * surface_shape),
                np.sum(self.negative_grad_z_coeff * surface_shape),
            ],
            dtype=float,
        )
        density = 2.0 * density_sum / (side_length * side_length)
        negative_gradient = -2.0 * np.pi * grad_sum / (side_length**3)
        return density, negative_gradient


def _centered_cube_face_probabilities(grid_size: int, series_terms: int) -> np.ndarray:
    """Tabulate one face of the centered-cube exit PDF on an N x N grid."""

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
    """Return the net containing a point, if the walk has hit metal."""

    for net_index, net in enumerate(nets):
        if any(conductor.box.contains_closed(point, tol=tol) for conductor in net.boxes):
            return net_index
    return None


def _distance_to_domain_boundary_linf(point: np.ndarray, domain: AxisAlignedBox) -> float:
    """L-infinity distance from a point to the absorbing outer box."""

    return float(np.min(np.minimum(point - domain.min_array, domain.max_array - point)))


def _nearest_conductor_linf(point: np.ndarray, nets: list[NetConductor]) -> tuple[float, int]:
    """Distance and net index for the nearest conductor in L-infinity metric."""

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
    """L-infinity distance from a point to an axis-aligned box."""

    lower_gap = np.maximum(box.min_array - point, 0.0)
    upper_gap = np.maximum(point - box.max_array, 0.0)
    return float(np.max(lower_gap + upper_gap))
