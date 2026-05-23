from __future__ import annotations

import numpy as np


def equivalent_disk_radius(area: float | np.ndarray) -> float | np.ndarray:
    return np.sqrt(np.asarray(area, dtype=float) / np.pi)


def point_center_panel_coefficient(
    area: float | np.ndarray,
    distance: float | np.ndarray,
    epsilon: float,
) -> float | np.ndarray:
    return np.asarray(area, dtype=float) / (4.0 * np.pi * epsilon * np.asarray(distance, dtype=float))


def disk_potential_coefficient_multipole(
    distance: float | np.ndarray,
    cos_theta: float | np.ndarray,
    radius: float | np.ndarray,
    epsilon: float,
    *,
    max_order: int = 24,
) -> float | np.ndarray:
    """Potential coefficient of a uniformly charged disk via multipole expansion.

    The returned value multiplies surface charge density sigma. The expansion is
    valid outside the smallest sphere containing the disk, i.e. distance > radius.
    """

    distance_arr = np.asarray(distance, dtype=float)
    cos_theta_arr = np.asarray(cos_theta, dtype=float)
    radius_arr = np.asarray(radius, dtype=float)
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if max_order < 0:
        raise ValueError("max_order must be non-negative")
    if np.any(radius_arr <= 0):
        raise ValueError("radius must be positive")
    if np.any(distance_arr <= radius_arr):
        raise ValueError("multipole expansion requires distance > radius")

    distance_arr, cos_theta_arr, radius_arr = np.broadcast_arrays(
        distance_arr,
        cos_theta_arr,
        radius_arr,
    )
    cos_theta_arr = np.clip(cos_theta_arr, -1.0, 1.0)

    result = np.zeros_like(distance_arr, dtype=float)
    p_l_minus_2 = np.ones_like(cos_theta_arr)
    p_l_minus_1 = cos_theta_arr
    p0_l_minus_2 = 1.0
    p0_l_minus_1 = 0.0

    for ell in range(max_order + 1):
        if ell == 0:
            p_l = p_l_minus_2
            p0_l = p0_l_minus_2
        elif ell == 1:
            p_l = p_l_minus_1
            p0_l = p0_l_minus_1
        else:
            p_l = ((2 * ell - 1) * cos_theta_arr * p_l_minus_1 - (ell - 1) * p_l_minus_2) / ell
            p0_l = -((ell - 1) * p0_l_minus_2) / ell
            p_l_minus_2, p_l_minus_1 = p_l_minus_1, p_l
            p0_l_minus_2, p0_l_minus_1 = p0_l_minus_1, p0_l

        if ell % 2 == 0:
            result += (
                radius_arr ** (ell + 2)
                / ((ell + 2) * distance_arr ** (ell + 1))
                * p0_l
                * p_l
            )

    return result / (2.0 * epsilon)


def point_center_relative_error_against_disk(
    distance_ratio: float,
    cos_theta: float | np.ndarray,
    *,
    max_order: int = 160,
) -> float | np.ndarray:
    """Relative error of point-center approximation against disk multipole."""

    if distance_ratio <= 1.0:
        raise ValueError("distance_ratio must be greater than 1")
    disk = disk_potential_coefficient_multipole(
        distance_ratio,
        cos_theta,
        1.0,
        1.0,
        max_order=max_order,
    )
    point = point_center_panel_coefficient(np.pi, distance_ratio, 1.0)
    return np.abs(point - disk) / np.abs(disk)


def worst_point_center_relative_error(
    distance_ratio: float,
    *,
    theta_samples: int = 721,
    max_order: int = 160,
) -> float:
    cos_theta = np.linspace(-1.0, 1.0, theta_samples)
    return float(
        np.max(
            point_center_relative_error_against_disk(
                distance_ratio,
                cos_theta,
                max_order=max_order,
            )
        )
    )


def find_point_center_distance_ratio(
    *,
    tolerance: float = 0.01,
    theta_samples: int = 721,
    max_order: int = 160,
) -> float:
    """Find k so point-center error is below tolerance for all sampled angles."""

    if tolerance <= 0:
        raise ValueError("tolerance must be positive")

    low = 1.0 + 1e-12
    high = 2.0
    while worst_point_center_relative_error(
        high,
        theta_samples=theta_samples,
        max_order=max_order,
    ) > tolerance:
        high *= 2.0

    for _ in range(60):
        mid = 0.5 * (low + high)
        error = worst_point_center_relative_error(
            mid,
            theta_samples=theta_samples,
            max_order=max_order,
        )
        if error <= tolerance:
            high = mid
        else:
            low = mid
    return high
