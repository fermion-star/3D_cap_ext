import numpy as np

from capext.panel_integrals import (
    disk_potential_coefficient_multipole,
    equivalent_disk_radius,
    find_point_center_distance_ratio,
    point_center_panel_coefficient,
    worst_point_center_relative_error,
)


def test_equivalent_disk_radius_preserves_area() -> None:
    area = 7.5
    radius = equivalent_disk_radius(area)

    assert np.isclose(np.pi * radius * radius, area)


def test_disk_multipole_monopole_term_matches_point_center() -> None:
    radius = 2.0
    distance = 10.0
    epsilon = 3.0

    multipole = disk_potential_coefficient_multipole(
        distance,
        0.3,
        radius,
        epsilon,
        max_order=0,
    )
    point = point_center_panel_coefficient(np.pi * radius * radius, distance, epsilon)

    assert np.isclose(multipole, point)


def test_disk_multipole_matches_axis_potential() -> None:
    radius = 1.0
    distance = 2.5
    epsilon = 1.7
    exact_axis = (np.sqrt(distance * distance + radius * radius) - distance) / (2.0 * epsilon)

    multipole = disk_potential_coefficient_multipole(
        distance,
        1.0,
        radius,
        epsilon,
        max_order=80,
    )

    assert np.isclose(multipole, exact_axis, rtol=1e-12, atol=1e-15)


def test_point_center_threshold_for_one_percent_error() -> None:
    assert worst_point_center_relative_error(5.0) < 0.01
    assert worst_point_center_relative_error(4.0) > 0.01

    threshold = find_point_center_distance_ratio(tolerance=0.01)
    assert 4.9 < threshold < 5.1
