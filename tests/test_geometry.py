from capext.geometry import AxisAlignedBox
from capext.problem import CapacitanceProblem


def test_face_contact_merges_into_same_net() -> None:
    problem = CapacitanceProblem.from_boxes(
        [
            ("a", (0, 0, 0), (1, 1, 1)),
            ("b", (1, 0.25, 0.25), (2, 0.75, 0.75)),
        ],
        domain_min=(-1, -1, -1),
        domain_max=(3, 3, 3),
    )

    assert len(problem.nets()) == 1


def test_edge_contact_does_not_merge() -> None:
    a = AxisAlignedBox((0, 0, 0), (1, 1, 1))
    b = AxisAlignedBox((1, 1, 0.25), (2, 2, 0.75))

    assert a.contact_dimension(b) == 1
    assert not a.same_net_contact(b)


def test_volume_overlap_merges_into_same_net() -> None:
    problem = CapacitanceProblem.from_boxes(
        [
            ("a", (0, 0, 0), (2, 2, 2)),
            ("b", (1, 1, 1), (3, 3, 3)),
        ],
        domain_min=(-1, -1, -1),
        domain_max=(4, 4, 4),
    )

    assert len(problem.nets()) == 1
