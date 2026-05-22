from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from capext.geometry import AxisAlignedBox, NetConductor


@dataclass(frozen=True)
class SurfacePanel:
    center: np.ndarray
    normal: np.ndarray
    area: float
    net_index: int


def mesh_net_surfaces(
    nets: list[NetConductor],
    *,
    max_panel_size: float,
    tol: float = 1e-12,
) -> list[SurfacePanel]:
    if max_panel_size <= 0:
        raise ValueError("max_panel_size must be positive")

    panels_by_key: dict[tuple[int, int, int, int, int, int, int], SurfacePanel] = {}
    for net_index, net in enumerate(nets):
        net_boxes = [conductor.box for conductor in net.boxes]
        for box in net_boxes:
            for axis in range(3):
                for side, normal_sign in (("min", -1.0), ("max", 1.0)):
                    face_value = box.min_array[axis] if side == "min" else box.max_array[axis]
                    normal = np.zeros(3)
                    normal[axis] = normal_sign
                    for center, area in _face_panels(box, axis, face_value, net_boxes, max_panel_size):
                        probe = center + normal * max(tol * 16.0, max_panel_size * 1e-9)
                        if _inside_any_box(probe, net_boxes, tol=tol):
                            continue
                        panel = SurfacePanel(center=center, normal=normal.copy(), area=area, net_index=net_index)
                        panels_by_key[_panel_key(panel)] = panel
    return list(panels_by_key.values())


def _face_panels(
    box: AxisAlignedBox,
    axis: int,
    face_value: float,
    net_boxes: list[AxisAlignedBox],
    max_panel_size: float,
) -> list[tuple[np.ndarray, float]]:
    axes = [idx for idx in range(3) if idx != axis]
    a0, a1 = axes
    breaks0 = _axis_breaks(box.min_array[a0], box.max_array[a0], net_boxes, a0, max_panel_size)
    breaks1 = _axis_breaks(box.min_array[a1], box.max_array[a1], net_boxes, a1, max_panel_size)

    panels = []
    for lo0, hi0 in zip(breaks0[:-1], breaks0[1:]):
        for lo1, hi1 in zip(breaks1[:-1], breaks1[1:]):
            center = np.zeros(3)
            center[axis] = face_value
            center[a0] = 0.5 * (lo0 + hi0)
            center[a1] = 0.5 * (lo1 + hi1)
            panels.append((center, float((hi0 - lo0) * (hi1 - lo1))))
    return panels


def _axis_breaks(
    lo: float,
    hi: float,
    boxes: list[AxisAlignedBox],
    axis: int,
    max_panel_size: float,
) -> np.ndarray:
    coords = {float(lo), float(hi)}
    for box in boxes:
        for value in (box.min_array[axis], box.max_array[axis]):
            if lo < value < hi:
                coords.add(float(value))

    base = sorted(coords)
    refined = [base[0]]
    for start, stop in zip(base[:-1], base[1:]):
        length = stop - start
        steps = max(1, int(np.ceil(length / max_panel_size)))
        for step in range(1, steps + 1):
            refined.append(start + length * step / steps)
    return np.asarray(refined, dtype=float)


def _inside_any_box(point: np.ndarray, boxes: list[AxisAlignedBox], tol: float) -> bool:
    return any(box.contains_closed(point, tol=tol) for box in boxes)


def _panel_key(panel: SurfacePanel) -> tuple[int, int, int, int, int, int, int]:
    scale = 1_000_000_000
    center_key = tuple(int(round(float(value) * scale)) for value in panel.center)
    normal_key = tuple(int(round(float(value))) for value in panel.normal)
    return (panel.net_index, *center_key, *normal_key)
