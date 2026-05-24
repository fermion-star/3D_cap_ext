from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from capext.geometry import BoxConductor, NetConductor
from capext.mesh import SurfacePanel, mesh_net_surfaces
from capext.problem import CapacitanceProblem
from capext.solvers import FRWSolver


NET_COLORS = [
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#E45756",
    "#72B7B2",
    "#B279A2",
    "#FF9DA6",
    "#9D755D",
    "#BAB0AC",
]


def box_faces(lo: tuple[float, float, float], hi: tuple[float, float, float]) -> list[list[tuple[float, float, float]]]:
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    vertices = [
        (x0, y0, z0),
        (x1, y0, z0),
        (x1, y1, z0),
        (x0, y1, z0),
        (x0, y0, z1),
        (x1, y0, z1),
        (x1, y1, z1),
        (x0, y1, z1),
    ]
    return [
        [vertices[idx] for idx in (0, 1, 2, 3)],
        [vertices[idx] for idx in (4, 5, 6, 7)],
        [vertices[idx] for idx in (0, 1, 5, 4)],
        [vertices[idx] for idx in (2, 3, 7, 6)],
        [vertices[idx] for idx in (1, 2, 6, 5)],
        [vertices[idx] for idx in (0, 3, 7, 4)],
    ]


def draw(problem: CapacitanceProblem, *, show: bool = True) -> Path:
    output = Path(__file__).with_name("run_bem_geometry.png")

    fig = plt.figure(figsize=(9, 7), dpi=180)
    ax = fig.add_subplot(111, projection="3d")

    all_min = []
    all_max = []
    legend_handles = []
    for net_index, net in enumerate(problem.nets()):
        color = NET_COLORS[net_index % len(NET_COLORS)]
        legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                label=net.name,
                markerfacecolor=color,
                markersize=9,
            )
        )
        for conductor in net.boxes:
            box = conductor.box
            lo = tuple(float(value) for value in box.min_array)
            hi = tuple(float(value) for value in box.max_array)
            all_min.append(box.min_array)
            all_max.append(box.max_array)

            faces = box_faces(lo, hi)
            poly = Poly3DCollection(
                faces,
                facecolors=color,
                edgecolors="#222222",
                linewidths=0.7,
                alpha=0.7,
            )
            ax.add_collection3d(poly)
            cx = 0.5 * (lo[0] + hi[0])
            cy = 0.5 * (lo[1] + hi[1])
            cz = hi[2] + 7.0
            ax.text(cx, cy, cz, conductor.name, ha="center", va="bottom", fontsize=8)

    mins, maxs = _plot_bounds(all_min, all_max)
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("examples/run_bem.py geometry")
    ax.view_init(elev=24, azim=-56)
    ax.set_box_aspect(maxs - mins)
    ax.legend(handles=legend_handles, loc="upper left")

    fig.tight_layout()
    fig.savefig(output)
    if show:
        print("Close the 3D geometry window to continue.")
        plt.show()
    return output


def draw_discretization(
    problem: CapacitanceProblem,
    *,
    max_panel_size: float,
    show: bool = True,
) -> Path:
    output = Path(__file__).with_name("run_bem_discretization.png")
    nets = problem.nets()
    panels = mesh_net_surfaces(nets, max_panel_size=max_panel_size)

    fig = plt.figure(figsize=(9, 7), dpi=180)
    ax = fig.add_subplot(111, projection="3d")

    all_min = []
    all_max = []
    for net in nets:
        for conductor in net.boxes:
            all_min.append(conductor.box.min_array)
            all_max.append(conductor.box.max_array)

    for panel in panels:
        color = NET_COLORS[panel.net_index % len(NET_COLORS)]
        poly = Poly3DCollection(
            [panel.corners],
            facecolors=color,
            edgecolors="#202020",
            linewidths=0.25,
            alpha=0.72,
        )
        ax.add_collection3d(poly)

    mins, maxs = _plot_bounds(all_min, all_max)
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(f"BEM surface discretization ({len(panels)} unknowns)")
    ax.view_init(elev=24, azim=-56)
    ax.set_box_aspect(maxs - mins)

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            label=net.name,
            markerfacecolor=NET_COLORS[index % len(NET_COLORS)],
            markersize=9,
        )
        for index, net in enumerate(nets)
    ]
    ax.legend(handles=legend_handles, loc="upper left")

    fig.tight_layout()
    fig.savefig(output)
    if show:
        print("Close the BEM discretization window to continue.")
        plt.show()
    return output


def count_bem_unknowns(
    problem: CapacitanceProblem,
    *,
    max_panel_size: float,
    include_physical_outer_box: bool = False,
) -> int:
    nets = problem.nets()
    if include_physical_outer_box:
        outer = BoxConductor("physical_outer_box", problem.domain)
        nets = [
            *nets,
            NetConductor("physical_outer_box", (outer,)),
        ]
    return len(mesh_net_surfaces(nets, max_panel_size=max_panel_size))


def draw_gaussian_surfaces(
    problem: CapacitanceProblem,
    solver: FRWSolver,
    *,
    show: bool = True,
) -> Path:
    output = Path(__file__).with_name("run_frw_gaussian_surfaces.png")
    nets = problem.nets()

    fig = plt.figure(figsize=(9, 7), dpi=180)
    ax = fig.add_subplot(111, projection="3d")

    all_min = []
    all_max = []
    legend_handles = []
    for net_index, net in enumerate(nets):
        color = NET_COLORS[net_index % len(NET_COLORS)]
        legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="s",
                color=color,
                label=f"{net.name} conductor",
                markerfacecolor=color,
                markersize=8,
            )
        )

        for conductor in net.boxes:
            box = conductor.box
            lo = tuple(float(value) for value in box.min_array)
            hi = tuple(float(value) for value in box.max_array)
            all_min.append(box.min_array)
            all_max.append(box.max_array)
            poly = Poly3DCollection(
                box_faces(lo, hi),
                facecolors=color,
                edgecolors="#222222",
                linewidths=0.7,
                alpha=0.45,
            )
            ax.add_collection3d(poly)

        gaussian = solver.gaussian_box(problem, net)
        all_min.append(gaussian.min_array)
        all_max.append(gaussian.max_array)
        _draw_box_wireframe(ax, gaussian.min_array, gaussian.max_array, color=color, linewidth=1.3)

    mins, maxs = _plot_bounds(all_min, all_max)
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(f"FRW Gaussian surfaces (padding={solver.gaussian_padding:g})")
    ax.view_init(elev=24, azim=-56)
    ax.set_box_aspect(maxs - mins)
    ax.legend(handles=legend_handles, loc="upper left")

    fig.tight_layout()
    fig.savefig(output)
    if show:
        print("Close the FRW Gaussian-surface window to continue.")
        plt.show()
    return output


def _draw_box_wireframe(ax, lo, hi, *, color: str, linewidth: float) -> None:
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    edges = [
        ((x0, y0, z0), (x1, y0, z0)),
        ((x1, y0, z0), (x1, y1, z0)),
        ((x1, y1, z0), (x0, y1, z0)),
        ((x0, y1, z0), (x0, y0, z0)),
        ((x0, y0, z1), (x1, y0, z1)),
        ((x1, y0, z1), (x1, y1, z1)),
        ((x1, y1, z1), (x0, y1, z1)),
        ((x0, y1, z1), (x0, y0, z1)),
        ((x0, y0, z0), (x0, y0, z1)),
        ((x1, y0, z0), (x1, y0, z1)),
        ((x1, y1, z0), (x1, y1, z1)),
        ((x0, y1, z0), (x0, y1, z1)),
    ]
    for start, stop in edges:
        ax.plot(
            [start[0], stop[0]],
            [start[1], stop[1]],
            [start[2], stop[2]],
            color=color,
            linewidth=linewidth,
            linestyle="--",
        )


def _plot_bounds(all_min, all_max):
    if not all_min:
        raise ValueError("problem has no conductor boxes to draw")

    import numpy as np

    mins = np.min(np.vstack(all_min), axis=0)
    maxs = np.max(np.vstack(all_max), axis=0)
    span = maxs - mins
    padding = np.maximum(span * 0.15, 10.0)
    return mins - padding, maxs + padding


if __name__ == "__main__":
    raise SystemExit("Call draw(problem) from examples/run_bem.py.")
