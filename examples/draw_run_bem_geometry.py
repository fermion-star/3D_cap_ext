from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


BOXES = [
    ("left", (100.0, 220.0, 220.0), (160.0, 280.0, 280.0), "#4C78A8"),
    ("left_extension", (160.0, 230.0, 230.0), (190.0, 270.0, 270.0), "#72B7B2"),
    ("right", (260.0, 220.0, 220.0), (320.0, 280.0, 280.0), "#F58518"),
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


def draw() -> Path:
    output = Path(__file__).with_name("run_bem_geometry.png")

    fig = plt.figure(figsize=(9, 7), dpi=180)
    ax = fig.add_subplot(111, projection="3d")

    for name, lo, hi, color in BOXES:
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
        ax.text(cx, cy, cz, name, ha="center", va="bottom", fontsize=8)

    ax.set_xlim(80, 340)
    ax.set_ylim(200, 300)
    ax.set_zlim(200, 300)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("examples/run_bem.py geometry")
    ax.view_init(elev=24, azim=-56)
    ax.set_box_aspect((260, 100, 100))

    legend_handles = [
        plt.Line2D([0], [0], marker="s", color="w", label="left net", markerfacecolor="#4C78A8", markersize=9),
        plt.Line2D([0], [0], marker="s", color="w", label="left extension same net", markerfacecolor="#72B7B2", markersize=9),
        plt.Line2D([0], [0], marker="s", color="w", label="right net", markerfacecolor="#F58518", markersize=9),
    ]
    ax.legend(handles=legend_handles, loc="upper left")

    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    return output


if __name__ == "__main__":
    print(draw())
