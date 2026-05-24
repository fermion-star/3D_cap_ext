# 3D Capacitance Extraction

Prototype solver stack for extracting capacitance matrices of axis-aligned cuboid conductors in a homogeneous dielectric.

The current implementation focuses on a dense low-order BEM solver. Coordinates are continuous values; integer coordinates are only a future dataset-generation convention for DNN training.

## Conductors and nets

Input boxes are merged into electrical nets when two boxes overlap in 3D volume or share a 2D face area. Edge-only and point-only contact are intentionally not merged.

## Run example

```bash
PYTHONPATH=src python examples/run_bem.py
```

The BEM solver uses rectangular constant-charge panels on the exterior surface of each merged net. It approximates panels as equal-area disks for self terms and near off-axis interactions, while far interactions use the faster point-center approximation.

## Solver switch

Use the solver factory when code should choose between solver backends:

```python
from capext.solvers import create_solver

bem = create_solver("bem", max_panel_size=10.0)
frw = create_solver("frw", samples_per_observation_net=10_000, seed=1)
```

`FRWSolver` is currently a framework placeholder. It exposes the intended API
and stochastic configuration, but raises `NotImplementedError` until the random
walk transition kernel and charge estimator are implemented.
