# 3D Capacitance Extraction

Prototype solver stack for extracting capacitance matrices of axis-aligned cuboid conductors in a homogeneous dielectric.

The current implementation focuses on a dense low-order BEM solver. Coordinates are continuous values; integer coordinates are only a future dataset-generation convention for DNN training.

## Conductors and nets

Input boxes are merged into electrical nets when two boxes overlap in 3D volume or share a 2D face area. Edge-only and point-only contact are intentionally not merged.

## Run example

```bash
PYTHONPATH=src python examples/run_bem.py
```

The BEM solver uses rectangular constant-charge panels on the exterior surface of each merged net. The first self-panel term is an equal-area disk approximation; this is good enough for a prototype and can later be replaced with rectangular analytic integrals.
