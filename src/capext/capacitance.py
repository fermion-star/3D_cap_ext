from __future__ import annotations

import numpy as np


def augment_with_reference_node(capacitance: np.ndarray) -> np.ndarray:
    """Add an explicit reference conductor so every row and column sums to zero.

    The input is the reduced Maxwell capacitance matrix for the non-reference
    conductors with the reference conductor held at 0 V. The returned matrix
    includes that reference conductor as the final node.
    """

    reduced = np.asarray(capacitance, dtype=float)
    if reduced.ndim != 2 or reduced.shape[0] != reduced.shape[1]:
        raise ValueError("capacitance must be a square matrix")

    net_count = reduced.shape[0]
    augmented = np.zeros((net_count + 1, net_count + 1), dtype=float)
    augmented[:net_count, :net_count] = reduced

    row_sums = np.sum(reduced, axis=1)
    col_sums = np.sum(reduced, axis=0)
    augmented[:net_count, net_count] = -row_sums
    augmented[net_count, :net_count] = -col_sums
    augmented[net_count, net_count] = float(np.sum(reduced))
    return augmented
