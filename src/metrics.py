"""
Multi-objective performance metrics.

NORMALISATION (identical for both MOEAs): for each benchmark instance the
ideal point z_min and nadir point z_max are taken over the union of ALL final
fronts of BOTH algorithms, ALL configurations and ALL runs on that instance.
Every objective vector is then scaled as
        f' = (f - z_min) / (z_max - z_min)          (so every f' is in [0, 1])
HYPERVOLUME is computed in this normalised space with the reference point
        r = (1.1, 1.1)
which is worse than every solution included in the comparison.
"""
from __future__ import annotations

import numpy as np

from src.pareto import nondominated_mask

REF_POINT = np.array([1.1, 1.1])


def bounds_from_fronts(fronts: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    allf = np.vstack(fronts)
    return allf.min(axis=0), allf.max(axis=0)


def normalise(F: np.ndarray, z_min: np.ndarray, z_max: np.ndarray) -> np.ndarray:
    span = np.where(z_max > z_min, z_max - z_min, 1.0)
    return (F - z_min) / span


def hypervolume_2d(F: np.ndarray, ref: np.ndarray = REF_POINT) -> float:
    """Exact hypervolume of a 2-objective minimisation front: the area
    dominated by the front and bounded by the reference point, computed as
    a sum of rectangles after sorting by f1."""
    F = F[np.all(F < ref, axis=1)]
    if len(F) == 0:
        return 0.0
    F = np.unique(F[nondominated_mask(F)], axis=0)
    F = F[np.argsort(F[:, 0])]                 # f1 ascending -> f2 descending
    x_next = np.append(F[1:, 0], ref[0])
    return float(np.sum((x_next - F[:, 0]) * (ref[1] - F[:, 1])))
