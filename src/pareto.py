"""Pareto-dominance utilities shared by both MOEAs and the metrics (minimisation)."""
from __future__ import annotations

import numpy as np


def dominance_matrix(F: np.ndarray) -> np.ndarray:
    """D[a, b] is True if solution a dominates solution b:
    a is no worse in every objective and strictly better in at least one."""
    le = np.all(F[:, None, :] <= F[None, :, :], axis=2)
    lt = np.any(F[:, None, :] < F[None, :, :], axis=2)
    return le & lt


def nondominated_mask(F: np.ndarray) -> np.ndarray:
    """True for solutions not dominated by any other solution in F."""
    return ~dominance_matrix(F).any(axis=0)


def final_front(F: np.ndarray) -> np.ndarray:
    """Final approximation set: non-dominated, duplicate objective vectors
    removed, sorted by f1. Its length is the reported 'number of
    non-dominated solutions'."""
    nd = np.unique(F[nondominated_mask(F)], axis=0)
    return nd[np.argsort(nd[:, 0], kind="stable")]
