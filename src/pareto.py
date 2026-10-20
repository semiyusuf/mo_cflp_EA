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


def fast_non_dominated_sort(F: np.ndarray) -> list[np.ndarray]:
    """Sort solutions into non-dominated fronts (lists of indices).
    fronts[0] = solutions nobody dominates, fronts[1] = solutions dominated
    only by fronts[0], and so on."""
    D = dominance_matrix(F)
    dominated_count = D.sum(axis=0)          # how many solutions dominate each one
    fronts = []
    current = np.flatnonzero(dominated_count == 0)
    while current.size:
        fronts.append(current)
        dominated_count = dominated_count - D[current].sum(axis=0)
        dominated_count[current] = -1        # already assigned to a front
        current = np.flatnonzero(dominated_count == 0)
    return fronts


class OfflineArchive:
    """Records every non-dominated solution found during a run.

    MOGA and NSGA are NOT elitist: a good solution can disappear from the
    population in the next generation. The archive only REMEMBERS the best
    solutions for the final result; it never feeds solutions back into the
    search, so it does not change how either algorithm behaves. Both
    algorithms use it in exactly the same way."""

    def __init__(self, m: int):
        self.F = np.empty((0, 2))
        self.Y = np.empty((0, m), dtype=np.int8)

    def update(self, F: np.ndarray, Y: np.ndarray) -> None:
        F_all, Y_all = np.vstack([self.F, F]), np.vstack([self.Y, Y])
        keep = nondominated_mask(F_all)
        F_all, Y_all = F_all[keep], Y_all[keep]
        _, idx = np.unique(F_all, axis=0, return_index=True)   # one copy per objective vector
        self.F, self.Y = F_all[idx], Y_all[idx]

    def result(self):
        order = np.argsort(self.F[:, 0], kind="stable")
        return self.F[order], self.Y[order]
