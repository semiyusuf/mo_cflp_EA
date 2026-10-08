"""Settings and result containers shared by both MOEAs."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EAConfig:
    """One parameter configuration. Both MOEAs receive the identical object."""
    name: str
    pop_size: int        # N (SPEA2 archive size is also N)
    max_evals: int       # termination: maximum number of objective evaluations
    pc: float            # crossover probability (per pair)
    pm_factor: float     # mutation probability per gene = pm_factor / m
    init: str            # initialisation method, see operators.init_population

    def pm(self, m: int) -> float:
        return self.pm_factor / m


@dataclass
class RunResult:
    algorithm: str
    front: np.ndarray        # final non-dominated objective vectors, shape (k, 2)
    front_genes: np.ndarray  # matching repaired chromosomes, shape (k, m)
    evals: int               # objective evaluations used
    time_s: float            # wall-clock execution time in seconds
    generations: int


def front_with_genes(F: np.ndarray, Y: np.ndarray, mask: np.ndarray):
    """Keep solutions in `mask`, drop duplicate objective vectors, sort by f1."""
    F, Y = F[mask], Y[mask]
    _, idx = np.unique(F, axis=0, return_index=True)
    F, Y = F[idx], Y[idx]
    order = np.argsort(F[:, 0], kind="stable")
    return F[order], Y[order]
