from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CFLPInstance:
    """All OR-Library data needed for the bi-objective CFLP."""

    name: str
    capacity: np.ndarray    # S_i, shape (m,)  - OR-Library "capacity" field
    fixed_cost: np.ndarray  # F_i, shape (m,)  - OR-Library "fixed cost" field  -> objective f1
    demand: np.ndarray      # d_j, shape (n,)  - OR-Library "demand" field
    alloc_cost: np.ndarray  # C_ij, shape (m, n) - allocation cost per customer -> objective f2

    @property
    def m(self) -> int:
        return len(self.capacity)

    @property
    def n(self) -> int:
        return len(self.demand)

    @property
    def total_demand(self) -> float:
        return float(self.demand.sum())


def load_instance(path: str | Path) -> CFLPInstance:
    """Parse an OR-Library cap*.txt file into a CFLPInstance."""
    path = Path(path)
    tokens = path.read_text().split()
    if len(tokens) < 2:
        raise ValueError(f"{path}: file is empty or truncated")

    m, n = int(float(tokens[0])), int(float(tokens[1]))
    expected = 2 + 2 * m + n * (1 + m)
    if len(tokens) != expected:
        raise ValueError(
            f"{path}: expected {expected} numbers for m={m}, n={n}, found {len(tokens)}. "
            "Is the file complete?"
        )

    try:
        values = np.array(tokens[2:], dtype=float)
    except ValueError as exc:
        # capa/capb/capc use the literal word 'capacity' - not used in this project.
        raise ValueError(f"{path}: non-numeric value found ({exc})") from exc

    fac = values[: 2 * m].reshape(m, 2)
    capacity, fixed_cost = fac[:, 0].copy(), fac[:, 1].copy()

    cust = values[2 * m:].reshape(n, 1 + m)
    demand = cust[:, 0].copy()
    alloc_cost = cust[:, 1:].T.copy()  # transpose to (m, n): row = facility, column = customer

    if capacity.sum() < demand.sum():
        raise ValueError(f"{path}: total capacity is smaller than total demand - instance infeasible")

    return CFLPInstance(path.stem, capacity, fixed_cost, demand, alloc_cost)
