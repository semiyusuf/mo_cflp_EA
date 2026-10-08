"""
Objective evaluation and feasibility checking for the bi-objective CFLP.

    f1 = sum_i F_i * y_i                 (facility-opening cost)
    f2 = sum_i sum_j C_ij * x_ij         (customer-allocation cost)

wwe try to store solution as a pair (y, X) where
    y : 0/1 vector of length m           (y_i = 1 if facility i is open)
    X : allocation matrix of shape (m, n), X[i, j] = x_ij
"""
from __future__ import annotations

import numpy as np

from src.data_loader import CFLPInstance

TOL = 1e-9


def oversized_customers(inst: CFLPInstance) -> np.ndarray:
    """Customers whose demand exceeds the largest facility capacity."""
    return np.flatnonzero(inst.demand > inst.capacity.max())


def is_feasible(inst: CFLPInstance, y: np.ndarray, X: np.ndarray) -> bool:
    """Check all constraints."""
    if X.shape != (inst.m, inst.n) or np.any(X < -TOL):
        return False
    if not np.allclose(X.sum(axis=0), 1.0):
        return False                                    # sum_i x_ij = 1 for every j
    if np.any(X[y == 0] > TOL):
        return False                                    # x_ij <= y_i
    single = np.ones(inst.n, bool)
    single[oversized_customers(inst)] = False
    if not np.all(np.isin(X[:, single], (0.0, 1.0))):
        return False                                    # normal customers: exactly one facility
    load = X @ inst.demand
    return bool(np.all(load <= inst.capacity * y + 1e-6))   # sum_j d_j x_ij <= S_i y_i


def evaluate(inst: CFLPInstance, y: np.ndarray, X: np.ndarray) -> tuple[float, float]:
    """Return (f1, f2). Must only be called on a feasible (decoded) solution."""
    f1 = float(np.dot(inst.fixed_cost, y))
    f2 = float(np.sum(inst.alloc_cost * X))   # C_ij already includes demand -> not multiplied by d_j
    return f1, f2
