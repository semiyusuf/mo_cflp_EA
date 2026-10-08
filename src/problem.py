"""
Objective evaluation and feasibility checking for the bi-objective CFLP.

    f1 = sum_i F_i * y_i                 (facility-opening cost)
    f2 = sum_i sum_j C_ij * x_ij         (customer-allocation cost)

A solution is stored as
    y      : 0/1 vector of length m   (y_i = 1 if facility i is open)
    assign : int vector of length n   (assign[j] = i  <=>  x_ij = 1)

Using an assignment vector guarantees "every customer is assigned exactly
once" by construction. The other two constraints are checked explicitly.
"""
from __future__ import annotations

import numpy as np

from src.data_loader import CFLPInstance


def is_feasible(inst: CFLPInstance, y: np.ndarray, assign: np.ndarray) -> bool:
    """Check all three mandatory constraints."""
    if assign.shape != (inst.n,) or assign.min() < 0 or assign.max() >= inst.m:
        return False                                   # sum_i x_ij = 1 for every j
    if not np.all(y[assign] == 1):
        return False                                   # x_ij <= y_i
    load = np.bincount(assign, weights=inst.demand, minlength=inst.m)
    return bool(np.all(load <= inst.capacity * y + 1e-9))  # sum_j d_j x_ij <= S_i y_i


def evaluate(inst: CFLPInstance, y: np.ndarray, assign: np.ndarray) -> tuple[float, float]:
    """Return (f1, f2). Must only be called on a feasible (decoded) solution."""
    f1 = float(np.dot(inst.fixed_cost, y))
    # C_ij already includes the demand of customer j -> NOT multiplied by d_j
    f2 = float(inst.alloc_cost[assign, np.arange(inst.n)].sum())
    return f1, f2
