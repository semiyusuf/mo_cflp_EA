"""
Chromosome representation and feasibility repair / decoding.

REPRESENTATION
    A chromosome is a binary vector y of length m (one gene per candidate
    facility, 1 = open). Customer assignments are NOT encoded; they are
    produced deterministically by the decoder below. Both MOEAs use this same
    representation and the same decoder, so the comparison is fair.

DECODER (repair + assignment), applied to every new chromosome:
    Step 1 - capacity repair: while total open capacity < total demand, open
             the closed facility with the lowest "unit cost"
                 u_i = F_i / S_i + mean_j (C_ij / d_j)
             (opening cost per unit of capacity + average transport cost per
             unit of demand). Deterministic, ties broken by lowest index.
    Step 2 - greedy assignment: customers are processed in decreasing order of
             demand (big items first makes the packing easier). Each customer
             goes to the open facility with the lowest C_ij that still has
             enough residual capacity.
             If no open facility can take it, the closed facility minimising
             F_i + C_ij among those with S_i >= d_j is opened (repair) and
             the customer is assigned there.
    Step 2b - fallback if Step 2 gets stuck (capacity fragmented: every
             facility has some room left but not enough for the next
             customer, and no closed facility is left to open):
             re-pack with best-fit decreasing (each customer, largest first,
             goes to the open facility with the SMALLEST residual capacity
             that fits; opens the lowest-unit-cost closed facility if needed),
             then improve f2 with shift and swap moves that keep every
             facility within capacity.
    Step 3 - clean-up: open facilities that received no customer are closed
             (they would only add to f1).
    The repaired y is written back into the chromosome (Lamarckian repair),
    so the genotype always matches the evaluated, feasible phenotype.
"""
from __future__ import annotations

import numpy as np

from src.data_loader import CFLPInstance
from src.problem import evaluate, is_feasible


class Decoder:
    """Turns any 0/1 vector into a feasible (y, assignment) pair and evaluates it."""

    def __init__(self, inst: CFLPInstance):
        self.inst = inst
        safe_d = np.maximum(inst.demand, 1e-12)
        self.unit_cost = inst.fixed_cost / inst.capacity + (inst.alloc_cost / safe_d).mean(axis=1)
        # stable sort -> ties in demand are broken by customer index
        self.customer_order = np.argsort(-inst.demand, kind="stable")

    def decode(self, genes: np.ndarray, trace: list | None = None):
        """Return (y_repaired, assign, (f1, f2)). If `trace` is a list, the
        repair steps are appended to it (used for the report example)."""
        inst = self.inst
        y = genes.astype(np.int8).copy()

        # Step 1: capacity repair
        while inst.capacity[y == 1].sum() < inst.total_demand:
            closed = np.flatnonzero(y == 0)
            i = closed[np.argmin(self.unit_cost[closed])]
            y[i] = 1
            if trace is not None:
                trace.append(f"capacity repair: open facility {i}")

        # Step 2: greedy assignment (cheapest facility with room)
        assign = self._greedy_assignment(y, trace)
        if assign is None:
            # Step 2b: greedy got stuck because capacity is fragmented (every
            # facility has some room left, but not enough for the next
            # customer). Re-pack with best-fit decreasing, then improve cost.
            if trace is not None:
                trace.append("capacity fragmented: re-pack with best-fit, then shift/swap improvement")
            assign = self._best_fit_assignment(y, trace)
            self._improve(y, assign)

        # Step 3: close unused facilities
        used = np.zeros(inst.m, dtype=bool)
        used[assign] = True
        if trace is not None:
            for i in np.flatnonzero((y == 1) & ~used):
                trace.append(f"clean-up: close unused facility {i}")
        y[~used] = 0

        assert is_feasible(inst, y, assign)
        return y, assign, evaluate(inst, y, assign)

    def _open_for(self, y, d, j, trace, rule):
        """Open the closed facility that can hold demand d with the best `rule` score."""
        inst = self.inst
        cand = np.where((y == 0) & (inst.capacity >= d), rule, np.inf)
        i = int(np.argmin(cand))
        if not np.isfinite(cand[i]):
            return None
        y[i] = 1
        if trace is not None:
            trace.append(f"assignment repair: open facility {i} for customer {j}")
        return i

    def _greedy_assignment(self, y, trace):
        """Customers by decreasing demand -> cheapest open facility with room;
        if none has room, open the closed facility minimising F_i + C_ij.
        Returns None if even that is impossible (fragmented capacity)."""
        inst = self.inst
        residual = inst.capacity * y
        assign = np.empty(inst.n, dtype=np.int64)
        for j in self.customer_order:
            d = inst.demand[j]
            cost = np.where((y == 1) & (residual >= d), inst.alloc_cost[:, j], np.inf)
            i = int(np.argmin(cost))
            if not np.isfinite(cost[i]):
                i = self._open_for(y, d, j, trace, inst.fixed_cost + inst.alloc_cost[:, j])
                if i is None:
                    return None
                residual[i] = inst.capacity[i]
            assign[j] = i
            residual[i] -= d
        return assign

    def _best_fit_assignment(self, y, trace):
        """Best-fit decreasing bin packing: each customer (decreasing demand)
        goes to the open facility with the SMALLEST residual capacity that
        still fits it, which keeps large gaps free for large customers.
        Opens the closed facility with the lowest unit cost if nothing fits."""
        inst = self.inst
        residual = inst.capacity * y
        assign = np.empty(inst.n, dtype=np.int64)
        for j in self.customer_order:
            d = inst.demand[j]
            room = np.where((y == 1) & (residual >= d), residual, np.inf)
            i = int(np.argmin(room))
            if not np.isfinite(room[i]):
                i = self._open_for(y, d, j, trace, self.unit_cost)
                if i is None:
                    raise RuntimeError(f"customer {j} cannot be packed even with all facilities open")
                residual[i] = inst.capacity[i]
            assign[j] = i
            residual[i] -= d
        return assign

    def _improve(self, y, assign):
        """Local search on f2 for a fixed set of open facilities, keeping
        capacities feasible: shift moves (move one customer to a cheaper
        facility with room) and swap moves (exchange two customers' facilities).
        Repeats until no move improves the allocation cost."""
        inst, C, d = self.inst, self.inst.alloc_cost, self.inst.demand
        residual = inst.capacity * y - np.bincount(assign, weights=d, minlength=inst.m)
        idx = np.arange(inst.n)
        improved = True
        while improved:
            improved = False
            for j in range(inst.n):                          # shift moves
                a = assign[j]
                cost = np.where((y == 1) & (residual >= d[j]), C[:, j], np.inf)
                i = int(np.argmin(cost))
                if cost[i] < C[a, j] - 1e-9:
                    residual[a] += d[j]
                    residual[i] -= d[j]
                    assign[j] = i
                    improved = True
            for j in range(inst.n):                          # swap moves
                a, b = assign[j], assign                     # b: facility of every other customer k
                delta = C[b, j] + C[a, idx] - C[a, j] - C[b, idx]
                ok = ((b != a) & (residual[a] + d[j] - d >= 0) & (residual[b] + d - d[j] >= 0)
                      & (delta < -1e-9))
                if ok.any():
                    k = int(np.argmin(np.where(ok, delta, np.inf)))
                    bk = assign[k]
                    residual[a] += d[j] - d[k]
                    residual[bk] += d[k] - d[j]
                    assign[j], assign[k] = bk, a
                    improved = True

    def decode_population(self, genes: np.ndarray):
        """Decode a (N, m) array. Returns repaired genes (N, m), assignments (N, n), objectives (N, 2)."""
        N = len(genes)
        Y = np.empty((N, self.inst.m), dtype=np.int8)
        A = np.empty((N, self.inst.n), dtype=np.int64)
        F = np.empty((N, 2))
        for k in range(N):
            Y[k], A[k], F[k] = self.decode(genes[k])
        return Y, A, F
