"""
Chromosome representation and feasibility repair / decoding.

REPRESENTATION
    A chromosome is a binary vector y of length m (one gene per candidate
    facility, 1 = open). Customer allocations are NOT encoded; they are
    produced deterministically by the decoder below. Both MOEAs use this same
    representation and the same decoder, so the comparison is fair.

DECODER (repair + allocation), applied to every new chromosome:
    Step 1 - capacity repair: while total open capacity < total demand, open
             the closed facility with the lowest "unit cost"
                 u_i = F_i / S_i + mean_j (C_ij / d_j)
             (opening cost per unit of capacity + average transport cost per
             unit of demand). Deterministic, ties broken by lowest index.
    Step 2 - oversized customers: a customer whose demand exceeds every
             facility capacity (cap41: customer 33, d = 12912 > S = 5000)
             cannot be served by one facility. Its demand is split over the
             open facilities, cheapest C_ij first, each taking as much as its
             residual capacity allows (opening the lowest-unit-cost closed
             facility if the open ones are full). x_ij = served fraction.
    Step 3 - greedy single-source assignment of all other customers, in
             decreasing order of demand: each goes to the open facility with
             the lowest C_ij that still has enough residual capacity. If none
             has room, the closed facility minimising F_i + C_ij among those
             with S_i >= d_j is opened and the customer is assigned there.
    Step 3b - fallback if Step 3 gets stuck (capacity fragmented: every
             facility has some room left but not enough for the next customer,
             and no suitable closed facility is left): re-pack with best-fit
             decreasing (each customer goes to the open facility with the
             SMALLEST residual capacity that fits), then improve f2 with
             shift and swap moves that keep every facility within capacity.
    Step 4 - clean-up: open facilities that serve no demand are closed
             (they would only add to f1).
    The repaired y is written back into the chromosome (Lamarckian repair),
    so the genotype always matches the evaluated, feasible phenotype.
"""
from __future__ import annotations

import numpy as np

from src.data_loader import CFLPInstance
from src.problem import evaluate, is_feasible, oversized_customers

EPS = 1e-9


class Decoder:
    """Turns any 0/1 vector into a feasible (y, X) pair and evaluates it."""

    def __init__(self, inst: CFLPInstance):
        self.inst = inst
        safe_d = np.maximum(inst.demand, 1e-12)
        self.unit_cost = inst.fixed_cost / inst.capacity + (inst.alloc_cost / safe_d).mean(axis=1)
        self.split_customers = oversized_customers(inst)
        regular = np.setdiff1d(np.arange(inst.n), self.split_customers)
        # decreasing demand; stable sort -> ties broken by customer index
        self.customer_order = regular[np.argsort(-inst.demand[regular], kind="stable")]

    # ------------------------------------------------------------------ main
    def decode(self, genes: np.ndarray, trace: list | None = None):
        """Return (y_repaired, X, (f1, f2)). If `trace` is a list, the repair
        steps are appended to it (used for the report example)."""
        inst = self.inst
        y = genes.astype(np.int8).copy()

        # Step 1: capacity repair
        while inst.capacity[y == 1].sum() < inst.total_demand:
            closed = np.flatnonzero(y == 0)
            i = closed[np.argmin(self.unit_cost[closed])]
            y[i] = 1
            if trace is not None:
                trace.append(f"capacity repair: open facility {i}")

        # Step 2: split the demand of oversized customers
        split = self._split_oversized(y, trace)          # (m, n) served amounts, 0 for others

        # Step 3: greedy single-source assignment of the other customers
        assign = self._greedy_assignment(y, split, trace)
        if assign is None:
            # Step 3b: fragmented capacity -> best-fit re-pack + local search
            if trace is not None:
                trace.append("capacity fragmented: re-pack with best-fit, then shift/swap improvement")
            assign = self._best_fit_assignment(y, split, trace)
            self._improve(y, split, assign)

        # build x_ij
        X = split / np.maximum(inst.demand, 1e-12)
        X[assign[self.customer_order], self.customer_order] = 1.0

        # Step 4: close facilities that serve nothing
        used = X.sum(axis=1) > EPS
        if trace is not None:
            for i in np.flatnonzero((y == 1) & ~used):
                trace.append(f"clean-up: close unused facility {i}")
        y[~used] = 0

        assert is_feasible(inst, y, X)
        return y, X, evaluate(inst, y, X)

    def decode_population(self, genes: np.ndarray):
        """Decode a (N, m) array. Returns repaired genes (N, m) and objectives (N, 2)."""
        N = len(genes)
        Y = np.empty((N, self.inst.m), dtype=np.int8)
        F = np.empty((N, 2))
        for k in range(N):
            Y[k], _, F[k] = self.decode(genes[k])
        return Y, F

    # --------------------------------------------------------------- helpers
    def _residual(self, y, split, assign=None):
        """Remaining capacity of every facility (0 for closed ones)."""
        inst = self.inst
        res = inst.capacity * y - split.sum(axis=1)
        if assign is not None:
            res -= np.bincount(assign[self.customer_order], weights=inst.demand[self.customer_order],
                               minlength=inst.m)
        return res

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

    def _split_oversized(self, y, trace):
        inst = self.inst
        split = np.zeros((inst.m, inst.n))
        residual = inst.capacity * y.astype(float)
        for j in self.split_customers:
            remaining = inst.demand[j]
            while remaining > EPS:
                cost = np.where((y == 1) & (residual > EPS), inst.alloc_cost[:, j], np.inf)
                i = int(np.argmin(cost))
                if not np.isfinite(cost[i]):
                    i = self._open_for(y, 0.0, j, trace, self.unit_cost)
                    if i is None:
                        raise RuntimeError(f"not enough capacity for oversized customer {j}")
                    residual[i] = inst.capacity[i]
                amount = min(residual[i], remaining)
                split[i, j] += amount
                residual[i] -= amount
                remaining -= amount
            if trace is not None:
                parts = ", ".join(f"fac {i}: {split[i, j]:.0f}" for i in np.flatnonzero(split[:, j]))
                trace.append(f"oversized customer {j} (demand {inst.demand[j]:.0f}) split -> {parts}")
        return split

    def _greedy_assignment(self, y, split, trace):
        """Cheapest open facility with room; if none, open the closed facility
        minimising F_i + C_ij. Returns None if that is impossible."""
        inst = self.inst
        residual = self._residual(y, split)
        assign = np.full(inst.n, -1, dtype=np.int64)
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

    def _best_fit_assignment(self, y, split, trace):
        """Best-fit decreasing: each customer goes to the open facility with the
        SMALLEST residual capacity that still fits it (keeps big gaps free for
        big customers). Opens the lowest-unit-cost closed facility if nothing fits."""
        inst = self.inst
        residual = self._residual(y, split)
        assign = np.full(inst.n, -1, dtype=np.int64)
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

    def _improve(self, y, split, assign):
        """Local search on f2 with fixed open facilities, keeping capacities
        feasible: shift moves (move a customer to a cheaper facility with room)
        and swap moves (exchange two customers' facilities), until no move
        improves the allocation cost."""
        inst, C, d = self.inst, self.inst.alloc_cost, self.inst.demand
        residual = self._residual(y, split, assign)
        cust = self.customer_order
        improved = True
        while improved:
            improved = False
            for j in cust:                                       # shift moves
                a = assign[j]
                cost = np.where((y == 1) & (residual >= d[j]), C[:, j], np.inf)
                i = int(np.argmin(cost))
                if cost[i] < C[a, j] - EPS:
                    residual[a] += d[j]
                    residual[i] -= d[j]
                    assign[j] = i
                    improved = True
            for j in cust:                                       # swap moves
                a, b = assign[j], assign[cust]                   # b: facility of each other customer
                delta = C[b, j] + C[a, cust] - C[a, j] - C[b, cust]
                ok = ((b != a) & (residual[a] + d[j] - d[cust] >= 0)
                      & (residual[b] + d[cust] - d[j] >= 0) & (delta < -EPS))
                if ok.any():
                    k = int(cust[np.argmin(np.where(ok, delta, np.inf))])
                    bk = assign[k]
                    residual[a] += d[j] - d[k]
                    residual[bk] += d[k] - d[j]
                    assign[j], assign[k] = bk, a
                    improved = True
