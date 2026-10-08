"""
NSGA-II (Deb et al., 2002) implemented from scratch.

Per generation:
    1. Binary tournament selection with the crowded-comparison operator
       (lower rank wins; equal rank -> larger crowding distance wins).
    2. Variation (shared operators) + repair/decoding (shared decoder).
    3. Elitist environmental selection on parents + offspring (size 2N):
       fill the new population front by front (fast non-dominated sort);
       the front that does not fit completely is truncated by keeping the
       solutions with the largest crowding distance.
"""
from __future__ import annotations

import time

import numpy as np

from src.common import EAConfig, RunResult, front_with_genes
from src.data_loader import CFLPInstance
from src.operators import init_population, variation
from src.pareto import dominance_matrix
from src.representation import Decoder


def fast_non_dominated_sort(F: np.ndarray) -> list[np.ndarray]:
    """Return a list of fronts (arrays of indices); fronts[0] is the best."""
    D = dominance_matrix(F)
    dominated_count = D.sum(axis=0)          # n_p: how many solutions dominate p
    fronts = []
    current = np.flatnonzero(dominated_count == 0)
    while current.size:
        fronts.append(current)
        # removing the current front reduces n_q for every q it dominates
        dominated_count = dominated_count - D[current].sum(axis=0)
        dominated_count[current] = -1        # mark as already assigned
        current = np.flatnonzero(dominated_count == 0)
    return fronts


def crowding_distance(F: np.ndarray) -> np.ndarray:
    """Crowding distance of every solution within one front."""
    n, M = F.shape
    dist = np.zeros(n)
    if n <= 2:
        dist[:] = np.inf
        return dist
    for k in range(M):
        order = np.argsort(F[:, k], kind="stable")
        fmin, fmax = F[order[0], k], F[order[-1], k]
        dist[order[0]] = dist[order[-1]] = np.inf      # keep boundary points
        if fmax > fmin:
            dist[order[1:-1]] += (F[order[2:], k] - F[order[:-2], k]) / (fmax - fmin)
    return dist


def rank_and_crowding(F: np.ndarray):
    rank = np.empty(len(F), dtype=int)
    crowd = np.empty(len(F))
    for r, front in enumerate(fast_non_dominated_sort(F)):
        rank[front] = r
        crowd[front] = crowding_distance(F[front])
    return rank, crowd


def tournament(rng, rank, crowd, n_select: int) -> np.ndarray:
    """Binary tournament with the crowded-comparison operator."""
    a = rng.integers(0, len(rank), n_select)
    b = rng.integers(0, len(rank), n_select)
    a_wins = (rank[a] < rank[b]) | ((rank[a] == rank[b]) & (crowd[a] > crowd[b]))
    tie = (rank[a] == rank[b]) & (crowd[a] == crowd[b])
    a_wins |= tie & (rng.random(n_select) < 0.5)      # full tie -> random
    return np.where(a_wins, a, b)


def environmental_selection(F: np.ndarray, N: int) -> np.ndarray:
    """Elitist (mu + lambda) survivor selection: indices of the N survivors."""
    survivors = []
    for front in fast_non_dominated_sort(F):
        if len(survivors) + len(front) <= N:
            survivors.extend(front)
        else:
            cd = crowding_distance(F[front])
            order = np.argsort(-cd, kind="stable")
            survivors.extend(front[order[: N - len(survivors)]])
            break
    return np.array(survivors)


def run_nsga2(inst: CFLPInstance, cfg: EAConfig, seed: int) -> RunResult:
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    dec = Decoder(inst)
    N, pm = cfg.pop_size, cfg.pm(inst.m)

    Y, F = dec.decode_population(init_population(rng, N, inst.m, cfg.init))
    evals, gen = N, 0
    rank, crowd = rank_and_crowding(F)

    while evals + N <= cfg.max_evals:
        parents = Y[tournament(rng, rank, crowd, N)]
        Yc, Fc = dec.decode_population(variation(rng, parents, cfg.pc, pm))
        evals += N
        gen += 1

        Yu, Fu = np.vstack([Y, Yc]), np.vstack([F, Fc])
        keep = environmental_selection(Fu, N)
        Y, F = Yu[keep], Fu[keep]
        rank, crowd = rank_and_crowding(F)

    front, genes = front_with_genes(F, Y, rank == 0)
    return RunResult("NSGA-II", front, genes, evals, time.perf_counter() - t0, gen)
