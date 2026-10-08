"""
SPEA2 (Zitzler, Laumanns & Thiele, 2001) implemented from scratch.

Population P (size N) and external archive A (size N_bar = N).
Per generation:
    1. Fitness of every member of P + A:
         strength  S(i) = number of solutions i dominates
         raw       R(i) = sum of S(j) over all j that dominate i  (0 = non-dominated)
         density   D(i) = 1 / (sigma_i^k + 2), sigma_i^k = distance to the
                    k-th nearest neighbour, k = floor(sqrt(|P| + |A|))
         fitness   F(i) = R(i) + D(i)   (lower is better; < 1 <=> non-dominated)
       Distances are Euclidean in objective space min-max normalised over P + A,
       so f1 and f2 (very different magnitudes) count equally.
    2. Environmental selection -> new archive of exactly N_bar members:
         copy all non-dominated solutions (F < 1);
         too few  -> fill with the best dominated ones by fitness;
         too many -> truncation: repeatedly remove the solution with the
                     smallest distance to its nearest neighbour (ties
                     broken by 2nd nearest, 3rd, ...).
    3. Termination check (evaluation budget) -> result = non-dominated archive.
    4. Mating selection: binary tournament on the archive by fitness.
    5. Variation (shared operators) + repair/decoding -> new population P.
"""
from __future__ import annotations

import time

import numpy as np

from src.common import EAConfig, RunResult, front_with_genes
from src.data_loader import CFLPInstance
from src.operators import init_population, variation
from src.pareto import dominance_matrix
from src.representation import Decoder


def normalised_distances(F: np.ndarray) -> np.ndarray:
    lo, hi = F.min(axis=0), F.max(axis=0)
    Fn = (F - lo) / np.where(hi > lo, hi - lo, 1.0)
    D = np.sqrt(((Fn[:, None, :] - Fn[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(D, np.inf)   # a solution is not its own neighbour
    return D


def spea2_fitness(F: np.ndarray):
    """Return (fitness, distance matrix) for all solutions in F."""
    dom = dominance_matrix(F)                 # dom[a, b]: a dominates b
    strength = dom.sum(axis=1)                # S(i)
    raw = (dom * strength[:, None]).sum(axis=0)   # R(i) = sum of S(j) for j dominating i
    D = normalised_distances(F)
    k = int(np.sqrt(len(F)))
    sigma_k = np.sort(D, axis=1)[:, k - 1]    # distance to k-th nearest neighbour
    density = 1.0 / (sigma_k + 2.0)
    return raw + density, D


def truncate(D: np.ndarray, idx: np.ndarray, size: int) -> np.ndarray:
    """SPEA2 archive truncation: iteratively remove the member whose sorted
    neighbour-distance list is lexicographically smallest."""
    idx = np.asarray(idx)
    while len(idx) > size:
        sub = np.sort(D[np.ix_(idx, idx)], axis=1)        # row: distances to others, ascending
        worst = np.lexsort(sub.T[::-1])[0]                # lexicographic minimum
        idx = np.delete(idx, worst)
    return idx


def environmental_selection(fitness: np.ndarray, D: np.ndarray, size: int) -> np.ndarray:
    nd = np.flatnonzero(fitness < 1.0)
    if len(nd) == size:
        return nd
    if len(nd) < size:
        return np.argsort(fitness, kind="stable")[:size]   # non-dominated first, then best dominated
    return truncate(D, nd, size)


def tournament(rng, fitness: np.ndarray, n_select: int) -> np.ndarray:
    """Binary tournament on fitness (lower wins, ties random)."""
    a = rng.integers(0, len(fitness), n_select)
    b = rng.integers(0, len(fitness), n_select)
    a_wins = (fitness[a] < fitness[b]) | ((fitness[a] == fitness[b]) & (rng.random(n_select) < 0.5))
    return np.where(a_wins, a, b)


def run_spea2(inst: CFLPInstance, cfg: EAConfig, seed: int) -> RunResult:
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    dec = Decoder(inst)
    N, pm = cfg.pop_size, cfg.pm(inst.m)
    archive_size = N

    Y, _, F = dec.decode_population(init_population(rng, N, inst.m, cfg.init))
    evals, gen = N, 0
    AY, AF = np.empty((0, inst.m), dtype=np.int8), np.empty((0, 2))   # empty archive

    while True:
        UY, UF = np.vstack([Y, AY]), np.vstack([F, AF])
        fit, D = spea2_fitness(UF)
        keep = environmental_selection(fit, D, archive_size)
        AY, AF, afit = UY[keep], UF[keep], fit[keep]

        if evals + N > cfg.max_evals:
            break

        parents = AY[tournament(rng, afit, N)]
        Y, _, F = dec.decode_population(variation(rng, parents, cfg.pc, pm))
        evals += N
        gen += 1

    front, genes = front_with_genes(AF, AY, afit < 1.0)
    return RunResult("SPEA2", front, genes, evals, time.perf_counter() - t0, gen)
