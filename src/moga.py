"""
MOGA - Multi-Objective Genetic Algorithm (Fonseca & Fleming, 1993),
implemented as described in Lecture 3.

Per generation:
    1. Rank:     r_i = 1 + (number of solutions that dominate i).
                 All non-dominated solutions get rank 1. Some ranks may be empty.
    2. Average fitness per rank (linear mapping from N for the best rank to
       1 for the worst, then averaged over each rank):
                 F_i = N - sum_{k=1}^{r_i - 1} mu(k) - 0.5 * (mu(r_i) - 1)
                 mu(k) = number of solutions with rank k.
    3. Niching (fitness sharing) WITHIN each rank, in OBJECTIVE space:
                 d_ij = Euclidean distance of the min-max normalised (f1, f2)
                 nc_i = sum over j with the same rank of Sh(d_ij)
                 F'_i = F_i / nc_i
    4. Scaling: inside each rank, multiply the shared fitness so the rank keeps
       its original total fitness (mu(r) * F_avg). Ranking keeps the
       convergence pressure; sharing only redistributes it within a rank.
    5. Selection: stochastic universal sampling (fitness proportional).
    6. Variation + repair (shared with NSGA) -> the children REPLACE the
       parents (generational GA, no elitism).
The final result is the offline archive of all non-dominated solutions found.
"""
from __future__ import annotations

import time

import numpy as np

from src.common import EAConfig, RunResult
from src.data_loader import CFLPInstance
from src.operators import init_population, sharing_function, stochastic_universal_sampling, variation
from src.pareto import OfflineArchive, dominance_matrix
from src.representation import Decoder


def moga_rank(F: np.ndarray) -> np.ndarray:
    """r_i = 1 + number of solutions dominating i."""
    return 1 + dominance_matrix(F).sum(axis=0)


def moga_average_fitness(rank: np.ndarray) -> np.ndarray:
    """F_i = N - sum_{k < r_i} mu(k) - 0.5 (mu(r_i) - 1)."""
    N = len(rank)
    mu = np.bincount(rank, minlength=N + 2)        # mu[k] = solutions with rank k (mu[0] = 0)
    better = np.cumsum(mu)[rank - 1]               # sum_{k=1}^{r_i - 1} mu(k)
    return N - better - 0.5 * (mu[rank] - 1)


def normalised_objective_distances(F: np.ndarray) -> np.ndarray:
    lo, hi = F.min(axis=0), F.max(axis=0)
    Fn = (F - lo) / np.where(hi > lo, hi - lo, 1.0)
    return np.sqrt(((Fn[:, None, :] - Fn[None, :, :]) ** 2).sum(axis=2))


def moga_fitness(F: np.ndarray, sigma_share: float):
    """Return (rank, final scaled shared fitness). Higher fitness = better."""
    rank = moga_rank(F)
    avg = moga_average_fitness(rank)
    same_rank = rank[:, None] == rank[None, :]
    niche_count = (sharing_function(normalised_objective_distances(F), sigma_share) * same_rank).sum(axis=1)
    shared = avg / niche_count                     # niche_count >= 1 (a solution shares with itself)
    fitness = shared.copy()
    for r in np.unique(rank):                      # scaling: restore each rank's total fitness
        idx = rank == r
        fitness[idx] *= avg[idx].sum() / shared[idx].sum()
    return rank, fitness


def run_moga(inst: CFLPInstance, cfg: EAConfig, seed: int) -> RunResult:
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    dec = Decoder(inst)
    N, pm = cfg.pop_size, cfg.pm(inst.m)
    archive = OfflineArchive(inst.m)

    Y, F = dec.decode_population(init_population(rng, N, inst.m, cfg.init))
    archive.update(F, Y)
    evals, gen = N, 0

    while evals + N <= cfg.max_evals:
        _, fitness = moga_fitness(F, cfg.sigma_moga)
        parents = Y[stochastic_universal_sampling(rng, fitness, N)]
        Y, F = dec.decode_population(variation(rng, parents, cfg.pc, pm))   # children replace parents
        archive.update(F, Y)
        evals += N
        gen += 1

    front, genes = archive.result()
    return RunResult("MOGA", front, genes, evals, time.perf_counter() - t0, gen)
