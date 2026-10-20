"""
NSGA - Non-dominated Sorting Genetic Algorithm (Srinivas & Deb, 1994),
implemented as described in Lecture 4. (The original NSGA, not NSGA-II.)

Per generation:
    1. Sort the population into non-dominated fronts (front 1 = nobody
       dominates them, front 2 = only front 1 does, ...).
    2. Front by front, starting with front 1:
         - give every member the same dummy fitness
           (front 1: N = population size;
            later fronts: slightly smaller than the minimum shared fitness
            of the previous front, so a better front always has higher fitness)
         - fitness sharing WITHIN the front, in DECISION space:
              d_ij = normalised Hamming distance between the chromosomes
                     (fraction of facilities that are open in one and closed
                      in the other)
              nc_i = sum over j in the same front of Sh(d_ij)
              shared fitness = dummy fitness / nc_i
    3. Selection: stochastic universal sampling (fitness proportional).
    4. Variation + repair (shared with MOGA) -> the children REPLACE the
       parents (generational GA, no elitism).
The final result is the offline archive of all non-dominated solutions found.

Differences to MOGA (Lecture 4): (1) fitness comes from fronts instead of
'1 + number of dominators', (2) niching uses decision-space distance instead
of objective-space distance.
"""
from __future__ import annotations

import time

import numpy as np

from src.common import EAConfig, RunResult
from src.data_loader import CFLPInstance
from src.operators import init_population, sharing_function, stochastic_universal_sampling, variation
from src.pareto import OfflineArchive, fast_non_dominated_sort
from src.representation import Decoder

EPSILON = 0.01   # "slightly smaller": next front's dummy fitness = (1 - EPSILON) * min shared fitness


def hamming_distances(Y: np.ndarray) -> np.ndarray:
    """Normalised Hamming distance: share of genes that differ (0 = identical, 1 = opposite)."""
    return (Y[:, None, :] != Y[None, :, :]).mean(axis=2)


def nsga_fitness(Y: np.ndarray, F: np.ndarray, sigma_share: float):
    """Return (front index of each solution, shared fitness). Higher fitness = better."""
    N = len(F)
    front_of = np.empty(N, dtype=int)
    fitness = np.empty(N)
    dummy = float(N)
    for r, front in enumerate(fast_non_dominated_sort(F)):
        niche_count = sharing_function(hamming_distances(Y[front]), sigma_share).sum(axis=1)
        shared = dummy / niche_count
        fitness[front] = shared
        front_of[front] = r
        dummy = (1.0 - EPSILON) * shared.min()
    return front_of, fitness


def run_nsga(inst: CFLPInstance, cfg: EAConfig, seed: int) -> RunResult:
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    dec = Decoder(inst)
    N, pm = cfg.pop_size, cfg.pm(inst.m)
    archive = OfflineArchive(inst.m)

    Y, F = dec.decode_population(init_population(rng, N, inst.m, cfg.init))
    archive.update(F, Y)
    evals, gen = N, 0

    while evals + N <= cfg.max_evals:
        _, fitness = nsga_fitness(Y, F, cfg.sigma_nsga)
        parents = Y[stochastic_universal_sampling(rng, fitness, N)]
        Y, F = dec.decode_population(variation(rng, parents, cfg.pc, pm))   # children replace parents
        archive.update(F, Y)
        evals += N
        gen += 1

    front, genes = archive.result()
    return RunResult("NSGA", front, genes, evals, time.perf_counter() - t0, gen)
