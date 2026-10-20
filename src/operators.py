"""
Problem-specific variation operators. Both MOGA and NSGA call exactly the
same functions here (same initialisation, crossover, mutation), followed by the
same decoder/repair (src/representation.py).
"""
from __future__ import annotations

import numpy as np


def init_population(rng: np.random.Generator, N: int, m: int, method: str) -> np.ndarray:
    """Return N random 0/1 chromosomes of length m (repaired later by the decoder).

    method = "random":  every gene is 1 with probability 0.5.
    method = "density": each individual first draws its own opening
                        probability p ~ U(0.1, 1.0), then each gene is 1 with
                        probability p. This spreads the initial population
                        over many different numbers of open facilities, i.e.
                        along the whole f1 axis.
    """
    if method == "random":
        p = np.full((N, 1), 0.5)
    elif method == "density":
        p = rng.uniform(0.1, 1.0, size=(N, 1))
    else:
        raise ValueError(f"unknown init method {method!r}")
    return (rng.random((N, m)) < p).astype(np.int8)


def uniform_crossover(rng, p1: np.ndarray, p2: np.ndarray, pc: float):
    """With probability pc swap each gene between the parents with prob. 0.5;
    otherwise return copies of the parents."""
    c1, c2 = p1.copy(), p2.copy()
    if rng.random() < pc:
        swap = rng.random(len(p1)) < 0.5
        c1[swap], c2[swap] = p2[swap], p1[swap]
    return c1, c2


def bit_flip_mutation(rng, genes: np.ndarray, pm: float) -> np.ndarray:
    """Flip each gene independently with probability pm (open <-> close)."""
    flip = rng.random(len(genes)) < pm
    out = genes.copy()
    out[flip] = 1 - out[flip]
    return out


def variation(rng, parents: np.ndarray, pc: float, pm: float) -> np.ndarray:
    """Create len(parents) offspring: consecutive parents are paired for
    crossover, then every child is mutated. Children are not yet repaired."""
    N = len(parents)
    children = np.empty_like(parents)
    for k in range(0, N, 2):
        a = parents[k]
        b = parents[k + 1] if k + 1 < N else parents[0]   # odd N: pair last with first
        c1, c2 = uniform_crossover(rng, a, b, pc)
        children[k] = bit_flip_mutation(rng, c1, pm)
        if k + 1 < N:
            children[k + 1] = bit_flip_mutation(rng, c2, pm)
    return children


def stochastic_universal_sampling(rng, fitness: np.ndarray, n_select: int) -> np.ndarray:
    """Fitness-proportional selection (Baker, 1987), as used by MOGA and NSGA.
    Higher fitness = larger slice of a roulette wheel. Instead of spinning the
    wheel n times, SUS spins ONCE and reads n equally spaced pointers, which
    gives every solution a number of copies very close to its expected value.
    The chosen indices are shuffled so that crossover pairs are random."""
    p = fitness / fitness.sum()
    cum = np.cumsum(p)
    cum[-1] = 1.0                                       # guard against rounding
    step = 1.0 / n_select
    pointers = rng.random() * step + step * np.arange(n_select)
    chosen = np.searchsorted(cum, pointers, side="right")
    return rng.permutation(chosen)


def sharing_function(d: np.ndarray, sigma_share: float, alpha: float = 1.0) -> np.ndarray:
    """Goldberg & Richardson (1987) sharing function:
        Sh(d) = 1 - (d / sigma_share)^alpha   if d < sigma_share
                0                             otherwise
    Sh(0) = 1 (a solution fully shares with itself / its copies)."""
    return np.where(d < sigma_share, 1.0 - (d / sigma_share) ** alpha, 0.0)
