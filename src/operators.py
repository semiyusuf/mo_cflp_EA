"""
Problem-specific variation operators. Both NSGA-II and SPEA2 call exactly the
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
