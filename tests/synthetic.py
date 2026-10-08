"""
TEST-ONLY helper: writes a small random instance in the OR-Library file format
so the unit tests can run without the real data. It is NEVER used by the
experiment scripts, which only read the real OR-Library files in data/.
"""
from pathlib import Path

import numpy as np


def write_synthetic_instance(path: Path, m: int = 8, n: int = 20, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    demand = rng.integers(10, 100, size=n)
    capacity = np.full(m, int(np.ceil(1.6 * demand.sum() / m * 2)))  # ~ half the facilities suffice
    fixed = rng.integers(500, 1500, size=m).astype(float)
    cost = rng.uniform(1.0, 20.0, size=(n, m)) * demand[:, None]

    lines = [f" {m} {n}"]
    lines += [f" {capacity[i]} {fixed[i]:.1f}" for i in range(m)]
    for j in range(n):
        lines.append(f" {demand[j]}")
        row = [f"{c:.3f}" for c in cost[j]]
        for k in range(0, m, 7):  # wrap like the real files
            lines.append(" " + " ".join(row[k:k + 7]))
    path.write_text("\n".join(lines) + "\n")
    return path
