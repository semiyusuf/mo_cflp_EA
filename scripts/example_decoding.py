"""
Show step by step how one encoded individual becomes a feasible solution
(for the report's representation/repair example).

    python scripts/example_decoding.py                  # cap41, random chromosome
    python scripts/example_decoding.py --instance cap101 --seed 7
    python scripts/example_decoding.py --genes 1010000000000001   # your own chromosome
"""
import argparse

import numpy as np

from config import DATA_DIR

from src.data_loader import load_instance
from src.problem import is_feasible
from src.representation import Decoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="cap41")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--genes", help="0/1 string of length m (default: random)")
    args = ap.parse_args()

    inst = load_instance(DATA_DIR / f"{args.instance}.txt")
    if args.genes:
        genes = np.array([int(c) for c in args.genes], dtype=np.int8)
        assert len(genes) == inst.m, f"need {inst.m} genes"
    else:
        genes = (np.random.default_rng(args.seed).random(inst.m) < 0.4).astype(np.int8)

    print(f"Instance {inst.name}: m={inst.m} facilities, n={inst.n} customers, "
          f"total demand={inst.total_demand:.0f}")
    print("Chromosome y (before repair): ", "".join(map(str, genes)))
    print(f"  open facilities {np.flatnonzero(genes).tolist()}, "
          f"open capacity {inst.capacity[genes == 1].sum():.0f}")

    trace = []
    y, X, (f1, f2) = Decoder(inst).decode(genes, trace)
    print("\nRepair / decoding steps:")
    for t in trace or ["(no repair needed)"]:
        print("  -", t)

    print("\nChromosome y (after repair):  ", "".join(map(str, y)))
    print("\nFacility | capacity | load  | customers (fraction of demand if split)")
    load = X @ inst.demand
    for i in np.flatnonzero(y):
        custs = [str(j) if X[i, j] == 1 else f"{j} ({X[i, j]:.2f})" for j in np.flatnonzero(X[i] > 0)]
        print(f"  {i:6d} | {inst.capacity[i]:8.0f} | {load[i]:5.0f} | {', '.join(custs)}")
    print(f"\nFeasible: {is_feasible(inst, y, X)}")
    print(f"f1 = sum F_i y_i       = {f1:,.2f}")
    print(f"f2 = sum C_ij x_ij     = {f2:,.2f}")


if __name__ == "__main__":
    main()
