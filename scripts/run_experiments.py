import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from config import ALL_INSTANCES, CONFIGS, DATA_DIR, N_RUNS, REPORT_INSTANCES, RESULTS_DIR, seed_for

from src.data_loader import load_instance
from src.nsga2 import run_nsga2
from src.spea2 import run_spea2

ALGORITHMS = {"NSGA-II": run_nsga2, "SPEA2": run_spea2}


def out_paths(instance, cfg, alg, run):
    d = RESULTS_DIR / "fronts" / instance / cfg
    stem = f"{alg}_run{run:02d}"
    return d / f"{stem}.csv", d / f"{stem}.json"


def one_run(instance, cfg_name, alg, run):
    inst = load_instance(DATA_DIR / f"{instance}.txt")
    cfg = CONFIGS[cfg_name]
    seed = seed_for(run)
    res = ALGORITHMS[alg](inst, cfg, seed)

    csv_path, json_path = out_paths(instance, cfg_name, alg, run)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    header = "f1,f2," + ",".join(f"y{i}" for i in range(inst.m))
    np.savetxt(csv_path, np.hstack([res.front, res.front_genes]), delimiter=",",
               header=header, comments="", fmt="%.6f")
    meta = dict(instance=instance, config=cfg_name, algorithm=alg, run=run, seed=seed,
                time_s=res.time_s, evals=res.evals, generations=res.generations,
                n_nondominated=len(res.front), **cfg.__dict__)
    json_path.write_text(json.dumps(meta, indent=1))
    return f"{instance} {cfg_name} {alg:7s} run {run:2d}: {len(res.front):3d} pts, {res.time_s:6.2f}s"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instances", nargs="+", default=REPORT_INSTANCES,
                    help="instance names, or 'all' (default: %(default)s)")
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    ap.add_argument("--algorithms", nargs="+", default=list(ALGORITHMS), choices=list(ALGORITHMS))
    ap.add_argument("--runs", type=int, default=N_RUNS)
    ap.add_argument("--jobs", type=int, default=1, help="parallel worker processes (default 1)")
    ap.add_argument("--force", action="store_true", help="re-run even if results exist")
    args = ap.parse_args()
    instances = ALL_INSTANCES if args.instances == ["all"] else args.instances

    for name in instances:
        if not (DATA_DIR / f"{name}.txt").exists():
            raise SystemExit(f"data/{name}.txt not found - run: python scripts/download_data.py")

    tasks = [(i, c, a, r) for i in instances for c in args.configs
             for r in range(args.runs) for a in args.algorithms
             if args.force or not out_paths(i, c, a, r)[1].exists()]
    print(f"{len(tasks)} runs to do ({args.jobs} worker(s))")

    if args.jobs == 1:
        for t in tasks:
            print(one_run(*t), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            for fut in as_completed([pool.submit(one_run, *t) for t in tasks]):
                print(fut.result(), flush=True)
    print("done - now run: python scripts/analyze_results.py")


if __name__ == "__main__":
    main()
