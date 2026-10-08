"""
Compute metrics, statistical tests, tables and figures from the saved fronts.

    python scripts/analyze_results.py

Outputs (all in results/):
    hv_normalisation.csv     ideal/nadir bounds and HV reference point per instance
    all_runs.csv             one row per run: HV, #non-dominated, time, ...
    summary_table.csv / .md  mean, std, best, worst of HV and #ND, mean time
    stats_tests.csv / .md    Mann-Whitney U test + A12 (NSGA-II vs SPEA2)
    config_effect.csv / .md  per algorithm: effect of the three configurations
    figures/front_<inst>_<cfg>.png   Pareto fronts, both MOEAs on the same axes
    figures/fronts_grid.png          all instances x configurations
    figures/box_hv.png, box_nd.png, box_time.png
"""
import json

import numpy as np
import pandas as pd

from config import ALL_INSTANCES, CONFIGS, RESULTS_DIR

from src.metrics import REF_POINT, bounds_from_fronts, hypervolume_2d, normalise
from src.plotting import save_front_figure, save_front_grid, save_metric_boxplots
from src.statistics_tests import compare, summary

ALG_ORDER = ["NSGA-II", "SPEA2"]


def load_runs():
    rows, fronts = [], {}
    for meta_path in sorted((RESULTS_DIR / "fronts").glob("*/*/*.json")):
        meta = json.loads(meta_path.read_text())
        F = np.loadtxt(meta_path.with_suffix(".csv"), delimiter=",", skiprows=1, ndmin=2)[:, :2]
        key = (meta["instance"], meta["config"], meta["algorithm"], meta["run"])
        fronts[key] = F
        rows.append(meta)
    if not rows:
        raise SystemExit("no results found - run scripts/run_experiments.py first")
    return pd.DataFrame(rows), fronts


def to_markdown(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        cells = [floatfmt.format(v) if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main():
    df, fronts = load_runs()
    (RESULTS_DIR / "figures").mkdir(parents=True, exist_ok=True)
    present = set(df.instance)
    instances = [i for i in ALL_INSTANCES if i in present] + sorted(present - set(ALL_INSTANCES))
    configs = [c for c in CONFIGS if c in set(df.config)]

    # 1) normalisation bounds per instance over ALL fronts of BOTH algorithms
    bounds, norm_rows = {}, []
    for inst in instances:
        lo, hi = bounds_from_fronts([F for k, F in fronts.items() if k[0] == inst])
        bounds[inst] = (lo, hi)
        norm_rows.append(dict(instance=inst, f1_min=lo[0], f1_max=hi[0], f2_min=lo[1], f2_max=hi[1],
                              ref_f1_normalised=REF_POINT[0], ref_f2_normalised=REF_POINT[1],
                              ref_f1_original=lo[0] + REF_POINT[0] * (hi[0] - lo[0]),
                              ref_f2_original=lo[1] + REF_POINT[1] * (hi[1] - lo[1])))
    pd.DataFrame(norm_rows).to_csv(RESULTS_DIR / "hv_normalisation.csv", index=False)

    # 2) hypervolume of every run
    df["HV"] = [hypervolume_2d(normalise(fronts[(r.instance, r.config, r.algorithm, r.run)], *bounds[r.instance]))
                for r in df.itertuples()]
    df = df.sort_values(["instance", "config", "algorithm", "run"])
    df[["instance", "config", "algorithm", "run", "seed", "HV", "n_nondominated", "time_s",
        "evals", "generations"]].to_csv(RESULTS_DIR / "all_runs.csv", index=False)

    # 3) summary table + statistical tests
    summ, tests = [], []
    for inst in instances:
        for cfg in configs:
            sub = df[(df.instance == inst) & (df.config == cfg)]
            for alg in ALG_ORDER:
                s = sub[sub.algorithm == alg]
                if s.empty:
                    continue
                hv, nd, t = summary(s.HV), summary(s.n_nondominated), summary(s.time_s)
                summ.append(dict(instance=inst, config=cfg, algorithm=alg, runs=len(s),
                                 HV_mean=hv["mean"], HV_std=hv["std"], HV_best=hv["best"], HV_worst=hv["worst"],
                                 ND_mean=nd["mean"], ND_std=nd["std"], ND_best=int(nd["best"]),
                                 ND_worst=int(nd["worst"]), time_mean_s=t["mean"], time_std_s=t["std"]))
            a, b = sub[sub.algorithm == "NSGA-II"], sub[sub.algorithm == "SPEA2"]
            if len(a) > 1 and len(b) > 1:
                for metric in ("HV", "n_nondominated", "time_s"):
                    r = compare(a[metric].values, b[metric].values)
                    r["winner"] = {"first": "NSGA-II", "second": "SPEA2"}.get(r["winner"], r["winner"])
                    tests.append(dict(instance=inst, config=cfg, metric=metric, **r))
    summ = pd.DataFrame(summ)
    summ.to_csv(RESULTS_DIR / "summary_table.csv", index=False)
    (RESULTS_DIR / "summary_table.md").write_text(to_markdown(summ))
    if tests:
        tests = pd.DataFrame(tests)
        tests.to_csv(RESULTS_DIR / "stats_tests.csv", index=False)
        (RESULTS_DIR / "stats_tests.md").write_text(to_markdown(tests))

    # 4) effect of the configurations (per instance and algorithm, averaged over runs)
    eff = (df.groupby(["instance", "algorithm", "config"])
             .agg(HV_mean=("HV", "mean"), ND_mean=("n_nondominated", "mean"),
                  time_mean_s=("time_s", "mean"), evals=("evals", "first"))
             .reset_index())
    eff.to_csv(RESULTS_DIR / "config_effect.csv", index=False)
    (RESULTS_DIR / "config_effect.md").write_text(to_markdown(eff))

    # 5) Pareto-front figures: for each algorithm the run with the median HV
    grid = {}
    for inst in instances:
        for cfg in configs:
            shown = {}
            for alg in ALG_ORDER:
                s = df[(df.instance == inst) & (df.config == cfg) & (df.algorithm == alg)]
                if s.empty:
                    continue
                med = s.iloc[int(np.argmin(np.abs(s.HV.values - s.HV.median())))]
                shown[alg] = fronts[(inst, cfg, alg, med.run)]
            grid[(inst, cfg)] = shown
            save_front_figure(RESULTS_DIR / "figures" / f"front_{inst}_{cfg}.png", shown,
                              f"{inst}, config {cfg}: median-HV run of each MOEA")
    save_front_grid(RESULTS_DIR / "figures" / "fronts_grid.png", grid, instances, configs)
    save_metric_boxplots(RESULTS_DIR / "figures" / "box_hv.png", df, "HV", "Hypervolume (normalised)", instances)
    save_metric_boxplots(RESULTS_DIR / "figures" / "box_nd.png", df, "n_nondominated", "# non-dominated solutions", instances)
    save_metric_boxplots(RESULTS_DIR / "figures" / "box_time.png", df, "time_s", "Execution time (s)", instances)

    pd.set_option("display.width", 200)
    print(summ.round(4).to_string(index=False))
    if len(tests):
        print()
        print(tests[tests.metric == "HV"].round(4).to_string(index=False))
    print(f"\nTables and figures written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
