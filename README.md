# Multi-Objective Capacitated Facility Location (CFLP) with NSGA-II and SPEA2

ACIT4610 – Mid-Term Group Project 2, 2026 – Group <NUMBER>

Two Multi-Objective Evolutionary Algorithms (NSGA-II and SPEA2) implemented
from scratch in Python to solve the bi-objective CFLP on OR-Library
instances (Beasley). Objectives: minimise total facility-opening cost
f1 = Σ F_i y_i and total customer-allocation cost f2 = Σ_i Σ_j C_ij x_ij.

## Project structure
```
data/                     OR-Library instances (cap41, cap42, cap101, cap102, cap121, cap122)
src/
  data_loader.py          reads the OR-Library files (S_i, F_i, d_j, C_ij)
  problem.py              objective evaluation (f1, f2) and feasibility check
  representation.py       binary chromosome + repair/decoding to a feasible solution
  operators.py            initialisation, uniform crossover, bit-flip mutation (shared)
  pareto.py               dominance utilities (shared)
  common.py               parameter configuration and run-result containers
  nsga2.py                NSGA-II: non-dominated sort, crowding distance, tournament, elitist selection
  spea2.py                SPEA2: strength/raw fitness, density, archive, truncation, tournament
  metrics.py              normalisation + exact 2-D hypervolume
  statistics_tests.py     Mann-Whitney U test and Vargha-Delaney A12
  plotting.py             Pareto-front and box plots
scripts/
  config.py               the three parameter configurations, seeds, number of runs
  download_data.py        downloads and validates the six instances
  run_experiments.py      runs both MOEAs (all configs x 10 runs), saves final fronts
  analyze_results.py      HV, tables, statistical tests and figures
  example_decoding.py     worked example of decoding one individual
tests/                    unit tests (pytest)
results/                  generated fronts, tables and figures
```

## Installation
Requires Python 3.11 or newer.

```bash
git clone <REPO_URL>
cd <REPO_FOLDER>
python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Running

All commands are run from the repository root with the virtual environment active.

```bash
# 1. Get the data (skips files already in data/) and check every file parses
python scripts/download_data.py

# 2. (optional) unit tests
python -m pytest -q

# 3. Run the experiments: cap41, cap101, cap121 x configs A, B, C x 10 runs x 2 MOEAs
python scripts/run_experiments.py
#    all six instances:          python scripts/run_experiments.py --instances all
#    quick smoke test (~10 s):   python scripts/run_experiments.py --instances cap41 --configs A --runs 1

# 4. Compute hypervolume, statistics, tables and figures
python scripts/analyze_results.py

# 5. Worked example of the representation and repair (report section 2)
python scripts/example_decoding.py --instance cap41
```

`run_experiments.py` skips runs whose results already exist, so it can be
interrupted and restarted. `--jobs N` runs N processes in parallel, which is
faster but makes the measured execution times less precise; the reported
times use the default `--jobs 1`.

### Outputs (in `results/`)
| File | Content |
|---|---|
| `fronts/<inst>/<cfg>/<alg>_runXX.csv/.json` | final non-dominated set of every run + seed, time, evaluations |
| `all_runs.csv` | HV, number of non-dominated solutions and time for every run |
| `summary_table.csv/.md` | mean, std, best, worst HV and #ND, mean time per instance/config/algorithm |
| `stats_tests.csv/.md` | Mann-Whitney U test (α = 0.05) and A12, NSGA-II vs SPEA2 |
| `config_effect.csv/.md` | effect of the three configurations |
| `hv_normalisation.csv` | ideal/nadir bounds and HV reference point per instance |
| `figures/front_<inst>_<cfg>.png` | Pareto fronts of both MOEAs on the same axes (median-HV run) |
| `figures/fronts_grid.png`, `figures/box_*.png` | overview grid and box plots of HV, #ND, time |

## Method summary

**Representation.** A chromosome is a binary vector y of length m (1 = facility
open). Customer assignments are produced by a deterministic decoder, so both
MOEAs search the same space with the same operators.

**Repair / decoding** (`src/representation.py`), applied to every new chromosome:
1. *Capacity repair*: while open capacity < total demand, open the closed facility
   with the lowest unit cost F_i/S_i + mean_j(C_ij/d_j).
2. *Oversized customers*: a customer whose demand exceeds every facility capacity
   (cap41/cap42: customer 33, d = 12,912 > S = 5,000) cannot be single-sourced. Its
   demand is split over open facilities, cheapest C_ij first; x_ij is the served
   fraction and costs x_ij · C_ij (C_ij is the cost of allocating *all* of j's demand).
   All other customers are single-sourced (x_ij ∈ {0, 1}).
3. *Greedy assignment*: remaining customers in decreasing order of demand go to the
   open facility with the lowest C_ij that still has room. If none has room, the closed
   facility minimising F_i + C_ij (with S_i ≥ d_j) is opened. If capacity is
   fragmented so that nothing fits, the customers are re-packed with best-fit
   decreasing and f2 is improved with capacity-feasible shift/swap moves.
4. *Clean-up*: open facilities that serve no demand are closed.

The repaired chromosome is written back (Lamarckian), and f1, f2 are computed
only on the resulting feasible solution. C_ij is used as given (it already
includes demand).

**Variation** (identical for both MOEAs): uniform crossover with probability pc,
bit-flip mutation with probability pm = pm_factor / m per gene.

**Parameter configurations** (`scripts/config.py`); within a configuration both
MOEAs use identical settings and the same seeds (run r uses seed 2026 + r):

| Config | Population (= SPEA2 archive) | Max evaluations | pc | pm | Initialisation |
|---|---|---|---|---|---|
| A | 50 | 5,000 | 0.9 | 1/m | density |
| B | 100 | 10,000 | 0.9 | 1/m | density |
| C | 100 | 20,000 | 0.7 | 3/m | random |

*random*: each gene is 1 with probability 0.5. *density*: each individual draws
its own p ~ U(0.1, 1.0), then each gene is 1 with probability p.

**Final approximation set**: NSGA-II, rank-1 solutions of the final population;
SPEA2, non-dominated members of the final archive. Duplicate objective vectors are
removed before counting.

**Hypervolume**: per instance, the ideal and nadir points are taken over the union
of all final fronts of both MOEAs, all configurations and all runs. Objectives are
min-max normalised to [0, 1] and HV is computed exactly (2-D) with reference
point (1.1, 1.1). Values are saved in `results/hv_normalisation.csv`.

## Data source
J. E. Beasley, OR-Library, Capacitated Warehouse Location:
https://people.brunel.ac.uk/~mastjjb/jeb/orlib/capinfo.html
