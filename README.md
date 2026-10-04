# Multi-Objective Capacitated Facility Location (CFLP) with NSGA-II and SPEA2

ACIT4610 – Mid-Term Group Project 2, 2026 – Group <NUMBER>

Two Multi-Objective Evolutionary Algorithms (NSGA-II and SPEA2) implemented
from scratch in Python to solve the bi-objective CFLP on OR-Library
instances (Beasley). Objectives: minimise total facility-opening cost (f1)
and total customer-allocation cost (f2).

## Project structure
```
data/      OR-Library instances (cap41, cap42, cap101, cap102, cap121, cap122)
src/       Core code: data loading, representation/repair, operators,
           NSGA-II, SPEA2, metrics, statistics, plotting
scripts/   Experiment configurations and runnable scripts
tests/     Unit tests
results/   Generated tables and figures
```

## Installation
```bash
git clone <REPO_URL>
cd cflp-moea
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Running
_TODO: commands to run experiments and reproduce figures._

## Data source
J. E. Beasley, OR-Library, Capacitated Warehouse Location:
https://people.brunel.ac.uk/~mastjjb/jeb/orlib/capinfo.html