import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import EAConfig  # noqa: E402

DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

ALL_INSTANCES = ["cap41", "cap42", "cap101", "cap102", "cap121", "cap122"]
# one small, one medium, one large instance for the report analysis
REPORT_INSTANCES = ["cap41", "cap101", "cap121"]

N_RUNS = 10
BASE_SEED = 2026

# pm = pm_factor / m  (m = number of candidate facilities)
CONFIGS = {
    # small population, small budget
    "A": EAConfig("A", pop_size=50, max_evals=5_000, pc=0.9, pm_factor=1.0, init="density"),
    # baseline: larger population, double budget
    "B": EAConfig("B", pop_size=100, max_evals=10_000, pc=0.9, pm_factor=1.0, init="density"),
    # long run, less crossover, more mutation, plain random initialisation
    "C": EAConfig("C", pop_size=100, max_evals=20_000, pc=0.7, pm_factor=3.0, init="random"),
}


def seed_for(run: int) -> int:
    return BASE_SEED + run
