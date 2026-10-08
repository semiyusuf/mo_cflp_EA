import numpy as np

from src.common import EAConfig
from src.data_loader import load_instance
from src.nsga2 import crowding_distance, environmental_selection, fast_non_dominated_sort, run_nsga2
from src.pareto import nondominated_mask
from src.problem import evaluate, is_feasible
from src.representation import Decoder
from tests.synthetic import write_synthetic_instance

F = np.array([[1, 5], [2, 3], [4, 1], [3, 4], [5, 5], [2, 3]], float)


def test_non_dominated_sort():
    fronts = [sorted(f.tolist()) for f in fast_non_dominated_sort(F)]
    assert fronts == [[0, 1, 2, 5], [3], [4]]


def test_crowding_distance():
    cd = crowding_distance(np.array([[0, 4], [1, 2], [2, 1], [4, 0]], float))
    assert np.isinf(cd[0]) and np.isinf(cd[3])
    np.testing.assert_allclose(cd[1:3], [(2 - 0) / 4 + (4 - 1) / 4, (4 - 1) / 4 + (2 - 0) / 4])


def test_environmental_selection_keeps_best_front():
    keep = environmental_selection(F, 4)
    assert sorted(keep.tolist()) == [0, 1, 2, 5]


def test_run_feasible_and_nondominated(tmp_path):
    inst = load_instance(write_synthetic_instance(tmp_path / "s.txt", m=16, n=50))
    cfg = EAConfig("t", pop_size=20, max_evals=400, pc=0.9, pm_factor=1.0, init="density")
    r = run_nsga2(inst, cfg, seed=1)
    assert r.evals == 400 and r.generations == 19
    assert nondominated_mask(r.front).all()
    dec = Decoder(inst)
    for g, f in zip(r.front_genes, r.front):
        y, a, obj = dec.decode(g)
        assert is_feasible(inst, y, a) and np.allclose(obj, f)
    # same seed -> identical result
    np.testing.assert_array_equal(r.front, run_nsga2(inst, cfg, seed=1).front)
