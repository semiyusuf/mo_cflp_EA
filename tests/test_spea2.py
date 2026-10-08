import numpy as np

from src.common import EAConfig
from src.data_loader import load_instance
from src.pareto import nondominated_mask
from src.problem import is_feasible
from src.representation import Decoder
from src.spea2 import environmental_selection, run_spea2, spea2_fitness, truncate
from tests.synthetic import write_synthetic_instance


def test_strength_and_raw_fitness():
    F = np.array([[1, 5], [2, 3], [3, 4], [5, 5]], float)
    fit, _ = spea2_fitness(F)
    # S = [1, 2, 1, 0]; R(2) = S(1) = 2 ; R(3) = S(0)+S(1)+S(2) = 4
    np.testing.assert_array_equal(np.floor(fit), [0, 0, 2, 4])
    assert np.all(fit - np.floor(fit) <= 0.5)      # density < 1/2


def test_truncation_removes_most_crowded():
    F = np.array([[0, 10], [1, 9], [1.1, 8.9], [5, 5], [10, 0]], float)
    fit, D = spea2_fitness(F)
    keep = truncate(D, np.arange(5), 4)
    assert len(keep) == 4 and 0 in keep and 4 in keep and 3 in keep
    assert (1 in keep) != (2 in keep)               # one of the close pair is removed


def test_archive_filled_with_dominated_when_too_few():
    F = np.array([[1, 1], [2, 2], [3, 3], [4, 4]], float)
    fit, D = spea2_fitness(F)
    assert sorted(environmental_selection(fit, D, 3).tolist()) == [0, 1, 2]


def test_run_feasible_and_nondominated(tmp_path):
    inst = load_instance(write_synthetic_instance(tmp_path / "s.txt", m=16, n=50))
    cfg = EAConfig("t", pop_size=20, max_evals=400, pc=0.9, pm_factor=1.0, init="density")
    r = run_spea2(inst, cfg, seed=1)
    assert r.evals == 400 and r.generations == 19
    assert nondominated_mask(r.front).all()
    dec = Decoder(inst)
    for g, f in zip(r.front_genes, r.front):
        y, a, obj = dec.decode(g)
        assert is_feasible(inst, y, a) and np.allclose(obj, f)
    np.testing.assert_array_equal(r.front, run_spea2(inst, cfg, seed=1).front)
