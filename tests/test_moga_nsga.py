import numpy as np

from src.common import EAConfig
from src.data_loader import load_instance
from src.moga import moga_average_fitness, moga_fitness, moga_rank, run_moga
from src.nsga import hamming_distances, nsga_fitness, run_nsga
from src.operators import sharing_function, stochastic_universal_sampling
from src.pareto import fast_non_dominated_sort, nondominated_mask
from src.problem import is_feasible
from src.representation import Decoder
from tests.synthetic import write_synthetic_instance

CFG = EAConfig("t", pop_size=20, max_evals=400, pc=0.9, pm_factor=1.0, init="density")


def test_moga_average_fitness_lecture_example():
    # Lecture 3, "MOGA: Step by Step": N = 6, mu(1) = 3, mu(2) = 2, mu(4) = 1
    rank = np.array([1, 2, 1, 4, 1, 2])
    np.testing.assert_allclose(moga_average_fitness(rank), [5, 2.5, 5, 1, 5, 2.5])


def test_moga_rank_and_scaling_keeps_rank_totals():
    F = np.array([[1, 5], [2, 3], [4, 1], [3, 4], [5, 5], [1.1, 4.9]], float)
    rank = moga_rank(F)
    np.testing.assert_array_equal(rank, [1, 1, 1, 2, 6, 1])  # [5,5] dominated by the other five
    r, fit = moga_fitness(F, sigma_share=0.5)
    avg = moga_average_fitness(rank)
    for k in np.unique(rank):
        assert np.isclose(fit[rank == k].sum(), avg[rank == k].sum())
    # the two crowded rank-1 points (0 and 5) get less fitness than the lonely ones
    assert fit[0] < fit[2] and fit[5] < fit[2]


def test_sharing_function():
    np.testing.assert_allclose(sharing_function(np.array([0.0, 0.25, 0.5, 0.9]), 0.5), [1, 0.5, 0, 0])


def test_nsga_better_front_always_fitter():
    rng = np.random.default_rng(0)
    Y = rng.integers(0, 2, (30, 10)).astype(np.int8)
    F = rng.random((30, 2))
    front_of, fit = nsga_fitness(Y, F, sigma_share=0.3)
    for a in range(front_of.max()):
        assert fit[front_of == a].min() > fit[front_of == a + 1].max()
    assert fit.max() <= 30
    assert sorted(np.flatnonzero(front_of == 0)) == sorted(fast_non_dominated_sort(F)[0])


def test_hamming():
    Y = np.array([[1, 0, 1, 1], [1, 1, 1, 0]], np.int8)
    np.testing.assert_allclose(hamming_distances(Y), [[0, 0.5], [0.5, 0]])


def test_sus_is_proportional():
    rng = np.random.default_rng(1)
    picks = stochastic_universal_sampling(rng, np.array([1.0, 3.0, 6.0]), 10)
    np.testing.assert_array_equal(np.bincount(picks, minlength=3), [1, 3, 6])


def _check_run(run, inst):
    r = run(inst, CFG, seed=1)
    assert r.evals == 400 and r.generations == 19
    assert nondominated_mask(r.front).all()
    dec = Decoder(inst)
    for g, f in zip(r.front_genes, r.front):
        y, X, obj = dec.decode(g)
        assert is_feasible(inst, y, X) and np.allclose(obj, f)
    np.testing.assert_array_equal(r.front, run(inst, CFG, seed=1).front)   # same seed -> same result


def test_runs_feasible_nondominated_reproducible(tmp_path):
    inst = load_instance(write_synthetic_instance(tmp_path / "s.txt", m=16, n=50))
    _check_run(run_moga, inst)
    _check_run(run_nsga, inst)
