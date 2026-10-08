import numpy as np
import pytest

from src.data_loader import load_instance
from src.problem import evaluate, is_feasible
from src.representation import Decoder
from tests.synthetic import write_synthetic_instance


@pytest.fixture
def inst(tmp_path):
    return load_instance(write_synthetic_instance(tmp_path / "syn.txt", m=16, n=50, seed=3))


def test_any_bitstring_decodes_to_feasible(inst):
    dec = Decoder(inst)
    rng = np.random.default_rng(0)
    for _ in range(300):
        y, X, (f1, f2) = dec.decode(rng.integers(0, 2, inst.m))
        assert is_feasible(inst, y, X)
        assert (f1, f2) == evaluate(inst, y, X)
        assert np.all(np.isin(X, (0.0, 1.0)))         # no oversized customers -> pure 0/1


def test_all_closed_and_all_open(inst):
    dec = Decoder(inst)
    for genes in (np.zeros(inst.m, int), np.ones(inst.m, int)):
        y, X, _ = dec.decode(genes)
        assert is_feasible(inst, y, X)
        assert np.array_equal(y == 1, X.sum(axis=1) > 0)   # no unused open facility


def test_objectives_by_hand(tmp_path):
    p = tmp_path / "toy.txt"
    p.write_text(" 2 2\n 10 100\n 10 200\n 6\n 1 5\n 6\n 2 3\n")
    inst = load_instance(p)
    y, X, (f1, f2) = Decoder(inst).decode(np.array([1, 0]))
    # open capacity 10 < demand 12 -> capacity repair opens facility 1
    assert list(y) == [1, 1]
    assert f1 == 300
    assert f2 == pytest.approx(1 + 3)   # cust 0 -> fac 0 (cost 1), cust 1 -> fac 1 (cost 3)


def test_decoder_is_deterministic(inst):
    dec = Decoder(inst)
    g = np.random.default_rng(1).integers(0, 2, inst.m)
    assert dec.decode(g)[2] == dec.decode(g)[2]


def test_greedy_fragmentation_is_repaired(tmp_path):
    # 2 facilities x cap 10, demands 5,5,4,4,2 (total 20 = all capacity).
    # Cost-greedy puts 5+4 in each facility -> 1 unit left in each -> the
    # last customer (demand 2) does not fit anywhere. The fallback must repack.
    p = tmp_path / "frag.txt"
    p.write_text(" 2 5\n 10 100\n 10 100\n"
                 " 5\n 1 9\n 5\n 9 1\n 4\n 1 9\n 4\n 9 1\n 2\n 1 1\n")
    inst = load_instance(p)
    y, X, (f1, f2) = Decoder(inst).decode(np.array([1, 1]))
    assert is_feasible(inst, y, X)
    assert f1 == 200


def test_oversized_customer_is_split(tmp_path):
    # capacities 10, 10, 10; customer 0 needs 15 (> any capacity), customer 1 needs 5.
    # Costs for customer 0 (all its demand): fac0 30, fac1 60, fac2 90.
    p = tmp_path / "big.txt"
    p.write_text(" 3 2\n 10 100\n 10 100\n 10 100\n"
                 " 15\n 30 60 90\n 5\n 1 1 1\n")
    inst = load_instance(p)
    y, X, (f1, f2) = Decoder(inst).decode(np.array([1, 1, 1]))
    assert is_feasible(inst, y, X)
    # cheapest first: 10 units at fac0 (2/3 of demand), 5 units at fac1 (1/3)
    np.testing.assert_allclose(X[:, 0], [2 / 3, 1 / 3, 0])
    # customer 1 (single-source) fits only in fac1 (5 left) -> fac2 closed
    np.testing.assert_allclose(X[:, 1], [0, 1, 0])
    assert list(y) == [1, 1, 0]
    assert f2 == pytest.approx(2 / 3 * 30 + 1 / 3 * 60 + 1)
