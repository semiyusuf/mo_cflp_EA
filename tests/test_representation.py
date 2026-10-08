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
        genes = rng.integers(0, 2, inst.m)
        y, assign, (f1, f2) = dec.decode(genes)
        assert is_feasible(inst, y, assign)
        assert (f1, f2) == evaluate(inst, y, assign)


def test_all_closed_and_all_open(inst):
    dec = Decoder(inst)
    for genes in (np.zeros(inst.m, int), np.ones(inst.m, int)):
        y, assign, _ = dec.decode(genes)
        assert is_feasible(inst, y, assign)
        assert set(np.flatnonzero(y)) == set(assign)   # no unused open facility


def test_objectives_by_hand(tmp_path):
    p = tmp_path / "toy.txt"
    p.write_text(" 2 2\n 10 100\n 10 200\n 6\n 1 5\n 6\n 2 3\n")
    inst = load_instance(p)
    y, assign, (f1, f2) = Decoder(inst).decode(np.array([1, 0]))
    # facility 0 (cap 10) capacity 10 < demand 12 -> capacity repair opens facility 1
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
    y, assign, (f1, f2) = Decoder(inst).decode(np.array([1, 1]))
    assert is_feasible(inst, y, assign)
    assert f1 == 200
