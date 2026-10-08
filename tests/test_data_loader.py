import numpy as np
import pytest

from src.data_loader import load_instance
from tests.synthetic import write_synthetic_instance


def test_load_shapes_and_values(tmp_path):
    p = tmp_path / "toy.txt"
    p.write_text(
        " 2 3\n"
        " 100 50.0\n"
        " 80 70.0\n"
        " 10\n 1.5 2.5\n"
        " 20\n 3.0 4.0\n"
        " 30\n 5.0\n 6.0\n"  # values may wrap over lines
    )
    inst = load_instance(p)
    assert (inst.m, inst.n) == (2, 3)
    np.testing.assert_allclose(inst.capacity, [100, 80])
    np.testing.assert_allclose(inst.fixed_cost, [50, 70])
    np.testing.assert_allclose(inst.demand, [10, 20, 30])
    # alloc_cost[i, j] = cost of customer j at facility i
    np.testing.assert_allclose(inst.alloc_cost, [[1.5, 3.0, 5.0], [2.5, 4.0, 6.0]])


def test_truncated_file_rejected(tmp_path):
    p = tmp_path / "bad.txt"
    p.write_text(" 2 3\n 100 50\n 80 70\n 10 1 2\n")
    with pytest.raises(ValueError):
        load_instance(p)


def test_synthetic_roundtrip(tmp_path):
    inst = load_instance(write_synthetic_instance(tmp_path / "syn.txt", m=16, n=50))
    assert inst.alloc_cost.shape == (16, 50)
    assert inst.capacity.sum() >= inst.total_demand
