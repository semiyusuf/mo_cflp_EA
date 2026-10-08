import numpy as np
import pytest

from src.metrics import bounds_from_fronts, hypervolume_2d, normalise
from src.statistics_tests import compare, vargha_delaney_a12


def test_hv_single_point():
    assert hypervolume_2d(np.array([[0.0, 0.0]]), np.array([1.0, 1.0])) == pytest.approx(1.0)


def test_hv_staircase():
    F = np.array([[0.0, 0.5], [0.5, 0.0]])
    # rectangles: 0.5 * 0.5 + 0.5 * 1.0 = 0.75
    assert hypervolume_2d(F, np.array([1.0, 1.0])) == pytest.approx(0.75)


def test_hv_ignores_dominated_and_outside_points():
    F = np.array([[0.0, 0.5], [0.5, 0.0], [0.6, 0.6], [2.0, 0.0]])
    assert hypervolume_2d(F, np.array([1.0, 1.0])) == pytest.approx(0.75)


def test_normalisation():
    fronts = [np.array([[10.0, 300.0], [20.0, 100.0]]), np.array([[30.0, 200.0]])]
    lo, hi = bounds_from_fronts(fronts)
    np.testing.assert_allclose(normalise(fronts[0], lo, hi), [[0, 1], [0.5, 0]])


def test_stats():
    a, b = np.arange(10) + 10.0, np.arange(10) * 1.0
    assert vargha_delaney_a12(a, b) == 1.0
    r = compare(a, b)
    assert r["significant"] and r["winner"] == "first"
    assert compare(a, a)["winner"] == "no significant difference"
