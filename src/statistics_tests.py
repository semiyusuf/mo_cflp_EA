"""
Statistical comparison of the two MOEAs over independent runs.

- Summary statistics: mean, standard deviation, best, worst.
- Mann-Whitney U test (Wilcoxon rank-sum), two-sided, alpha = 0.05: are the
  HV values of NSGA-II and SPEA2 drawn from the same distribution? Non-
  parametric, so no normality assumption is needed for 10 runs.
- Vargha-Delaney A12 effect size: probability that a random NSGA-II run has
  a larger value than a random SPEA2 run (0.5 = no difference;
  |A12 - 0.5| >= 0.21 is usually called a large effect).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import mannwhitneyu


def summary(values) -> dict:
    v = np.asarray(values, float)
    return {"mean": v.mean(), "std": v.std(ddof=1) if len(v) > 1 else 0.0,
            "best": v.max(), "worst": v.min()}


def vargha_delaney_a12(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    greater = (a[:, None] > b[None, :]).sum()
    equal = (a[:, None] == b[None, :]).sum()
    return float((greater + 0.5 * equal) / (len(a) * len(b)))


def compare(a, b, alpha: float = 0.05) -> dict:
    """Two-sided Mann-Whitney U test of samples a vs b plus A12 effect size."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.all(a == a[0]) and np.all(b == a[0]):
        p = 1.0                                   # identical samples
    else:
        p = float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
    a12 = vargha_delaney_a12(a, b)
    if p >= alpha:
        winner = "no significant difference"
    else:
        winner = "first" if a12 > 0.5 else "second"
    return {"p_value": p, "A12": a12, "significant": p < alpha, "winner": winner}
