"""Plotting of Pareto fronts and metric distributions (matplotlib only)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")   # no window needed; figures are written to files
import matplotlib.pyplot as plt
import numpy as np

STYLE = {
    "NSGA-II": dict(color="#2a6fdb", marker="o"),
    "SPEA2": dict(color="#e0731f", marker="s"),
}


def plot_fronts(ax, fronts: dict[str, np.ndarray], title: str = ""):
    """Scatter f1 against f2 for each algorithm on the same axes."""
    for alg, F in fronts.items():
        st = STYLE.get(alg, {})
        ax.plot(F[:, 0], F[:, 1], "-", color=st.get("color"), alpha=0.35, lw=1)
        ax.scatter(F[:, 0], F[:, 1], s=28, label=f"{alg} ({len(F)} pts)", alpha=0.85,
                   edgecolors="white", linewidths=0.5, **st)
    ax.set_xlabel("f1: facility-opening cost")
    ax.set_ylabel("f2: customer-allocation cost")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)


def save_front_figure(path: Path, fronts: dict[str, np.ndarray], title: str):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    plot_fronts(ax, fronts, title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_front_grid(path: Path, grid: dict[tuple[str, str], dict[str, np.ndarray]],
                    instances: list[str], configs: list[str]):
    """Rows = instances, columns = configurations."""
    fig, axes = plt.subplots(len(instances), len(configs),
                             figsize=(4.6 * len(configs), 3.8 * len(instances)), squeeze=False)
    for r, inst in enumerate(instances):
        for c, cfg in enumerate(configs):
            if (inst, cfg) in grid:
                plot_fronts(axes[r][c], grid[(inst, cfg)], f"{inst} - config {cfg}")
            else:
                axes[r][c].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_metric_boxplots(path: Path, df, metric: str, ylabel: str, instances: list[str] | None = None):
    """One panel per instance; boxes grouped by configuration and algorithm."""
    instances = instances or list(dict.fromkeys(df["instance"]))
    configs = list(dict.fromkeys(df["config"]))
    algs = list(STYLE)
    fig, axes = plt.subplots(1, len(instances), figsize=(4.6 * len(instances), 3.8), squeeze=False)
    for ax, inst in zip(axes[0], instances):
        for a, alg in enumerate(algs):
            data = [df[(df.instance == inst) & (df.config == c) & (df.algorithm == alg)][metric].values
                    for c in configs]
            pos = np.arange(len(configs)) * 3 + a
            bp = ax.boxplot(data, positions=pos, widths=0.8, patch_artist=True,
                            medianprops=dict(color="black"))
            for box in bp["boxes"]:
                box.set_facecolor(STYLE[alg]["color"])
                box.set_alpha(0.6)
            ax.plot([], [], "s", color=STYLE[alg]["color"], label=alg)
        ax.set_xticks(np.arange(len(configs)) * 3 + 0.5)
        ax.set_xticklabels([f"config {c}" for c in configs])
        ax.set_title(inst)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
