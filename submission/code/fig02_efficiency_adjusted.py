#!/usr/bin/env python3
"""
Figure 2: Efficiency-Adjusted Quality Comparison
==================================================

Generates a bar chart showing the efficiency-adjusted quality metric
ratio/evals for each method.  This metric captures both the solution
quality (approximation ratio) and the query efficiency (how quickly
the method reaches that quality).

Experimental setup:
  - Graph size n = 14, QAOA depth p = 3, budget Q = 18.
  - 8 test instances (2 per family).
  - Metric: mean( best_ratio / budget ) across instances.

Output: code/figures/fig02_efficiency_adjusted.pdf
"""
from __future__ import annotations

import os
import pathlib

BASE = pathlib.Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(BASE / ".mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from uq_qaoa_core import (
    CANDIDATE_BUDGET, COLORS, DEPTH, FAMILIES, METHODS, METHOD_LABELS,
    build_training_library, eval_all_methods, get_trained_gin,
    figure_path, make_instances,
)


def gen_fig02():
    """Generate the efficiency-adjusted quality bar chart.

    Computes ratio/evals for each method on 8 held-out n=14 instances,
    then plots the mean as a bar chart.
    """
    path = figure_path("fig02_efficiency_adjusted")
    n = 14; budget = CANDIDATE_BUDGET
    instances = make_instances(FAMILIES, n, 2)
    res = eval_all_methods(instances, DEPTH, budget)
    eff = {m: np.mean([r[0] / len(r[3]) for r in res[m]]) for m in METHODS}

    fig, ax = plt.subplots(figsize=(8, 4))
    vals = [eff[m] for m in METHODS]
    bars = ax.bar(METHOD_LABELS, vals, color=COLORS, edgecolor="black",
                  linewidth=0.5)
    ax.set_ylabel("Efficiency-adjusted quality (ratio/evals)", fontsize=14)
    ax.set_title(f"Efficiency-adjusted comparison \u2013 $n={n}$, $p={DEPTH}$",
                 fontsize=15, pad=12)
    ax.tick_params(axis='x', labelsize=11, rotation=15)
    ax.tick_params(axis='y', labelsize=12)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + max(vals) * 0.01,
                f"{val:.4f}", ha="center", va="bottom", fontsize=11,
                fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  [done] {path.name}")


if __name__ == "__main__":
    print("=" * 60)
    print("Figure 2: Efficiency-adjusted quality comparison")
    print(f"  QAOA depth p={DEPTH}, n=14, budget Q={CANDIDATE_BUDGET}")
    print("=" * 60, flush=True)
    build_training_library(depth=DEPTH)
    get_trained_gin(depth=DEPTH)
    gen_fig02()
