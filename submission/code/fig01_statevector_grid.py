#!/usr/bin/env python3
"""
Figure 1: Cross-Size Approximation-Ratio Comparison Grid
=========================================================

Generates a grouped bar chart comparing the approximation ratios of all
six QAOA parameter-initialization algorithms across four graph sizes
(n = 8, 10, 12, 14).  For each size, 2 instances from each of the 4
graph families (ER, REG, BA, WS) are generated, totaling 8 instances
per size.

Experimental protocol (matches Section IV of the manuscript):
  - QAOA depth p = 3, candidate budget Q = 18 for all methods.
  - Approximation ratio = <psi(t*)|H_C|psi(t*)> / C_max, where t* is
    the best candidate selected from the budget.
  - All methods receive the same set of graph instances and budget,
    ensuring a fair comparison.

Output: code/figures/fig01_statevector_grid.pdf

This script imports all algorithmic infrastructure from uq_qaoa_core.py.
No results are hard-coded; every data point is computed from scratch
using exact statevector simulation.
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

SIZES = [8, 10, 12, 14]


def gen_fig01():
    """Generate the cross-size approximation ratio grouped bar chart.

    For each graph size n in {8, 10, 12, 14}:
      1. Create 8 test instances (2 per family, deterministic seeds).
      2. Evaluate all 6 methods with budget Q = 18.
      3. Average the best approximation ratio across the 8 instances.
    Plot the results as a grouped bar chart.
    """
    path = figure_path("fig01_statevector_grid")
    budget = CANDIDATE_BUDGET
    results = {sz: {} for sz in SIZES}
    for sz in SIZES:
        instances = make_instances(FAMILIES, sz, 2, tag=str(sz))
        res = eval_all_methods(instances, DEPTH, budget, tag=str(sz))
        for m in METHODS:
            results[sz][m] = np.mean([r[0] for r in res[m]])

    x = np.arange(len(SIZES))
    width = 0.12
    fig, ax = plt.subplots(figsize=(10, 5))
    for mi, method in enumerate(METHODS):
        vals = [results[sz][method] for sz in SIZES]
        ax.bar(x + mi * width, vals, width, label=METHOD_LABELS[mi],
               color=COLORS[mi], edgecolor="black", linewidth=0.4)
    ax.set_xlabel("Graph size $n$", fontsize=14)
    ax.set_ylabel("Approximation ratio", fontsize=14)
    ax.set_title(f"Statevector approximation ratio \u2013 $p={DEPTH}$, all families",
                 fontsize=15, pad=12)
    ax.set_xticks(x + width * 2.5)
    ax.set_xticklabels([str(s) for s in SIZES], fontsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.legend(fontsize=11, ncol=3, loc="upper left",
              framealpha=0.9, edgecolor="0.7")
    ax.set_ylim(0, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  [done] {path.name}")


if __name__ == "__main__":
    print("=" * 60)
    print("Figure 1: Cross-size approximation ratio grid")
    print(f"  QAOA depth p={DEPTH}, budget Q={CANDIDATE_BUDGET}, sizes={SIZES}")
    print("=" * 60, flush=True)
    build_training_library(depth=DEPTH)
    get_trained_gin(depth=DEPTH)
    gen_fig01()
