#!/usr/bin/env python3
"""
Figure 3: Held-Out Quality Validation
=======================================

Generates a bar chart comparing approximation ratios on an independent
held-out test set (n = 12, tag="abl") to verify that the UQ-QAOA
advantage is not due to overfitting to specific graph instances.

Experimental setup:
  - Graph size n = 12, QAOA depth p = 3, budget Q = 18.
  - 8 held-out instances (2 per family, tag="abl" for distinct seeds).
  - These graphs have NO overlap with the training library.

Output: code/figures/fig03_heldout_quality.pdf
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


def gen_fig03():
    """Generate the held-out quality validation bar chart.

    Creates 8 test instances with tag="abl" (ablation) to ensure
    different seeds from the main evaluation, then evaluates all
    6 methods with budget Q = 18.
    """
    path = figure_path("fig03_heldout_quality")
    n = 12; budget = CANDIDATE_BUDGET
    instances = make_instances(FAMILIES, n, 2, tag="abl")
    res = eval_all_methods(instances, DEPTH, budget, tag="abl")
    mean_ratio = {m: np.mean([r[0] for r in res[m]]) for m in METHODS}

    fig, ax = plt.subplots(figsize=(8, 4))
    vals_r = [mean_ratio[m] for m in METHODS]
    bars = ax.bar(METHOD_LABELS, vals_r, color=COLORS, edgecolor="black",
                  linewidth=0.5)
    ax.set_ylabel("Approximation ratio", fontsize=14)
    ax.set_title(f"Held-out quality check \u2013 $n={n}$, $p={DEPTH}$",
                 fontsize=15, pad=12)
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis='x', labelsize=11, rotation=15)
    ax.tick_params(axis='y', labelsize=12)
    for bar, val in zip(bars, vals_r):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=11,
                fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  [done] {path.name}")


if __name__ == "__main__":
    print("=" * 60)
    print("Figure 3: Held-out quality validation")
    print(f"  QAOA depth p={DEPTH}, n=12, budget Q={CANDIDATE_BUDGET}, tag=abl")
    print("=" * 60, flush=True)
    build_training_library(depth=DEPTH)
    get_trained_gin(depth=DEPTH)
    gen_fig03()
