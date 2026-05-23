#!/usr/bin/env python3
"""
Figure 4: Trust-Region Concept Illustration on C4
===================================================

Generates a bar chart demonstrating the trust-region mechanism on the
4-cycle graph (C_4) at depth p = 1.  This is a pedagogical figure
showing that even on a small, well-understood graph, the UQ-QAOA
method identifies high-quality parameters within a modest budget.

Experimental setup:
  - Graph: 4-cycle (deterministic, no randomness in construction).
  - QAOA depth p = 1, budget Q = 6.
  - All six methods evaluated for comparison.

Output: code/figures/fig04_trust_region_concept.pdf
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
    COLORS, DEPTH, METHODS, METHOD_LABELS,
    build_training_library, get_trained_gin,
    angle_candidates, graph_edges, graph_features,
    figure_path, qaoa_cost_values, qaoa_ratio, stable_seed,
)


def gen_fig04():
    """Generate the trust-region concept figure on C4.

    Uses the deterministic cycle graph (4 nodes) at depth p=1 to show
    how each method performs with a small budget of Q=6 candidates.
    """
    path = figure_path("fig04_trust_region_concept")
    n = 4; d = 1
    edges = graph_edges("cycle", n, stable_seed("C4"))
    feats = graph_features(n, edges)
    c_vals = qaoa_cost_values(edges, n)

    budget = 6
    results = {}
    for method in METHODS:
        cands = angle_candidates(method, budget, n, edges, feats,
                                 stable_seed(method, "C4"), depth=d)
        results[method] = max(
            qaoa_ratio(c[:d], c[d:], edges, n, c_vals) for c in cands)

    labels = METHOD_LABELS
    values = [results[m] for m in METHODS]
    colors = COLORS
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, values, color=colors, edgecolor="black",
                  linewidth=0.5)
    ax.set_ylabel("Approximation ratio", fontsize=14)
    ax.set_title("Trust-region mechanism \u2013 $p=1$ QAOA on $C_4$",
                 fontsize=15, pad=12)
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis='x', labelsize=11, rotation=15)
    ax.tick_params(axis='y', labelsize=12)
    for bar, val in zip(bars, values):
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
    print("Figure 4: Trust-region concept on C4")
    print("  QAOA depth p=1, budget Q=6, graph=C4")
    print("=" * 60, flush=True)
    build_training_library(depth=DEPTH)
    get_trained_gin(depth=DEPTH)
    gen_fig04()
