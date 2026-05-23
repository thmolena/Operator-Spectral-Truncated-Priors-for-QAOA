#!/usr/bin/env python3
"""
Table 2: Analytical Bound Verification (Appendix)
===================================================

Generates Table 10 in the appendix, which verifies the analytical
bounds derived in the theoretical analysis (Section III):

  1. Lipschitz constant L_G (Proposition 1):
     L_G = 2 * |E| * p for a MaxCut QAOA circuit.

  2. Expected-quality lower bound (Theorem 1):
     E[C(theta)] >= r* C_max - L_G sigma sqrt(d)

  3. Best-of-K gap (Proposition 2):
     gap <= L_G sigma sqrt(d) sqrt(2 ln K) / K

  4. Volume ratio of trust region to full domain.

  5. Noisy evaluation lower bound (Proposition 3).

  6. Generalization gap (Corollary 1).

Experimental setup:
  - Reference graph: 3-regular graph on n=14 vertices.
  - sigma_j = 0.15, r* = 0.851, K = 3, epsilon = 0.01, n_train = 48.

Output: code/tables/table02_bound_verification.csv
"""
from __future__ import annotations

import csv
import math
import os
import pathlib

BASE = pathlib.Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(BASE / ".mplconfig"))

import numpy as np

from uq_qaoa_core import (
  CANDIDATE_BUDGET, DEPTH, graph_edges, stable_seed, table_path,
)


def gen_table02():
    """Generate the analytical bound verification table.

    All bounds are computed from the theoretical formulas in the
    manuscript (Section III).  No simulation is needed -- this table
    is purely analytical.
    """
    path = table_path("table02_bound_verification")
    n = 14; p = DEPTH; dim = 2 * p
    sigma_j = 0.15; r_star = 0.865
    edges = graph_edges("REG", n, stable_seed("reg14"), degree=3)
    m = len(edges)
    # Proposition 1: Lipschitz constant for MaxCut QAOA
    L_G = 2.0 * m * p
    domain_widths = [math.pi] * p + [math.pi / 2] * p
    # Trust-region volume (Gaussian sigma per dimension)
    vol_trust = np.prod([sigma_j * math.sqrt(2 * math.pi * math.e)] * dim)
    vol_domain = np.prod(domain_widths)
    vol_ratio = vol_trust / vol_domain
    # Theorem 1: noise penalty
    noise_penalty = L_G * sigma_j * math.sqrt(dim)
    # Proposition 2: best-of-K gap
    K = 3
    bestofk_gap = (L_G * sigma_j * math.sqrt(dim) *
                   math.sqrt(2 * math.log(K)) / K)
    # Proposition 3: noisy evaluation bound
    eps = 0.01
    noise_noisy = L_G * (sigma_j + eps) * math.sqrt(dim)
    # Corollary 1: generalization gap
    n_train = 48
    gen_gap = L_G * sigma_j * math.sqrt(2 * dim * math.log(2) / n_train)
    rows = [
        {"Quantity": "Lipschitz L_G (Prop. 1)", "Bound": f"{L_G:.1f}"},
        {"Quantity": "Expected-quality lower bound",
         "Bound": f">= {r_star:.3f} C_max - {noise_penalty:.2f}"},
        {"Quantity": f"Best-of-K gap (K={K})",
         "Bound": f"<= {bestofk_gap:.2f}"},
        {"Quantity": "Volume ratio",
         "Bound": f"{vol_ratio:.2e}"},
        {"Quantity": f"Expected noisy lower bound (eps={eps})",
         "Bound": f">= {r_star - eps:.3f} C_max - {noise_noisy:.2f}"},
        {"Quantity": "Generalization gap",
         "Bound": f"<= {gen_gap:.2f}"},
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Quantity", "Bound"])
        w.writeheader()
        w.writerows(rows)
    print(f"  [done] {path.name}")


if __name__ == "__main__":
    print("=" * 60)
    print("Table 2: Analytical bound verification")
    print(f"  QAOA depth p={DEPTH}, n=14, REG-3 graph, budget Q={CANDIDATE_BUDGET}")
    print("=" * 60, flush=True)
    gen_table02()
