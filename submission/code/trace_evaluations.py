#!/usr/bin/env python3
"""
Trace-Level Evidence Logger
============================

Produces a per-candidate CSV trace of every objective evaluation in the
UQ-QAOA experimental pipeline.  Each row records:
  - method: algorithm name
  - family: graph family (ER, REG, BA, WS)
  - instance: instance index within family
  - n: graph size
  - query_index: evaluation number within this method-instance pair
  - gamma_0, gamma_1, ...: angle parameters (phase)
  - beta_0, beta_1, ...: angle parameters (mixer)
  - ratio: exact approximation ratio from statevector simulation
  - best_so_far: running maximum ratio for this method-instance pair
  - graph_seed: deterministic seed used to generate this graph

This trace enables:
  - Independent verification of every reported number
  - Convergence analysis (best-so-far vs query index)
  - Method comparison at any sub-budget
  - Detection of anomalous evaluations

Output: code/tables/trace_all_evaluations.csv

Usage:
    python code/trace_evaluations.py
"""
from __future__ import annotations

import csv
import os
import pathlib

BASE = pathlib.Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(BASE / ".mplconfig"))

import numpy as np

from uq_qaoa_core import (
    CANDIDATE_BUDGET, DEPTH, FAMILIES, METHODS,
    _eval_uq_qaoa_sequential, _build_uq_qaoa_posterior,
    angle_candidates, build_training_library, get_trained_gin,
    graph_edges, graph_features, make_instances, qaoa_cost_values,
    qaoa_ratio, stable_seed, table_path,
)


def gen_trace():
    """Generate full evaluation trace for all methods and instances."""
    print("=" * 60)
    print("Trace-Level Evidence: per-candidate evaluation log")
    print("=" * 60, flush=True)

    build_training_library(depth=DEPTH)
    get_trained_gin(depth=DEPTH)

    n = 14
    budget = CANDIDATE_BUDGET
    n_inst = 2  # per family

    instances = make_instances(FAMILIES, n, n_inst)
    path = table_path("trace_all_evaluations")

    # Header adapts to the active depth so higher-depth studies remain auditable.
    angle_cols = [f"gamma_{l}" for l in range(DEPTH)] + \
                 [f"beta_{l}" for l in range(DEPTH)]
    fieldnames = ["method", "family", "instance", "n", "query_index",
                  *angle_cols, "ratio", "best_so_far", "graph_seed"]

    total_rows = 0
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for fam, inst, edges, feats, graph_n in instances:
            c_vals = qaoa_cost_values(edges, graph_n)
            graph_seed = stable_seed("trace", fam, inst)

            for method in METHODS:
                if method == "uq_qaoa":
                    # Sequential evaluation — log actual evaluation order
                    seed = stable_seed(method, fam, inst)
                    safety_prefix = angle_candidates(
                        "tqa", min(6, budget), graph_n, edges, feats,
                        stable_seed("tqa", fam, inst), depth=DEPTH)
                    result = _eval_uq_qaoa_sequential(
                        graph_n, edges, feats, c_vals, DEPTH, budget, seed,
                        safety_prefix)
                    _, _, seq_ratios, seq_cands = result
                    best_so_far = 0.0
                    for qi, (c, r) in enumerate(zip(seq_cands, seq_ratios)):
                        gamma = c[:DEPTH]
                        beta = c[DEPTH:]
                        best_so_far = max(best_so_far, r)
                        row = {
                            "method": method,
                            "family": fam,
                            "instance": inst,
                            "n": graph_n,
                            "query_index": qi,
                            "ratio": f"{r:.6f}",
                            "best_so_far": f"{best_so_far:.6f}",
                            "graph_seed": graph_seed,
                        }
                        for l in range(DEPTH):
                            row[f"gamma_{l}"] = f"{gamma[l]:.6f}"
                            row[f"beta_{l}"] = f"{beta[l]:.6f}"
                        writer.writerow(row)
                        total_rows += 1
                    continue

                seed = stable_seed(method, fam, inst)
                cands = angle_candidates(method, budget, graph_n, edges,
                                         feats, seed, depth=DEPTH)
                best_so_far = 0.0
                for qi, c in enumerate(cands):
                    gamma = c[:DEPTH]
                    beta = c[DEPTH:]
                    r = qaoa_ratio(gamma, beta, edges, graph_n, c_vals)
                    best_so_far = max(best_so_far, r)

                    row = {
                        "method": method,
                        "family": fam,
                        "instance": inst,
                        "n": graph_n,
                        "query_index": qi,
                        "ratio": f"{r:.6f}",
                        "best_so_far": f"{best_so_far:.6f}",
                        "graph_seed": graph_seed,
                    }
                    for l in range(DEPTH):
                        row[f"gamma_{l}"] = f"{gamma[l]:.6f}"
                        row[f"beta_{l}"] = f"{beta[l]:.6f}"
                    writer.writerow(row)
                    total_rows += 1

    print(f"\nWrote {path} ({total_rows} rows)")
    print(f"  {len(METHODS)} methods x {len(instances)} instances x {budget} queries")
    print(f"  = {len(METHODS) * len(instances) * budget} expected rows")


if __name__ == "__main__":
    gen_trace()
