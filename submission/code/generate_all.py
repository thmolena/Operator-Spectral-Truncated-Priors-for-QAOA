#!/usr/bin/env python3
"""
Master Script: Generate All Figures and Tables
================================================

Runs all figure and table generation scripts in sequence, with timing
information for each step.  The training library and GIN predictor are
built once and cached in memory, then reused by all generators.

Usage:
    python generate_all.py

This produces:
  code/figures/fig01_statevector_grid.pdf
  code/figures/fig02_efficiency_adjusted.pdf
  code/figures/fig03_heldout_quality.pdf
  code/figures/fig04_trust_region_concept.pdf
  code/figures/fig05_expanded_benchmark.pdf
  code/tables/table01_computational_efficiency.csv
  code/tables/table01_expanded.csv
  code/tables/paired_uq_vs_tqa_expanded.csv
  code/tables/table02_bound_verification.csv
  code/tables/table03_ablation.csv

Total runtime on Apple M2 Pro: ~2-5 minutes depending on GIN training
convergence and the number of QAOA evaluations.
"""
from __future__ import annotations

import time

from uq_qaoa_core import (
    CANDIDATE_BUDGET, DEPTH, GLOBAL_SEED,
    build_training_library, get_trained_gin,
)

# Import generators from their respective modules
from fig01_statevector_grid import gen_fig01
from fig02_efficiency_adjusted import gen_fig02
from fig03_heldout_quality import gen_fig03
from fig04_trust_region_concept import gen_fig04
from fig05_expanded_benchmark import gen_fig05
from table01_computational_efficiency import gen_table01
from table01_expanded import main as gen_table01_expanded
from table02_bound_verification import gen_table02
from table03_ablation import gen_table03
from trace_evaluations import gen_trace


if __name__ == "__main__":
    print("=" * 60)
    print("Generating all figures and tables")
    print(f"  QAOA depth p={DEPTH}, budget Q={CANDIDATE_BUDGET}, seed={GLOBAL_SEED}")
    print("=" * 60, flush=True)

    # Build shared state once (cached in uq_qaoa_core module globals)
    build_training_library(depth=DEPTH)
    get_trained_gin(depth=DEPTH)

    generators = [
        ("Fig 01: Cross-size ratio grid", gen_fig01),
        ("Fig 02: Efficiency-adjusted quality", gen_fig02),
        ("Fig 03: Held-out quality check", gen_fig03),
        ("Fig 04: Trust-region concept (C4)", gen_fig04),
        ("Table 01: Computational efficiency", gen_table01),
        ("Table 01-expanded: 48-instance replication", gen_table01_expanded),
        ("Fig 05: Expanded 48-instance benchmark", gen_fig05),
        ("Table 02: Bound verification", gen_table02),
        ("Table 03: Ablation study", gen_table03),
        ("Trace: Per-candidate evaluation log", gen_trace),
    ]
    t_total = time.perf_counter()
    for name, fn in generators:
        print(f"\n[{name}]", flush=True)
        t0 = time.perf_counter()
        fn()
        elapsed = time.perf_counter() - t0
        print(f"  ({elapsed:.1f}s)", flush=True)
    total = time.perf_counter() - t_total
    print(f"\nAll done in {total:.1f}s total.", flush=True)
