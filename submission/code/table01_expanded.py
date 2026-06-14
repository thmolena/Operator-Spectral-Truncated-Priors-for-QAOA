#!/usr/bin/env python3
"""
Expanded primary benchmark (larger held-out test set)
=====================================================

Identical protocol to ``table01_computational_efficiency.py`` (n=14, p=3,
Q=18, exact statevector, four families) but evaluated on a substantially
larger held-out test set to tighten confidence intervals and provide a
higher-powered paired test of UQ-QAOA vs. the strongest baseline (TQA).

Test instances use the SAME deterministic seeding namespace as the original
benchmark (``stable_seed(fam, inst, "")``), which is disjoint from the
training-library namespace (``stable_seed("train", fam, idx)``); increasing
the instance count therefore adds new held-out graphs without any train/test
leakage.

Outputs (real, reproducible):
  - tables/table01_expanded.csv          per-method mean ratio + 95% bootstrap CI
  - tables/paired_uq_vs_tqa_expanded.csv  paired advantage, CI, win rate

Usage:
    UQ_QAOA_NPF=12 python table01_expanded.py     # 12 per family = 48 instances
"""
from __future__ import annotations

import csv
import os
import pathlib

BASE = pathlib.Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(BASE / ".mplconfig"))

import numpy as np

from uq_qaoa_core import (
    CANDIDATE_BUDGET, DEPTH, FAMILIES, METHODS, METHOD_LABELS,
    build_training_library, eval_all_methods, get_trained_gin,
    make_instances, table_path,
)
from table01_computational_efficiency import (
    bootstrap_ci, eval_de_baseline, eval_bo_baseline,
    eval_multiseed_random, eval_nelder_mead, eval_cmaes,
)


def bootstrap_ci_diff(diffs, n_boot=10_000, ci=0.95, seed=0):
    """Bootstrap CI for the mean of paired differences."""
    arr = np.asarray(diffs, dtype=float)
    rng = np.random.RandomState(seed)
    means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, len(arr), size=len(arr))
        means[i] = arr[idx].mean()
    alpha = (1 - ci) / 2
    return float(np.percentile(means, 100 * alpha)), \
           float(np.percentile(means, 100 * (1 - alpha)))


def main():
    npf = int(os.environ.get("UQ_QAOA_NPF", "12"))
    n = 14
    budget = CANDIDATE_BUDGET
    print("=" * 64)
    print(f"Expanded benchmark: n={n}, p={DEPTH}, Q={budget}, "
          f"{npf} instances/family = {npf * len(FAMILIES)} held-out instances")
    print("=" * 64, flush=True)

    build_training_library(depth=DEPTH)
    get_trained_gin(depth=DEPTH)

    instances = make_instances(FAMILIES, n, npf)
    print(f"  built {len(instances)} test instances", flush=True)

    res = eval_all_methods(instances, DEPTH, budget)
    per_inst = {m: np.array([r[0] for r in res[m]]) for m in METHODS}

    # Black-box optimizer baselines (same as primary table)
    per_inst["de"] = np.array(eval_de_baseline(instances, DEPTH, budget))
    per_inst["bo"] = np.array(eval_bo_baseline(instances, DEPTH, budget))
    per_inst["multiseed"] = np.array(
        eval_multiseed_random(instances, DEPTH, budget, n_seeds=5))
    per_inst["nelder_mead"] = np.array(eval_nelder_mead(instances, DEPTH, budget))
    per_inst["cmaes"] = np.array(eval_cmaes(instances, DEPTH, budget))

    all_methods = METHODS + ["de", "bo", "multiseed", "nelder_mead", "cmaes"]
    all_labels = METHOD_LABELS + ["DE (scipy)", "BO (GP-EI)",
                                   "Multi-seed random", "Nelder-Mead", "CMA-ES"]

    tqa_mean = per_inst["tqa"].mean()

    # --- per-method table ---
    rows = []
    for mi, m in enumerate(all_methods):
        vals = per_inst[m]
        lo, hi = bootstrap_ci(vals)
        rows.append({
            "Method": all_labels[mi] + (" (ours)" if m == "uq_qaoa" else ""),
            "Ratio": f"{vals.mean():.3f}",
            "CI_95": f"[{lo:.3f}, {hi:.3f}]",
            "Delta_vs_TQA": f"{(vals.mean() - tqa_mean):+.3f}",
            "N_instances": str(len(instances)),
            "Candidate_budget": str(budget),
        })
    path = table_path("table01_expanded")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # --- paired UQ vs TQA ---
    uq = per_inst["uq_qaoa"]; tqa = per_inst["tqa"]
    diffs = uq - tqa
    plo, phi = bootstrap_ci_diff(diffs)
    wins = int((diffs > 0).sum()); ties = int((diffs == 0).sum())
    losses = int((diffs < 0).sum())
    paired = {
        "n_instances": len(instances),
        "uq_mean": f"{uq.mean():.4f}",
        "tqa_mean": f"{tqa.mean():.4f}",
        "paired_advantage": f"{diffs.mean():+.4f}",
        "paired_CI95_lo": f"{plo:+.4f}",
        "paired_CI95_hi": f"{phi:+.4f}",
        "wins": wins, "ties": ties, "losses": losses,
        "win_rate": f"{wins / len(instances):.3f}",
    }
    ppath = table_path("paired_uq_vs_tqa_expanded")
    with open(ppath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(paired.keys()))
        w.writeheader(); w.writerow(paired)

    # --- per-instance long-format dump (for figures) ---
    ipath = table_path("per_instance_expanded")
    with open(ipath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["family", "instance", "method", "ratio"])
        w.writeheader()
        for mi, (fam, inst, edges, feats, gn) in enumerate(instances):
            for m in all_methods:
                w.writerow({"family": fam, "instance": inst, "method": m,
                            "ratio": f"{per_inst[m][mi]:.6f}"})

    print("\n--- per-method (real, this run) ---")
    for r in rows:
        print(f"  {r['Method']:22s} {r['Ratio']}  {r['CI_95']}  "
              f"d_vs_TQA={r['Delta_vs_TQA']}")
    print("\n--- paired UQ-QAOA vs TQA ---")
    print(f"  instances     : {paired['n_instances']}")
    print(f"  UQ mean       : {paired['uq_mean']}")
    print(f"  TQA mean      : {paired['tqa_mean']}")
    print(f"  advantage     : {paired['paired_advantage']} "
          f"CI95 [{paired['paired_CI95_lo']}, {paired['paired_CI95_hi']}]")
    print(f"  win/tie/loss  : {wins}/{ties}/{losses}  "
          f"(win rate {paired['win_rate']})")
    print(f"\nWrote {path.name} and {ppath.name}")


if __name__ == "__main__":
    main()
