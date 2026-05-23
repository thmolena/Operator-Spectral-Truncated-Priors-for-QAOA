#!/usr/bin/env python3
"""
Table 3: Ablation Study
========================

Evaluates UQ-QAOA ablation variants on n = 14 graphs to show the
contribution of each method component.

Ablation variants:
  1. Mean only:        Use GIN mean as single candidate (no covariance).
  2. Mean + fixed box: Uniform samples in fixed [mu-0.2, mu+0.2] box.
  3. Mean + isotropic: Replace learned covariance with sigma^2 I.
  4. No local prior:   Remove k-NN source from posterior.
  5. No global prior:  Remove population source from posterior.
  6. No coord probes:  Replace coordinate probes with random samples.
  7. No sequential:    Static probing without adaptive refinement.
  8. Full UQ-QAOA:     The complete proposed method (sequential).

Output: code/tables/table03_ablation.csv
"""
from __future__ import annotations

import csv
import os
import pathlib

BASE = pathlib.Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(BASE / ".mplconfig"))

import numpy as np

from uq_qaoa_core import (
    CANDIDATE_BUDGET, DEPTH, FAMILIES,
    build_training_library, get_trained_gin, make_instances,
    predict_gaussian, qaoa_cost_values, qaoa_ratio, stable_seed,
    angle_candidates, table_path,
)


def ablation_candidates(variant, budget, n, edges, features, seed, depth=DEPTH):
    """Generate candidates for a specific ablation variant."""
    dim = 2 * depth
    rng = np.random.RandomState(seed)
    cands = []

    mu_gin, sigma2_gin = predict_gaussian(n, edges, depth)

    library = build_training_library(depth=depth)
    lib_feats = np.array([e["features"] for e in library])
    all_angles_full = np.array([e["angles"] for e in library])
    all_angles_d = np.column_stack([all_angles_full[:, :depth],
                                    all_angles_full[:, depth:depth + depth]])

    # Global population prior
    mu_global = np.median(all_angles_d, axis=0)
    sigma2_global = np.var(all_angles_d, axis=0)
    sigma2_global = np.maximum(sigma2_global, 0.01)

    # Local k-NN prior (k=5)
    k = 5
    feat_std = lib_feats.std(axis=0)
    feat_std[feat_std < 1e-8] = 1.0
    dists = np.linalg.norm(lib_feats / feat_std - features / feat_std, axis=1)
    nn_idx = np.argsort(dists)[:k]
    mu_local = np.median(all_angles_d[nn_idx], axis=0)
    sigma2_local = np.var(all_angles_d[nn_idx], axis=0)
    sigma2_local = np.maximum(sigma2_local, 0.01)

    def _clip(theta):
        clipped = theta.copy()
        clipped[:depth] = np.clip(clipped[:depth], 0, np.pi)
        clipped[depth:] = np.clip(clipped[depth:], 0, np.pi)
        return clipped

    if variant == "mean_only":
        # Single GIN mean candidate, rest random in small box
        cands.append(_clip(mu_gin))
        for _ in range(budget - 1):
            cands.append(_clip(mu_gin + rng.uniform(-0.3, 0.3, dim)))

    elif variant == "mean_fixed_box":
        # Uniform samples in a fixed-width box around GIN mean
        cands.append(_clip(mu_gin))
        for _ in range(budget - 1):
            cands.append(_clip(mu_gin + rng.uniform(-0.2, 0.2, dim)))

    elif variant == "mean_isotropic":
        # Replace learned per-dimension covariance with isotropic
        avg_var = sigma2_gin.mean()
        sigma2_iso = np.full(dim, avg_var)
        # Three-source posterior with isotropic GIN
        prec_gin = 1.0 / np.maximum(sigma2_iso, sigma2_local)
        prec_local = 1.0 / sigma2_local
        prec_global = 1.0 / sigma2_global
        prec_post = prec_gin + prec_local + prec_global
        sigma2_post = 1.0 / prec_post
        mu_post = sigma2_post * (prec_gin * mu_gin
                                 + prec_local * mu_local
                                 + prec_global * mu_global)
        cands.append(_clip(mu_global))
        cands.append(_clip(mu_local))
        cands.append(_clip(mu_post))
        while len(cands) < budget:
            noise = rng.normal(0, 1, dim) * np.sqrt(sigma2_post)
            cands.append(_clip(mu_post + noise))

    elif variant == "no_local_prior":
        # Remove local k-NN source: two-source posterior (GIN + global)
        sigma2_gin_cal = np.maximum(sigma2_gin, sigma2_global)
        prec_gin = 1.0 / sigma2_gin_cal
        prec_global = 1.0 / sigma2_global
        prec_post = prec_gin + prec_global
        sigma2_post = 1.0 / prec_post
        mu_post = sigma2_post * (prec_gin * mu_gin
                                 + prec_global * mu_global)
        cands.append(_clip(mu_global))
        cands.append(_clip(mu_post))
        cands.append(_clip(mu_gin))
        # Deterministic coordinate probes (same as full method)
        delta = 0.10
        for anchor in [mu_global, mu_post]:
            for scale in [1.0, 2.0, 0.5, 3.0, 4.0]:
                for j in range(dim):
                    for sign in [1.0, -1.0]:
                        if len(cands) >= budget: break
                        cand = anchor.copy()
                        cand[j] += sign * delta * scale
                        cands.append(_clip(cand))
                    if len(cands) >= budget: break
                if len(cands) >= budget: break
            if len(cands) >= budget: break
        while len(cands) < budget:
            noise = rng.normal(0, 1, dim) * np.sqrt(sigma2_post)
            cands.append(_clip(mu_post + noise))

    elif variant == "no_global_prior":
        # Remove global source: two-source posterior (GIN + local)
        sigma2_gin_cal = np.maximum(sigma2_gin, sigma2_local)
        prec_gin = 1.0 / sigma2_gin_cal
        prec_local = 1.0 / sigma2_local
        prec_post = prec_gin + prec_local
        sigma2_post = 1.0 / prec_post
        mu_post = sigma2_post * (prec_gin * mu_gin
                                 + prec_local * mu_local)
        cands.append(_clip(mu_local))
        cands.append(_clip(mu_post))
        cands.append(_clip(mu_gin))
        delta = 0.10
        for anchor in [mu_local, mu_post]:
            for scale in [1.0, 2.0, 0.5, 3.0, 4.0]:
                for j in range(dim):
                    for sign in [1.0, -1.0]:
                        if len(cands) >= budget: break
                        cand = anchor.copy()
                        cand[j] += sign * delta * scale
                        cands.append(_clip(cand))
                    if len(cands) >= budget: break
                if len(cands) >= budget: break
            if len(cands) >= budget: break
        while len(cands) < budget:
            noise = rng.normal(0, 1, dim) * np.sqrt(sigma2_post)
            cands.append(_clip(mu_post + noise))

    elif variant == "no_coord_probes":
        # Full posterior but random samples instead of coordinate probes
        sigma2_gin_cal = np.maximum(sigma2_gin, sigma2_local)
        prec_gin = 1.0 / sigma2_gin_cal
        prec_local = 1.0 / sigma2_local
        prec_global = 1.0 / sigma2_global
        prec_post = prec_gin + prec_local + prec_global
        sigma2_post = 1.0 / prec_post
        mu_post = sigma2_post * (prec_gin * mu_gin
                                 + prec_local * mu_local
                                 + prec_global * mu_global)
        cands.append(_clip(mu_global))
        cands.append(_clip(mu_local))
        cands.append(_clip(mu_post))
        cands.append(_clip(mu_gin))
        while len(cands) < budget:
            noise = rng.normal(0, 1, dim) * np.sqrt(sigma2_post)
            cands.append(_clip(mu_post + noise))

    elif variant == "no_sequential":
        # Full four-source posterior with static (non-adaptive) probing.
        # This uses the pre-committed interleaved probes from the original
        # method, without sequential refinement.
        cands = angle_candidates("uq_qaoa", budget, n, edges, features,
                                  seed, depth=depth)

    return cands[:budget]


ABLATION_VARIANTS = [
    ("mean_only",        "GIN mean only"),
    ("mean_fixed_box",   "Mean + fixed box"),
    ("mean_isotropic",   "Mean + isotropic $\\sigma^2 I$"),
    ("no_local_prior",   "No local (k-NN) prior"),
    ("no_global_prior",  "No global prior"),
    ("no_coord_probes",  "No coord. probes"),
    ("no_sequential",    "No sequential refine"),
]


def gen_table03():
    """Generate Table 3: ablation study results."""
    path = table_path("table03_ablation")
    n = 14; budget = CANDIDATE_BUDGET
    instances = make_instances(FAMILIES, n, 2)

    # Full UQ-QAOA reference (from eval_all_methods)
    from uq_qaoa_core import eval_all_methods
    res_full = eval_all_methods(instances, DEPTH, budget)
    full_ratios = [r[0] for r in res_full["uq_qaoa"]]
    full_mean = np.mean(full_ratios)

    rows = []
    for variant, label in ABLATION_VARIANTS:
        per_inst = []
        for fam, inst, edges, feats, n_v in instances:
            c_vals = qaoa_cost_values(edges, n_v)
            seed = stable_seed(variant, fam, inst)
            cands = ablation_candidates(variant, budget, n_v, edges, feats,
                                         seed, depth=DEPTH)
            ratios = [qaoa_ratio(c[:DEPTH], c[DEPTH:], edges, n_v, c_vals)
                      for c in cands]
            per_inst.append(max(ratios))
        mr = np.mean(per_inst)
        rows.append({
            "Variant": label,
            "Ratio": f"{mr:.3f}",
            "Delta_vs_full": f"{(mr - full_mean):+.3f}",
        })

    # Add full UQ-QAOA row
    rows.append({
        "Variant": "Full UQ-QAOA",
        "Ratio": f"{full_mean:.3f}",
        "Delta_vs_full": "+0.000",
    })

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Variant", "Ratio", "Delta_vs_full"])
        w.writeheader()
        w.writerows(rows)
    print(f"  [done] {path.name}")
    for r in rows:
        print(f"    {r}")


if __name__ == "__main__":
    print("=" * 60)
    print("Table 3: Ablation study")
    print(f"  QAOA depth p={DEPTH}, n=14, budget Q={CANDIDATE_BUDGET}")
    print("=" * 60, flush=True)
    build_training_library(depth=DEPTH)
    get_trained_gin(depth=DEPTH)
    gen_table03()
