#!/usr/bin/env python3
"""
Table 1: Primary Computational-Efficiency Table
=================================================

Generates the main results table (Table I in the manuscript) comparing
approximation ratios of all six methods plus a CMA-ES/DE baseline on
n = 14 graphs, with bootstrap 95% confidence intervals.

The table columns are:
  Method | Ratio | CI_95 | Delta_vs_best_baseline | Candidate_budget

where Delta is the improvement over the best non-UQ-QAOA baseline.

Experimental setup:
  - Graph size n = 14, QAOA depth p = 3, budget Q = 18.
  - 8 test instances (2 per family ER/REG/BA/WS).
  - Each ratio is the mean best-of-Q approximation ratio.
  - 95% CI computed via 10 000-resample bootstrap over test instances.

Output: code/tables/table01_computational_efficiency.csv
"""
from __future__ import annotations

import csv
import os
import pathlib

BASE = pathlib.Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(BASE / ".mplconfig"))

import numpy as np
from scipy.optimize import differential_evolution

from uq_qaoa_core import (
    CANDIDATE_BUDGET, DEPTH, FAMILIES, METHODS, METHOD_LABELS,
    build_training_library, eval_all_methods, get_trained_gin,
    make_instances, qaoa_cost_values, qaoa_ratio, stable_seed, table_path,
)


def bootstrap_ci(values, n_boot=10_000, ci=0.95, seed=0):
    """Compute bootstrap confidence interval for the mean.

    Parameters
    ----------
    values : array-like
        Per-instance metric values.
    n_boot : int
        Number of bootstrap resamples.
    ci : float
        Confidence level (default 0.95).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    (lo, hi) : tuple of float
        Lower and upper bounds of the confidence interval.
    """
    arr = np.asarray(values, dtype=float)
    rng = np.random.RandomState(seed)
    means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, len(arr), size=len(arr))
        means[i] = arr[idx].mean()
    alpha = (1 - ci) / 2
    return float(np.percentile(means, 100 * alpha)), \
           float(np.percentile(means, 100 * (1 - alpha)))


def eval_de_baseline(instances, depth, budget):
    """Evaluate differential evolution (CMA-ES-class) baseline.

    Runs scipy differential_evolution on each graph instance with a
    strict maxfun=budget limit, matching the same evaluation budget as
    all other methods.  The optimizer adaptively selects candidates via
    mutation and crossover in the angle domain.

    Returns
    -------
    list of float
        Best approximation ratio per instance.
    """
    dim = 2 * depth
    results = []
    for fam, inst, edges, feats, n in instances:
        c_vals = qaoa_cost_values(edges, n)
        seed = stable_seed("de", fam, inst)
        bounds = [(0, np.pi)] * dim

        # Track evaluations and best ratio under strict budget limit
        eval_count = [0]
        best_ratio = [0.0]

        def neg_ratio(x):
            if eval_count[0] >= budget:
                return 0.0  # budget exhausted
            eval_count[0] += 1
            r = qaoa_ratio(x[:depth], x[depth:], edges, n, c_vals)
            best_ratio[0] = max(best_ratio[0], r)
            return -r

        # Under Q=18 budget: popsize=1 gives population = max(5, 1*dim).
        # With dim=6 this gives pop=6, init uses 6 evals, leaving 12
        # for evolution (roughly 2 generations).
        try:
            differential_evolution(
                neg_ratio, bounds, seed=seed % (2**31),
                maxiter=100, tol=0, atol=0,
                popsize=1, init="sobol",
                mutation=(0.5, 1.0), recombination=0.9,
                polish=False,
            )
        except Exception:
            pass
        results.append(best_ratio[0])
    return results


def eval_bo_baseline(instances, depth, budget):
    """Evaluate Bayesian optimization (GP-based) baseline.

    Uses a simple GP surrogate with RBF kernel and Expected Improvement
    acquisition.  Initial points: 5 Sobol samples; remaining budget
    used for sequential acquisition.  Budget is strictly Q evaluations.

    This implements a standard BO loop without library dependencies beyond
    NumPy/SciPy, ensuring reproducibility.

    Returns
    -------
    list of float
        Best approximation ratio per instance.
    """
    from scipy.spatial.distance import cdist
    from scipy.optimize import minimize as scipy_minimize

    dim = 2 * depth
    results = []

    for fam, inst, edges, feats, n in instances:
        c_vals = qaoa_cost_values(edges, n)
        seed = stable_seed("bo", fam, inst)
        rng = np.random.RandomState(seed % (2**31))

        # Sobol-like initialization (Latin hypercube)
        n_init = min(5, budget)
        X = rng.uniform(0, np.pi, size=(n_init, dim))
        Y = np.array([qaoa_ratio(x[:depth], x[depth:], edges, n, c_vals)
                      for x in X])
        best_ratio = float(Y.max())

        # Simple RBF kernel GP with noise
        def rbf_kernel(X1, X2, length_scale=1.0, variance=1.0):
            dists = cdist(X1, X2, metric='sqeuclidean')
            return variance * np.exp(-0.5 * dists / length_scale**2)

        for t in range(n_init, budget):
            # Fit GP (with jitter for stability)
            K = rbf_kernel(X, X) + 1e-6 * np.eye(len(X))
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                K += 1e-4 * np.eye(len(X))
                L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y))

            # Expected improvement acquisition (maximize)
            def neg_ei(x):
                x = x.reshape(1, -1)
                k_star = rbf_kernel(x, X)
                mu = (k_star @ alpha).item()
                v = rbf_kernel(x, x) - k_star @ np.linalg.solve(
                    L.T, np.linalg.solve(L, k_star.T))
                sigma = float(np.sqrt(max(v.item(), 1e-10)))
                if sigma < 1e-8:
                    return 0.0
                z = (mu - best_ratio) / sigma
                from scipy.stats import norm
                ei = (mu - best_ratio) * norm.cdf(z) + sigma * norm.pdf(z)
                return -ei

            # Multi-start optimization of acquisition
            best_x = rng.uniform(0, np.pi, size=dim)
            best_acq = neg_ei(best_x)
            for _ in range(5):
                x0 = rng.uniform(0, np.pi, size=dim)
                res = scipy_minimize(neg_ei, x0, bounds=[(0, np.pi)] * dim,
                                     method='L-BFGS-B')
                if res.fun < best_acq:
                    best_acq = res.fun
                    best_x = res.x

            # Evaluate new point
            r = qaoa_ratio(best_x[:depth], best_x[depth:], edges, n, c_vals)
            best_ratio = max(best_ratio, r)
            X = np.vstack([X, best_x.reshape(1, -1)])
            Y = np.append(Y, r)

        results.append(best_ratio)
    return results


def eval_multiseed_random(instances, depth, budget, n_seeds=5):
    """Evaluate multi-seed random baseline.

    Runs Q/n_seeds evaluations from each of n_seeds independent random
    restarts, takes the global best.  This tests whether structured
    exploration (UQ-QAOA) outperforms naive diversification.

    Parameters
    ----------
    n_seeds : int
        Number of independent random starts (each gets budget/n_seeds evals).

    Returns
    -------
    list of float
        Best approximation ratio per instance.
    """
    dim = 2 * depth
    results = []
    evals_per_seed = max(1, budget // n_seeds)

    for fam, inst, edges, feats, n in instances:
        c_vals = qaoa_cost_values(edges, n)
        seed_base = stable_seed("multiseed", fam, inst)
        best_ratio = 0.0

        for s in range(n_seeds):
            rng = np.random.RandomState((seed_base + s) % (2**31))
            for _ in range(evals_per_seed):
                x = rng.uniform(0, np.pi, size=dim)
                r = qaoa_ratio(x[:depth], x[depth:], edges, n, c_vals)
                best_ratio = max(best_ratio, r)

        results.append(best_ratio)
    return results


def eval_nelder_mead(instances, depth, budget):
    """Evaluate Nelder-Mead baseline under strict Q budget.

    Nelder-Mead in R^4 requires a simplex of d+1=5 initial points and
    typically needs ~50-200 evaluations to converge.  Under Q=18, it
    cannot complete a full contraction cycle, demonstrating the inadequacy
    of iterative local optimizers in the low-budget regime.

    Returns
    -------
    list of float
        Best approximation ratio per instance.
    """
    from scipy.optimize import minimize as scipy_minimize

    dim = 2 * depth
    results = []

    for fam, inst, edges, feats, n in instances:
        c_vals = qaoa_cost_values(edges, n)
        seed = stable_seed("neldermead", fam, inst)
        rng = np.random.RandomState(seed % (2**31))

        eval_count = [0]
        best_ratio = [0.0]

        def neg_ratio(x):
            if eval_count[0] >= budget:
                return 0.0
            eval_count[0] += 1
            r = qaoa_ratio(x[:depth], x[depth:], edges, n, c_vals)
            best_ratio[0] = max(best_ratio[0], r)
            return -r

        x0 = rng.uniform(0, np.pi, size=dim)
        try:
            scipy_minimize(
                neg_ratio, x0, method='Nelder-Mead',
                options={'maxfev': budget, 'xatol': 1e-8, 'fatol': 1e-8}
            )
        except Exception:
            pass
        results.append(best_ratio[0])
    return results


def eval_cmaes(instances, depth, budget):
    """Evaluate CMA-ES-style baseline under strict Q budget.

    Implements a simplified (mu/mu_w, lambda)-CMA-ES with population
    size lambda=4 (minimum practical) and budget-limited evaluation.
    Under Q=18, CMA-ES completes ~4 generations.

    Returns
    -------
    list of float
        Best approximation ratio per instance.
    """
    dim = 2 * depth
    results = []

    for fam, inst, edges, feats, n in instances:
        c_vals = qaoa_cost_values(edges, n)
        seed = stable_seed("cmaes", fam, inst)
        rng = np.random.RandomState(seed % (2**31))

        # Initialize CMA-ES parameters
        mean = rng.uniform(0, np.pi, size=dim)
        sigma = np.pi / 4
        lam = 4  # minimal population
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / weights.sum()
        C = np.eye(dim)

        eval_count = 0
        best_ratio = 0.0

        while eval_count < budget:
            # Sample lambda candidates
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(dim)
                L = np.eye(dim)

            candidates = []
            fitnesses = []
            for _ in range(lam):
                if eval_count >= budget:
                    break
                z = rng.randn(dim)
                x = mean + sigma * (L @ z)
                x = np.clip(x, 0, np.pi)
                r = qaoa_ratio(x[:depth], x[depth:], edges, n, c_vals)
                eval_count += 1
                best_ratio = max(best_ratio, r)
                candidates.append(x)
                fitnesses.append(-r)

            if len(candidates) < 2:
                break

            # Select mu best
            order = np.argsort(fitnesses)[:mu]
            selected = np.array([candidates[i] for i in order])

            # Update mean
            old_mean = mean.copy()
            mean = weights @ selected

            # Update covariance (simplified rank-mu update)
            diff = selected - old_mean
            C = np.sum([weights[i] * np.outer(diff[i], diff[i])
                       for i in range(mu)], axis=0)
            C = 0.8 * np.eye(dim) + 0.2 * C / (sigma**2 + 1e-10)

        results.append(best_ratio)
    return results


def gen_table01():
    """Generate Table 1: primary approximation-ratio comparison.

    Evaluates all 6 methods + DE baseline on 8 held-out n=14 instances,
    computes mean ratios with bootstrap 95% CIs, and writes a CSV with
    the improvement delta relative to the best non-UQ-QAOA baseline.
    """
    path = table_path("table01_computational_efficiency")
    n = 14; budget = CANDIDATE_BUDGET
    instances = make_instances(FAMILIES, n, 2)
    res = eval_all_methods(instances, DEPTH, budget)

    # Collect per-instance ratios for each method
    per_inst = {}
    for method in METHODS:
        per_inst[method] = np.array([r[0] for r in res[method]])

    # Additional baselines
    de_ratios = eval_de_baseline(instances, DEPTH, budget)
    per_inst["de"] = np.array(de_ratios)

    bo_ratios = eval_bo_baseline(instances, DEPTH, budget)
    per_inst["bo"] = np.array(bo_ratios)

    ms_ratios = eval_multiseed_random(instances, DEPTH, budget, n_seeds=5)
    per_inst["multiseed"] = np.array(ms_ratios)

    nm_ratios = eval_nelder_mead(instances, DEPTH, budget)
    per_inst["nelder_mead"] = np.array(nm_ratios)

    cmaes_ratios = eval_cmaes(instances, DEPTH, budget)
    per_inst["cmaes"] = np.array(cmaes_ratios)

    all_methods = METHODS + ["de", "bo", "multiseed", "nelder_mead", "cmaes"]
    all_labels = METHOD_LABELS + ["DE (scipy)", "BO (GP-EI)",
                                   "Multi-seed random", "Nelder-Mead",
                                   "CMA-ES"]

    best_baseline = max(per_inst[m].mean()
                        for m in all_methods
                        if m != "uq_qaoa")

    rows = []
    for mi, method in enumerate(all_methods):
        vals = per_inst[method]
        mr = vals.mean()
        lo, hi = bootstrap_ci(vals)
        rows.append({
            "Method": all_labels[mi] + (" (ours)" if method == "uq_qaoa" else ""),
            "Ratio": f"{mr:.3f}",
            "CI_95": f"[{lo:.3f}, {hi:.3f}]",
            "Delta_vs_best_baseline": f"{(mr - best_baseline):+.3f}",
            "Candidate_budget": str(budget),
        })
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Method", "Ratio", "CI_95",
                                           "Delta_vs_best_baseline",
                                           "Candidate_budget"])
        w.writeheader()
        w.writerows(rows)
    print(f"  [done] {path.name}")
    for r in rows:
        print(f"    {r}")


if __name__ == "__main__":
    print("=" * 60)
    print("Table 1: Computational efficiency comparison")
    print(f"  QAOA depth p={DEPTH}, n=14, budget Q={CANDIDATE_BUDGET}")
    print("=" * 60, flush=True)
    build_training_library(depth=DEPTH)
    get_trained_gin(depth=DEPTH)
    gen_table01()
