#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
import uuid
from typing import Any
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from uq_qaoa.config import load_config
from uq_qaoa.graphs import generate_graph, graph_features
from uq_qaoa.maxcut import exact_maxcut_bruteforce, normalize_objective, all_cut_values
from uq_qaoa.statevector import qaoa_expectation
from uq_qaoa.priors import theta_tqa, theta_global, theta_gin_prior, theta_knn_prior
from uq_qaoa.posterior import fuse_diagonal_priors, ablate_variance
from uq_qaoa.search_policy import uq_qaoa_search
from uq_qaoa.baselines import random_search, tqa_refine, tqa, global_prior, posterior_mean_only, posterior_anchors_only, spsa
from uq_qaoa.traces import TRACE_COLUMNS, git_commit, now_iso
from uq_qaoa.metrics import bootstrap_ci, standard_error


def build_anchors(p, layout, features):
    tqa0 = theta_tqa(p, layout)
    gin = theta_gin_prior(p, features, layout)
    knn = theta_knn_prior(p, features, layout)
    glob = theta_global(p, layout)
    mu, var = fuse_diagonal_priors([tqa0, gin, knn, glob], None, p, layout)
    anchors = [("posterior", mu), ("tqa", tqa0), ("gin", gin), ("knn", knn), ("global", glob)]
    return anchors, mu, var


def run_config(config_path: str, smoke: bool = False):
    cfg = load_config(config_path)
    out_root = (ROOT.parent / cfg.outputs.root) if not Path(cfg.outputs.root).is_absolute() else Path(cfg.outputs.root)
    out_root.mkdir(parents=True, exist_ok=True)
    trace_path = out_root / f"traces_p{cfg.qaoa_depth}.csv"
    summary_path = out_root / f"summary_p{cfg.qaoa_depth}.csv"
    p = cfg.qaoa_depth
    methods = cfg.methods.get("include", [])
    budgets = cfg.query_budgets[:1] if smoke else cfg.query_budgets
    families = cfg.graphs.families[:1] if smoke else cfg.graphs.families
    sizes = cfg.graphs.sizes[:1] if smoke else cfg.graphs.sizes
    seeds = cfg.graphs.seeds[:1] if smoke else cfg.graphs.seeds[:cfg.graphs.test_instances_per_family]
    rows = []
    trace_rows = []
    commit = git_commit()
    for family in families:
        for n in sizes:
            for seed in seeds:
                graph = generate_graph(family, n, seed)
                ref, ref_type = exact_maxcut_bruteforce(n, graph.edges)
                cvals = all_cut_values(n, graph.edges)
                feats = graph_features(graph)
                anchors, mu, var = build_anchors(p, cfg.theta_layout, feats)
                def objective(theta):
                    return qaoa_expectation(n, graph.edges, p, theta, cfg.theta_layout, cost_values=cvals)
                for Q in budgets:
                    method_calls = {
                        "random_search": lambda: random_search(objective, p, Q, seed, cfg.theta_layout),
                        "tqa": lambda: tqa(objective, p, Q, cfg.theta_layout),
                        "tqa_refine": lambda: tqa_refine(objective, p, Q, cfg.theta_layout, seed),
                        "global_prior": lambda: global_prior(objective, p, Q, cfg.theta_layout),
                        "posterior_mean_only": lambda: posterior_mean_only(objective, p, Q, mu, cfg.theta_layout),
                        "posterior_anchors_only": lambda: posterior_anchors_only(objective, p, Q, anchors, cfg.theta_layout),
                        "spsa": lambda: spsa(objective, p, Q, seed, cfg.theta_layout),
                        "uq_qaoa_full": lambda: uq_qaoa_search(objective, p, Q, anchors, var, cfg.theta_layout, rng_seed=seed),
                        "uq_no_covariance": lambda: uq_qaoa_search(objective, p, Q, anchors, ablate_variance(var, "isotropic"), cfg.theta_layout, rng_seed=seed),
                        "uq_shuffled_covariance": lambda: uq_qaoa_search(objective, p, Q, anchors, ablate_variance(var, "shuffled", seed), cfg.theta_layout, rng_seed=seed),
                        "uq_no_refinement": lambda: posterior_anchors_only(objective, p, Q, anchors, cfg.theta_layout),
                        "uq_no_tqa_prior": lambda: uq_qaoa_search(objective, p, Q, [a for a in anchors if a[0] != "tqa"], var, cfg.theta_layout, rng_seed=seed),
                        "uq_no_knn_prior": lambda: uq_qaoa_search(objective, p, Q, [a for a in anchors if a[0] != "knn"], var, cfg.theta_layout, rng_seed=seed),
                        "uq_no_global_prior": lambda: uq_qaoa_search(objective, p, Q, [a for a in anchors if a[0] != "global"], var, cfg.theta_layout, rng_seed=seed),
                    }
                    for method in methods:
                        if method not in method_calls:
                            continue
                        run_id = str(uuid.uuid4())
                        result = method_calls[method]()
                        ratio = normalize_objective(result.y_hat, ref)
                        rows.append({"run_id": run_id, "method": method, "p": p, "dim": 2*p, "n": n, "graph_family": family, "graph_seed": seed, "query_budget_Q": Q, "best_objective": result.y_hat, "reference_value": ref, "reference_type": ref_type, "approximation_ratio": ratio, "backend": cfg.backend, "threads": 1})
                        for tr in result.trace:
                            row: dict[str, Any] = {k: "" for k in TRACE_COLUMNS}
                            row.update({"run_id": run_id, "timestamp": now_iso(), "git_commit": commit, "config_path": config_path, "method": method, "ablation": method if method.startswith("uq_no") else "", "p": p, "dim": 2*p, "n": n, "graph_family": family, "graph_seed": seed, "graph_id": graph.graph_id, "edge_count": graph.edge_count, "weighted": False, "query_budget_Q": Q, "theta_layout": cfg.theta_layout, "reference_value": ref, "reference_type": ref_type, "approximation_ratio": ratio, "backend": cfg.backend, "threads": 1, "dtype": cfg.dtype, "rng_seed": seed})
                            row.update(tr)
                            trace_rows.append(row)
    with trace_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=TRACE_COLUMNS)
        w.writeheader(); w.writerows(trace_rows)
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["method", "p", "dim", "n", "graph_family", "num_instances", "query_budget_Q", "mean_ratio", "std_ratio", "stderr_ratio", "ci95_low", "ci95_high", "backend", "threads", "elapsed_total_s"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        groups = {}
        for r in rows:
            key = (r["method"], r["p"], r["n"], r["graph_family"], r["query_budget_Q"])
            groups.setdefault(key, []).append(r["approximation_ratio"])
        for (method, p0, n0, fam, Q), vals in groups.items():
            vals = np.asarray(vals, dtype=float)
            lo, hi = bootstrap_ci(vals, n_boot=200)
            w.writerow({"method": method, "p": p0, "dim": 2*p0, "n": n0, "graph_family": fam, "num_instances": vals.size, "query_budget_Q": Q, "mean_ratio": vals.mean(), "std_ratio": vals.std(ddof=1) if vals.size > 1 else 0, "stderr_ratio": standard_error(vals), "ci95_low": lo, "ci95_high": hi, "backend": cfg.backend, "threads": 1, "elapsed_total_s": ""})
    print(f"wrote {summary_path}")
    print(f"wrote {trace_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "p3_main.yaml"))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    run_config(args.config, args.smoke)
