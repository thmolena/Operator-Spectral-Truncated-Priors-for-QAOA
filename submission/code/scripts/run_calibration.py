#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from typing import Any
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from uq_qaoa.config import load_config
from uq_qaoa.graphs import generate_graph, graph_features
from uq_qaoa.maxcut import all_cut_values
from uq_qaoa.priors import theta_tqa, theta_global, theta_gin_prior, theta_knn_prior
from uq_qaoa.posterior import fuse_diagonal_priors
from uq_qaoa.qaoa_angles import random_theta
from uq_qaoa.statevector import qaoa_expectation
from uq_qaoa.calibration import normalized_squared_error, reliability_bins, expected_calibration_error, uncertainty_error_correlation


def as_float(value: Any) -> float:
    if isinstance(value, tuple):
        return float(value[0])
    return float(value)


def posterior_for_graph(cfg, graph):
    p = cfg.qaoa_depth
    features = graph_features(graph)
    anchors = [
        theta_tqa(p, cfg.theta_layout),
        theta_gin_prior(p, features, cfg.theta_layout),
        theta_knn_prior(p, features, cfg.theta_layout),
        theta_global(p, cfg.theta_layout),
    ]
    return fuse_diagonal_priors(anchors, None, p, cfg.theta_layout)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "p3_main.yaml"))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    p = cfg.qaoa_depth
    rng = np.random.default_rng(cfg.random_seed)
    families = cfg.graphs.families[:1] if args.smoke else cfg.graphs.families
    sizes = cfg.graphs.sizes[:1] if args.smoke else cfg.graphs.sizes
    seeds = cfg.graphs.seeds[:2] if args.smoke else cfg.graphs.seeds[:cfg.graphs.test_instances_per_family]
    candidate_count = 24 if args.smoke else 96
    rows = []
    theta_refs = []
    mus = []
    vars_ = []
    for family in families:
        for n in sizes:
            for seed in seeds:
                graph = generate_graph(family, n, seed)
                cvals = all_cut_values(n, graph.edges)
                mu, var = posterior_for_graph(cfg, graph)
                best_theta = mu
                best_y = as_float(qaoa_expectation(n, graph.edges, p, mu, cfg.theta_layout, cost_values=cvals))
                for _ in range(candidate_count):
                    theta = random_theta(p, rng, cfg.theta_layout)
                    y = as_float(qaoa_expectation(n, graph.edges, p, theta, cfg.theta_layout, cost_values=cvals))
                    if y > best_y:
                        best_y = y
                        best_theta = theta
                nse = normalized_squared_error(best_theta, mu, var)
                score = float(nse.sum())
                predicted_confidence = float(math.exp(-0.5 * min(score, 60.0)))
                observed_success = float(score <= 2.0 * p)
                theta_refs.append(best_theta)
                mus.append(mu)
                vars_.append(var)
                rows.append({
                    "p": p,
                    "dim": 2 * p,
                    "n": n,
                    "graph_family": family,
                    "graph_seed": seed,
                    "score": score,
                    "predicted_confidence": predicted_confidence,
                    "observed_success": observed_success,
                    "mean_variance": float(np.mean(var)),
                    "best_objective_proxy": float(best_y),
                })
    results = ROOT / "results"
    tables = ROOT / "tables"
    results.mkdir(exist_ok=True)
    tables.mkdir(exist_ok=True)
    calib = pd.DataFrame(rows)
    calib_path = results / f"calibration_p{p}.csv"
    calib.to_csv(calib_path, index=False)
    bins = pd.DataFrame(reliability_bins(calib["predicted_confidence"], calib["observed_success"], bins=8))
    bins_path = results / f"calibration_curve_p{p}.csv"
    bins.to_csv(bins_path, index=False)
    radii = [0.5 * p, p, 2 * p, 4 * p, 8 * p]
    coverage_rows = []
    scores = calib["score"].to_numpy(dtype=float)
    for radius in radii:
        coverage_rows.append({"p": p, "radius": radius, "empirical_coverage": float(np.mean(scores <= radius)), "num_instances": int(scores.size)})
    coverage = pd.DataFrame(coverage_rows)
    coverage_path = results / f"trust_region_coverage_p{p}.csv"
    coverage.to_csv(coverage_path, index=False)
    summary = pd.DataFrame([{
        "p": p,
        "dim": 2 * p,
        "num_instances": len(rows),
        "ece": expected_calibration_error(calib["predicted_confidence"], calib["observed_success"], bins=8),
        "uncertainty_error_correlation": uncertainty_error_correlation(np.asarray(vars_), scores),
    }])
    summary_path = tables / "table_calibration.tex"
    summary.to_latex(summary_path, index=False, float_format="%.4f")
    print(f"wrote {calib_path}")
    print(f"wrote {bins_path}")
    print(f"wrote {coverage_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
