"""Multi-seed robustness study for OST-QAOA.

Re-runs the matched-budget benchmark (depth 3, budget 24, rank 4, omega 4.0)
across many global seeds and aggregates the OST-QAOA advantage distribution.
This directly addresses the single-seed limitation of the original manuscript.
Uses the installed package API; no numbers are fabricated.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from uq_qaoa.operator_spectral import (
    DEFAULT_FAMILIES,
    DEFAULT_SIZES,
    build_operator_library,
    build_test_graphs,
    evaluate_methods_on_graph,
    stable_seed,
)
from uq_qaoa.paper_artifacts import _best_so_far, _q_to_target

METHOD_ORDER = ["Random", "TQA", "TQA+coordinate", "kNN+coordinate", "OST diagonal", "OST-QAOA"]
BASELINE = "TQA+coordinate"

DEPTH = 3
BUDGET = 24
RANK = 4
OMEGA = 4.0
TRAIN_PER_FAMILY = 6
TEST_PER_FAMILY = 4
OPT_BUDGET = 42


def run_one_seed(seed: int) -> dict:
    families = tuple(DEFAULT_FAMILIES)
    sizes = tuple(DEFAULT_SIZES)
    library = build_operator_library(
        p=DEPTH, families=families, sizes=sizes,
        train_per_family=TRAIN_PER_FAMILY, rank=RANK,
        commutator_weight=OMEGA, optimizer_budget=OPT_BUDGET, seed=seed,
    )
    graphs = build_test_graphs(families=families, sizes=sizes, test_per_family=TEST_PER_FAMILY, seed=seed)

    # per-graph ratio per method, plus best-so-far trace for q-to-target
    ratios = {m: {} for m in METHOD_ORDER}
    traces = {m: {} for m in METHOD_ORDER}
    for graph in graphs:
        results = evaluate_methods_on_graph(
            graph, library, budget=BUDGET,
            seed=stable_seed(seed, "random", graph.graph_id),
        )
        for method, result in results.items():
            ratios[method][graph.graph_id] = float(result.y_hat)
            traces[method][graph.graph_id] = result.trace

    gids = [g.graph_id for g in graphs]
    # oracle target = 0.98 * best across methods per graph
    oracle = {gid: max(ratios[m][gid] for m in METHOD_ORDER) for gid in gids}
    target = {gid: 0.98 * oracle[gid] for gid in gids}

    out = {"seed": seed}
    base_vec = np.array([ratios[BASELINE][gid] for gid in gids])
    for method in METHOD_ORDER:
        vec = np.array([ratios[method][gid] for gid in gids])
        out[f"{method}__mean"] = float(vec.mean())
        deltas = vec - base_vec
        out[f"{method}__delta"] = float(deltas.mean())
        out[f"{method}__wins"] = int(np.sum(deltas > 1e-12))
        qtt = []
        for gid in gids:
            curve = _best_so_far(sorted(traces[method][gid], key=lambda r: r["query_index"]), BUDGET)
            qtt.append(_q_to_target(curve, target[gid]))
        out[f"{method}__qtt"] = float(np.mean(qtt))
    return out


def main() -> int:
    seeds = [int(s) for s in sys.argv[1:]] if len(sys.argv) > 1 else [
        260424803, 7, 101, 2024, 31337, 555, 99991, 12345, 424242, 8675309,
    ]
    records = []
    for i, seed in enumerate(seeds, 1):
        rec = run_one_seed(seed)
        records.append(rec)
        print(f"[{i}/{len(seeds)}] seed={seed}  "
              f"OST={rec['OST-QAOA__mean']:.4f}  "
              f"delta={rec['OST-QAOA__delta']:+.4f}  "
              f"wins={rec['OST-QAOA__wins']}/16  "
              f"q98={rec['OST-QAOA__qtt']:.1f}", flush=True)

    df = pd.DataFrame(records)
    out_csv = sys.argv[0].rsplit("/", 1)[0] + "/multiseed_results.csv"
    df.to_csv(out_csv, index=False)

    # aggregate across seeds
    print("\n=== AGGREGATE ACROSS", len(seeds), "SEEDS ===")
    agg = {}
    for method in METHOD_ORDER:
        means = df[f"{method}__mean"].to_numpy()
        deltas = df[f"{method}__delta"].to_numpy()
        wins = df[f"{method}__wins"].to_numpy()
        qtt = df[f"{method}__qtt"].to_numpy()
        agg[method] = {
            "mean_ratio_mean": float(means.mean()),
            "mean_ratio_std": float(means.std(ddof=1)),
            "delta_mean": float(deltas.mean()),
            "delta_std": float(deltas.std(ddof=1)),
            "delta_min": float(deltas.min()),
            "delta_max": float(deltas.max()),
            "seeds_with_positive_delta": int(np.sum(deltas > 0)),
            "total_wins": int(wins.sum()),
            "total_instances": int(16 * len(seeds)),
            "qtt_mean": float(qtt.mean()),
            "qtt_std": float(qtt.std(ddof=1)),
        }
        print(f"{method:16s} ratio={agg[method]['mean_ratio_mean']:.4f}"
              f"±{agg[method]['mean_ratio_std']:.4f}  "
              f"delta={agg[method]['delta_mean']:+.4f}±{agg[method]['delta_std']:.4f}  "
              f"[{agg[method]['delta_min']:+.4f},{agg[method]['delta_max']:+.4f}]  "
              f"wins={agg[method]['total_wins']}/{agg[method]['total_instances']}  "
              f"q98={agg[method]['qtt_mean']:.1f}")

    with open(sys.argv[0].rsplit("/", 1)[0] + "/multiseed_agg.json", "w") as f:
        json.dump({"seeds": seeds, "aggregate": agg}, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
