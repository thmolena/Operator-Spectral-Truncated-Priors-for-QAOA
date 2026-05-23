#!/usr/bin/env python3
from pathlib import Path
import argparse, csv, sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from uq_qaoa.config import load_config
from uq_qaoa.graphs import generate_graph
from uq_qaoa.maxcut import all_cut_values
from uq_qaoa.priors import theta_tqa
from uq_qaoa.statevector import qaoa_expectation
from uq_qaoa.finite_shots import sample_objective

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default=str(ROOT/"configs"/"p3_main.yaml")); args = ap.parse_args()
    cfg = load_config(args.config); out = ROOT/"results"/f"finite_shots_p{cfg.qaoa_depth}.csv"; out.parent.mkdir(parents=True, exist_ok=True)
    rows=[]; p=cfg.qaoa_depth
    graph=generate_graph(cfg.graphs.families[0], cfg.graphs.sizes[0], cfg.graphs.seeds[0]); c=all_cut_values(graph.n, graph.edges); theta=theta_tqa(p,cfg.theta_layout)
    exact=qaoa_expectation(graph.n, graph.edges,p,theta,cfg.theta_layout,cost_values=c)
    for shots in cfg.finite_shots.shots:
        est=sample_objective(graph.n,graph.edges,p,theta,shots,np.random.default_rng(cfg.random_seed),cfg.theta_layout,c)
        rows.append({"p":p,"shots":shots,"method":"tqa","noisy_selected_objective":est["mean"],"standard_error":est["standard_error"],"exact_reevaluated_objective":exact,"misranking_rate":""})
    with out.open("w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print(f"wrote {out}")
