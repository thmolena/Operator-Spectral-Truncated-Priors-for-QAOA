#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from uq_qaoa.config import load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "p3_main.yaml"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    results = ROOT / "results"
    tables = ROOT / "tables"
    tables.mkdir(exist_ok=True)
    summary = results / f"summary_p{cfg.qaoa_depth}.csv"
    if not summary.exists():
        raise FileNotFoundError(f"missing {summary}; run scripts/run_all_small.py first")
    df = pd.read_csv(summary)
    ablation_methods = [m for m in df["method"].unique() if m.startswith("uq_")]
    keep = df[df["method"].isin(ablation_methods + ["tqa_refine"])]
    keys = ["p", "n", "graph_family", "query_budget_Q"]
    full = keep[keep["method"] == "uq_qaoa_full"][keys + ["mean_ratio"]].rename(columns={"mean_ratio": "uq_full_mean_ratio"})
    out = keep.merge(full, on=keys, how="left")
    out["delta_vs_uq_full"] = out["mean_ratio"] - out["uq_full_mean_ratio"]
    out_path = tables / f"table_ablation_p{cfg.qaoa_depth}.csv"
    out.to_csv(out_path, index=False)
    tex_path = tables / "table_ablation.tex"
    cols = ["method", "p", "n", "graph_family", "query_budget_Q", "mean_ratio", "delta_vs_uq_full"]
    out[cols].to_latex(tex_path, index=False, float_format="%.4f")
    print(f"wrote {out_path}")
    print(f"wrote {tex_path}")


if __name__ == "__main__":
    main()
