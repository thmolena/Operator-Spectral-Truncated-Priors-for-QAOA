#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Global style: larger text, slightly darker palette
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 15,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 10,
    "figure.titlesize": 15,
    "lines.linewidth": 1.8,
    "lines.markersize": 7,
})

# Slightly darker color cycle
DARK_COLORS = ["#155fa0", "#c44e00", "#1e7a1e", "#b81d1d", "#7a4fad",
               "#8c5600", "#c43782", "#555555", "#1a8a8a", "#6b6b00"]

ROOT = Path(__file__).resolve().parents[1]
res = ROOT / "results"; fig = ROOT / "figures"; fig.mkdir(exist_ok=True)
summaries = []
for csv_path in sorted(res.glob("summary_p*.csv")):
    df = pd.read_csv(csv_path)
    if df.empty:
        continue
    summaries.append(df)
    p = int(df["p"].iloc[0])
    if p != 3 or (ROOT / "tables" / "trace_all_evaluations.csv").exists():
        continue
    plt.figure(figsize=(7, 4.5))
    for ci, (method, g) in enumerate(df.groupby("method")):
        h = g.groupby("query_budget_Q")["mean_ratio"].mean().reset_index()
        plt.plot(h["query_budget_Q"], h["mean_ratio"],
                 label=method, color=DARK_COLORS[ci % len(DARK_COLORS)])
    plt.xlabel("Query budget $Q$")
    plt.ylabel("Mean approximation ratio")
    plt.title(f"Best-so-far ratio vs. query budget \u2013 depth $p={p}$", pad=10)
    plt.legend(fontsize=10, framealpha=0.9, edgecolor="0.7",
               loc="lower right")
    plt.tight_layout()
    out = fig / f"best_so_far_ratio_vs_Q_p{p}.pdf"
    plt.savefig(out, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"wrote {out}")

trace_path = ROOT / "tables" / "trace_all_evaluations.csv"
if trace_path.exists():
    trace = pd.read_csv(trace_path)
    if not trace.empty and {"method", "query_index", "best_so_far"}.issubset(trace.columns):
        trace["Q"] = trace["query_index"] + 1
        method_order = ["random", "heuristic", "knn", "tqa", "gnn_point", "uq_qaoa"]
        method_labels = {
            "random": "Random",
            "heuristic": "Heuristic",
            "knn": "k-NN",
            "tqa": "TQA",
            "gnn_point": "GNN point",
            "uq_qaoa": "UQ-QAOA",
        }
        method_colors = {
            "random": "#737373",
            "heuristic": "#c48a00",
            "knn": "#1e7a1e",
            "tqa": "#b81d1d",
            "gnn_point": "#7a4fad",
            "uq_qaoa": "#155fa0",
        }
        plt.figure(figsize=(8, 4.8))
        for method in method_order:
            g = trace[trace["method"] == method]
            if g.empty:
                continue
            h = g.groupby("Q")["best_so_far"].mean().reset_index()
            plt.plot(h["Q"], h["best_so_far"], label=method_labels[method],
                     color=method_colors[method], linewidth=2.3)
        plt.xlabel("Query budget $Q$")
        plt.ylabel("Mean best-so-far approximation ratio")
        plt.title("Best-so-far ratio vs. query budget -- $n=14$, $p=3$", pad=10)
        plt.xlim(1, int(trace["Q"].max()))
        plt.xticks([1, 3, 6, 9, 12, 15, 18])
        plt.ylim(0.58, 0.90)
        plt.legend(fontsize=10, framealpha=0.9, edgecolor="0.7",
                   loc="lower right", ncol=2)
        plt.tight_layout()
        out = fig / "best_so_far_ratio_vs_Q_p3.pdf"
        plt.savefig(out, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"wrote {out}")

if summaries:
    all_df = pd.concat(summaries, ignore_index=True)
    p3 = all_df[all_df["p"] == 3]
    if not p3.empty and {"method", "query_budget_Q", "mean_ratio"}.issubset(p3.columns):
        pivot = p3.pivot_table(index="query_budget_Q", columns="method",
                               values="mean_ratio", aggfunc="mean")
        if "tqa_refine" in pivot and "uq_qaoa_full" in pivot:
            delta = (pivot["uq_qaoa_full"] - pivot["tqa_refine"]).dropna()
            plt.figure(figsize=(6, 4))
            plt.axhline(0, color="0.4", lw=1, ls="--")
            plt.plot(delta.index, delta.values,
                     color=DARK_COLORS[0], linewidth=2)
            plt.xlabel("Query budget $Q$")
            plt.ylabel("UQ-QAOA minus TQA+refine")
            plt.title("Paired difference vs. TQA+refine \u2013 $p=3$", pad=10)
            plt.tight_layout()
            out = fig / "paired_difference_vs_tqa_p3.pdf"
            plt.savefig(out, bbox_inches="tight", dpi=300)
            plt.close()
            print(f"wrote {out}")
        abl = p3[p3["method"].str.startswith("uq_")]
        if not abl.empty:
            h = abl.groupby("method")["mean_ratio"].mean().sort_values()
            plt.figure(figsize=(8, 4))
            h.plot(kind="barh", color=DARK_COLORS[4], edgecolor="black",
                   linewidth=0.5)
            plt.xlabel("Mean approximation ratio")
            plt.title("Ablation variants \u2013 $p=3$", pad=10)
            plt.tight_layout()
            out = fig / "ablation_p3.pdf"
            plt.savefig(out, bbox_inches="tight", dpi=300)
            plt.close()
            print(f"wrote {out}")
cal = next(iter(sorted(res.glob("calibration_curve_p*.csv"))), None)
if cal:
    df = pd.read_csv(cal)
    if not df.empty:
        plt.figure(figsize=(5, 5))
        plt.plot([0, 1], [0, 1], color="0.4", lw=1, ls="--",
                 label="Perfect calibration")
        plt.plot(df["predicted"], df["observed"],
                 color=DARK_COLORS[0], linewidth=2, label="Observed")
        plt.xlabel("Predicted coverage")
        plt.ylabel("Observed coverage")
        plt.title("Calibration curve", pad=10)
        plt.legend(fontsize=11, framealpha=0.9)
        plt.tight_layout()
        out = fig / "calibration_curve.pdf"
        plt.savefig(out, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"wrote {out}")

cov = next(iter(sorted(res.glob("trust_region_coverage_p*.csv"))), None)
if cov:
    df = pd.read_csv(cov)
    if not df.empty:
        plt.figure(figsize=(6, 4))
        plt.plot(df["radius"], df["empirical_coverage"],
                 color=DARK_COLORS[0], linewidth=2)
        plt.xlabel("Trust-region radius $\\rho$")
        plt.ylabel("Empirical coverage")
        plt.title("Trust-region coverage curve", pad=10)
        plt.tight_layout()
        out = fig / "trust_region_coverage.pdf"
        plt.savefig(out, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"wrote {out}")

hpc_files = list((res / "hpc").glob("*.csv")) if (res / "hpc").exists() else []
if hpc_files:
    hpc = pd.concat([pd.read_csv(p) for p in hpc_files], ignore_index=True)
    for name, xcol, ylabel, title in [
        ("hpc_ms_per_query_vs_n.pdf", "n", "ms per query",
         "Per-query time vs. graph size"),
        ("hpc_ms_per_query_vs_p.pdf", "p", "ms per query",
         "Per-query time vs. QAOA depth"),
        ("hpc_thread_speedup.pdf", "threads", "ms per query",
         "Thread scaling"),
    ]:
        if xcol in hpc and "ms_per_query" in hpc:
            plt.figure(figsize=(6, 4))
            h = hpc.groupby(xcol)["ms_per_query"].mean().reset_index()
            plt.plot(h[xcol], h["ms_per_query"],
                     color=DARK_COLORS[0], linewidth=2)
            plt.xlabel(xcol if xcol != "n" else "Graph size $n$")
            plt.ylabel(ylabel)
            plt.title(title, pad=10)
            plt.tight_layout()
            out = fig / name
            plt.savefig(out, bbox_inches="tight", dpi=300)
            plt.close()
            print(f"wrote {out}")
    kernel_cols = [c for c in ["phase_ms", "mixer_ms", "expectation_ms",
                               "reduction_ms", "allocation_ms"] if c in hpc]
    if kernel_cols:
        vals = hpc[kernel_cols].mean()
        plt.figure(figsize=(7, 4))
        bars = plt.bar(range(len(vals)), vals.values,
                       color=DARK_COLORS[:len(vals)], edgecolor="black",
                       linewidth=0.5)
        plt.xticks(range(len(vals)),
                   [c.replace("_ms", "").replace("_", " ").title()
                    for c in kernel_cols], fontsize=11)
        plt.ylabel("Time (ms)")
        plt.title("Kernel breakdown", pad=10)
        for bar, v in zip(bars, vals.values):
            plt.text(bar.get_x() + bar.get_width() / 2, v + 0.01 * max(vals),
                     f"{v:.2f}", ha="center", va="bottom", fontsize=11)
        plt.tight_layout()
        out = fig / "hpc_kernel_breakdown.pdf"
        plt.savefig(out, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"wrote {out}")
    if "memory_bytes" in hpc and "n" in hpc:
        plt.figure(figsize=(6, 4))
        h = hpc.groupby("n")["memory_bytes"].mean().reset_index()
        plt.plot(h["n"], h["memory_bytes"],
                 color=DARK_COLORS[2], linewidth=2)
        plt.xlabel("Graph size $n$")
        plt.ylabel("Memory (bytes)")
        plt.title("Memory footprint vs. graph size", pad=10)
        plt.tight_layout()
        out = fig / "memory_footprint_vs_n.pdf"
        plt.savefig(out, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"wrote {out}")

landscape = fig / "landscape_trust_region.pdf"
if not landscape.exists():
    plt.figure(figsize=(5, 4.5))
    xs = [-1, 0, 1]
    ys = [0.3, 1, 0.3]
    plt.plot(xs, ys, color=DARK_COLORS[0], linewidth=2)
    plt.fill_between(xs, [0, 0, 0], ys, alpha=0.2, color=DARK_COLORS[0])
    plt.xlabel("Principal trust-region coordinate")
    plt.ylabel("Relative objective")
    plt.title("Landscape trust-region illustration", pad=10)
    plt.tight_layout()
    plt.savefig(landscape, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"wrote {landscape}")
