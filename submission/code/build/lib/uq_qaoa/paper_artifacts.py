"""Deterministic regeneration of every OST-QAOA manuscript artifact.

This module is the single source of truth for the figures, tables, and CSV
results consumed by ``submission/main.tex``. It builds the rank/commutator
independent training core once, evaluates the matched-budget benchmark,
performs the spectral-truncation sweep (the central experiment), and renders all
figures and LaTeX tables. Every output is a deterministic function of the global
seed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import binomtest  # noqa: E402

from .baselines import tqa_refine  # noqa: E402,F401
from .graphs import GraphInstance  # noqa: E402
from .operator_spectral import (  # noqa: E402
    DEFAULT_FAMILIES,
    DEFAULT_SEED,
    DEFAULT_SIZES,
    OperatorLibrary,
    build_operator_library,
    build_test_graphs,
    build_training_core,
    diagonal_operator_search,
    effective_dimension,
    evaluate_methods_on_graph,
    operator_prior,
    ost_qaoa_search,
    spectral_operator_matrix,
    stable_seed,
)

METHOD_ORDER = ["Random", "TQA", "TQA+coordinate", "kNN+coordinate", "OST diagonal", "OST-QAOA"]
BASELINE_METHOD = "TQA+coordinate"
METHOD_COLORS = {
    "Random": "#707070",
    "TQA": "#9b3a2f",
    "TQA+coordinate": "#c47f2a",
    "kNN+coordinate": "#3a7d44",
    "OST diagonal": "#5e63a9",
    "OST-QAOA": "#0f5b8f",
}

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi": 160,
        "savefig.bbox": "tight",
    }
)


# --------------------------------------------------------------------------- #
# small deterministic statistics helpers
# --------------------------------------------------------------------------- #
def _mean_ci(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(values.mean()) if values.size else 0.0
    if values.size <= 1:
        return mean, 0.0
    ci = 1.96 * float(values.std(ddof=1)) / np.sqrt(values.size)
    return mean, ci


def _bootstrap_paired_ci(deltas: np.ndarray, seed: int, reps: int = 10000) -> tuple[float, float]:
    deltas = np.asarray(deltas, dtype=float)
    if deltas.size == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, deltas.size, size=(reps, deltas.size))
    means = deltas[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def _sign_test_p(deltas: np.ndarray) -> tuple[int, int, float]:
    deltas = np.asarray(deltas, dtype=float)
    wins = int(np.sum(deltas > 1e-12))
    losses = int(np.sum(deltas < -1e-12))
    trials = wins + losses
    p = 1.0 if trials == 0 else float(binomtest(max(wins, losses), trials, p=0.5).pvalue)
    return wins, losses, p


def _format_p(value: float) -> str:
    if value < 1e-4:
        return "$<10^{-4}$"
    return f"${value:.4f}$"


def _q_to_target(curve: np.ndarray, target: float) -> int:
    for i, v in enumerate(curve, start=1):
        if v >= target:
            return i
    return len(curve) + 1


def _best_so_far(trace: list[dict], budget: int) -> np.ndarray:
    out = np.full(budget, np.nan)
    for row in trace:
        q = int(row["query_index"])
        if 1 <= q <= budget:
            out[q - 1] = float(row["incumbent_objective"])
    last = np.nan
    for i in range(budget):
        if np.isnan(out[i]):
            out[i] = last
        else:
            last = out[i]
    first = np.nanmin(out) if np.any(~np.isnan(out)) else 0.0
    out[np.isnan(out)] = first
    return out


def _ensure_dirs(root: Path) -> dict[str, Path]:
    paths = {
        "root": root,
        "figures": root / "figures",
        "tables": root / "tables",
        "results": root / "results",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _latex_table(path: Path, header: list[str], rows: list[list[str]], align: str | None = None) -> None:
    if align is None:
        align = "l" + "r" * (len(header) - 1)
    lines = ["\\begin{tabular}{" + align + "}", "\\toprule"]
    lines.append(" & ".join(header) + " \\\\")
    lines.append("\\midrule")
    for row in rows:
        lines.append(" & ".join(row) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- #
# experiment 1: matched-budget benchmark
# --------------------------------------------------------------------------- #
def run_benchmark(
    args: argparse.Namespace, paths: dict[str, Path]
) -> tuple[pd.DataFrame, pd.DataFrame, OperatorLibrary, list[GraphInstance]]:
    families = tuple(args.families.split(","))
    sizes = tuple(int(x) for x in args.sizes.split(","))
    library = build_operator_library(
        p=args.depth,
        families=families,
        sizes=sizes,
        train_per_family=args.train_per_family,
        rank=args.rank,
        commutator_weight=args.commutator_weight,
        optimizer_budget=args.optimizer_budget,
        seed=args.seed,
    )
    graphs = build_test_graphs(families=families, sizes=sizes, test_per_family=args.test_per_family, seed=args.seed)

    rows: list[dict] = []
    traces: list[dict] = []
    for graph in graphs:
        results = evaluate_methods_on_graph(
            graph,
            library,
            budget=args.budget,
            seed=stable_seed(args.seed, "random", graph.graph_id),
        )
        for method, result in results.items():
            rows.append(
                {
                    "graph_id": graph.graph_id,
                    "family": graph.family,
                    "n": graph.n,
                    "method": method,
                    "ratio": result.y_hat,
                    "budget": args.budget,
                    "depth": args.depth,
                    "rank": args.rank,
                    "commutator_weight": args.commutator_weight,
                }
            )
            for entry in result.trace:
                traces.append(
                    {
                        "graph_id": graph.graph_id,
                        "family": graph.family,
                        "n": graph.n,
                        "method": method,
                        "depth": args.depth,
                        "budget": args.budget,
                        **entry,
                    }
                )
    result_df = pd.DataFrame(rows)
    trace_df = pd.DataFrame(traces)
    result_df.to_csv(paths["results"] / "operator_spectral_results.csv", index=False)
    trace_df.to_csv(paths["results"] / "operator_spectral_traces.csv", index=False)
    return result_df, trace_df, library, graphs


def summarize_results(
    result_df: pd.DataFrame, trace_df: pd.DataFrame, args: argparse.Namespace, paths: dict[str, Path]
) -> pd.DataFrame:
    baseline = (
        result_df[result_df["method"] == BASELINE_METHOD][["graph_id", "ratio"]].rename(columns={"ratio": "baseline"})
    )
    merged = result_df.merge(baseline, on="graph_id", how="left")
    merged["delta_vs_baseline"] = merged["ratio"] - merged["baseline"]

    oracle = result_df.groupby("graph_id")["ratio"].max().rename("oracle").reset_index()
    oracle["target"] = 0.98 * oracle["oracle"]
    target_map = dict(zip(oracle["graph_id"], oracle["target"]))

    q_to_target: dict[str, list[int]] = {m: [] for m in METHOD_ORDER}
    for method in METHOD_ORDER:
        sub = trace_df[trace_df["method"] == method]
        for gid, gdf in sub.groupby("graph_id"):
            curve = _best_so_far(gdf.sort_values("query_index").to_dict("records"), args.budget)
            q_to_target[method].append(_q_to_target(curve, target_map.get(gid, np.inf)))

    summary_rows: list[dict] = []
    for method in METHOD_ORDER:
        sub = merged[merged["method"] == method]
        mean, ci = _mean_ci(sub["ratio"].to_numpy())
        deltas = sub["delta_vs_baseline"].to_numpy()
        delta_mean, _ = _mean_ci(deltas)
        lo, hi = _bootstrap_paired_ci(deltas, seed=stable_seed(args.seed, "boot", method))
        wins, losses, p_value = _sign_test_p(deltas)
        qtt = np.array(q_to_target[method], dtype=float)
        summary_rows.append(
            {
                "method": method,
                "mean_ratio": mean,
                "ci95": ci,
                "delta_vs_baseline": delta_mean,
                "delta_lo": lo,
                "delta_hi": hi,
                "wins": wins,
                "losses": losses,
                "sign_test_p": p_value,
                "mean_q_to_target": float(np.mean(qtt)) if qtt.size else float("nan"),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(paths["tables"] / "table01_headline.csv", index=False)

    latex_rows = []
    for row in summary_rows:
        delta_cell = (
            "$0.000$"
            if row["method"] == BASELINE_METHOD
            else f"${row['delta_vs_baseline']:+.3f}$ [{row['delta_lo']:+.3f}, {row['delta_hi']:+.3f}]"
        )
        latex_rows.append(
            [
                row["method"],
                f"${row['mean_ratio']:.3f}\\pm{row['ci95']:.3f}$",
                delta_cell,
                f"{row['wins']}/{row['wins'] + row['losses']}",
                _format_p(row["sign_test_p"]),
                f"{row['mean_q_to_target']:.1f}",
            ]
        )
    _latex_table(
        paths["tables"] / "table01_headline.tex",
        ["Method", "Mean ratio", "$\\Delta$ vs.\\ TQA+coord.\\ [95\\% CI]", "Wins", "Sign $p$", "$\\bar{Q}_{0.98}$"],
        latex_rows,
        align="lrlrrr",
    )
    return summary


def summarize_per_family(result_df: pd.DataFrame, paths: dict[str, Path]) -> pd.DataFrame:
    baseline = (
        result_df[result_df["method"] == BASELINE_METHOD][["graph_id", "ratio"]].rename(columns={"ratio": "baseline"})
    )
    merged = result_df.merge(baseline, on="graph_id", how="left")
    merged["delta_vs_baseline"] = merged["ratio"] - merged["baseline"]
    ost = merged[merged["method"] == "OST-QAOA"]
    rows: list[dict] = []
    for family, sub in ost.groupby("family", sort=False):
        deltas = sub["delta_vs_baseline"].to_numpy()
        wins, losses, p_value = _sign_test_p(deltas)
        rows.append(
            {
                "family": family,
                "n_instances": int(sub.shape[0]),
                "ost_mean": float(sub["ratio"].mean()),
                "tqa_coord_mean": float(sub["baseline"].mean()),
                "delta": float(deltas.mean()),
                "wins": wins,
                "losses": losses,
                "sign_test_p": p_value,
            }
        )
    per_family = pd.DataFrame(rows)
    per_family.to_csv(paths["tables"] / "table04_per_family.csv", index=False)
    latex_rows = [
        [
            r["family"].replace("_", "\\_"),
            str(r["n_instances"]),
            f"${r['tqa_coord_mean']:.3f}$",
            f"${r['ost_mean']:.3f}$",
            f"${r['delta']:+.3f}$",
            f"{r['wins']}/{r['wins'] + r['losses']}",
            _format_p(r["sign_test_p"]),
        ]
        for r in rows
    ]
    _latex_table(
        paths["tables"] / "table04_per_family.tex",
        ["Family", "$N$", "TQA+coord.", "OST-QAOA", "$\\Delta$", "Wins", "Sign $p$"],
        latex_rows,
    )
    return per_family


# --------------------------------------------------------------------------- #
# experiment 2: spectral-truncation sweep  (the central experiment)
# --------------------------------------------------------------------------- #
def run_truncation_sweep(
    args: argparse.Namespace, paths: dict[str, Path], graphs: list[GraphInstance]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    families = tuple(args.families.split(","))
    sizes = tuple(int(x) for x in args.sizes.split(","))
    d = 2 * args.depth
    # Full truncation sweep over every retained-direction count 1..2p so the
    # representation-vs-query-budget tradeoff (interior optimum, effective
    # dimension) is resolved smoothly rather than at a few sampled ranks.
    ranks = list(range(1, d + 1))

    build_training_core(
        args.depth,
        families=families,
        sizes=sizes,
        train_per_family=args.train_per_family,
        optimizer_budget=args.optimizer_budget,
        seed=args.seed,
    )

    rows: list[dict] = []
    for rank in ranks:
        for comm_on in (True, False):
            comm_weight = args.commutator_weight if comm_on else 0.0
            library = build_operator_library(
                p=args.depth,
                families=families,
                sizes=sizes,
                train_per_family=args.train_per_family,
                rank=rank,
                commutator_weight=comm_weight,
                optimizer_budget=args.optimizer_budget,
                seed=args.seed,
            )
            for graph in graphs:
                prior = operator_prior(graph, library)
                d_eff = effective_dimension(prior.covariance)
                result = ost_qaoa_search(graph, library, budget=args.budget)
                rows.append(
                    {
                        "rank": rank,
                        "commutator": "on" if comm_on else "off",
                        "graph_id": graph.graph_id,
                        "family": graph.family,
                        "ratio": result.y_hat,
                        "effective_dimension": d_eff,
                    }
                )
    sweep = pd.DataFrame(rows)
    sweep.to_csv(paths["tables"] / "table05_truncation_raw.csv", index=False)

    summary_rows: list[dict] = []
    for (rank, comm), sub in sweep.groupby(["rank", "commutator"], sort=True):
        mean, ci = _mean_ci(sub["ratio"].to_numpy())
        summary_rows.append(
            {
                "rank": int(rank),
                "commutator": comm,
                "mean_ratio": mean,
                "ci95": ci,
                "mean_effective_dimension": float(sub["effective_dimension"].mean()),
            }
        )
    sweep_summary = pd.DataFrame(summary_rows)
    sweep_summary.to_csv(paths["tables"] / "table05_truncation.csv", index=False)

    on = sweep_summary[sweep_summary["commutator"] == "on"].sort_values("rank")
    off = sweep_summary[sweep_summary["commutator"] == "off"].set_index("rank")
    best_rank = int(on.loc[on["mean_ratio"].idxmax(), "rank"])
    latex_rows = []
    for _, r in on.iterrows():
        rank = int(r["rank"])
        off_mean = float(off.loc[rank, "mean_ratio"]) if rank in off.index else float("nan")
        marker = "$^{\\star}$" if rank == best_rank else ""
        latex_rows.append(
            [
                f"${rank}${marker}",
                f"${r['mean_ratio']:.3f}$",
                f"${off_mean:.3f}$",
                f"${r['mean_ratio'] - off_mean:+.3f}$",
                f"${r['mean_effective_dimension']:.2f}$",
            ]
        )
    _latex_table(
        paths["tables"] / "table05_truncation.tex",
        ["Truncation $n$", "Noncomm.\\ (on)", "Comm.\\ (off)", "$\\Delta_{\\mathrm{nc}}$", "$d_{\\mathrm{eff}}$"],
        latex_rows,
    )
    return sweep, sweep_summary


# --------------------------------------------------------------------------- #
# experiment 3: design ablation at the operating point
# --------------------------------------------------------------------------- #
def run_ablation(
    args: argparse.Namespace, paths: dict[str, Path], graphs: list[GraphInstance], result_df: pd.DataFrame
) -> pd.DataFrame:
    families = tuple(args.families.split(","))
    sizes = tuple(int(x) for x in args.sizes.split(","))
    library = build_operator_library(
        p=args.depth,
        families=families,
        sizes=sizes,
        train_per_family=args.train_per_family,
        rank=args.rank,
        commutator_weight=args.commutator_weight,
        optimizer_budget=args.optimizer_budget,
        seed=args.seed,
    )
    library_off = build_operator_library(
        p=args.depth,
        families=families,
        sizes=sizes,
        train_per_family=args.train_per_family,
        rank=args.rank,
        commutator_weight=0.0,
        optimizer_budget=args.optimizer_budget,
        seed=args.seed,
    )
    rows: list[dict] = []
    for graph in graphs:
        full = ost_qaoa_search(graph, library, budget=args.budget).y_hat
        diag = diagonal_operator_search(graph, library, budget=args.budget).y_hat
        comm_off = ost_qaoa_search(graph, library_off, budget=args.budget).y_hat
        rows.append({"graph_id": graph.graph_id, "variant": "OST-QAOA (full)", "ratio": full})
        rows.append({"graph_id": graph.graph_id, "variant": "diagonal directions", "ratio": diag})
        rows.append({"graph_id": graph.graph_id, "variant": "commutator off", "ratio": comm_off})
    base = result_df[result_df["method"] == BASELINE_METHOD][["graph_id", "ratio"]]
    for _, r in base.iterrows():
        rows.append({"graph_id": r["graph_id"], "variant": "TQA + coordinate", "ratio": r["ratio"]})

    df = pd.DataFrame(rows)
    full_sub = df[df["variant"] == "OST-QAOA (full)"]
    full_map = dict(zip(full_sub["graph_id"], full_sub["ratio"]))
    df["delta_vs_full"] = df.apply(lambda x: x["ratio"] - full_map.get(x["graph_id"], np.nan), axis=1)
    order = ["OST-QAOA (full)", "diagonal directions", "commutator off", "TQA + coordinate"]
    summary_rows = []
    for variant in order:
        sub = df[df["variant"] == variant]
        mean, ci = _mean_ci(sub["ratio"].to_numpy())
        delta, _ = _mean_ci(sub["delta_vs_full"].to_numpy())
        summary_rows.append({"variant": variant, "mean_ratio": mean, "ci95": ci, "delta_vs_full": delta})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(paths["tables"] / "table02_ablation.csv", index=False)
    latex_rows = [
        [row["variant"], f"${row['mean_ratio']:.3f}\\pm{row['ci95']:.3f}$", f"${row['delta_vs_full']:+.3f}$"]
        for row in summary_rows
    ]
    _latex_table(
        paths["tables"] / "table02_ablation.tex",
        ["Variant", "Mean ratio", "$\\Delta$ vs.\\ full"],
        latex_rows,
    )
    return summary


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def plot_operator_spectrum(
    args: argparse.Namespace, paths: dict[str, Path], library: OperatorLibrary, graph: GraphInstance
) -> None:
    prior = operator_prior(graph, library)
    op, comm_norm, eff_rank = spectral_operator_matrix(graph, args.depth, args.rank, args.commutator_weight)
    evals = np.linalg.eigvalsh(op)[::-1]
    cov_evals = prior.eigvals
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.2))
    im = axes[0].imshow(op, cmap="viridis")
    axes[0].set_title("(a) truncated noncommutative operator")
    axes[0].set_xlabel("angle coordinate")
    axes[0].set_ylabel("angle coordinate")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    axes[1].bar(np.arange(1, len(evals) + 1), evals, color="#0f5b8f")
    axes[1].axvline(args.rank + 0.5, color="#9b3a2f", linestyle="--", linewidth=1.2, label=f"truncation $n={args.rank}$")
    axes[1].set_title("(b) operator spectrum")
    axes[1].set_xlabel("eigenvalue index")
    axes[1].set_ylabel("eigenvalue")
    axes[1].legend()
    axes[2].bar(np.arange(1, len(cov_evals) + 1), cov_evals, color="#5e63a9")
    axes[2].set_title("(c) collective search widths")
    axes[2].set_xlabel("operator direction")
    axes[2].set_ylabel("variance")
    fig.suptitle(
        f"{graph.family}, $n={graph.n}$: commutator norm $={comm_norm:.3f}$, effective rank $={eff_rank:.2f}$", y=1.04
    )
    fig.tight_layout()
    fig.savefig(paths["figures"] / "fig01_operator_spectrum.pdf")
    plt.close(fig)


def plot_benchmark(summary: pd.DataFrame, paths: dict[str, Path]) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    x = np.arange(len(summary))
    colors = [METHOD_COLORS.get(m, "#707070") for m in summary["method"]]
    ax.bar(x, summary["mean_ratio"], yerr=summary["ci95"], color=colors, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(summary["method"], rotation=20, ha="right")
    ax.set_ylabel("mean approximation ratio")
    ax.set_ylim(
        max(0.0, float(summary["mean_ratio"].min()) - 0.08),
        min(1.02, float(summary["mean_ratio"].max()) + 0.06),
    )
    ax.grid(axis="y", color="#d8d8d8", linewidth=0.8)
    for xi, v in zip(x, summary["mean_ratio"]):
        ax.text(xi, v + 0.006, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(paths["figures"] / "fig02_benchmark.pdf")
    plt.close(fig)


def plot_truncation_sweep(sweep_summary: pd.DataFrame, paths: dict[str, Path], baseline_mean: float) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    on = sweep_summary[sweep_summary["commutator"] == "on"].sort_values("rank")
    off = sweep_summary[sweep_summary["commutator"] == "off"].sort_values("rank")
    ax.errorbar(
        on["rank"], on["mean_ratio"], yerr=on["ci95"], marker="o", color="#0f5b8f",
        capsize=3, linewidth=2.0, label="noncommutative (commutator on)",
    )
    ax.errorbar(
        off["rank"], off["mean_ratio"], yerr=off["ci95"], marker="s", color="#9b3a2f",
        capsize=3, linewidth=1.6, linestyle="--", label="commutative (commutator off)",
    )
    ax.axhline(baseline_mean, color="#707070", linestyle=":", linewidth=1.4, label="TQA + coordinate baseline")
    best_idx = on["mean_ratio"].idxmax()
    best_rank = float(on.loc[best_idx, "rank"])
    best_val = float(on.loc[best_idx, "mean_ratio"])
    ax.scatter([best_rank], [best_val], s=120, facecolors="none", edgecolors="#0f5b8f", linewidths=2.0, zorder=5)
    ax.annotate(f"$n^\\star={int(best_rank)}$", (best_rank, best_val), textcoords="offset points", xytext=(6, 8))
    ax.set_xlabel("spectral truncation parameter $n$ (retained operator directions)")
    ax.set_ylabel("mean approximation ratio")
    ax.grid(color="#e0e0e0", linewidth=0.8)
    ax.legend(loc="lower center", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(paths["figures"] / "fig03_truncation_sweep.pdf")
    plt.close(fig)


def plot_query_curves(trace_df: pd.DataFrame, args: argparse.Namespace, paths: dict[str, Path]) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for method in METHOD_ORDER:
        sub = trace_df[trace_df["method"] == method]
        curves = []
        for _gid, gdf in sub.groupby("graph_id"):
            rows = gdf.sort_values("query_index").to_dict("records")
            if rows:
                curves.append(_best_so_far(rows, args.budget))
        if not curves:
            continue
        arr = np.vstack(curves)
        mean = arr.mean(axis=0)
        ax.plot(np.arange(1, args.budget + 1), mean, label=method, color=METHOD_COLORS.get(method, None), linewidth=2.0)
    ax.set_xlabel("objective queries $Q$")
    ax.set_ylabel("mean best-so-far approximation ratio")
    ax.grid(color="#e0e0e0", linewidth=0.8)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(paths["figures"] / "fig04_query_curves.pdf")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# manifest + metadata
# --------------------------------------------------------------------------- #
def write_manifest(args: argparse.Namespace, paths: dict[str, Path], counts: dict[str, int]) -> None:
    rows = [
        ["Fig.~1", "\\texttt{plot\\_operator\\_spectrum}", "\\texttt{fig01\\_operator\\_spectrum.pdf}"],
        ["Fig.~2", "\\texttt{plot\\_benchmark}", "\\texttt{fig02\\_benchmark.pdf}"],
        ["Fig.~3", "\\texttt{plot\\_truncation\\_sweep}", "\\texttt{fig03\\_truncation\\_sweep.pdf}"],
        ["Fig.~4", "\\texttt{plot\\_query\\_curves}", "\\texttt{fig04\\_query\\_curves.pdf}"],
        ["Table~1", "\\texttt{summarize\\_results}", "\\texttt{table01\\_headline.tex}"],
        ["Table~2", "\\texttt{run\\_ablation}", "\\texttt{table02\\_ablation.tex}"],
        ["Table~3", "\\texttt{run\\_truncation\\_sweep}", "\\texttt{table05\\_truncation.tex}"],
        ["Table~4", "\\texttt{summarize\\_per\\_family}", "\\texttt{table04\\_per\\_family.tex}"],
    ]
    _latex_table(paths["tables"] / "table03_manifest.tex", ["Element", "Generator", "Output"], rows)
    metadata = {
        "depth": args.depth,
        "budget": args.budget,
        "rank": args.rank,
        "commutator_weight": args.commutator_weight,
        "seed": args.seed,
        "families": args.families,
        "sizes": args.sizes,
        "train_per_family": args.train_per_family,
        "test_per_family": args.test_per_family,
        "optimizer_budget": args.optimizer_budget,
        **counts,
    }
    (paths["results"] / "reproduction_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Regenerate the OST-QAOA manuscript figures, tables, and CSV results.")
    parser.add_argument("--output-dir", type=Path, default=Path.cwd(), help="Artifact root holding figures/, tables/, results/.")
    parser.add_argument("--depth", type=int, default=3, help="QAOA depth p (operator dimension is 2p).")
    parser.add_argument("--budget", type=int, default=24, help="Matched objective-query budget.")
    parser.add_argument("--rank", type=int, default=4, help="Operating-point spectral truncation parameter n.")
    parser.add_argument("--commutator-weight", type=float, default=4.0, help="Weight of the noncommutative commutator interaction.")
    parser.add_argument("--families", default=",".join(DEFAULT_FAMILIES), help="Comma-separated graph families.")
    parser.add_argument("--sizes", default=",".join(str(x) for x in DEFAULT_SIZES), help="Comma-separated graph sizes (qubits).")
    parser.add_argument("--train-per-family", type=int, default=6, help="Training graphs per family.")
    parser.add_argument("--test-per-family", type=int, default=4, help="Held-out test graphs per family.")
    parser.add_argument("--optimizer-budget", type=int, default=42, help="Offline optimizer budget for training-library targets.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Global deterministic seed.")
    parser.add_argument("--quick", action="store_true", help="Smaller complete run for package/CI checks.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.quick:
        args.depth = min(args.depth, 2)
        args.budget = min(args.budget, 14)
        args.rank = min(args.rank, 2)
        args.train_per_family = min(args.train_per_family, 3)
        args.test_per_family = min(args.test_per_family, 2)
        args.optimizer_budget = min(args.optimizer_budget, 18)
        args.sizes = "6,8"
    paths = _ensure_dirs(args.output_dir.resolve())

    result_df, trace_df, library, graphs = run_benchmark(args, paths)
    summary = summarize_results(result_df, trace_df, args, paths)
    per_family = summarize_per_family(result_df, paths)
    ablation = run_ablation(args, paths, graphs, result_df)
    sweep, sweep_summary = run_truncation_sweep(args, paths, graphs)

    baseline_mean = float(summary.loc[summary["method"] == BASELINE_METHOD, "mean_ratio"].iloc[0])
    plot_operator_spectrum(args, paths, library, graphs[0])
    plot_benchmark(summary, paths)
    plot_truncation_sweep(sweep_summary, paths, baseline_mean)
    plot_query_curves(trace_df, args, paths)

    counts = {
        "n_test_graphs": len(graphs),
        "summary_rows": int(summary.shape[0]),
        "ablation_rows": int(ablation.shape[0]),
        "per_family_rows": int(per_family.shape[0]),
        "sweep_rows": int(sweep.shape[0]),
    }
    write_manifest(args, paths, counts)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
