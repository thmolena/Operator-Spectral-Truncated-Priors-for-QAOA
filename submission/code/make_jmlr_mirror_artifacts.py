"""Generate the display items for the JMLR-mirror version of the manuscript.

The manuscript mirrors the section/figure/table structure of
Hashimoto et al., "Spectral Truncation Kernels" (JMLR; arXiv:2405.17823):
seven main-body figures and two main-body tables.  Three figures are reused
unchanged from the package output (operator construction, truncation sweep,
query curves); the four built here are produced from the same result CSVs the
package already wrote, so every display item remains regenerable from data.

Run:  python make_jmlr_mirror_artifacts.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 200,
    }
)

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
TAB = os.path.join(HERE, "tables")
RES = os.path.join(HERE, "results")
os.makedirs(FIG, exist_ok=True)

ACCENT = "#1A4E8A"
GREY = "#7a7a7a"
ORANGE = "#D1670B"

# ----------------------------------------------------------------------------
# Figure 2 (corresponds to STK Fig. 2 "Fejer kernel"): convergence of the
# truncated angle operator to the full operator, measured by the effective
# search dimension as the truncation parameter n grows.
# ----------------------------------------------------------------------------
def figure_convergence():
    df = pd.read_csv(os.path.join(TAB, "table05_truncation.csv"))
    g = df.groupby("rank", as_index=False)["mean_effective_dimension"].mean()
    n = g["rank"].to_numpy()
    deff = g["mean_effective_dimension"].to_numpy()
    fig, ax = plt.subplots(figsize=(4.0, 2.9))
    ax.plot(n, deff, "o-", color=ACCENT, lw=1.8, ms=5, label=r"$d_{\mathrm{eff}}(\Sigma_G^{(n)})$")
    ax.plot(n, n, "--", color=GREY, lw=1.0, label=r"$d_{\mathrm{eff}}=n$ (no concentration)")
    nstar = 4
    ax.axvline(nstar, color=ORANGE, lw=1.0, ls=":")
    ax.annotate(r"$n^{\star}$", xy=(nstar, deff[nstar - 1]), xytext=(nstar + 0.25, deff[nstar - 1] - 0.45),
                color=ORANGE, fontsize=9)
    ax.set_xlabel(r"truncation parameter $n$")
    ax.set_ylabel(r"effective search dimension $d_{\mathrm{eff}}$")
    ax.set_xticks(n)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_jmlr_convergence.pdf"))
    plt.close(fig)
    print("wrote fig_jmlr_convergence.pdf")


# ----------------------------------------------------------------------------
# Figure 4 (corresponds to STK Fig. 4, multi-panel (a)(b)(c)): the second
# experiment -- robustness of the angle-search task across independent seeds,
# with (a) approximation ratio, (b) queries-to-target, (c) per-family gain.
# ----------------------------------------------------------------------------
def figure_robustness():
    ms = pd.read_csv(os.path.join(RES, "multiseed_results.csv"))
    methods = ["Random", "TQA", "TQA+coordinate", "kNN+coordinate", "OST diagonal", "OST-QAOA"]
    short = ["Rand", "TQA", "TQA+c", "kNN+c", "OST-d", "OST"]
    colors = [GREY, "#b0b0b0", "#8aa0bf", "#5f7fae", "#3a6098", ACCENT]
    fam = pd.read_csv(os.path.join(TAB, "table04_per_family.csv"))

    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.7))
    # (a) ratio across seeds
    data_r = [ms[f"{m}__mean"].to_numpy() for m in methods]
    bp = axes[0].boxplot(data_r, patch_artist=True, widths=0.6, showfliers=False)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.85); patch.set_edgecolor("k"); patch.set_linewidth(0.6)
    for med in bp["medians"]:
        med.set_color("k")
    axes[0].set_xticks(range(1, len(short) + 1)); axes[0].set_xticklabels(short, rotation=40, ha="right")
    axes[0].set_ylabel("approximation ratio"); axes[0].set_title("(a) ratio across seeds")
    # (b) queries-to-target across seeds
    data_q = [ms[f"{m}__qtt"].to_numpy() for m in methods]
    bp2 = axes[1].boxplot(data_q, patch_artist=True, widths=0.6, showfliers=False)
    for patch, c in zip(bp2["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.85); patch.set_edgecolor("k"); patch.set_linewidth(0.6)
    for med in bp2["medians"]:
        med.set_color("k")
    axes[1].set_xticks(range(1, len(short) + 1)); axes[1].set_xticklabels(short, rotation=40, ha="right")
    axes[1].set_ylabel(r"$\bar{Q}_{0.98}$ (queries)"); axes[1].set_title("(b) queries to target")
    # (c) per-family paired gain of OST-QAOA over TQA+coordinate
    famnames = {"er": "ER", "random_regular": "Reg", "watts_strogatz": "WS", "barabasi_albert": "BA"}
    labels = [famnames.get(f, f) for f in fam["family"]]
    axes[2].bar(range(len(fam)), fam["delta"].to_numpy(), color=ACCENT, alpha=0.85, width=0.6)
    axes[2].axhline(0, color="k", lw=0.6)
    axes[2].set_xticks(range(len(fam))); axes[2].set_xticklabels(labels)
    axes[2].set_ylabel(r"$\Delta$ ratio vs.\ TQA+coord."); axes[2].set_title("(c) per-family gain")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_jmlr_robustness.pdf"))
    plt.close(fig)
    print("wrote fig_jmlr_robustness.pdf")


# ----------------------------------------------------------------------------
# Figure 5 (corresponds to STK Fig. 5 "output images" grid): qualitative
# per-instance output of the angle-search task -- approximation ratio of every
# method on each of the 16 held-out graphs.
# ----------------------------------------------------------------------------
def figure_per_instance():
    o = pd.read_csv(os.path.join(RES, "operator_spectral_results.csv"))
    methods = ["Random", "TQA", "TQA+coordinate", "kNN+coordinate", "OST diagonal", "OST-QAOA"]
    fam_order = ["er", "random_regular", "watts_strogatz", "barabasi_albert"]
    o = o.sort_values(["family", "graph_id"], key=lambda s: s.map({f: i for i, f in enumerate(fam_order)}) if s.name == "family" else s)
    graphs = list(dict.fromkeys(o["graph_id"]))
    M = np.full((len(methods), len(graphs)), np.nan)
    for i, m in enumerate(methods):
        sub = o[o["method"] == m].set_index("graph_id")["ratio"]
        for j, gid in enumerate(graphs):
            if gid in sub.index:
                M[i, j] = float(sub.loc[gid])
    fig, ax = plt.subplots(figsize=(7.4, 2.6))
    im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=np.nanmin(M), vmax=1.0)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(["Random", "TQA", "TQA+coord.", "kNN+coord.", "OST diagonal", "OST-QAOA"])
    # family separators / labels on x
    fam_of = [o[o["graph_id"] == g]["family"].iloc[0] for g in graphs]
    ax.set_xticks(range(len(graphs)))
    ax.set_xticklabels([{"er": "ER", "random_regular": "Reg", "watts_strogatz": "WS", "barabasi_albert": "BA"}[f] for f in fam_of],
                       rotation=0, fontsize=6)
    ax.set_xlabel("held-out graph instance (grouped by family)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label("approximation ratio")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_jmlr_per_instance.pdf"))
    plt.close(fig)
    print("wrote fig_jmlr_per_instance.pdf")


# ----------------------------------------------------------------------------
# Figure 7 (corresponds to STK Fig. 7 "pointwise error"): pointwise optimality
# gap 1 - best-so-far ratio of the incumbent angle as a function of the query
# index, for the proposed policy and the baselines.
# ----------------------------------------------------------------------------
def figure_query_gap():
    t = pd.read_csv(os.path.join(TAB, "trace_all_evaluations.csv"))
    # The query traces use the package's per-evaluation method labels; map only
    # the policies that correspond unambiguously to the manuscript baselines.
    name = {"uq_qaoa": "OST-QAOA", "tqa": "TQA", "random": "Random"}
    colors = {"OST-QAOA": ACCENT, "TQA": "#b0b0b0", "Random": GREY}
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for raw, disp in name.items():
        sub = t[t["method"] == raw]
        if sub.empty:
            continue
        piv = sub.pivot_table(index="query_index", values="best_so_far", aggfunc="mean")
        gap = 1.0 - piv["best_so_far"]
        ax.plot(gap.index.to_numpy(), gap.to_numpy(), "-o", ms=3, lw=1.5, color=colors[disp], label=disp)
    ax.set_xlabel("objective-query index")
    ax.set_ylabel(r"pointwise optimality gap $1-r_G(\theta)$")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_jmlr_query_gap.pdf"))
    plt.close(fig)
    print("wrote fig_jmlr_query_gap.pdf")


# ----------------------------------------------------------------------------
# Table 1 (corresponds to STK Table 1 "Summary of the existing and the proposed
# kernels"): qualitative summary of the existing and proposed angle-search
# priors along the properties that distinguish them.
# ----------------------------------------------------------------------------
def table_prior_summary():
    rows = [
        ("Random",            r"---",             r"\xmark", r"\xmark", r"\xmark"),
        ("TQA",               r"fixed schedule",  r"\xmark", r"\xmark", r"\xmark"),
        ("TQA+coordinate",    r"fixed schedule",  r"\xmark", r"\xmark", r"\cmark"),
        ("kNN+coordinate",    r"graph features",  r"\xmark", r"\xmark", r"\cmark"),
        ("OST diagonal",      r"$\mathcal{O}_{G,r}$ (diagonal)", r"\cmark", r"\xmark", r"\cmark"),
        (r"\OSTQAOA{} (proposed)", r"$\mathcal{O}_{G,r}$ (full)", r"\cmark", r"\cmark", r"\cmark"),
    ]
    lines = [
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Prior / policy & Graph operator & Noncommuting & Off-diagonal & Query-",
        r" & used & generators & search & ranked \\",
        r"\midrule",
    ]
    for name, op, nc, od, qr in rows:
        lines.append(f"{name} & {op} & {nc} & {od} & {qr} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out = os.path.join(TAB, "table_prior_summary.tex")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote table_prior_summary.tex")


if __name__ == "__main__":
    figure_convergence()
    figure_robustness()
    figure_per_instance()
    figure_query_gap()
    table_prior_summary()
    print("done.")
