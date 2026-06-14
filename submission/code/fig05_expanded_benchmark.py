#!/usr/bin/env python3
"""
Figure 5: Higher-powered benchmark on a 6x larger held-out set
==============================================================

Reads the real outputs of ``table01_expanded.py`` (48 held-out instances,
12 per family, n=14, p=3, Q=18) and renders a two-panel publication figure:

  (a) mean approximation ratio with 95% bootstrap CIs for every method,
      ranked, with UQ-QAOA highlighted;
  (b) the per-instance paired advantage (UQ-QAOA minus TQA) across all 48
      instances, with the bootstrap-mean and its 95% CI band.

No numbers are hard-coded; the figure is a faithful rendering of the CSVs
written by the expanded benchmark run.

Output: code/figures/fig05_expanded_benchmark.pdf
"""
from __future__ import annotations

import csv
import os
import pathlib

BASE = pathlib.Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(BASE / ".mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from uq_qaoa_core import figure_path, table_path


def _load_per_method():
    rows = []
    with open(table_path("table01_expanded"), newline="") as f:
        for r in csv.DictReader(f):
            lo, hi = (float(x) for x in
                      r["CI_95"].strip("[]").split(","))
            rows.append((r["Method"], float(r["Ratio"]), lo, hi))
    return rows


def _load_per_instance():
    by_method = {}
    with open(table_path("per_instance_expanded"), newline="") as f:
        for r in csv.DictReader(f):
            by_method.setdefault(r["method"], []).append(
                (r["family"], int(r["instance"]), float(r["ratio"])))
    return by_method


def _load_paired():
    with open(table_path("paired_uq_vs_tqa_expanded"), newline="") as f:
        return next(csv.DictReader(f))


def gen_fig05():
    per_method = _load_per_method()
    per_inst = _load_per_instance()
    paired = _load_paired()

    # rank ascending by ratio
    per_method.sort(key=lambda t: t[1])
    labels = [t[0] for t in per_method]
    means = np.array([t[1] for t in per_method])
    los = np.array([t[2] for t in per_method])
    his = np.array([t[3] for t in per_method])
    err = np.vstack([means - los, his - means])
    is_uq = np.array(["ours" in lab for lab in labels])

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.2),
                                   gridspec_kw={"width_ratios": [1.15, 1.0]})

    # --- Panel (a): ranked methods with 95% CIs ---
    ypos = np.arange(len(labels))
    bar_colors = ["#1d4ed8" if u else "#9aa3b2" for u in is_uq]
    axA.barh(ypos, means, color=bar_colors, edgecolor="black", linewidth=0.4)
    axA.errorbar(means, ypos, xerr=err, fmt="none", ecolor="black",
                 elinewidth=1.0, capsize=3)
    for y, m in zip(ypos, means):
        axA.text(m + 0.012, y, f"{m:.3f}", va="center", ha="left", fontsize=9)
    axA.set_yticks(ypos)
    axA.set_yticklabels([lab.replace(" (ours)", "★") for lab in labels],
                        fontsize=10)
    axA.set_xlabel("Mean approximation ratio (95% bootstrap CI)", fontsize=12)
    axA.set_xlim(0.55, 0.92)
    axA.set_title("(a) All methods, 48 held-out instances", fontsize=12, pad=8)
    axA.spines["top"].set_visible(False)
    axA.spines["right"].set_visible(False)

    # --- Panel (b): per-instance paired advantage UQ - TQA ---
    uq = {(f, i): r for f, i, r in per_inst["uq_qaoa"]}
    tqa = {(f, i): r for f, i, r in per_inst["tqa"]}
    keys = sorted(uq.keys())
    diffs = np.array([uq[k] - tqa[k] for k in keys])
    order = np.argsort(diffs)
    diffs_sorted = diffs[order]
    xpos = np.arange(len(diffs_sorted))
    colors = ["#1d4ed8" if d > 0 else "#d1495b" for d in diffs_sorted]
    axB.bar(xpos, diffs_sorted, color=colors, width=0.9)
    adv = float(paired["paired_advantage"])
    clo = float(paired["paired_CI95_lo"])
    chi = float(paired["paired_CI95_hi"])
    axB.axhline(0, color="black", linewidth=0.8)
    axB.axhline(adv, color="#1d4ed8", linestyle="--", linewidth=1.2,
                label=f"mean {adv:+.4f}")
    axB.axhspan(clo, chi, color="#1d4ed8", alpha=0.12,
                label=f"95% CI [{clo:+.4f}, {chi:+.4f}]")
    wins = int(paired["wins"]); losses = int(paired["losses"])
    axB.set_xlabel(f"Held-out instance (sorted)  —  "
                   f"{wins} wins / {losses} losses", fontsize=12)
    axB.set_ylabel("Paired advantage  UQ-QAOA $-$ TQA", fontsize=12)
    axB.set_title("(b) Per-instance paired advantage", fontsize=12, pad=8)
    axB.legend(fontsize=10, loc="upper left", framealpha=0.9, edgecolor="0.7")
    axB.spines["top"].set_visible(False)
    axB.spines["right"].set_visible(False)

    fig.tight_layout()
    path = figure_path("fig05_expanded_benchmark")
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  [done] {path.name}")


if __name__ == "__main__":
    print("Figure 5: expanded 48-instance benchmark")
    gen_fig05()
