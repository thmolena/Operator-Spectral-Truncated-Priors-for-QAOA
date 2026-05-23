from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def plot_query_curve(summary_csv: str | Path, out_pdf: str | Path) -> None:
    df = pd.read_csv(summary_csv)
    plt.figure(figsize=(6, 4))
    for method, g in df.groupby("method"):
        h = g.groupby("query_budget_Q")["mean_ratio"].mean().reset_index()
        plt.plot(h["query_budget_Q"], h["mean_ratio"], label=method)
    plt.xlabel("query budget Q")
    plt.ylabel("mean approximation ratio")
    plt.legend(fontsize=7)
    plt.tight_layout()
    Path(out_pdf).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_pdf)
    plt.close()
