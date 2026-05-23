from __future__ import annotations

from pathlib import Path
import pandas as pd


def summary_to_latex(summary_csv: str | Path, out_tex: str | Path) -> None:
    df = pd.read_csv(summary_csv)
    cols = [c for c in ["method", "query_budget_Q", "mean_ratio", "ci95_low", "ci95_high", "num_instances"] if c in df.columns]
    Path(out_tex).parent.mkdir(parents=True, exist_ok=True)
    Path(out_tex).write_text(df[cols].to_latex(index=False, float_format="%.4f"), encoding="utf-8")
