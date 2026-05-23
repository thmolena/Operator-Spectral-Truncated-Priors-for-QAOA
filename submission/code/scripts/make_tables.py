#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
res=ROOT/"results"
tab=ROOT/"tables"; tab.mkdir(exist_ok=True)
summaries=[]
for csv_path in sorted(res.glob("summary_p*.csv")):
    df=pd.read_csv(csv_path)
    if not df.empty: summaries.append(df)
    p=str(df["p"].iloc[0]) if not df.empty else csv_path.stem.split("p")[-1]
    out=tab/f"table_summary_p{p}.tex"
    cols=[c for c in ["method","query_budget_Q","mean_ratio","ci95_low","ci95_high","num_instances"] if c in df]
    out.write_text(df[cols].to_latex(index=False,float_format="%.4f"),encoding="utf-8")
    print(f"wrote {out}")

if summaries:
    all_df=pd.concat(summaries, ignore_index=True)
    p3=all_df[all_df["p"]==3]
    if not p3.empty:
        cols=[c for c in ["method","n","graph_family","query_budget_Q","mean_ratio","stderr_ratio","ci95_low","ci95_high","num_instances"] if c in p3]
        out=tab/"table_main_p3.tex"; p3[cols].to_latex(out,index=False,float_format="%.4f"); print(f"wrote {out}")
    higher=all_df[all_df["p"]>3]
    if not higher.empty:
        agg=higher.groupby(["p","method","query_budget_Q"],as_index=False).agg(mean_ratio=("mean_ratio","mean"),num_rows=("mean_ratio","size"))
        out=tab/"table_higher_depth.tex"; agg.to_latex(out,index=False,float_format="%.4f"); print(f"wrote {out}")

ab_csv=next(iter(sorted(tab.glob("table_ablation_p*.csv"))), None)
if ab_csv:
    df=pd.read_csv(ab_csv)
    cols=[c for c in ["method","p","n","graph_family","query_budget_Q","mean_ratio","delta_vs_uq_full"] if c in df]
    out=tab/"table_ablation.tex"; df[cols].to_latex(out,index=False,float_format="%.4f"); print(f"wrote {out}")

fs=res/"finite_shots_p3.csv"
if fs.exists():
    df=pd.read_csv(fs)
    out=tab/"table_finite_shots.tex"; df.to_latex(out,index=False,float_format="%.4f"); print(f"wrote {out}")

val=res/"python_cpp_validation.csv"
if val.exists():
    df=pd.read_csv(val)
    cols=[c for c in ["n","p","dim","threads","python_value","cpp_value","max_abs_error","thread_invariance_error","pass"] if c in df]
    out=tab/"table_python_cpp_validation.tex"; df[cols].to_latex(out,index=False,float_format="%.3e"); print(f"wrote {out}")

hpc_dir=res/"hpc"
if hpc_dir.exists():
    files=sorted(hpc_dir.glob("*.csv"))
    if files:
        df=pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
        cols=[c for c in ["n","p","dim","batch_size","threads","backend","ms_per_query","queries_per_second","memory_bytes"] if c in df]
        out=tab/"table_hpc_benchmark.tex"; df[cols].to_latex(out,index=False,float_format="%.4f"); print(f"wrote {out}")
