#!/usr/bin/env python3
from pathlib import Path
import argparse, csv, subprocess, sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"python"))
from uq_qaoa.graphs import generate_graph, save_edge_list
from uq_qaoa.maxcut import all_cut_values
from uq_qaoa.qaoa_angles import random_theta
from uq_qaoa.statevector import qaoa_expectation

if __name__=="__main__":
    out=ROOT/"results"/"python_cpp_validation.csv"; out.parent.mkdir(exist_ok=True)
    exe=ROOT/"cpp"/"build"/"eval_qaoa_batch"
    rows=[]
    for n in [8,10,12,14]:
      for p in [3,6,7]:
        g=generate_graph("er",n,123+p+n); gp=ROOT/"results"/f"graph_n{n}_p{p}.csv"; save_edge_list(g,gp)
        th=random_theta(p,np.random.default_rng(1000+n+p),"blocked"); py=qaoa_expectation(n,g.edges,p,th,cost_values=all_cut_values(n,g.edges))
        thread_vals=[]
        if exe.exists():
          for threads in [1,2,4,8]:
            cmd=[str(exe),"--n",str(n),"--p",str(p),"--graph",str(gp),"--theta",",".join(map(str,th)),"--threads",str(threads)]
            cp=subprocess.run(cmd,text=True,capture_output=True,check=True); thread_vals.append(float(cp.stdout.strip().splitlines()[-1]))
          cpp=thread_vals[0]
        else: cpp=float("nan")
        err=abs(py-cpp) if np.isfinite(cpp) else float("nan")
        terr=max(abs(v-cpp) for v in thread_vals) if thread_vals else float("nan")
        rows.append({"n":n,"p":p,"python":py,"cpp":cpp,"max_abs_error":err,"thread_invariance_error":terr,"pass": bool(err<1e-8 and terr<1e-12) if np.isfinite(err) else False})
    with out.open("w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print(f"wrote {out}")
