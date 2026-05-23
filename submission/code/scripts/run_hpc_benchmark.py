#!/usr/bin/env python3
from pathlib import Path
import argparse, subprocess, sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"python"))
from uq_qaoa.config import load_config
from uq_qaoa.graphs import generate_graph, save_edge_list

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default=str(ROOT/"configs"/"p7_hpc_stress.yaml")); ap.add_argument("--smoke",action="store_true"); args=ap.parse_args()
    cfg=load_config(args.config); exe=ROOT/"cpp"/"build"/"bench_qaoa"; outdir=ROOT/"results"/"hpc"; outdir.mkdir(parents=True,exist_ok=True)
    if not exe.exists(): raise SystemExit("build C++ backend first")
    ns=cfg.hpc.benchmark_ns[:1] if args.smoke else cfg.hpc.benchmark_ns; ps=cfg.hpc.benchmark_ps[:1] if args.smoke else cfg.hpc.benchmark_ps
    for n in ns:
      g=generate_graph("er",n,42+n); gp=outdir/f"graph_n{n}.csv"; save_edge_list(g,gp)
      for p in ps:
        for batch in (cfg.hpc.batch_sizes[:1] if args.smoke else cfg.hpc.batch_sizes):
          for th in (cfg.hpc.thread_counts[:1] if args.smoke else cfg.hpc.thread_counts):
            out=outdir/f"bench_n{n}_p{p}_b{batch}_t{th}.csv"
            subprocess.run([str(exe),"--n",str(n),"--p",str(p),"--batch",str(batch),"--threads",str(th),"--graph",str(gp),"--out",str(out)],check=True)
            print(f"wrote {out}")
