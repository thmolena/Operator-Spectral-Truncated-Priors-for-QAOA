#!/usr/bin/env python3
from run_all_small import run_config
from pathlib import Path
import argparse

ROOT = Path(__file__).resolve().parents[1]
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="*", default=[str(ROOT / "configs" / f"p{p}_validation.yaml") for p in (4,5,6)] + [str(ROOT / "configs" / "p7_hpc_stress.yaml")])
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    for cfg in args.configs:
        run_config(cfg, args.smoke)
