#!/usr/bin/env python3
from run_all_small import run_config
from pathlib import Path
import argparse

ROOT = Path(__file__).resolve().parents[1]
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "p3_main.yaml"))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    run_config(args.config, args.smoke)
