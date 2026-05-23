#!/usr/bin/env python3
import subprocess, sys
subprocess.run([sys.executable, "code/scripts/run_main_experiment.py", "--smoke"], check=True)
subprocess.run([sys.executable, "code/scripts/run_higher_depth.py", "--smoke"], check=True)
subprocess.run([sys.executable, "code/scripts/run_finite_shots.py"], check=True)
subprocess.run([sys.executable, "code/scripts/make_figures.py"], check=True)
subprocess.run([sys.executable, "code/scripts/make_tables.py"], check=True)
