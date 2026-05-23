#!/usr/bin/env python3
from pathlib import Path
import subprocess, sys
ROOT=Path(__file__).resolve().parents[1]
subprocess.run([sys.executable,"-m","pytest",str(ROOT/"tests")],check=True)
